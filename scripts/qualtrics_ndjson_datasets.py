#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate 10 brand-new synthetic WHS claims-experience datasets as Qualtrics
NDJSON response exports, replay them through the redaction module, and review results.

Each dataset is a distinct "wave" with its own seed, cohort profile (sentiment,
abandonment, identifier density, language mix) and size. The script:

1. generates the responses (WHS claims instrument, embedded claims data, planted
   identifiers) and writes a Qualtrics-style **NDJSON export** — one API-shaped
   response object per line — plus a ground-truth sidecar;
2. reads each file back through the Qualtrics NDJSON reader and verifies the
   round trip;
3. replays each response through the hybrid redaction module (cross-field
   propagation, schema-governed embedded data, second pass on queued responses);
4. grades identifier recall, character-level recall, escape rates,
   over-redaction, review load, idempotence and the acceptance gates, per
   dataset and cumulatively.

Examples
--------
    python3 scripts/qualtrics_ndjson_datasets.py
    python3 scripts/qualtrics_ndjson_datasets.py --limit 3 --out /tmp/ndjson
    python3 scripts/qualtrics_ndjson_datasets.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from qualtrics_batch_check import BatchChecker, BatchMetrics, _gates, _merge_metrics
from run_settings_matrix import Palette, bar, render_table, section

from gov_survey_redact import AcceptanceGates
from gov_survey_redact.qualtrics import read_ndjson, text_answers, write_ndjson
from gov_survey_redact.whs import (
    WHS_CLAIMS_INSTRUMENT,
    WHSBatch,
    WHSBatchGenerator,
    WHSGroundTruth,
)

DEFAULT_OUT = ROOT / "data" / "qualtrics_ndjson"
DEFAULT_SALT = b"whs-claims-qualtrics-2026"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One synthetic NDJSON dataset (a distinct survey "wave")."""

    name: str
    description: str
    seed: int
    responses: int
    negative_rate: float = 0.30
    partial_rate: float = 0.10
    pii_rate: float = 0.75
    language_weights: tuple[float, float, float, float] = (80.0, 8.0, 6.0, 6.0)


DEFAULT_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "wave2026q3_national_baseline",
        "Q3 2026 national baseline wave, English-dominant, standard identifier density",
        58001,
        96,
    ),
    DatasetSpec(
        "wave2026q3_settled_claims",
        "Q3 2026 settled claims with mostly positive experiences and few typed identifiers",
        58002,
        72,
        negative_rate=0.15,
        partial_rate=0.05,
        pii_rate=0.55,
    ),
    DatasetSpec(
        "wave2026q3_escalated_disputes",
        "Q3 2026 escalated disputes: negative sentiment, dense identifying detail",
        58003,
        68,
        negative_rate=0.55,
        pii_rate=0.85,
    ),
    DatasetSpec(
        "wave2026q3_self_insured",
        "Q3 2026 self-insured employer cohort, very high identifier density",
        58004,
        64,
        negative_rate=0.25,
        pii_rate=0.95,
    ),
    DatasetSpec(
        "wave2026q4_multilingual",
        "Q4 2026 bilingual wave (Vietnamese and Arabic responses included)",
        58005,
        60,
        language_weights=(35.0, 10.0, 35.0, 20.0),
    ),
    DatasetSpec(
        "wave2026q4_abandonment_heavy",
        "Q4 2026 wave with a high share of abandoned/partial responses",
        58006,
        80,
        negative_rate=0.30,
        partial_rate=0.40,
    ),
    DatasetSpec(
        "wave2026q4_contact_optin",
        "Q4 2026 callback-consent-heavy wave: contact block is often filled in",
        58007,
        60,
        pii_rate=0.95,
        partial_rate=0.05,
    ),
    DatasetSpec(
        "wave2025q2_long_tail_archive",
        "Legacy 2025 archive wave, largest dataset and longer narratives",
        58008,
        120,
        negative_rate=0.25,
        pii_rate=0.80,
    ),
    DatasetSpec(
        "wave2026q4_deidentified_pilot",
        "Q4 2026 pilot where respondents rarely typed identifying details",
        58009,
        50,
        negative_rate=0.20,
        pii_rate=0.15,
    ),
    DatasetSpec(
        "wave2026q4_sparse_feedback",
        "Q4 2026 sparse-feedback wave: shorter answers, mixed sentiment",
        58010,
        48,
        negative_rate=0.35,
        partial_rate=0.20,
        pii_rate=0.60,
    ),
)


def generate_dataset(
    spec: DatasetSpec,
    out_dir: Path,
    *,
    batch_index: int,
) -> tuple[WHSBatch, list[str]]:
    """Generate one dataset and write its NDJSON export plus truth sidecar."""
    generator = WHSBatchGenerator(
        seed=spec.seed,
        instrument=WHS_CLAIMS_INSTRUMENT,
        negative_rate=spec.negative_rate,
        pii_rate=spec.pii_rate,
        partial_rate=spec.partial_rate,
        language_weights=spec.language_weights,
    )
    batch = generator.batch(spec.responses, batch_index=batch_index)
    ndjson_path = write_ndjson(
        out_dir / f"{spec.name}.ndjson", batch.responses, WHS_CLAIMS_INSTRUMENT
    )
    truth_path = out_dir / f"{spec.name}.ground_truth.json"
    truth_path.write_text(
        json.dumps(
            [batch.ground_truth[r.response_id].as_dict() for r in batch.responses],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return batch, [str(ndjson_path), str(truth_path)]


def load_dataset(spec: DatasetSpec, out_dir: Path, *, batch_index: int) -> WHSBatch:
    """Read a dataset back from its NDJSON export and truth sidecar."""
    responses = read_ndjson(out_dir / f"{spec.name}.ndjson")
    truth_entries = json.loads(
        (out_dir / f"{spec.name}.ground_truth.json").read_text(encoding="utf-8")
    )
    ground_truth = {}
    for entry in truth_entries:
        truth = WHSGroundTruth.from_dict(entry)
        ground_truth[truth.response_id] = truth
    return WHSBatch(batch_index=batch_index, responses=responses, ground_truth=ground_truth)


def verify_round_trip(batch: WHSBatch, reloaded: WHSBatch) -> int:
    """Count mismatches between the in-memory batch and the NDJSON round trip."""
    failures = 0
    if len(batch.responses) != len(reloaded.responses):
        return 1
    for original, restored in zip(batch.responses, reloaded.responses, strict=True):
        if original.response_id != restored.response_id:
            failures += 1
        if text_answers(original, WHS_CLAIMS_INSTRUMENT) != text_answers(
            restored, WHS_CLAIMS_INSTRUMENT
        ):
            failures += 1
        if original.values != restored.values:
            failures += 1
    return failures


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _build_report(
    palette: Palette,
    args: argparse.Namespace,
    specs: list[DatasetSpec],
    metrics_by_dataset: list[BatchMetrics],
    combined: BatchMetrics,
    file_sizes_kb: list[float],
    total_elapsed: float,
) -> str:
    rules = "━" * 78
    lines: list[str] = []
    lines.append(palette.bold(rules))
    lines.append(palette.bold(" WHS CLAIMS EXPERIENCE · 10 NEW QUALTRICS NDJSON DATASETS — REDACTION REVIEW"))
    lines.append(palette.bold(rules))
    lines.append(f" instrument     {WHS_CLAIMS_INSTRUMENT.name} ({WHS_CLAIMS_INSTRUMENT.survey_id})")
    lines.append(
        f" datasets       {len(specs)} NDJSON exports · {combined.responses} responses"
        f" ({combined.partials} partial) · {combined.text_fields} free-text fields"
    )
    lines.append(f" output         {args.out}")
    lines.append(
        " pipeline       survey profile · second pass on · schema-governed embedded data · "
        "IP metadata governed"
    )
    lines.append(
        f" runtime        {total_elapsed:.1f}s total · "
        f"{combined.responses / max(total_elapsed, 1e-6):.1f} responses/s"
    )

    lines.append(section(palette, 1, "DATASETS — per-wave effectiveness"))
    rows = []
    for spec, metrics, size_kb in zip(specs, metrics_by_dataset, file_sizes_kb, strict=True):
        rows.append(
            [
                spec.name,
                str(metrics.responses),
                str(metrics.partials),
                str(metrics.text_fields),
                f"{metrics.caught}/{metrics.planted}",
                f"{metrics.mandatory_recall * 100:.1f}%",
                f"{metrics.char_recall * 100:.1f}%",
                str(metrics.escaped_first),
                str(metrics.escaped_final),
                str(metrics.must_survive_lost),
                f"{metrics.review_rate * 100:.0f}%",
                f"{size_kb:.0f}",
                f"{metrics.elapsed_s * 1000 / max(metrics.responses, 1):.2f}",
            ]
        )
    rows.append(
        [
            palette.bold("Σ ALL"),
            str(combined.responses),
            str(combined.partials),
            str(combined.text_fields),
            f"{combined.caught}/{combined.planted}",
            f"{combined.mandatory_recall * 100:.1f}%",
            f"{combined.char_recall * 100:.1f}%",
            str(combined.escaped_first),
            str(combined.escaped_final),
            str(combined.must_survive_lost),
            f"{combined.review_rate * 100:.0f}%",
            f"{sum(file_sizes_kb):.0f}",
            f"{combined.elapsed_s * 1000 / max(combined.responses, 1):.2f}",
        ]
    )

    def paint(ri: int, ci: int, plain: str, padded: str) -> str:
        if ri == len(rows) - 1:
            return palette.bold(padded)
        if ci in (7, 8, 9):
            return palette.count(int(plain), padded)
        if ci == 10 and float(plain.rstrip("%")) > 85:
            return palette.warn(padded)
        return padded

    lines.append(
        render_table(
            [
                "dataset",
                "resp",
                "part",
                "fields",
                "caught",
                "mandatory",
                "chars",
                "esc1",
                "esc2",
                "over",
                "review",
                "KB",
                "ms/resp",
            ],
            rows,
            aligns=["l"] + ["r"] * 12,
            paint=paint,
        )
    )
    lines.append(
        palette.dim(
            "  esc1 = responses with a missed mandatory identifier on pass 1 · "
            "esc2 = still missing after second pass"
        )
    )

    lines.append(section(palette, 2, "ACCEPTANCE GATES — cumulative across all 10 datasets"))
    passed, gate_lines = _gates(palette, combined, AcceptanceGates())
    lines.extend(gate_lines)
    lines.append("")
    lines.append(
        "  "
        + (
            palette.good("VERDICT: PASS — every wave is safe to release after the review flow")
            if passed
            else palette.bad("VERDICT: FAIL — investigate the failing gates above")
        )
    )
    lines.append(
        f"  embedded claims data governed: {combined.embedded_caught}/{combined.embedded_planted}"
        f"   · IP metadata masked: {combined.metadata_governed}/{combined.metadata_checked}"
        f"   · idempotence failures: {combined.idempotence_failures}"
        f"   · round-trip failures: {combined.round_trip_failures}"
    )

    lines.append(section(palette, 3, "PER-CLASS IDENTIFIER RECALL (all datasets)"))
    recall = {
        k: combined.per_class_caught.get(k, 0) / v
        for k, v in sorted(combined.per_class_planted.items())
        if v
    }
    for name, value in sorted(recall.items(), key=lambda kv: kv[1]):
        lines.append(
            f"    {name:<22} {palette.score(value * 100.0, bar(value, 26))} {value * 100:5.1f}%"
            f"  {palette.dim(f'n={combined.per_class_planted[name]}')}"
        )

    section_no = 3
    if combined.misses or combined.over_examples:
        section_no += 1
        lines.append(section(palette, section_no, "TRIAGE — surviving identifiers and over-redaction"))
        for miss in combined.misses[:12]:
            lines.append(f"    {palette.warn('!')} {miss}")
        for example in combined.over_examples[:12]:
            lines.append(f"    {palette.info('~')} {example}")

    section_no += 1
    lines.append(section(palette, section_no, "ARTIFACTS"))
    lines.append(
        f"    {len(specs)} NDJSON exports + ground-truth sidecars + instrument.json + manifest.json"
        f" written under {args.out}"
    )
    for spec in specs[:3]:
        lines.append(palette.dim(f"      {spec.name}.ndjson"))
    if len(specs) > 3:
        lines.append(palette.dim("      …"))

    notes: list[str] = []
    if combined.partials:
        notes.append(
            f"{combined.partials} partial responses exercised missing-answer handling end to end"
        )
    if not combined.round_trip_failures:
        notes.append(palette.good("every NDJSON export round-tripped through the Qualtrics reader"))
    if combined.review_rate > 0.5:
        notes.append(
            palette.warn(
                f"review load is {combined.review_rate * 100:.0f}% with the untrained gazetteer "
                f"backend; trained encoder weights will lower confidence-driven routing"
            )
        )
    if notes:
        section_no += 1
        lines.append(section(palette, section_no, "NOTES"))
        for note in notes:
            lines.append(f"    {note}")

    lines.append("")
    lines.append(palette.dim(rules))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qualtrics_ndjson_datasets.py",
        description=(
            "Generate 10 new synthetic WHS claims-experience datasets as Qualtrics NDJSON "
            "exports, replay them through the redaction module, and review effectiveness."
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N datasets")
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--quiet", action="store_true", help="hide progress output")
    parser.add_argument("--json", action="store_true", help="write effectiveness_report.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    palette = Palette(
        enabled=args.color == "always" or (args.color == "auto" and sys.stdout.isatty())
    )
    specs = list(DEFAULT_SPECS)
    if args.limit > 0:
        specs = specs[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "instrument.json").write_text(
        json.dumps(WHS_CLAIMS_INSTRUMENT.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    checker = BatchChecker(WHS_CLAIMS_INSTRUMENT, salt=DEFAULT_SALT)
    metrics_by_dataset: list[BatchMetrics] = []
    sizes_kb: list[float] = []
    manifest: dict[str, object] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "instrument": WHS_CLAIMS_INSTRUMENT.survey_id,
        "format": "qualtrics-ndjson",
        "datasets": [],
    }
    started = time.perf_counter()
    for index, spec in enumerate(specs, start=1):
        if not args.quiet:
            sys.stderr.write(
                f"\r  generating+checking {index}/{len(specs)} · {spec.name:<40}"
            )
            sys.stderr.flush()
        dataset_started = time.perf_counter()
        batch, files = generate_dataset(spec, out_dir, batch_index=index)
        reloaded = load_dataset(spec, out_dir, batch_index=index)
        round_trip = verify_round_trip(batch, reloaded)
        metrics = checker.evaluate(reloaded)
        metrics.elapsed_s = time.perf_counter() - dataset_started
        metrics.round_trip_failures = round_trip
        metrics.files = [Path(f).name for f in files]
        metrics_by_dataset.append(metrics)
        sizes_kb.append((out_dir / f"{spec.name}.ndjson").stat().st_size / 1024.0)
        manifest["datasets"].append(  # type: ignore[union-attr]
            {
                "name": spec.name,
                "description": spec.description,
                "seed": spec.seed,
                "responses": spec.responses,
                "negative_rate": spec.negative_rate,
                "partial_rate": spec.partial_rate,
                "pii_rate": spec.pii_rate,
                "metrics": metrics.as_dict(),
            }
        )
    total_elapsed = time.perf_counter() - started
    if not args.quiet:
        sys.stderr.write("\r" + " " * 76 + "\r")
        sys.stderr.flush()

    combined = _merge_metrics(metrics_by_dataset)
    manifest["responses_total"] = combined.responses
    manifest["effectiveness"] = combined.as_dict()
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    report = _build_report(
        palette, args, specs, metrics_by_dataset, combined, sizes_kb, total_elapsed
    )
    sys.stdout.write(report + "\n")

    if args.json:
        payload = {
            "generated": manifest["generated"],
            "instrument": WHS_CLAIMS_INSTRUMENT.survey_id,
            "datasets": [m.as_dict() for m in metrics_by_dataset],
            "cumulative": combined.as_dict(),
        }
        (out_dir / "effectiveness_report.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        sys.stderr.write(f"wrote {out_dir / 'effectiveness_report.json'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
