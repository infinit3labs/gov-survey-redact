#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate, "submit", and check batches of Qualtrics WHS claims-experience responses.

Simulates a survey programme submitting successive batches of responses exported
from Qualtrics, then measures how effective the redaction pipeline is on each
batch and cumulatively:

1. **Generate** — reproducible batches of Qualtrics-shaped responses (WHS claims
   experience instrument) with planted ground-truth identifiers, partial
   responses, and embedded claims-system data.
2. **Submit** — write each batch in the Qualtrics formats requested (API JSON,
   wide CSV, three-line-header TSV) plus a ground-truth sidecar, and validate the
   files round-trip back through the readers.
3. **Redact** — run the hybrid pipeline over open-text answers (with cross-field
   propagation from embedded data), govern embedded/metadata columns through
   schema policy (claim numbers tokenised, contact details labelled, IP addresses
   masked), and run the second pass on queued responses.
4. **Check effectiveness** — identifier recall, character-level recall, response
   escape rate, over-redaction, review load, idempotence, and the acceptance
   gates, per batch and cumulatively.

Examples
--------
    python3 scripts/qualtrics_batch_check.py
    python3 scripts/qualtrics_batch_check.py --batches 8 --per-batch 50 --seed 7
    python3 scripts/qualtrics_batch_check.py --out data/qualtrics_whs_batches --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from run_settings_matrix import Palette, bar, render_table, section

from gov_survey_redact import (
    AcceptanceGates,
    Audience,
    ReviewState,
    RiskBand,
    SurveyRedactor,
    SurveySchema,
    survey_redactor,
)
from gov_survey_redact.qualtrics import (
    QualtricsInstrument,
    QualtricsResponse,
    embedded_values,
    read_api_json,
    read_csv,
    read_ndjson,
    read_tsv,
    text_answers,
)
from gov_survey_redact.whs import (
    WHS_CLAIMS_INSTRUMENT,
    WHSBatch,
    WHSBatchGenerator,
    write_batch_files,
)

DEFAULT_SALT = b"whs-claims-qualtrics-2026"
DEFAULT_OUT = ROOT / "data" / "qualtrics_whs_batches"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class BatchMetrics:
    """Effectiveness metrics for one submitted batch."""

    batch_index: int
    responses: int = 0
    partials: int = 0
    text_fields: int = 0
    # Text-answer identifiers (graded on the released text)
    planted: int = 0
    caught: int = 0
    mandatory_planted: int = 0
    mandatory_caught: int = 0
    chars_planted: int = 0
    chars_redacted: int = 0
    # Embedded / metadata governance
    embedded_planted: int = 0
    embedded_caught: int = 0
    metadata_checked: int = 0
    metadata_governed: int = 0
    # Utility
    must_survive_total: int = 0
    must_survive_lost: int = 0
    # Routing / safety
    queued: int = 0
    escaped_first: int = 0
    escaped_final: int = 0
    idempotence_failures: int = 0
    round_trip_failures: int = 0
    elapsed_s: float = 0.0
    per_class_planted: dict[str, int] = field(default_factory=dict)
    per_class_caught: dict[str, int] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)
    over_examples: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.caught / self.planted if self.planted else 1.0

    @property
    def mandatory_recall(self) -> float:
        return self.mandatory_caught / self.mandatory_planted if self.mandatory_planted else 1.0

    @property
    def char_recall(self) -> float:
        return self.chars_redacted / self.chars_planted if self.chars_planted else 1.0

    @property
    def escape_rate(self) -> float:
        return self.escaped_first / self.responses if self.responses else 0.0

    @property
    def post_review_escape_rate(self) -> float:
        return self.escaped_final / self.responses if self.responses else 0.0

    @property
    def over_rate(self) -> float:
        return (
            self.must_survive_lost / self.must_survive_total
            if self.must_survive_total
            else 0.0
        )

    @property
    def review_rate(self) -> float:
        return self.queued / self.responses if self.responses else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "batch_index": self.batch_index,
            "responses": self.responses,
            "partials": self.partials,
            "text_fields": self.text_fields,
            "planted": self.planted,
            "caught": self.caught,
            "recall": round(self.recall, 5),
            "mandatory_planted": self.mandatory_planted,
            "mandatory_caught": self.mandatory_caught,
            "mandatory_recall": round(self.mandatory_recall, 5),
            "char_recall": round(self.char_recall, 5),
            "embedded_planted": self.embedded_planted,
            "embedded_caught": self.embedded_caught,
            "metadata_checked": self.metadata_checked,
            "metadata_governed": self.metadata_governed,
            "must_survive_total": self.must_survive_total,
            "must_survive_lost": self.must_survive_lost,
            "over_redaction_rate": round(self.over_rate, 5),
            "queued": self.queued,
            "review_rate": round(self.review_rate, 5),
            "escaped_first": self.escaped_first,
            "escaped_final": self.escaped_final,
            "escape_rate": round(self.escape_rate, 5),
            "post_review_escape_rate": round(self.post_review_escape_rate, 5),
            "idempotence_failures": self.idempotence_failures,
            "round_trip_failures": self.round_trip_failures,
            "elapsed_s": round(self.elapsed_s, 3),
            "per_class_recall": {
                k: round(self.per_class_caught.get(k, 0) / v, 5)
                for k, v in sorted(self.per_class_planted.items())
                if v
            },
            "misses": self.misses[:20],
            "over_redaction_examples": self.over_examples[:20],
            "files": self.files,
        }


def _merge_metrics(batches: list[BatchMetrics]) -> BatchMetrics:
    combined = BatchMetrics(batch_index=-1)
    for metrics in batches:
        combined.responses += metrics.responses
        combined.partials += metrics.partials
        combined.text_fields += metrics.text_fields
        combined.planted += metrics.planted
        combined.caught += metrics.caught
        combined.mandatory_planted += metrics.mandatory_planted
        combined.mandatory_caught += metrics.mandatory_caught
        combined.chars_planted += metrics.chars_planted
        combined.chars_redacted += metrics.chars_redacted
        combined.embedded_planted += metrics.embedded_planted
        combined.embedded_caught += metrics.embedded_caught
        combined.metadata_checked += metrics.metadata_checked
        combined.metadata_governed += metrics.metadata_governed
        combined.must_survive_total += metrics.must_survive_total
        combined.must_survive_lost += metrics.must_survive_lost
        combined.queued += metrics.queued
        combined.escaped_first += metrics.escaped_first
        combined.escaped_final += metrics.escaped_final
        combined.idempotence_failures += metrics.idempotence_failures
        combined.round_trip_failures += metrics.round_trip_failures
        combined.elapsed_s += metrics.elapsed_s
        for key, value in metrics.per_class_planted.items():
            combined.per_class_planted[key] = combined.per_class_planted.get(key, 0) + value
        for key, value in metrics.per_class_caught.items():
            combined.per_class_caught[key] = combined.per_class_caught.get(key, 0) + value
        combined.misses.extend(metrics.misses)
        combined.over_examples.extend(metrics.over_examples)
    return combined


# --------------------------------------------------------------------------- #
# Batch pipeline
# --------------------------------------------------------------------------- #


def _language_code(user_language: str) -> str:
    base = user_language.split("-")[0].lower()
    return base if base in {"en", "vi", "ar"} else "en"


class BatchChecker:
    """Generates, submits (writes + round-trips), redacts, and grades batches."""

    def __init__(
        self,
        instrument: QualtricsInstrument,
        *,
        salt: bytes = DEFAULT_SALT,
        second_pass: bool = True,
        govern_metadata_ip: bool = True,
    ) -> None:
        self.instrument = instrument
        self.salt = salt
        self.second_pass = second_pass
        self.schema = SurveySchema(redact_metadata_ip=govern_metadata_ip)
        self.survey: SurveyRedactor = SurveyRedactor(
            survey_redactor(salt=salt, nlp_backend="gazetteer"),
            audience=Audience.INTERNAL,
            review_threshold=RiskBand.HIGH,
        )

    # -- submission ------------------------------------------------------------

    def submit(
        self,
        batch: WHSBatch,
        out_dir: Path,
        *,
        formats: tuple[str, ...],
        round_trip: bool = True,
    ) -> BatchMetrics:
        started = time.perf_counter()
        files = write_batch_files(str(out_dir), batch, self.instrument, formats=formats)
        metrics = self._grade(batch)
        metrics.files = [Path(f).name for f in files]
        metrics.round_trip_failures = (
            self._round_trip(out_dir, batch, formats) if round_trip else 0
        )
        metrics.elapsed_s = time.perf_counter() - started
        return metrics

    def evaluate(self, batch: WHSBatch) -> BatchMetrics:
        """Grade a pre-generated batch (no file writes) for effectiveness."""
        return self._grade(batch)

    def _round_trip(
        self,
        out_dir: Path,
        batch: WHSBatch,
        formats: tuple[str, ...],
    ) -> int:
        """Read the written files back and compare with the in-memory batch."""
        failures = 0
        expected = {r.response_id: text_answers(r, self.instrument) for r in batch.responses}
        expected_embedded = {
            r.response_id: embedded_values(r, self.instrument) for r in batch.responses
        }
        readers = {"api": read_api_json, "csv": read_csv, "tsv": read_tsv, "ndjson": read_ndjson}
        stem = f"batch_{batch.batch_index + 1:02d}"
        suffixes = {
            "api": "api.json",
            "csv": "responses.csv",
            "tsv": "responses.tsv",
            "ndjson": "ndjson",
        }
        for fmt in formats:
            reader = readers.get(fmt)
            path = out_dir / f"{stem}.{suffixes[fmt]}"
            if reader is None or not path.exists():
                failures += 1
                continue
            rows: list[QualtricsResponse] = reader(path)
            if len(rows) != len(batch.responses):
                failures += 1
            for row in rows:
                if text_answers(row, self.instrument) != expected.get(row.response_id, {}):
                    failures += 1
                if embedded_values(row, self.instrument) != expected_embedded.get(
                    row.response_id, {}
                ):
                    failures += 1
        return failures

    # -- redaction + grading ---------------------------------------------------

    def _grade(self, batch: WHSBatch) -> BatchMetrics:
        metrics = BatchMetrics(batch_index=batch.batch_index)
        for response in batch.responses:
            truth = batch.ground_truth[response.response_id]
            embedded = embedded_values(response, self.instrument)
            language = _language_code(response.language)
            first_texts: list[str] = []
            final_texts: list[str] = []
            queued = False

            for qid, answer_text in text_answers(response, self.instrument).items():
                result = self.survey.redact_record(
                    answer_text,
                    record_id=response.response_id,
                    field_id=qid,
                    structured_context=embedded or None,
                    language=language,
                )
                first_texts.append(result.text)
                final = result.text
                if result.review_state in (
                    ReviewState.PENDING_REVIEW,
                    ReviewState.ESCALATED,
                ):
                    queued = True
                    if self.second_pass:
                        final = self.survey.run_second_pass(
                            result,
                            field_id=qid,
                            structured_context=embedded or None,
                            language=language,
                        ).text
                final_texts.append(final)
                metrics.text_fields += 1
                if self.survey.redact_record(final).text != final:
                    metrics.idempotence_failures += 1

            first_joined = "\n".join(first_texts)
            final_joined = "\n".join(final_texts)

            missed_mandatory_first = False
            missed_mandatory_final = False
            for secret in truth.secrets:
                metrics.planted += 1
                metrics.chars_planted += len(secret.literal)
                metrics.per_class_planted[secret.kind] = (
                    metrics.per_class_planted.get(secret.kind, 0) + 1
                )
                if secret.mandatory:
                    metrics.mandatory_planted += 1
                caught_first = secret.literal not in first_joined
                caught_final = secret.literal not in final_joined
                if caught_first:
                    metrics.caught += 1
                    metrics.chars_redacted += len(secret.literal)
                    metrics.per_class_caught[secret.kind] = (
                        metrics.per_class_caught.get(secret.kind, 0) + 1
                    )
                    if secret.mandatory:
                        metrics.mandatory_caught += 1
                else:
                    if secret.mandatory:
                        missed_mandatory_first = True
                        if not caught_final:
                            missed_mandatory_final = True
                    if len(metrics.misses) < 12:
                        metrics.misses.append(
                            f"{response.response_id} {secret.field or '?'} · {secret.kind}: "
                            f"{secret.literal[:48]!r}"
                        )

            # Embedded claims data + metadata governance.
            for secret in truth.embedded_secrets:
                metrics.embedded_planted += 1
                raw = response.values.get(secret.field or "", "")
                policy = self.schema.policy_for(secret.field or "", raw)
                governed = str(
                    self.schema.apply_whole_column(
                        policy, raw, salt=self.salt
                    )
                )
                if secret.literal not in governed:
                    metrics.embedded_caught += 1
            ip_address = response.values.get("IPAddress")
            if ip_address:
                metrics.metadata_checked += 1
                policy = self.schema.policy_for("IPAddress", ip_address)
                governed_ip = str(self.schema.apply_whole_column(policy, ip_address, salt=self.salt))
                if str(ip_address) not in governed_ip:
                    metrics.metadata_governed += 1

            for fragment in truth.must_survive:
                metrics.must_survive_total += 1
                if fragment not in final_joined:
                    metrics.must_survive_lost += 1
                    if len(metrics.over_examples) < 12:
                        metrics.over_examples.append(
                            f"{response.response_id}: {fragment!r} was removed"
                        )

            metrics.responses += 1
            metrics.partials += int(response.partial)
            metrics.queued += int(queued)
            metrics.escaped_first += int(missed_mandatory_first)
            metrics.escaped_final += int(missed_mandatory_final)
        return metrics


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _gates(palette: Palette, combined: BatchMetrics, gates: AcceptanceGates) -> tuple[bool, list[str]]:
    lines: list[str] = []
    passed = True

    def check(name: str, value: float, threshold: float, *, higher_is_better: bool) -> None:
        nonlocal passed
        ok = value >= threshold if higher_is_better else value <= threshold
        passed = passed and ok
        mark = palette.good("PASS") if ok else palette.bad("FAIL")
        comparator = "≥" if higher_is_better else "≤"
        lines.append(
            f"  {mark}  {name:<38} {value:8.4f}   (target {comparator} {threshold})"
        )

    check(
        "mandatory identifier recall",
        combined.mandatory_recall,
        gates.mandatory_direct_id_recall_min,
        higher_is_better=True,
    )
    check("response escape rate", combined.escape_rate, gates.record_escape_rate_max, higher_is_better=False)
    check(
        "over-redaction rate",
        combined.over_rate,
        gates.over_redaction_loss_max,
        higher_is_better=False,
    )
    check(
        "post-review escape rate",
        combined.post_review_escape_rate,
        gates.record_escape_rate_max,
        higher_is_better=False,
    )
    return passed, lines


def _build_report(
    palette: Palette,
    args: argparse.Namespace,
    metrics_by_batch: list[BatchMetrics],
    combined: BatchMetrics,
    out_dir: Path,
    total_elapsed: float,
) -> str:
    rules = "━" * 78
    lines: list[str] = []
    lines.append(palette.bold(rules))
    lines.append(palette.bold(" WHS CLAIMS EXPERIENCE · QUALTRICS BATCH SUBMISSION & REDACTION CHECK"))
    lines.append(palette.bold(rules))
    lines.append(f" instrument     {WHS_CLAIMS_INSTRUMENT.name} ({WHS_CLAIMS_INSTRUMENT.survey_id})")
    lines.append(
        f" batches        {len(metrics_by_batch)} × {args.per_batch} responses"
        f" · {combined.responses} total ({combined.partials} partial) · seed {args.seed}"
    )
    lines.append(
        f" formats        {', '.join(fmt.strip() for fmt in args.formats.split(',') if fmt.strip())}"
        f" + ground truth sidecars · out: {out_dir}"
    )
    lines.append(
        f" pipeline       survey profile · second pass {'on' if not args.no_second_pass else 'off'}"
        f" · schema-governed embedded data · IP metadata {'governed' if not args.no_govern_metadata_ip else 'passthrough'}"
    )
    lines.append(f" runtime        {total_elapsed:.1f}s total · {combined.responses / max(total_elapsed, 1e-6):.1f} responses/s")

    lines.append(section(palette, 1, "BATCH SUBMISSIONS — per-batch effectiveness"))
    rows = []
    for metrics in metrics_by_batch:
        rows.append(
            [
                str(metrics.batch_index + 1),
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
                "✓" if not metrics.round_trip_failures else "✗",
                f"{metrics.elapsed_s * 1000 / max(metrics.responses, 1):.2f}",
            ]
        )
    rows.append(
        [
            palette.bold("Σ"),
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
            "✓" if not combined.round_trip_failures else "✗",
            f"{combined.elapsed_s * 1000 / max(combined.responses, 1):.2f}",
        ]
    )

    def paint(ri: int, ci: int, plain: str, padded: str) -> str:
        if ri == len(rows) - 1:
            return palette.bold(padded)
        if ci in (7, 8, 9):
            return palette.count(int(plain), padded)
        if ci == 11:
            return palette.good(padded) if plain == "✓" else palette.bad(padded)
        return padded

    lines.append(
        render_table(
            [
                "batch",
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
                "files",
                "ms/resp",
            ],
            rows,
            aligns=["r"] * 13,
            paint=paint,
        )
    )
    lines.append(
        palette.dim(
            "  esc1 = responses with a missed mandatory identifier on pass 1 · "
            "esc2 = still missing after second pass · files = round-trip check"
        )
    )

    lines.append(section(palette, 2, "ACCEPTANCE GATES — cumulative across all batches"))
    passed, gate_lines = _gates(palette, combined, AcceptanceGates())
    lines.extend(gate_lines)
    lines.append("")
    verdict = (
        palette.good("VERDICT: PASS — batches are safe to release after the standard review flow")
        if passed
        else palette.bad("VERDICT: FAIL — investigate the failing gates above")
    )
    lines.append("  " + verdict)
    lines.append(
        f"  embedded claims data governed: {combined.embedded_caught}/{combined.embedded_planted}"
        f"   · IP metadata masked: {combined.metadata_governed}/{combined.metadata_checked}"
        f"   · idempotence failures: {combined.idempotence_failures}"
        f"   · round-trip failures: {combined.round_trip_failures}"
    )

    section_no = 3
    lines.append(section(palette, section_no, "PER-CLASS IDENTIFIER RECALL (all batches)"))
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

    if combined.misses or combined.over_examples:
        section_no += 1
        lines.append(section(palette, section_no, "TRIAGE — surviving identifiers and over-redaction"))
        for miss in combined.misses[:12]:
            lines.append(f"    {palette.warn('!')} {miss}")
        for example in combined.over_examples[:12]:
            lines.append(f"    {palette.info('~')} {example}")

    section_no += 1
    lines.append(section(palette, section_no, "ARTIFACTS"))
    file_count = sum(len(metrics.files) for metrics in metrics_by_batch)
    lines.append(f"    {file_count} response files + sidecars written under {out_dir}")
    for metrics in metrics_by_batch[:2]:
        for name in metrics.files:
            lines.append(palette.dim(f"      {name}"))
    if len(metrics_by_batch) > 2:
        lines.append(palette.dim("      …"))

    notes: list[str] = []
    if combined.partials:
        notes.append(
            f"{combined.partials} partial responses (abandoned mid-survey) exercised "
            f"missing-answer handling end to end"
        )
    if combined.review_rate > 0.5:
        notes.append(
            palette.warn(
                f"review load is {combined.review_rate * 100:.0f}% of responses with the untrained "
                f"gazetteer backend; trained encoders will lower confidence-driven routing"
            )
        )
    if not combined.round_trip_failures:
        notes.append(palette.good("every batch file round-tripped through the Qualtrics readers"))
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
        prog="qualtrics_batch_check.py",
        description=(
            "Generate and submit batches of Qualtrics WHS claims-experience responses, "
            "then check redaction effectiveness per batch and cumulatively."
        ),
    )
    parser.add_argument("--batches", type=int, default=6, help="number of batches (default: 6)")
    parser.add_argument(
        "--per-batch", type=int, default=40, help="responses per batch (default: 40)"
    )
    parser.add_argument("--seed", type=int, default=20261018, help="generator seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--formats",
        default="api,csv,tsv",
        help="Qualtrics export formats to write: api,ndjson,csv,tsv (default: api,csv,tsv)",
    )
    parser.add_argument(
        "--no-second-pass",
        action="store_true",
        help="skip the second pass for queued responses",
    )
    parser.add_argument(
        "--no-govern-metadata-ip",
        action="store_true",
        help="do not apply schema IP redaction to metadata columns",
    )
    parser.add_argument(
        "--negative-rate",
        type=float,
        default=0.30,
        help="share of negative-sentiment responses (default: 0.30)",
    )
    parser.add_argument(
        "--partial-rate",
        type=float,
        default=0.10,
        help="share of partial (abandoned) responses (default: 0.10)",
    )
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--quiet", action="store_true", help="hide progress output")
    parser.add_argument(
        "--json", action="store_true", help="also write effectiveness_report.json to --out"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    palette = Palette(
        enabled=args.color == "always" or (args.color == "auto" and sys.stdout.isatty())
    )
    formats = tuple(fmt.strip() for fmt in args.formats.split(",") if fmt.strip())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    instrument = WHS_CLAIMS_INSTRUMENT
    (out_dir / "instrument.json").write_text(
        json.dumps(instrument.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    generator = WHSBatchGenerator(
        seed=args.seed,
        instrument=instrument,
        negative_rate=args.negative_rate,
        partial_rate=args.partial_rate,
    )
    checker = BatchChecker(
        instrument,
        salt=DEFAULT_SALT,
        second_pass=not args.no_second_pass,
        govern_metadata_ip=not args.no_govern_metadata_ip,
    )

    metrics_by_batch: list[BatchMetrics] = []
    started = time.perf_counter()
    for index in range(args.batches):
        if not args.quiet:
            sys.stderr.write(f"\r  submitting batch {index + 1}/{args.batches} …")
            sys.stderr.flush()
        batch = generator.batch(args.per_batch, batch_index=index)
        metrics = checker.submit(batch, out_dir, formats=formats)
        metrics_by_batch.append(metrics)
    total_elapsed = time.perf_counter() - started
    if not args.quiet:
        sys.stderr.write("\r" + " " * 48 + "\r")
        sys.stderr.flush()

    combined = _merge_metrics(metrics_by_batch)
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "instrument": instrument.survey_id,
        "batches": len(metrics_by_batch),
        "per_batch": args.per_batch,
        "seed": args.seed,
        "formats": list(formats),
        "responses": combined.responses,
        "partial_responses": combined.partials,
        "round_trip_failures": combined.round_trip_failures,
        "effectiveness": combined.as_dict(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    report = _build_report(palette, args, metrics_by_batch, combined, out_dir, total_elapsed)
    sys.stdout.write(report + "\n")

    if args.json:
        payload = {
            "generated": manifest["generated"],
            "instrument": instrument.survey_id,
            "seed": args.seed,
            "batches": [m.as_dict() for m in metrics_by_batch],
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
