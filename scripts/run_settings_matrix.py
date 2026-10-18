#!/usr/bin/env python3
# ruff: noqa: E402
"""Matrix-run every checked-in test dataset through configurable redaction pipelines.

The script rebuilds a :class:`SurveyRedactor` for each configuration in a settings
matrix, replays all four datasets in ``data/`` plus the 80-record stratified
benchmark, and collates:

* **Leakage** — planted secrets / ``must_not_appear`` literals surviving in the
  *final released* text (second pass applied to queued records when enabled),
* **Identifier traces** — digit-dominant fragments (4+ digits) or email fragments
  (5+ characters) of direct identifiers still visible (catches ``style=partial``
  suffix leaks),
* **Over-redaction** — ``must_appear`` / ``must_survive`` fragments destroyed,
* **Expectation failures** — missing categories, wrong review routing, OCR
  polygons that fail to map back to the right pages,
* **Operations** — review-queue rate, runtime per record, idempotence, and the
  synthetic benchmark acceptance gates.

It then renders a console report that ranks configurations, shows which setting
moves the needle, and recommends the best profile for the chosen objective.

Examples
--------
    python3 scripts/run_settings_matrix.py --profile quick
    python3 scripts/run_settings_matrix.py --profile standard --axes nlp,overlap_mode,style
    python3 scripts/run_settings_matrix.py --profile full --json /tmp/matrix.json --no-color
    python3 scripts/run_settings_matrix.py --list
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from gov_survey_redact import (
    BoundingPolygon,
    Category,
    Lexicon,
    MaskStyle,
    OCRFieldLayout,
    OCRWord,
    ReviewAction,
    ReviewerDecision,
    ReviewState,
    Span,
    SurveyRedactor,
    Term,
    survey_redactor,
)
from gov_survey_redact.benchmark import AcceptanceGates, BenchmarkMetrics
from gov_survey_redact.generate import GeneratedRecord, Secret
from gov_survey_redact.qualtrics import QualtricsInstrument, row_values_from_mapping
from gov_survey_redact.risk import Audience, RiskBand
from gov_survey_redact.survey import DEFAULT_GENERALISED

# --------------------------------------------------------------------------- #
# Configuration matrix
# --------------------------------------------------------------------------- #

AXIS_CHOICES: dict[str, list[Any]] = {
    "nlp": [True, False],
    "overlap_mode": ["longest", "union"],
    "ocr_mode": [False, True],
    "generalise": ["default", "dates", "none"],
    "review_threshold": ["high", "moderate"],
    "style": ["mask", "label", "partial"],
    "second_pass": [True, False],
    "audience": ["internal", "public"],
    "min_confidence": [0.0, 0.55],
}

AXIS_LABELS: dict[str, str] = {
    "nlp": "NLP",
    "overlap_mode": "overlap",
    "ocr_mode": "OCR",
    "generalise": "generalise",
    "review_threshold": "review",
    "style": "style",
    "second_pass": "2nd-pass",
    "audience": "audience",
    "min_confidence": "min-conf",
}

PROFILE_AXES: dict[str, list[str]] = {
    "quick": ["nlp", "overlap_mode", "style", "second_pass"],
    "standard": ["nlp", "overlap_mode", "generalise", "style", "second_pass"],
    "full": [
        "nlp",
        "overlap_mode",
        "ocr_mode",
        "generalise",
        "review_threshold",
        "style",
        "second_pass",
    ],
}

EXTRA_AXES = ["audience", "min_confidence"]

#: Embedded Qualtrics fields that act as structured cross-field context.
EMBEDDED_STRUCTURED = (
    "respondent_name",
    "household_members",
    "street_address",
    "employer_name",
    "respondent_email",
)


def _row_language(record: dict) -> str:
    return str(record.get("UserLanguage", "EN")).split("-")[0].lower()


def _embedded_context(record: dict) -> dict | None:
    context = {key: record[key] for key in EMBEDDED_STRUCTURED if key in record}
    return context or None

SCORE_WEIGHTS = {"leak": 0.60, "trace": 0.10, "over": 0.20, "review": 0.10}
GATE_PENALTY = 100.0
IDEMPOTENCE_PENALTY = 25.0

STYLE_MAP = {
    "mask": MaskStyle.MASK,
    "label": MaskStyle.LABEL,
    "partial": MaskStyle.PARTIAL,
}
RISK_MAP = {
    "high": RiskBand.HIGH,
    "moderate": RiskBand.MODERATE,
    "critical": RiskBand.CRITICAL,
}
GENERALISE_SETS = {
    "default": DEFAULT_GENERALISED,
    "dates": frozenset({Category.DATE_AU, Category.DATE_ISO}),
    "none": frozenset(),
}


@dataclass(frozen=True)
class RunConfig:
    """One point in the settings matrix."""

    nlp: bool = True
    nlp_backend: str = "gazetteer"
    overlap_mode: str = "longest"
    ocr_mode: bool = False
    generalise: str = "default"
    review_threshold: str = "high"
    style: str = "mask"
    second_pass: bool = True
    audience: str = "internal"
    min_confidence: float = 0.0

    def label(self, axes: Sequence[str]) -> str:
        parts = []
        for axis in axes:
            value = getattr(self, axis)
            shown = "on" if value is True else "off" if value is False else value
            parts.append(f"{AXIS_LABELS[axis]}={shown}")
        return " · ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in (
                "nlp",
                "nlp_backend",
                "overlap_mode",
                "ocr_mode",
                "generalise",
                "review_threshold",
                "style",
                "second_pass",
                "audience",
                "min_confidence",
            )
        }


def expand_runs(axes: Sequence[str], pinned: RunConfig) -> list[RunConfig]:
    """Cartesian product of the chosen axes, starting from ``pinned`` defaults."""
    choices = [AXIS_CHOICES[axis] for axis in axes]
    runs: list[RunConfig] = []
    for combo in itertools.product(*choices):
        runs.append(replace(pinned, **dict(zip(axes, combo, strict=True))))
    return runs


# --------------------------------------------------------------------------- #
# Leak-trace detection
# --------------------------------------------------------------------------- #

DATE_LIKE = re.compile(r"^\s*\d{1,4}(?:[/.\-:]\d{1,4})+\s*$")
TRACE_NGRAM_DIGITS = 4
TRACE_NGRAM_LETTERS = 5

#: Benchmark secret kinds that are direct identifiers worth tracing.
TRACE_SECRET_KINDS = frozenset(
    {
        "phone",
        "phone_obfuscated",
        "email",
        "email_spaced",
        "tfn",
        "ocr_tfn",
        "medicare",
        "medicare_enrolment",
        "acn",
        "credit_card",
        "bsb",
        "passport",
        "drivers_licence",
        "centrelink_crn",
        "employee_ref",
        "iban",
        "ipv6",
        "mac_address",
        "street_address",
        "cross_field_person",
    }
)

#: Canonical ontology entity types that are direct identifiers worth tracing.
TRACE_ENTITY_TYPES = frozenset(
    {
        "PERSON",
        "TAX_ID",
        "HEALTH_ID",
        "GOV_ID",
        "EMAIL",
        "PHONE",
        "DRIVERS_LICENCE",
        "PASSPORT",
        "CREDIT_CARD",
        "IBAN",
        "STREET_ADDRESS",
        "ADDRESS",
        "BANK_ACCOUNT",
        "IP_ADDRESS",
        "MAC_ADDRESS",
    }
)

def _compact(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _trace_params(literal: str) -> tuple[int, int] | None:
    """Return ``(digits, ngram)`` when ``literal`` is worth trace analysis.

    Traces catch partially masked direct identifiers (e.g. the last four digits
    kept by ``style=partial``). Only digit-dominant values and emails are
    eligible: phrases such as ``"18 years"`` or ``"WorkSafe Victoria"`` would
    otherwise produce false positives from common letter n-grams.
    """
    compact = _compact(literal)
    if not compact or DATE_LIKE.match(literal.strip()):
        return None
    digits = sum(ch.isdigit() for ch in compact)
    digit_dominant = digits * 2 >= len(compact)
    ngram = TRACE_NGRAM_DIGITS if digit_dominant else TRACE_NGRAM_LETTERS
    if len(compact) < ngram:
        return None
    if digit_dominant or "@" in literal:
        return digits, ngram
    return None


def trace_candidate(literal: str) -> bool:
    return _trace_params(literal) is not None


def trace_leaked(literal: str, text: str, benign_text: str = "") -> bool:
    """True when an ``n``-character fragment of ``literal`` survives outside benign text.

    ``benign_text`` holds fragments that are *expected* to survive (``must_appear``
    / ``must_survive``), so coincidental collisions such as the ``1234`` inside a
    legitimate account number are not reported as identifier traces.
    """
    params = _trace_params(literal)
    if params is None:
        return False
    _, ngram = params
    needle = _compact(literal)
    haystack = _compact(text)
    benign = _compact(benign_text)
    for index in range(len(needle) - ngram + 1):
        fragment = needle[index : index + ngram]
        if fragment in haystack and fragment not in benign:
            return True
    return False


# --------------------------------------------------------------------------- #
# Console palette / rendering
# --------------------------------------------------------------------------- #


class Palette:
    """Minimal ANSI colour helper with graceful plain-text fallback."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def paint(self, text: str, code: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text: str) -> str:
        return self.paint(text, "1")

    def dim(self, text: str) -> str:
        return self.paint(text, "90")

    def good(self, text: str) -> str:
        return self.paint(text, "32")

    def bad(self, text: str) -> str:
        return self.paint(text, "31")

    def warn(self, text: str) -> str:
        return self.paint(text, "33")

    def info(self, text: str) -> str:
        return self.paint(text, "36")

    def score(self, value: float, text: str | None = None) -> str:
        text = text if text is not None else f"{value:.1f}"
        if value >= 90:
            return self.good(text)
        if value >= 75:
            return self.info(text)
        if value >= 55:
            return self.warn(text)
        return self.bad(text)

    def count(self, value: int, text: str | None = None) -> str:
        text = text if text is not None else str(value)
        return self.dim(text) if value == 0 else self.bad(text)


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    aligns: Sequence[str] | None = None,
    paint: Callable[[int, int, str, str], str] | None = None,
) -> str:
    """Box-drawing table; optional ``paint(row, col, plain, padded)`` for colour."""
    column_count = len(headers)
    aligns = list(aligns) if aligns is not None else ["l"] + ["r"] * (column_count - 1)
    plain_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in plain_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def fmt(cells: Sequence[str], row_index: int | None = None) -> str:
        rendered = []
        for index, cell in enumerate(cells):
            padded = cell.rjust(widths[index]) if aligns[index] == "r" else cell.ljust(widths[index])
            if paint is not None and row_index is not None:
                padded = paint(row_index, index, cell, padded)
            rendered.append(" " + padded + " ")
        return "│".join(rendered)

    top = "┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
    mid = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
    bottom = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"
    lines = [top, fmt([str(header) for header in headers]), mid]
    lines.extend(fmt(row, index) for index, row in enumerate(plain_rows))
    lines.append(bottom)
    return "\n".join(lines)


_PARTIAL_BLOCKS = "▏▎▍▌▋▊▉"


def bar(fraction: float, width: int = 18, fill: str = "█", empty: str = "░") -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = fraction * width
    whole = int(filled)
    remainder = filled - whole
    text = fill * whole
    if whole < width and remainder >= 0.06:
        text += _PARTIAL_BLOCKS[min(len(_PARTIAL_BLOCKS) - 1, int(remainder * len(_PARTIAL_BLOCKS)))]
        whole += 1
    return text + empty * (width - whole)


def section(palette: Palette, number: int, title: str) -> str:
    rule = "─" * 78
    return f"\n{palette.dim(rule)}\n{palette.bold(f' ▌ {number} · {title}')}\n{palette.dim(rule)}"


# --------------------------------------------------------------------------- #
# Run results
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    """Collated metrics for one configuration across every dataset."""

    config: RunConfig
    # Safety (graded on the final released text)
    secrets_checked: int = 0
    secrets_leaked: int = 0
    leaks_mandatory: int = 0
    leaks_optional: int = 0
    trace_checked: int = 0
    secrets_traced: int = 0
    # Utility
    benign_checked: int = 0
    benign_lost: int = 0
    # Expectations
    category_checks: int = 0
    category_misses: int = 0
    state_checks: int = 0
    state_mismatches: int = 0
    polygon_failures: int = 0
    reviewer_failures: int = 0
    # Operations
    records_processed: int = 0
    review_units: int = 0
    review_queued: int = 0
    idempotence_failures: int = 0
    elapsed_s: float = 0.0
    # Official benchmark metrics + breakdowns
    benchmark: dict[str, Any] = field(default_factory=dict)
    per_dataset: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def leak_rate(self) -> float:
        return self.secrets_leaked / self.secrets_checked if self.secrets_checked else 0.0

    @property
    def trace_rate(self) -> float:
        return self.secrets_traced / self.trace_checked if self.trace_checked else 0.0

    @property
    def over_rate(self) -> float:
        return self.benign_lost / self.benign_checked if self.benign_checked else 0.0

    @property
    def review_rate(self) -> float:
        return self.review_queued / self.review_units if self.review_units else 0.0

    @property
    def gates_passed(self) -> bool:
        gates = self.benchmark.get("gates") if self.benchmark else None
        return True if gates is None else bool(gates.get("passed"))

    @property
    def score(self) -> float:
        score = 100.0 * (
            SCORE_WEIGHTS["leak"] * (1.0 - self.leak_rate)
            + SCORE_WEIGHTS["trace"] * (1.0 - self.trace_rate)
            + SCORE_WEIGHTS["over"] * (1.0 - self.over_rate)
            + SCORE_WEIGHTS["review"] * (1.0 - self.review_rate)
        )
        if not self.gates_passed:
            score -= GATE_PENALTY
        if self.idempotence_failures:
            score -= IDEMPOTENCE_PENALTY
        return score

    def as_dict(self, axes: Sequence[str]) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "label": self.config.label(axes),
            "score": round(self.score, 3),
            "leak_rate": round(self.leak_rate, 5),
            "trace_rate": round(self.trace_rate, 5),
            "over_rate": round(self.over_rate, 5),
            "review_rate": round(self.review_rate, 5),
            "secrets_checked": self.secrets_checked,
            "secrets_leaked": self.secrets_leaked,
            "leaks_mandatory": self.leaks_mandatory,
            "leaks_optional": self.leaks_optional,
            "trace_checked": self.trace_checked,
            "secrets_traced": self.secrets_traced,
            "benign_checked": self.benign_checked,
            "benign_lost": self.benign_lost,
            "category_misses": self.category_misses,
            "state_mismatches": self.state_mismatches,
            "polygon_failures": self.polygon_failures,
            "reviewer_failures": self.reviewer_failures,
            "idempotence_failures": self.idempotence_failures,
            "review_queued": self.review_queued,
            "review_units": self.review_units,
            "records_processed": self.records_processed,
            "elapsed_s": round(self.elapsed_s, 3),
            "benchmark": self.benchmark,
            "per_dataset": self.per_dataset,
        }


# --------------------------------------------------------------------------- #
# Dataset loading + harness
# --------------------------------------------------------------------------- #


@dataclass
class DatasetBundle:
    """All test data loaded once and reused for every configuration."""

    aps: dict
    citizen: dict
    canonical: dict
    benchmark: list[GeneratedRecord]
    aps_lexicon: Lexicon

    @property
    def small_record_count(self) -> int:
        return (
            len(self.aps["records"]) + len(self.citizen["records"]) + len(self.canonical["annotations"])
        )


def load_bundle(benchmark_limit: int = 0, datasets: Sequence[str] = ()) -> tuple[DatasetBundle, dict[str, int]]:
    """Load the datasets in ``data/`` (mirrors the pytest dataset harnesses)."""
    wanted = set(datasets) or {"aps", "citizen", "canonical", "benchmark"}

    def load(name: str) -> dict:
        return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))

    aps = load("aps_workplace_census_dirty.json") if "aps" in wanted else {"records": []}
    citizen = load("citizen_services_claims_and_ocr.json") if "citizen" in wanted else {"records": []}
    canonical = (
        load("canonical_annotations_and_edge_cases.json") if "canonical" in wanted else {"annotations": []}
    )

    benchmark: list[GeneratedRecord] = []
    if "benchmark" in wanted:
        manifest = load("synthetic_stratified_benchmark.json")
        instrument = QualtricsInstrument.from_dict(manifest["qualtrics"])
        benchmark = [_record_from_manifest(entry, instrument) for entry in manifest["records"]]
        if benchmark_limit > 0:
            benchmark = benchmark[:benchmark_limit]

    lexicon = Lexicon.from_terms(
        [
            Term(
                term=entry["term"],
                category=Category(entry["category"]),
                aliases=entry.get("aliases", ()),
                label=entry.get("label", "internal term"),
            )
            for entry in aps.get("lexicon_terms", [])
        ]
    )
    bundle = DatasetBundle(
        aps=aps, citizen=citizen, canonical=canonical, benchmark=benchmark, aps_lexicon=lexicon
    )
    inventory = {
        "APS census": len(aps["records"]),
        "Citizen services": len(citizen["records"]),
        "Canonical": len(canonical["annotations"]),
        "Synthetic benchmark": len(benchmark),
    }
    return bundle, {name: count for name, count in inventory.items() if count > 0}


def _record_from_manifest(entry: dict, instrument: QualtricsInstrument) -> GeneratedRecord:
    """Rebuild a :class:`GeneratedRecord` from the checked-in Qualtrics benchmark manifest."""
    answers, embedded = row_values_from_mapping(entry, instrument)
    return GeneratedRecord(
        response_id=entry["ResponseId"],
        answers=answers,
        secrets=[
            Secret(
                kind=str(secret["kind"]),
                literal=str(secret["literal"]),
                expect_category=secret.get("expect_category"),
                checksummed=bool(secret.get("checksummed", False)),
                degradation=str(secret.get("degradation", "clean")),
                mandatory=bool(secret.get("mandatory", True)),
                field=secret.get("field"),
            )
            for secret in entry.get("secrets", [])
        ],
        must_survive=list(entry.get("must_survive", [])),
        template=str(entry.get("template", "")),
        stratum=str(entry.get("stratum", "short_open_text")),
        language=_row_language(entry),
        ocr_confidence=entry.get("ocr_confidence"),
        structured_fields={
            key: value for key, value in embedded.items() if key in EMBEDDED_STRUCTURED
        },
        household_id=str(entry.get("household_id", "HH0000")),
    )


def _layout_from_spec(spec: dict, text: str, field_confidence: float | None) -> OCRFieldLayout:
    words = []
    for word in spec["words"]:
        polygon = None
        if word.get("polygon") is not None:
            polygon = BoundingPolygon(
                page=word["polygon"]["page"],
                points=tuple(tuple(point) for point in word["polygon"]["points"]),
            )
        words.append(
            OCRWord(
                text=word["text"],
                span=Span(word["start"], word["end"]),
                confidence=word["confidence"],
                polygon=polygon,
            )
        )
    return OCRFieldLayout(text=text, words=tuple(words), field_confidence=field_confidence)


class Runner:
    """Executes one :class:`RunConfig` across every loaded dataset."""

    def __init__(self, bundle: DatasetBundle) -> None:
        self.bundle = bundle

    def surveys_for(self, config: RunConfig) -> tuple[SurveyRedactor, SurveyRedactor]:
        """Return ``(base_survey, aps_survey)`` for the configuration."""

        def make(lexicon: Lexicon | None) -> SurveyRedactor:
            redactor = survey_redactor(
                lexicon=lexicon,
                nlp_enabled=config.nlp,
                nlp_backend=config.nlp_backend,
                ocr_mode=config.ocr_mode,
                overlap_mode=config.overlap_mode,
                generalise_categories=GENERALISE_SETS[config.generalise],
                default_style=STYLE_MAP[config.style],
                min_confidence=config.min_confidence,
            )
            return SurveyRedactor(
                redactor,
                audience=Audience(config.audience),
                review_threshold=RISK_MAP[config.review_threshold],
            )

        return make(None), make(self.bundle.aps_lexicon)

    def run(self, config: RunConfig) -> RunResult:
        result = RunResult(config=config)
        base_survey, aps_survey = self.surveys_for(config)
        started = time.perf_counter()

        if self.bundle.aps["records"]:
            self._run_expectation_dataset("aps", self.bundle.aps, aps_survey, config, result)
        if self.bundle.citizen["records"]:
            self._run_expectation_dataset("citizen", self.bundle.citizen, base_survey, config, result)
        if self.bundle.canonical["annotations"]:
            self._run_canonical(base_survey, config, result)
        if self.bundle.benchmark:
            self._run_benchmark(base_survey, config, result)

        result.elapsed_s = time.perf_counter() - started
        return result

    # -- dataset harnesses -----------------------------------------------------

    def _final_text_and_categories(
        self,
        res: Any,
        survey: SurveyRedactor,
        config: RunConfig,
        *,
        field_id: str,
        structured_context: Any = None,
        language: str = "en",
        categories: set[str],
    ) -> str:
        categories.update(category.value for category in res.redacted_categories)
        final = res.text
        if config.second_pass and res.review_state in (
            ReviewState.PENDING_REVIEW,
            ReviewState.ESCALATED,
        ):
            second = survey.run_second_pass(
                res,
                field_id=field_id,
                structured_context=structured_context,
                language=language,
            )
            final = second.text
            categories.update(category.value for category in second.redacted_categories)
        return final

    def _run_expectation_dataset(
        self,
        name: str,
        dataset: dict,
        survey: SurveyRedactor,
        config: RunConfig,
        result: RunResult,
    ) -> None:
        stats = {"records": len(dataset["records"]), "leaks": 0, "traced": 0, "over": 0,
                 "state": 0, "category": 0, "review": 0}
        instrument = QualtricsInstrument.from_dict(dataset["qualtrics"])
        for record in dataset["records"]:
            rec_id = record["ResponseId"]
            expectations = record["expectations"]
            answers, _embedded = row_values_from_mapping(record, instrument)
            layout_spec = record.get("ocr_layout")
            structured = _embedded_context(record)
            language = _row_language(record)
            redacted: list[str] = []
            categories: set[str] = set()
            states: set[ReviewState] = set()
            pages_seen: set[int] = set()
            boxes = 0

            for col, answer in answers.items():
                layout = None
                if layout_spec and layout_spec.get("field") == col:
                    layout = _layout_from_spec(layout_spec, answer, record.get("ocr_confidence"))
                res = survey.redact_record(
                    answer,
                    record_id=rec_id,
                    field_id=col,
                    structured_context=structured,
                    ocr_layout=layout,
                    ocr_confidence=record.get("ocr_confidence"),
                    language=language,
                )
                states.add(res.review_state)
                final = self._final_text_and_categories(
                    res,
                    survey,
                    config,
                    field_id=col,
                    structured_context=structured,
                    language=language,
                    categories=categories,
                )
                redacted.append(final)
                boxes += len(res.visual_boxes)
                for box in res.visual_boxes:
                    pages_seen.update(polygon.page for polygon in box.polygons)

            joined = "\n".join(redacted)
            benign_text = " ".join(expectations.get("must_appear", []))
            for forbidden in expectations.get("must_not_appear", []):
                result.secrets_checked += 1
                if forbidden in joined:
                    result.secrets_leaked += 1
                    result.leaks_mandatory += 1
                    stats["leaks"] += 1
                if trace_candidate(forbidden):
                    result.trace_checked += 1
                    if trace_leaked(forbidden, joined, benign_text):
                        result.secrets_traced += 1
                        stats["traced"] += 1
            for required in expectations.get("must_appear", []):
                result.benign_checked += 1
                if required not in joined:
                    result.benign_lost += 1
                    stats["over"] += 1
            for category in expectations.get("expected_categories", []):
                result.category_checks += 1
                if category not in categories:
                    result.category_misses += 1
                    stats["category"] += 1
            expected_state = ReviewState(expectations["expected_review_state"])
            queued = bool(states & {ReviewState.PENDING_REVIEW, ReviewState.ESCALATED})
            result.state_checks += 1
            if (expected_state is ReviewState.PENDING_REVIEW and not queued) or (expected_state is ReviewState.AUTO_ACCEPTED and states != {ReviewState.AUTO_ACCEPTED}):
                result.state_mismatches += 1
                stats["state"] += 1
            if layout_spec:
                expected_pages = set(layout_spec.get("expected_pages", ()))
                if boxes < 1 or (expected_pages and pages_seen != expected_pages):
                    result.polygon_failures += 1
            result.review_units += 1
            result.review_queued += int(queued)
            stats["review"] += int(queued)
            result.records_processed += 1
        result.per_dataset[name] = stats

    def _run_canonical(self, survey: SurveyRedactor, config: RunConfig, result: RunResult) -> None:
        stats = {"records": len(self.bundle.canonical["annotations"]), "leaks": 0, "traced": 0,
                 "over": 0, "state": 0, "category": 0, "review": 0}
        for item in self.bundle.canonical["annotations"]:
            text = item["text"]
            res = survey.redact_record(
                text,
                record_id=item["document_id"],
                field_id=item["field_id"],
                language=item["language"],
                ocr_confidence=item["ocr_confidence"],
            )
            categories: set[str] = set()
            final = self._final_text_and_categories(
                res,
                survey,
                config,
                field_id=item["field_id"],
                language=item["language"],
                categories=categories,
            )
            for entity in item["entities"]:
                if not entity["mandatory_redact"]:
                    continue
                literal = text[entity["start_char"] : entity["end_char"]]
                result.secrets_checked += 1
                if literal in final:
                    result.secrets_leaked += 1
                    result.leaks_mandatory += 1
                    stats["leaks"] += 1
                if entity["entity_type"] in TRACE_ENTITY_TYPES and trace_candidate(literal):
                    result.trace_checked += 1
                    if trace_leaked(literal, final):
                        result.secrets_traced += 1
                        stats["traced"] += 1

            expected_states = []
            if item["language"] != "en":
                expected_states.append("PENDING_REVIEW")
            if item["ocr_confidence"] is not None and item["ocr_confidence"] < 0.8:
                expected_states.append("PENDING_REVIEW")
            if item.get("expect_review_state"):
                expected_states.append(item["expect_review_state"])
            for expected in expected_states:
                result.state_checks += 1
                if res.review_state.value != expected:
                    result.state_mismatches += 1
                    stats["state"] += 1

            reviewer_spec = item.get("reviewer_decision")
            expected_model = item.get("expect_second_pass_model")
            if reviewer_spec is not None:
                decision = ReviewerDecision(
                    span=Span(reviewer_spec["start_char"], reviewer_spec["end_char"]),
                    action=ReviewAction(reviewer_spec["action"]),
                    category=Category(reviewer_spec["category"]),
                    reviewer_principal=reviewer_spec["reviewer_principal"],
                    reason_code=reviewer_spec["reason_code"],
                )
                reviewed, annotation = survey.apply_review_decisions(
                    res, [decision], field_id=item["field_id"], language=item["language"]
                )
                if reviewed.review_state is not ReviewState.REVIEWED or not annotation.entities:
                    result.reviewer_failures += 1
                second = survey.run_second_pass(
                    reviewed, field_id=item["field_id"], language=item["language"]
                )
                expected_after = item.get("expect_review_state_after_second_pass", "RELEASED")
                if second.review_state.value != expected_after:
                    result.reviewer_failures += 1
                for entity in item["entities"]:
                    if not entity["mandatory_redact"]:
                        continue
                    literal = text[entity["start_char"] : entity["end_char"]]
                    if literal in second.text:
                        result.reviewer_failures += 1
            elif expected_model and config.second_pass:
                second = survey.run_second_pass(res, field_id=item["field_id"], language=item["language"])
                if second.second_pass_model_used != expected_model:
                    result.reviewer_failures += 1

            queued = res.review_state in (ReviewState.PENDING_REVIEW, ReviewState.ESCALATED)
            result.review_units += 1
            result.review_queued += int(queued)
            stats["review"] += int(queued)
            result.records_processed += 1
        result.per_dataset["canonical"] = stats

    def _run_benchmark(self, survey: SurveyRedactor, config: RunConfig, result: RunResult) -> None:
        records = self.bundle.benchmark
        metrics = BenchmarkMetrics(records_total=len(records))
        latencies_ms: list[float] = []
        total_tokens = 0
        stats = {"records": len(records), "leaks": 0, "traced": 0, "over": 0,
                 "state": 0, "category": 0, "review": 0}

        for record in records:
            started = time.perf_counter()
            first_parts: list[str] = []
            final_parts: list[str] = []
            queued = False

            for column, text in record.answers.items():
                total_tokens += len(text.split())
                res = survey.redact_record(
                    text,
                    record_id=record.response_id,
                    field_id=column,
                    structured_context=record.structured_fields or None,
                    ocr_confidence=record.ocr_confidence,
                    language=record.language,
                )
                first_parts.append(res.text)
                final = res.text
                if res.review_state in (ReviewState.PENDING_REVIEW, ReviewState.ESCALATED):
                    queued = True
                    if config.second_pass:
                        second = survey.run_second_pass(
                            res,
                            structured_context=record.structured_fields or None,
                            language=record.language,
                            field_id=column,
                        )
                        final = second.text
                final_parts.append(final)

            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            joined_first = "\n".join(first_parts)
            joined_final = "\n".join(final_parts)
            benign_fragments = " ".join(record.must_survive)

            if record.secrets:
                metrics.records_with_secrets += 1
                record_missed_first = False
                record_missed_second = False
                for secret in record.secrets:
                    metrics.secrets_planted += 1
                    metrics.pii_chars_planted += len(secret.literal)
                    metrics.per_class_planted[secret.kind] = (
                        metrics.per_class_planted.get(secret.kind, 0) + 1
                    )
                    if record.ocr_confidence is not None:
                        metrics.ocr_planted += 1
                    caught_first = secret.literal not in joined_first
                    caught_second = secret.literal not in joined_final
                    if caught_first:
                        metrics.secrets_caught += 1
                        metrics.pii_chars_redacted += len(secret.literal)
                        metrics.per_class_caught[secret.kind] = (
                            metrics.per_class_caught.get(secret.kind, 0) + 1
                        )
                        if record.ocr_confidence is not None:
                            metrics.ocr_caught += 1
                    else:
                        if secret.mandatory:
                            record_missed_first = True
                        metrics.first_pass_misses_total += 1
                        if queued:
                            metrics.first_pass_misses_routed_to_review += 1
                        if secret.mandatory and not caught_second and not queued:
                            record_missed_second = True
                    result.secrets_checked += 1
                    if not caught_second:
                        result.secrets_leaked += 1
                        if secret.mandatory:
                            result.leaks_mandatory += 1
                        else:
                            result.leaks_optional += 1
                        stats["leaks"] += 1
                    if secret.kind in TRACE_SECRET_KINDS and trace_candidate(secret.literal):
                        result.trace_checked += 1
                        if trace_leaked(secret.literal, joined_final, benign_fragments):
                            result.secrets_traced += 1
                            stats["traced"] += 1
                if record_missed_first:
                    metrics.records_escaped_first_pass += 1
                if record_missed_second:
                    metrics.records_escaped_post_review += 1

            for fragment in record.must_survive:
                metrics.must_survive_total += 1
                metrics.non_pii_chars_total += len(fragment)
                if fragment not in joined_first:
                    metrics.must_survive_lost += 1
                    metrics.non_pii_chars_lost += len(fragment)
                result.benign_checked += 1
                if fragment not in joined_final:
                    result.benign_lost += 1
                    stats["over"] += 1

            metrics.review_queue_count += int(queued)
            result.review_units += 1
            result.review_queued += int(queued)
            stats["review"] += int(queued)
            result.records_processed += 1

            # Idempotence: re-redacting each released column must be a no-op.
            for part in final_parts:
                if survey.redact_record(part).text != part:
                    result.idempotence_failures += 1
                    break

        if latencies_ms:
            ordered = sorted(latencies_ms)
            metrics.p50_ms = ordered[len(ordered) // 2]
            metrics.p95_ms = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        metrics.records_per_sec = (
            len(records) / max(1e-6, sum(latencies_ms) / 1000.0) if latencies_ms else 0.0
        )
        metrics.tokens_per_sec = metrics.records_per_sec * (
            total_tokens / len(records) if records else 0.0
        )
        summary = metrics.as_dict()
        summary["gates"] = metrics.verify_gates(AcceptanceGates())
        result.benchmark = summary
        result.per_dataset["benchmark"] = stats


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _rank_key(run: RunResult) -> tuple[float, int, int, int, float]:
    """Rank primarily by score, then by mandatory leaks, total leaks, traces, review load."""
    return (-run.score, run.leaks_mandatory, run.secrets_leaked, run.secrets_traced, run.review_rate)


def _leaderboard(palette: Palette, results: Sequence[RunResult], axes: Sequence[str], top: int) -> str:
    ranked = sorted(results, key=_rank_key)
    rows = []
    for index, run in enumerate(ranked[:top], 1):
        rows.append(
            [
                str(index),
                run.config.label(axes),
                str(run.secrets_leaked),
                str(run.secrets_traced),
                str(run.benign_lost),
                str(run.state_mismatches + run.category_misses + run.polygon_failures),
                _fmt_pct(run.review_rate),
                "✓" if run.gates_passed else "✗",
                "✓" if not run.idempotence_failures else "✗",
                f"{run.elapsed_s / max(1, run.records_processed) * 1000:.2f}",
                f"{run.score:.1f}",
                bar(run.score / 100.0, width=10),
            ]
        )

    def paint(row_index: int, col: int, plain: str, padded: str) -> str:
        run = ranked[row_index]
        if col in (2, 3, 4, 5):
            return palette.count(int(plain), padded)
        if col in (7, 8):
            return palette.good(padded) if plain == "✓" else palette.bad(padded)
        if col in (10, 11):
            return palette.score(run.score, padded)
        if col == 0:
            return palette.bold(padded)
        return padded

    headers = [
        "#",
        "configuration",
        "leak",
        "trace",
        "over",
        "expect",
        "review",
        "gates",
        "idem",
        "ms/rec",
        "score",
        "",
    ]
    aligns = ["r", "l"] + ["r"] * 9 + ["l"]
    return f"{palette.dim('top ' + str(min(top, len(ranked))) + ' of ' + str(len(results)) + ' runs')}\n" + render_table(
        headers, rows, aligns=aligns, paint=paint
    )


def _full_matrix(palette: Palette, results: Sequence[RunResult], axes: Sequence[str], show: int) -> str:
    ranked = sorted(results, key=_rank_key)
    shown = ranked if show <= 0 else ranked[:show]
    rows = []
    for index, run in enumerate(shown, 1):
        rows.append(
            [
                str(index),
                run.config.label(axes),
                str(run.secrets_leaked),
                str(run.secrets_traced),
                str(run.benign_lost),
                str(run.category_misses),
                str(run.state_mismatches),
                _fmt_pct(run.review_rate),
                "✓" if run.gates_passed else "✗",
                f"{run.score:.1f}",
                bar(run.score / 100.0, width=14),
            ]
        )

    def paint(row_index: int, col: int, plain: str, padded: str) -> str:
        run = shown[row_index]
        if col in (2, 3, 4, 5, 6):
            return palette.count(int(plain), padded)
        if col == 8:
            return palette.good(padded) if plain == "✓" else palette.bad(padded)
        if col in (9, 10):
            return palette.score(run.score, padded)
        return padded

    headers = [
        "#",
        "configuration",
        "leak",
        "trace",
        "over",
        "cat",
        "route",
        "review",
        "gates",
        "score",
        "",
    ]
    aligns = ["r", "l"] + ["r"] * 8 + ["l"]
    note = "" if show <= 0 or len(ranked) <= show else f"\n{palette.dim(f'… {len(ranked) - show} more runs hidden (use --show 0 for all)')}"
    return render_table(headers, rows, aligns=aligns, paint=paint) + note


def _sensitivity(palette: Palette, results: Sequence[RunResult], axes: Sequence[str]) -> str:
    blocks: list[str] = []
    for axis in axes:
        values = AXIS_CHOICES[axis]
        if len(values) < 2:
            continue
        groups: dict[Any, list[RunResult]] = {}
        for run in results:
            groups.setdefault(getattr(run.config, axis), []).append(run)
        if len(groups) < 2:
            continue
        means = {value: sum(r.score for r in runs) / len(runs) for value, runs in groups.items()}
        best_value = max(means, key=lambda value: (means[value], -sum(r.secrets_leaked for r in groups[value])))
        worst_value = min(means, key=means.get)  # type: ignore[arg-type]
        delta = means[best_value] - means[worst_value]
        lo, hi = min(means.values()), max(means.values())

        def display(value: Any) -> str:
            return "on" if value is True else "off" if value is False else str(value)

        lines = [f"  {palette.bold(AXIS_LABELS[axis].ljust(12))}"]
        score_span = hi - lo
        for value in values:
            if value not in groups:
                continue
            leak = sum(r.secrets_leaked for r in groups[value]) / max(1, len(groups[value]))
            star = palette.good("★") if value == best_value else " "
            # Bars are scaled between the best and worst setting for this axis so the
            # relative effect of the switch is visible regardless of absolute score.
            fraction = 1.0 if score_span <= 0 else (means[value] - lo) / score_span
            lines.append(
                f"    {star} {display(value):<10} {palette.score(means[value], bar(fraction, 22))} "
                f"{means[value]:6.1f}  {palette.dim(f'n={len(groups[value])}, avg leaks={leak:.1f}')}"
            )
        lines.append(
            f"      {palette.info('→')} prefer {palette.bold(display(best_value))} "
            f"(Δ {delta:+.1f} score vs worst{' — tie' if abs(delta) < 0.05 else ''})"
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "  (no varying axes)"


def _per_class_bars(palette: Palette, recall: dict[str, float], *, limit: int = 14) -> str:
    if not recall:
        return ""
    ordered = sorted(recall.items(), key=lambda item: item[1])
    lines = []
    for name, value in ordered[:limit]:
        lines.append(
            f"    {name:<22} {palette.score(value * 100.0, bar(value, 26))} {_fmt_pct(value)}"
        )
    return "\n".join(lines)


def _recommendation(
    palette: Palette,
    results: Sequence[RunResult],
    axes: Sequence[str],
    runner: Runner,
    bundle: DatasetBundle,
) -> str:
    ranked = sorted(results, key=_rank_key)
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    lines: list[str] = []

    config_rows = [[key, str(value)] for key, value in best.config.as_dict().items()]
    lines.append(
        render_table(
            ["setting", "value"],
            config_rows,
            aligns=["l", "l"],
            paint=lambda ri, ci, plain, padded: palette.bold(padded) if ci == 0 else padded,
        )
    )
    lines.append("")
    leak_detail = f"{best.secrets_leaked}/{best.secrets_checked} ({_fmt_pct(best.leak_rate)})"
    if best.leaks_optional and not best.leaks_mandatory:
        leak_detail += " — all optional adversarial phrases, 0 mandatory identifiers"
    lines.append(
        f"  leakage {palette.count(best.secrets_leaked, leak_detail)}"
        f"   traces {palette.count(best.secrets_traced, f'{best.secrets_traced}/{best.trace_checked}')}"
        f"   over-redaction {palette.count(best.benign_lost, f'{best.benign_lost}/{best.benign_checked}')}"
        f"   review {_fmt_pct(best.review_rate)}"
        f"   score {palette.score(best.score)}"
    )
    if runner_up is not None:
        lines.append(
            palette.dim(
                f"  runner-up: {runner_up.config.label(axes)}  (score {runner_up.score:.1f})"
            )
        )

    dataset_rows = []
    for name, stats in best.per_dataset.items():
        dataset_rows.append(
            [
                name,
                str(stats["records"]),
                str(stats["leaks"]),
                str(stats["traced"]),
                str(stats["over"]),
                str(stats["state"]),
                str(stats["category"]),
                _fmt_pct(stats["review"] / max(1, stats["records"])),
            ]
        )
    if dataset_rows:
        lines.append("")
        lines.append(palette.bold("  per-dataset breakdown"))

        def paint_dataset(ri: int, ci: int, plain: str, padded: str) -> str:
            if ci in (2, 3, 4, 5, 6):
                return palette.count(int(plain), padded)
            return padded

        lines.append(
            render_table(
                ["dataset", "records", "leak", "trace", "over", "route", "cat", "review"],
                dataset_rows,
                aligns=["l", "r", "r", "r", "r", "r", "r", "r"],
                paint=paint_dataset,
            )
        )

    recall = (best.benchmark or {}).get("per_class_recall", {})
    if recall:
        lines.append("")
        lines.append(palette.bold("  benchmark per-class recall (weakest first)"))
        lines.append(_per_class_bars(palette, recall))

    notes = _notes(palette, best, ranked, axes)
    if notes:
        lines.append("")
        lines.append(palette.bold("  notes"))
        lines.extend(f"    {note}" for note in notes)

    samples = _sample_outputs(palette, runner, best, bundle)
    if samples:
        lines.append("")
        lines.append(palette.bold("  sample outputs (winner configuration)"))
        lines.extend(samples)
    return "\n".join(lines)


def _notes(palette: Palette, best: RunResult, ranked: Sequence[RunResult], axes: Sequence[str]) -> list[str]:
    notes: list[str] = []
    zero_leak = [run for run in ranked if run.secrets_leaked == 0]
    if not best.secrets_leaked and not best.secrets_traced:
        notes.append(palette.good("zero identifier leakage and zero surviving identifier traces"))
    elif not best.secrets_leaked:
        notes.append(
            palette.warn(
                f"no full-literal leaks, but {best.secrets_traced} identifier fragment trace(s) "
                f"survive — check partial masking styles"
            )
        )
    elif not best.leaks_mandatory:
        notes.append(
            palette.dim(
                f"all {best.leaks_optional} surviving leaks are optional adversarial implicating "
                f"phrases (heavily misspelled / character-spaced); no mandatory direct identifier escaped"
            )
        )
    if best.benign_lost:
        notes.append(
            palette.warn(f"{best.benign_lost} benign fragment(s) were over-redacted")
        )
    if best.review_rate > 0.6:
        if "review_threshold" in axes or "audience" in axes:
            best_review = min(ranked, key=lambda run: run.review_rate)
            notes.append(
                palette.warn(
                    f"review load is high ({_fmt_pct(best.review_rate)} of records); the lowest-load "
                    f"run uses review={best_review.config.review_threshold} at {_fmt_pct(best_review.review_rate)}"
                )
            )
        else:
            notes.append(
                palette.warn(
                    f"review load is high ({_fmt_pct(best.review_rate)} of records); re-run with "
                    f"--axes review_threshold,audience to explore lower-load routing"
                )
            )
    if best.idempotence_failures:
        notes.append(palette.bad(f"{best.idempotence_failures} idempotence failure(s)"))
    if not best.gates_passed:
        notes.append(palette.bad("synthetic benchmark acceptance gates failed"))
    if zero_leak:
        notes.append(
            palette.dim(f"{len(zero_leak)}/{len(ranked)} runs are completely leak-free")
        )
        if best not in zero_leak:
            example = zero_leak[0]
            notes.append(
                palette.warn(
                    f"a leak-free alternative scoring {example.score:.1f} exists: "
                    f"{example.config.label(axes)}"
                )
            )
    elif best.leaks_mandatory:
        notes.append(palette.bad("mandatory identifier leaks remain for every configuration"))
    # optional-only leaks are already called out above
    if not _module_available("transformers"):
        notes.append(
            palette.dim(
                "transformers is not installed: once trained DeBERTa/mDeBERTa/ModernBERT weights are "
                "available, re-run this matrix to compare encoder backends"
            )
        )
    return notes


def _sample_outputs(
    palette: Palette,
    runner: Runner,
    best: RunResult,
    bundle: DatasetBundle,
    limit: int = 120,
) -> list[str]:
    if not bundle.aps["records"] or not bundle.citizen["records"]:
        return []
    base_survey, aps_survey = runner.surveys_for(best.config)
    samples: list[str] = []

    def show(
        tag: str,
        survey: SurveyRedactor,
        text: str,
        *,
        record_id: str,
        field_id: str,
        structured_context: dict | None = None,
    ) -> None:
        res = survey.redact_record(
            text, record_id=record_id, field_id=field_id, structured_context=structured_context
        )
        first = res.text
        final = first
        if best.config.second_pass and res.review_state in (
            ReviewState.PENDING_REVIEW,
            ReviewState.ESCALATED,
        ):
            final = survey.run_second_pass(
                res, field_id=field_id, structured_context=structured_context
            ).text
        samples.append(f"    {palette.dim(tag)}  original : {text[:limit]}")
        samples.append(f"    {' ' * len(tag)}  redacted : {final[:limit]}")
        samples.append("")

    aps = bundle.aps["records"][0]
    aps_instrument = QualtricsInstrument.from_dict(bundle.aps["qualtrics"])
    aps_answers, _ = row_values_from_mapping(aps, aps_instrument)
    aps_field = next(iter(aps_answers))
    show(
        f"APS {aps['ResponseId']}",
        aps_survey,
        aps_answers[aps_field],
        record_id=aps["ResponseId"],
        field_id=aps_field,
    )
    citizen = bundle.citizen["records"][0]
    citizen_instrument = QualtricsInstrument.from_dict(bundle.citizen["qualtrics"])
    citizen_answers, _ = row_values_from_mapping(citizen, citizen_instrument)
    citizen_field = next(iter(citizen_answers))
    show(
        f"CLM {citizen['ResponseId']}",
        base_survey,
        citizen_answers[citizen_field],
        record_id=citizen["ResponseId"],
        field_id=citizen_field,
        structured_context=_embedded_context(citizen),
    )
    return samples


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_settings_matrix.py",
        description=(
            "Execute all test datasets against a matrix of redaction settings and "
            "render a console report identifying the best configuration."
        ),
        epilog=(
            "Scoring: score = 100 × (0.60·(1−leak_rate) + 0.10·(1−trace_rate) + "
            "0.20·(1−over_redaction) + 0.10·(1−review_rate)); failing benchmark gates "
            "subtracts 100, idempotence failures subtract 25."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_AXES),
        default="quick",
        help="matrix size preset (default: quick)",
    )
    parser.add_argument(
        "--axes",
        help="comma-separated axes overriding the profile, e.g. nlp,overlap_mode,style,audience",
    )
    parser.add_argument(
        "--datasets",
        default="aps,citizen,canonical,benchmark",
        help="comma-separated datasets: aps,citizen,canonical,benchmark (default: all)",
    )
    parser.add_argument(
        "--benchmark-limit",
        type=int,
        default=0,
        help="use only the first N benchmark records (0 = all)",
    )
    parser.add_argument("--nlp-backend", default="gazetteer", help="NLP backend for every run")
    parser.add_argument("--top", type=int, default=10, help="leaderboard size (default: 10)")
    parser.add_argument(
        "--show", type=int, default=0, help="matrix rows to print (0 = all, default: 0)"
    )
    parser.add_argument("--json", dest="json_path", help="also write raw results as JSON")
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="ANSI colour output (default: auto)",
    )
    parser.add_argument("--quiet", action="store_true", help="hide the per-run progress line")
    parser.add_argument("--list", action="store_true", help="list profiles and axes, then exit")
    return parser.parse_args(argv)


def _resolve_axes(args: argparse.Namespace) -> list[str]:
    if args.axes:
        axes = [axis.strip() for axis in args.axes.split(",") if axis.strip()]
        unknown = [axis for axis in axes if axis not in AXIS_CHOICES]
        if unknown:
            raise SystemExit(f"unknown axes: {', '.join(unknown)} (choose from {', '.join(AXIS_CHOICES)})")
        return axes
    return list(PROFILE_AXES[args.profile])


def _print_list() -> None:
    print("profiles:")
    for name, axes in PROFILE_AXES.items():
        count = 1
        for axis in axes:
            count *= len(AXIS_CHOICES[axis])
        print(f"  {name:<9} {count:>4} runs  axes: {', '.join(axes)}")
    print(f"\nextra axes available via --axes: {', '.join(EXTRA_AXES)}")
    print("\naxes and values:")
    for axis, values in AXIS_CHOICES.items():
        shown = ", ".join("on" if v is True else "off" if v is False else str(v) for v in values)
        print(f"  {axis:<17} {shown}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list:
        _print_list()
        return 0

    palette = Palette(
        enabled=args.color == "always" or (args.color == "auto" and sys.stdout.isatty())
    )
    axes = _resolve_axes(args)
    dataset_filter = [name.strip() for name in args.datasets.split(",") if name.strip()]
    bundle, inventory = load_bundle(benchmark_limit=args.benchmark_limit, datasets=dataset_filter)
    pinned = RunConfig(nlp_backend=args.nlp_backend)
    runs = expand_runs(axes, pinned)

    runner = Runner(bundle)
    results: list[RunResult] = []
    started = time.perf_counter()
    for index, config in enumerate(runs, 1):
        if not args.quiet:
            sys.stderr.write(
                f"\r  running {index:>4}/{len(runs)} · {config.label(axes)[:78]:<78}"
            )
            sys.stderr.flush()
        results.append(runner.run(config))
    elapsed = time.perf_counter() - started
    if not args.quiet and runs:
        sys.stderr.write("\r" + " " * 96 + "\r")
        sys.stderr.flush()

    report = _build_report(palette, args, axes, runs, results, bundle, inventory, elapsed)
    sys.stdout.write(report + "\n")

    if args.json_path:
        payload = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "profile": args.profile,
            "axes": axes,
            "datasets": inventory,
            "runs": [run.as_dict(axes) for run in results],
        }
        Path(args.json_path).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        sys.stderr.write(f"wrote {args.json_path}\n")
    return 0


def _build_report(
    palette: Palette,
    args: argparse.Namespace,
    axes: Sequence[str],
    runs: Sequence[RunConfig],
    results: Sequence[RunResult],
    bundle: DatasetBundle,
    inventory: dict[str, int],
    elapsed: float,
) -> str:
    ranked = sorted(results, key=_rank_key)
    best = ranked[0]
    total_records = sum(run.records_processed for run in results)
    rules = "━" * 78
    lines: list[str] = []
    lines.append(palette.bold(rules))
    lines.append(palette.bold(" gov-survey-redact · SETTINGS MATRIX REPORT"))
    lines.append(palette.bold(rules))
    dataset_line = " · ".join(f"{name} {count}" for name, count in inventory.items())
    lines.append(f" datasets       {dataset_line}  ({bundle.small_record_count + len(bundle.benchmark)} records)")
    limit_note = (
        f" · benchmark limited to {args.benchmark_limit}" if args.benchmark_limit > 0 else ""
    )
    lines.append(
        f" configurations {len(runs)} runs · axes: {', '.join(axes)} · nlp-backend={args.nlp_backend}{limit_note}"
    )
    encoder_note = (
        "transformers installed — encoder backends available"
        if _module_available("transformers")
        else "transformers/spaCy not installed — encoder backends fall back to the gazetteer"
    )
    lines.append(f" environment    Python {platform.python_version()} · {encoder_note}")
    lines.append(
        " grading        leak/trace/over on final released text (2nd pass on queued records when"
        " enabled); review/category expectations on pass 1"
    )
    lines.append(
        f" runtime        {elapsed:.1f}s total · {total_records / max(elapsed, 1e-6):.1f} records/s"
        f" · best score {palette.score(best.score)} ({best.config.label(axes)})"
    )

    lines.append(section(palette, 1, "LEADERBOARD — ranked by composite score"))
    lines.append(_leaderboard(palette, results, axes, max(1, args.top)))

    lines.append(section(palette, 2, "FULL RUN MATRIX — every configuration evaluated"))
    lines.append(_full_matrix(palette, results, axes, args.show))

    lines.append(section(palette, 3, "SENSITIVITY — which settings move the needle"))
    lines.append(_sensitivity(palette, results, axes))

    lines.append(section(palette, 4, "RECOMMENDED SETTINGS"))
    lines.append(_recommendation(palette, results, axes, Runner(bundle), bundle))

    lines.append("")
    lines.append(palette.dim(rules))
    lines.append(
        palette.dim(
            " leak = identifiers surviving release · trace = 4-char digit / 5-char email "
            "fragments surviving · over = benign content destroyed · gate/idempotence "
            "failures are penalised"
        )
    )
    lines.append(palette.dim(rules))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
