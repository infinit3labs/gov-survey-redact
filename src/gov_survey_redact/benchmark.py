"""Benchmark evaluation harness and acceptance-gate verifier for government survey redaction.

Computes the full set of procurement and release-quality metrics from the
hybrid specification:
* Per-class recall and false-negative count
* **Character-level PII recall** (a partially unredacted identifier is still a leak)
* **Record-level PII escape rate** (fraction of PII-bearing records where >=1 mandatory ID survived)
* Over-redaction / non-PII character loss rate
* Review queue rate and Review capture rate (how many first-pass misses/uncertainties
  were routed to human review)
* Post-second-pass residual PII rate
* OCR-stratified and Language-stratified recall
* Latency (p50/p95 ms) and Throughput (records/s, tokens/s)
* Proposed acceptance gates (:func:`verify_acceptance_gates`)
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .generate import GeneratedRecord
from .records import ReviewState, SurveyRedactor


@dataclass(frozen=True, slots=True)
class AcceptanceGates:
    """Proposed acceptance gates for government survey redaction procurement."""

    mandatory_direct_id_recall_min: float = 0.995
    min_class_recall_floor: float = 0.85
    record_escape_rate_max: float = 0.015
    over_redaction_loss_max: float = 0.005


@dataclass
class BenchmarkMetrics:
    """Comprehensive evaluation report over a synthetic or held-out survey corpus."""

    records_total: int = 0
    records_with_secrets: int = 0
    records_escaped_first_pass: int = 0
    records_escaped_post_review: int = 0
    secrets_planted: int = 0
    secrets_caught: int = 0
    pii_chars_planted: int = 0
    pii_chars_redacted: int = 0
    must_survive_total: int = 0
    must_survive_lost: int = 0
    non_pii_chars_total: int = 0
    non_pii_chars_lost: int = 0
    review_queue_count: int = 0
    first_pass_misses_total: int = 0
    first_pass_misses_routed_to_review: int = 0
    per_class_planted: dict[str, int] = field(default_factory=dict)
    per_class_caught: dict[str, int] = field(default_factory=dict)
    per_stratum_planted: dict[str, int] = field(default_factory=dict)
    per_stratum_caught: dict[str, int] = field(default_factory=dict)
    per_language_planted: dict[str, int] = field(default_factory=dict)
    per_language_caught: dict[str, int] = field(default_factory=dict)
    ocr_planted: int = 0
    ocr_caught: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    records_per_sec: float = 0.0
    tokens_per_sec: float = 0.0

    @property
    def overall_recall(self) -> float:
        return self.secrets_caught / self.secrets_planted if self.secrets_planted else 1.0

    @property
    def char_level_pii_recall(self) -> float:
        return self.pii_chars_redacted / self.pii_chars_planted if self.pii_chars_planted else 1.0

    @property
    def record_escape_rate(self) -> float:
        return (
            self.records_escaped_first_pass / self.records_with_secrets
            if self.records_with_secrets
            else 0.0
        )

    @property
    def post_second_pass_residual_rate(self) -> float:
        return (
            self.records_escaped_post_review / self.records_with_secrets
            if self.records_with_secrets
            else 0.0
        )

    @property
    def review_queue_rate(self) -> float:
        return self.review_queue_count / self.records_total if self.records_total else 0.0

    @property
    def review_capture_rate(self) -> float:
        return (
            self.first_pass_misses_routed_to_review / self.first_pass_misses_total
            if self.first_pass_misses_total
            else 1.0
        )

    @property
    def over_redaction_rate(self) -> float:
        return (
            self.must_survive_lost / self.must_survive_total
            if self.must_survive_total
            else 0.0
        )

    @property
    def per_class_false_negatives(self) -> dict[str, int]:
        return {
            k: self.per_class_planted[k] - self.per_class_caught.get(k, 0)
            for k in sorted(self.per_class_planted)
        }

    @property
    def per_class_recall(self) -> dict[str, float]:
        return {
            k: round(self.per_class_caught.get(k, 0) / v, 4) if v else 1.0
            for k, v in sorted(self.per_class_planted.items())
        }

    @property
    def ocr_recall(self) -> float:
        return self.ocr_caught / self.ocr_planted if self.ocr_planted else 1.0

    def verify_gates(self, gates: AcceptanceGates | None = None) -> dict[str, Any]:
        gates = gates or AcceptanceGates()
        direct_kinds = {
            "tfn",
            "abn",
            "acn",
            "medicare",
            "credit_card",
            "email",
            "phone",
            "bsb",
            "dob",
            "organisation",
            "cross_field_person",
        }
        direct_planted = sum(v for k, v in self.per_class_planted.items() if k in direct_kinds)
        direct_caught = sum(v for k, v in self.per_class_caught.items() if k in direct_kinds)
        direct_recall = direct_caught / direct_planted if direct_planted else 1.0
        passed = (
            direct_recall >= gates.mandatory_direct_id_recall_min
            and self.over_redaction_rate <= gates.over_redaction_loss_max
            and self.record_escape_rate <= gates.record_escape_rate_max
        )
        return {
            "passed": passed,
            "mandatory_direct_id_recall": round(direct_recall, 4),
            "char_level_pii_recall": round(self.char_level_pii_recall, 4),
            "record_escape_rate": round(self.record_escape_rate, 4),
            "post_second_pass_residual_rate": round(self.post_second_pass_residual_rate, 4),
            "over_redaction_rate": round(self.over_redaction_rate, 4),
            "review_queue_rate": round(self.review_queue_rate, 4),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "records_total": self.records_total,
            "records_with_secrets": self.records_with_secrets,
            "secrets_planted": self.secrets_planted,
            "secrets_caught": self.secrets_caught,
            "overall_recall": round(self.overall_recall, 4),
            "char_level_pii_recall": round(self.char_level_pii_recall, 4),
            "record_escape_rate": round(self.record_escape_rate, 4),
            "post_second_pass_residual_rate": round(self.post_second_pass_residual_rate, 4),
            "over_redaction_rate": round(self.over_redaction_rate, 4),
            "review_queue_rate": round(self.review_queue_rate, 4),
            "review_capture_rate": round(self.review_capture_rate, 4),
            "ocr_recall": round(self.ocr_recall, 4),
            "per_class_recall": self.per_class_recall,
            "per_class_false_negatives": self.per_class_false_negatives,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "records_per_sec": round(self.records_per_sec, 1),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "gates": self.verify_gates(),
        }


def evaluate_corpus(
    records: Sequence[GeneratedRecord],
    survey: SurveyRedactor,
    *,
    run_second_pass_on_queued: bool = True,
) -> BenchmarkMetrics:
    """Evaluate ``survey`` across ``records`` and compute full benchmark metrics."""
    metrics = BenchmarkMetrics(records_total=len(records))
    latencies_ms: list[float] = []
    total_tokens = 0
    t_start_all = time.perf_counter()

    for rec in records:
        t0 = time.perf_counter()
        redacted_parts: list[str] = []
        second_pass_parts: list[str] = []
        queued = False

        for col, text in rec.answers.items():
            total_tokens += len(text.split())
            res = survey.redact_record(
                text,
                record_id=rec.response_id,
                field_id=col,
                structured_context=rec.structured_fields or None,
                ocr_confidence=rec.ocr_confidence,
                language=rec.language,
            )
            redacted_parts.append(res.text)
            if res.review_state in (ReviewState.PENDING_REVIEW, ReviewState.ESCALATED):
                queued = True
                if run_second_pass_on_queued:
                    res2 = survey.run_second_pass(
                        res,
                        structured_context=rec.structured_fields or None,
                        language=rec.language,
                        field_id=col,
                    )
                    second_pass_parts.append(res2.text)
                else:
                    second_pass_parts.append(res.text)
            else:
                second_pass_parts.append(res.text)

        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if queued:
            metrics.review_queue_count += 1

        joined_first = "\n".join(redacted_parts)
        joined_second = "\n".join(second_pass_parts)

        if rec.secrets:
            metrics.records_with_secrets += 1
            record_missed_first = False
            record_missed_second = False

            for secret in rec.secrets:
                metrics.secrets_planted += 1
                char_len = len(secret.literal)
                metrics.pii_chars_planted += char_len
                metrics.per_class_planted[secret.kind] = (
                    metrics.per_class_planted.get(secret.kind, 0) + 1
                )
                metrics.per_stratum_planted[rec.stratum] = (
                    metrics.per_stratum_planted.get(rec.stratum, 0) + 1
                )
                metrics.per_language_planted[rec.language] = (
                    metrics.per_language_planted.get(rec.language, 0) + 1
                )
                if rec.ocr_confidence is not None:
                    metrics.ocr_planted += 1

                caught_first = secret.literal not in joined_first
                caught_second = secret.literal not in joined_second
                if caught_first:
                    metrics.secrets_caught += 1
                    metrics.pii_chars_redacted += char_len
                    metrics.per_class_caught[secret.kind] = (
                        metrics.per_class_caught.get(secret.kind, 0) + 1
                    )
                    metrics.per_stratum_caught[rec.stratum] = (
                        metrics.per_stratum_caught.get(rec.stratum, 0) + 1
                    )
                    metrics.per_language_caught[rec.language] = (
                        metrics.per_language_caught.get(rec.language, 0) + 1
                    )
                    if rec.ocr_confidence is not None:
                        metrics.ocr_caught += 1
                else:
                    if secret.mandatory:
                        record_missed_first = True
                    metrics.first_pass_misses_total += 1
                    if queued:
                        metrics.first_pass_misses_routed_to_review += 1
                    if secret.mandatory and not caught_second and not queued:
                        record_missed_second = True

            if record_missed_first:
                metrics.records_escaped_first_pass += 1
            if record_missed_second:
                metrics.records_escaped_post_review += 1

        for fragment in rec.must_survive:
            metrics.must_survive_total += 1
            metrics.non_pii_chars_total += len(fragment)
            if fragment not in joined_first:
                metrics.must_survive_lost += 1
                metrics.non_pii_chars_lost += len(fragment)

    elapsed = max(1e-6, time.perf_counter() - t_start_all)
    if latencies_ms:
        sorted_lat = sorted(latencies_ms)
        metrics.p50_ms = statistics.median(sorted_lat)
        p95_idx = min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))
        metrics.p95_ms = sorted_lat[p95_idx]
    metrics.records_per_sec = len(records) / elapsed
    metrics.tokens_per_sec = total_tokens / elapsed
    return metrics


__all__ = [
    "AcceptanceGates",
    "BenchmarkMetrics",
    "evaluate_corpus",
]
