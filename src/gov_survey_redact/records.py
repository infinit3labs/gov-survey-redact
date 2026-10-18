"""Asymmetric two-pass record redaction, review state machine, and hard-constraint adjudication.

Implements the hybrid two-pass workflow:
1. **First pass** (:meth:`SurveyRedactor.redact_record`):
   * Applies cross-field deterministic propagation from structured columns
     (:class:`CrossFieldDictionary`).
   * Runs deterministic recognisers + first-pass compact token classifier
     (``DeBERTa-v3-small`` / ``Gazetteer``) with ``[Question: ...]`` context.
   * Evaluates class-specific confidence routing, OCR confidence, checksum-failed
     OCR candidates, and re-identification risk band.
   * Transitions unambiguous records to ``ReviewState.AUTO_ACCEPTED`` and
     uncertain/high-risk records to ``ReviewState.PENDING_REVIEW``.
2. **Human review** (:meth:`SurveyRedactor.apply_review_decisions`):
   * Reviewers ``APPROVE``, ``EXTEND``, ``REMOVE``, or ``RELABEL`` spans with a
     ``reason_code`` and ``reviewer_principal``.
   * Transitions the record to ``ReviewState.REVIEWED`` and emits a canonical
     character-offset :class:`AnnotationRecord` for curated offline retraining.
3. **Second pass & Residual QA** (:meth:`SurveyRedactor.run_second_pass`):
   * Runs only on ``REVIEWED`` or ambiguous records.
   * Locks reviewer-approved spans as **immutable hard constraints** (the second-pass
     model may add more spans in unclaimed regions, and ``REMOVE`` decisions
     suppress false positives, but it can never silently remove a human-approved span).
   * Selects the specialist second-pass model: ``ModernBERT-base`` for long
     English narratives (> 512 tokens) or ``mDeBERTa-v3-base`` for multilingual
     or semantically ambiguous responses.
   * Re-runs deterministic residual validation (:func:`validate_residual_deterministic`)
     plus optional cohort disclosure control (:func:`protect`).
   * Transitions clean records to ``ReviewState.RELEASED`` and any record with
     residual risk to ``ReviewState.ESCALATED``.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .audit import AnnotationRecord, AuditLogEntry, LineageVersionSet
from .engine import RedactionResult, Redactor
from .ocr import OCRFieldLayout, VisualRedactionBox
from .propagation import CrossFieldDictionary, KnownRecordDetector
from .residual import (
    DisclosureDecision,
    DisclosurePolicy,
    protect,
    validate_residual_deterministic,
)
from .risk import Audience, RiskAssessment, RiskBand, assess
from .types import Category, Finding, Span, Stage

_LABEL_CATEGORIES = frozenset({Category.INTERNAL_TERM, Category.AWARD})


class ReviewState(str, enum.Enum):
    """State machine for the hybrid two-pass redaction lifecycle."""

    FIRST_PASS = "FIRST_PASS"
    AUTO_ACCEPTED = "AUTO_ACCEPTED"
    PENDING_REVIEW = "PENDING_REVIEW"
    REVIEWED = "REVIEWED"
    SECOND_PASS = "SECOND_PASS"
    RESIDUAL_QA = "RESIDUAL_QA"
    RELEASED = "RELEASED"
    ESCALATED = "ESCALATED"

    def __str__(self) -> str:
        return self.value


class ReviewAction(str, enum.Enum):
    """Human reviewer decision on a proposed or added span."""

    APPROVE = "APPROVE"
    EXTEND = "EXTEND"
    REMOVE = "REMOVE"
    RELABEL = "RELABEL"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ReviewerDecision:
    """One human reviewer action that becomes a hard constraint in the second pass."""

    span: Span
    action: ReviewAction
    category: Category
    reviewer_principal: str = "reviewer@agency.gov.au"
    reason_code: str = "CONFIRMED_PII"

    def clamped_span(self, text: str) -> Span:
        """Return the decision span trimmed to ``text`` bounds (may be empty)."""
        return clamp_span(self.span, len(text))

    def to_finding(self, text: str) -> Finding:
        span = self.clamped_span(text)
        return Finding(
            span=span,
            category=self.category,
            detector=f"human_reviewer:{self.action.value.lower()}",
            stage=Stage.DETERMINISTIC,
            confidence=1.0,
            value=text[span.start : span.end],
            metadata=(
                ("hard_constraint", True),
                ("reviewer_principal", self.reviewer_principal),
                ("reason_code", self.reason_code),
                ("recognizer", "HUMAN_REVIEW_HARD_CONSTRAINT"),
            ),
        )


def clamp_span(span: Span, text_length: int) -> Span:
    """Trim ``span`` to ``[0, text_length)``; may return an empty span."""
    start = max(0, min(span.start, text_length))
    end = max(start, min(span.end, text_length))
    return Span(start, end)


@dataclass(frozen=True, slots=True)
class RecordRedaction:
    """One redacted survey response with provenance, review state, and audit trail."""

    record_id: str | None
    text: str
    original_text: str
    findings: tuple[Finding, ...]
    risk: RiskAssessment
    uncertain_findings: tuple[Finding, ...] = ()
    review_state: ReviewState = ReviewState.AUTO_ACCEPTED
    review_reasons: tuple[str, ...] = ()
    disclosure: DisclosureDecision | None = None
    visual_boxes: tuple[VisualRedactionBox, ...] = ()
    audit_entries: tuple[AuditLogEntry, ...] = ()
    reviewer_decisions: tuple[ReviewerDecision, ...] = ()
    second_pass_model_used: str | None = None
    residual_leaks: tuple[Finding, ...] = ()
    #: True when Pass 1 switched to the OCR-tolerant recogniser profile because
    #: of low OCR confidence; the second pass must reuse that profile.
    used_ocr_mode: bool = False

    @property
    def redacted_categories(self) -> frozenset[Category]:
        return frozenset(f.category for f in self.findings)

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def withheld(self) -> bool:
        return self.disclosure is not None and self.disclosure.action.value == "suppress"

    def findings_json(self) -> str:
        """Compact findings as JSON with offsets into ``original_text``."""
        return json.dumps(
            [
                {
                    "category": f.category.value,
                    "start": f.span.start,
                    "end": f.span.end,
                    "detector": f.detector,
                    "stage": f.stage.value,
                    "confidence": round(f.confidence, 4),
                    "metadata": f.meta(),
                }
                for f in self.findings
            ],
            separators=(",", ":"),
            default=str,
        )

    def audit_json(self) -> str:
        """Privacy-safe JSON summary of findings, risk, review state, and disclosure."""
        return json.dumps(
            {
                "record_id": self.record_id,
                "review_state": self.review_state.value,
                "review_reasons": list(self.review_reasons),
                "second_pass_model": self.second_pass_model_used,
                "finding_count": self.finding_count,
                "uncertain_count": len(self.uncertain_findings),
                "categories": sorted(c.value for c in self.redacted_categories),
                "risk": self.risk.as_dict(),
                "disclosure": (
                    None if self.disclosure is None else self.disclosure.as_dict()
                ),
                "findings": json.loads(self.findings_json()),
                "audit_log": [e.as_dict() for e in self.audit_entries],
            },
            separators=(",", ":"),
            default=str,
        )


class SurveyRedactor:
    """Orchestrates the two-pass hybrid redaction workflow for government surveys."""

    def __init__(
        self,
        redactor: Redactor,
        *,
        audience: Audience = Audience.INTERNAL,
        review_threshold: RiskBand = RiskBand.HIGH,
        residual_policy: DisclosurePolicy | None = None,
        ocr_confidence_threshold: float = 0.80,
        long_narrative_tokens: int = 512,
        lineage: LineageVersionSet | None = None,
    ) -> None:
        self.redactor = redactor
        self.audience = audience
        self.review_threshold = review_threshold
        self.residual_policy = residual_policy
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.long_narrative_tokens = long_narrative_tokens
        self.lineage = lineage or LineageVersionSet(model_name=redactor.policy.nlp_backend)
        self._ocr_redactor: Redactor | None = None
        self._second_pass_redactors: dict[str, Redactor] = {}

    def redact_record(
        self,
        text: str | None,
        *,
        record_id: str | None = None,
        field_id: str = "Q1",
        question_context: str | None = None,
        structured_context: Mapping[str, object] | CrossFieldDictionary | None = None,
        ocr_layout: OCRFieldLayout | None = None,
        ocr_confidence: float | None = None,
        language: str = "en",
        job_run_id: str = "local-run",
    ) -> RecordRedaction:
        """Execute Pass 1 (fast deterministic + cross-field + compact NER + routing)."""
        if text is None:
            text = ""
        elif not isinstance(text, str):
            # Survey frames occasionally carry numeric/boolean cells; redact their
            # textual form rather than crashing the batch mid-partition.
            text = str(text)
        if not text.strip():
            return RecordRedaction(
                record_id=record_id,
                text=text,
                original_text=text,
                findings=(),
                risk=RiskAssessment(score=0.0, band=RiskBand.LOW, signals=(), word_count=0),
                review_state=ReviewState.AUTO_ACCEPTED,
            )

        extra_detectors = self._build_cross_field_detectors(structured_context)
        effective_ocr_conf = (
            ocr_layout.min_confidence
            if ocr_layout is not None
            else ocr_confidence
        )
        use_ocr_mode = (
            effective_ocr_conf is not None
            and effective_ocr_conf < self.ocr_confidence_threshold
            and not self.redactor.policy.ocr_mode
        )
        active_redactor = (
            self._redactor_with_ocr_mode()
            if (effective_ocr_conf is not None and effective_ocr_conf < self.ocr_confidence_threshold)
            else self.redactor
        )

        result: RedactionResult = active_redactor.redact(
            text,
            extra_detectors=extra_detectors,
            question_context=question_context,
        )
        labels = _labels_for(result.findings)
        risk = assess(text, result.findings, audience=self.audience, lexicon_labels=labels)

        review_reasons = self._evaluate_routing(
            findings=result.findings,
            uncertain_findings=result.uncertain_findings,
            risk=risk,
            ocr_confidence=effective_ocr_conf,
            language=language,
        )
        review_state = (
            ReviewState.PENDING_REVIEW if review_reasons else ReviewState.AUTO_ACCEPTED
        )

        released = result.text
        disclosure: DisclosureDecision | None = None
        if self.residual_policy is not None:
            released, disclosure = protect(result.text, self.residual_policy)
            if disclosure.escalated and "residual_disclosure_escalation" not in review_reasons:
                review_reasons = (*review_reasons, "residual_disclosure_escalation")
                review_state = ReviewState.PENDING_REVIEW

        visual_boxes = (
            tuple(ocr_layout.visual_boxes_for(result.findings))
            if ocr_layout is not None and ocr_layout.text == text
            else ()
        )

        doc_id = record_id or "unknown"
        audit_entries = tuple(
            AuditLogEntry.from_finding(
                f,
                job_run_id=job_run_id,
                document_id=doc_id,
                field_id=field_id,
                lineage=self.lineage,
                action="FIRST_PASS_REDACT",
                review_status=review_state.value,
                salt=self.redactor.policy.salt,
            )
            for f in result.findings
        )

        return RecordRedaction(
            record_id=record_id,
            text=released,
            original_text=text,
            findings=result.findings,
            uncertain_findings=result.uncertain_findings,
            risk=risk,
            review_state=review_state,
            review_reasons=review_reasons,
            disclosure=disclosure,
            visual_boxes=visual_boxes,
            audit_entries=audit_entries,
            used_ocr_mode=use_ocr_mode,
        )

    def apply_review_decisions(
        self,
        redaction: RecordRedaction,
        decisions: Sequence[ReviewerDecision],
        *,
        field_id: str = "Q1",
        language: str = "en",
        template_id: str = "gov-survey-v1",
    ) -> tuple[RecordRedaction, AnnotationRecord]:
        """Record human reviewer decisions and transition to ``ReviewState.REVIEWED``.

        Also produces a canonical character-offset :class:`AnnotationRecord` for
        the curated offline retraining dataset (never used for uncontrolled
        online learning).
        """
        hard_findings: list[Finding] = []
        removed_spans: list[Span] = []
        for decision in decisions:
            span = decision.clamped_span(redaction.original_text)
            if span.length == 0:
                # Stale, reversed, or zero-length reviewer span: nothing to act on.
                continue
            if decision.action is ReviewAction.REMOVE:
                removed_spans.append(span)
            else:
                hard_findings.append(
                    replace(decision, span=span).to_finding(redaction.original_text)
                )
        kept_existing = [
            f
            for f in redaction.findings
            if not any(f.span.overlaps(r) for r in removed_spans)
            and not any(f.span.overlaps(h.span) for h in hard_findings)
        ]
        merged_findings = tuple(sorted([*hard_findings, *kept_existing], key=lambda f: f.span.start))
        annotation = AnnotationRecord.from_findings(
            document_id=redaction.record_id or "unknown",
            field_id=field_id,
            text=redaction.original_text,
            findings=merged_findings,
            language=language,
            template_id=template_id,
            adjudication_status="FINAL",
            annotation_source="HUMAN_ADJUDICATED",
        )
        updated = replace(
            redaction,
            review_state=ReviewState.REVIEWED,
            reviewer_decisions=tuple(decisions),
        )
        return updated, annotation

    def run_second_pass(
        self,
        redaction: RecordRedaction,
        *,
        question_context: str | None = None,
        structured_context: Mapping[str, object] | CrossFieldDictionary | None = None,
        language: str = "en",
        field_id: str = "Q1",
        job_run_id: str = "second-pass-run",
    ) -> RecordRedaction:
        """Execute Pass 2 on ``REVIEWED`` or ambiguous records.

        * Locks reviewer ``APPROVE`` / ``EXTEND`` / ``RELABEL`` decisions as
          immutable hard constraints (never removed or shrunk by the second model).
        * Suppresses spans explicitly marked ``REMOVE`` by the reviewer.
        * Selects ``ModernBERT-base`` for long English narratives (> ``long_narrative_tokens``)
          and ``mDeBERTa-v3-base`` for multilingual or semantically ambiguous records.
        * Re-runs deterministic residual validation (:func:`validate_residual_deterministic`)
          after the second pass and transitions to ``RELEASED`` or ``ESCALATED``.
        """
        text = redaction.original_text
        if not text.strip():
            return replace(redaction, review_state=ReviewState.RELEASED)

        approx_tokens = len(text.split())
        if language.lower() == "en" and approx_tokens > self.long_narrative_tokens:
            second_model = "modernbert-base"
        else:
            second_model = "mdeberta-v3-base"

        text_length = len(text)
        hard_constraints: list[Finding] = []
        suppressed_spans: list[Span] = []
        for decision in redaction.reviewer_decisions:
            span = clamp_span(decision.span, text_length)
            if span.length == 0:
                # Reviewer span no longer aligns with this text version: ignore it
                # rather than masking at stale offsets.
                continue
            if decision.action is ReviewAction.REMOVE:
                suppressed_spans.append(span)
            else:
                hard_constraints.append(
                    replace(decision, span=span).to_finding(text)
                )

        extra_detectors = self._build_cross_field_detectors(structured_context)
        # Reuse the OCR-tolerant recogniser profile when Pass 1 enabled it,
        # otherwise review would silently un-mask OCR-corrupted identifiers.
        base_redactor = (
            self._redactor_with_ocr_mode() if redaction.used_ocr_mode else self.redactor
        )
        cache_key = f"{second_model}|ocr={redaction.used_ocr_mode}"
        second_redactor = self._second_pass_redactors.get(cache_key)
        if second_redactor is None:
            second_redactor = base_redactor.with_policy(
                nlp_enabled=True,
                nlp_backend=second_model,
                overlap_mode="union",
            )
            self._second_pass_redactors[cache_key] = second_redactor

        result = second_redactor.redact(
            text,
            extra_detectors=extra_detectors,
            question_context=question_context,
            hard_constraints=hard_constraints,
            suppressed_spans=suppressed_spans,
        )

        # Residual deterministic QA pass (including cross-field known values)
        all_detectors = (*extra_detectors, *base_redactor._detectors)
        residual_leaks = [
            leak
            for leak in validate_residual_deterministic(result.text, all_detectors)
            if not any(leak.span.overlaps(s) for s in suppressed_spans)
        ]

        released = result.text
        disclosure = redaction.disclosure
        if self.residual_policy is not None:
            released, disclosure = protect(result.text, self.residual_policy)

        final_state = (
            ReviewState.ESCALATED
            if (residual_leaks or (disclosure is not None and disclosure.escalated))
            else ReviewState.RELEASED
        )

        labels = _labels_for(result.findings)
        risk = assess(text, result.findings, audience=self.audience, lexicon_labels=labels)
        doc_id = redaction.record_id or "unknown"
        reviewer = (
            redaction.reviewer_decisions[0].reviewer_principal
            if redaction.reviewer_decisions
            else None
        )
        reason = (
            redaction.reviewer_decisions[0].reason_code
            if redaction.reviewer_decisions
            else None
        )
        audit_entries = tuple(
            AuditLogEntry.from_finding(
                f,
                job_run_id=job_run_id,
                document_id=doc_id,
                field_id=field_id,
                lineage=self.lineage,
                action="SECOND_PASS_REDACT",
                review_status=final_state.value,
                salt=self.redactor.policy.salt,
                reviewer_principal=reviewer,
                review_reason=reason,
                second_pass_model=second_model,
                final_disposition=final_state.value,
            )
            for f in result.findings
        )

        return replace(
            redaction,
            text=released,
            findings=result.findings,
            uncertain_findings=(),
            risk=risk,
            review_state=final_state,
            disclosure=disclosure,
            audit_entries=audit_entries,
            second_pass_model_used=second_model,
            residual_leaks=tuple(residual_leaks),
        )

    def _build_cross_field_detectors(
        self,
        structured_context: Mapping[str, object] | CrossFieldDictionary | None,
    ) -> tuple[KnownRecordDetector, ...]:
        if structured_context is None:
            return ()
        dictionary = (
            structured_context
            if isinstance(structured_context, CrossFieldDictionary)
            else CrossFieldDictionary.from_record(structured_context)
        )
        if not dictionary:
            return ()
        return (KnownRecordDetector(dictionary),)

    def _redactor_with_ocr_mode(self) -> Redactor:
        """Reuse one OCR-tolerant redactor variant rather than rebuilding per record."""
        if self.redactor.policy.ocr_mode:
            return self.redactor
        if self._ocr_redactor is None:
            self._ocr_redactor = self.redactor.with_policy(ocr_mode=True)
        return self._ocr_redactor

    def _evaluate_routing(
        self,
        *,
        findings: Sequence[Finding],
        uncertain_findings: Sequence[Finding],
        risk: RiskAssessment,
        ocr_confidence: float | None,
        language: str,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        order = list(RiskBand)
        if order.index(risk.band) >= order.index(self.review_threshold):
            reasons.append(f"risk_band_{risk.band.value}")
        if uncertain_findings:
            reasons.append("low_confidence_or_ambiguous_span")
        if any(f.meta().get("checksum") == "failed_ocr_candidate" for f in findings):
            reasons.append("ocr_checksum_failure")
        if any(f.meta().get("ocr_repaired") for f in findings):
            reasons.append("ocr_repaired_identifier")
        if ocr_confidence is not None and ocr_confidence < self.ocr_confidence_threshold:
            reasons.append(f"low_ocr_confidence({ocr_confidence:.2f})")
        if language.lower() not in {"en", "eng", "en-au", "en_au"}:
            reasons.append(f"non_english_language({language})")
        return tuple(reasons)

    def needs_review(self, redaction: RecordRedaction) -> bool:
        """True when a record is routed to human review or supervisor escalation."""
        if redaction.review_state in (ReviewState.PENDING_REVIEW, ReviewState.ESCALATED):
            return True
        order = list(RiskBand)
        return order.index(redaction.risk.band) >= order.index(self.review_threshold)

    def redact_many(
        self,
        texts: Sequence[str | None],
        record_ids: Sequence[str | None] | None = None,
    ) -> list[RecordRedaction]:
        """Redact a batch, padding missing ``record_ids`` instead of dropping texts."""
        ids: list[str | None] = list(record_ids) if record_ids is not None else []
        if len(ids) < len(texts):
            ids.extend([None] * (len(texts) - len(ids)))
        return [
            self.redact_record(text, record_id=record_id)
            for text, record_id in zip(texts, ids, strict=False)
        ]


def _labels_for(findings: Sequence[Finding]) -> list[str]:
    return [
        str(f.meta()["label"])
        for f in findings
        if f.category in _LABEL_CATEGORIES and "label" in f.meta()
    ]


__all__ = [
    "RecordRedaction",
    "ReviewAction",
    "ReviewState",
    "ReviewerDecision",
    "SurveyRedactor",
    "clamp_span",
]
