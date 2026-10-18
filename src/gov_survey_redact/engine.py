"""Redaction policy, overlap resolution (longest & union-of-spans), and core engine.

Execution order:
1. **Cross-field known-record dictionary** (when supplied from structured columns)
   and **deterministic detectors** run first.
2. Deterministic findings are resolved for overlap using either ``longest`` or
   ``union`` (union-of-spans) semantics.
3. **NLP / Encoder Token-Classification** runs on unresolved regions with
   optional ``[Question: ...]`` context. Any NLP span touching a deterministic
   span is either discarded (in ``longest`` mode) or merged only when unioning
   same-class person/org spans without splitting validated checksum IDs.
4. **Confidence routing**: confidence is treated as a routing threshold rather
   than an erase threshold. Spans above ``class_thresholds[category]`` are
   auto-redacted; spans between ``review_floor`` and ``class_thresholds[category]``
   are retained in ``uncertain_findings`` to route the record to ``PENDING_REVIEW``.
5. Surviving spans are rewritten right-to-left (generalising quasi-identifiers
   and masking/labelling direct & organisational identifiers) while respecting
   already-redacted placeholders for strict idempotence.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

from . import masking
from .detectors import au_ids, generic, nlp
from .detectors.base import Detector
from .generalise import generalise
from .masking import build_masker
from .types import (
    Category,
    Finding,
    MaskStyle,
    RedactionError,
    Span,
    Stage,
)

StyleMap = dict[Category | str, MaskStyle | str]

#: Process-wide cache so ``with_policy`` variants (OCR mode, second-pass
#: specialist model) reuse the same encoder/gazetteer backend instead of
#: reloading transformer weights per record or per record batch.
_BACKEND_CACHE: dict[tuple[str, str], object] = {}

# Class-specific auto-redaction confidence thresholds.
# Findings scoring below these (down to review_floor) trigger human review routing.
DEFAULT_CLASS_THRESHOLDS: dict[Category, float] = {
    Category.TFN: 0.85,
    Category.ABN: 0.85,
    Category.ACN: 0.85,
    Category.MEDICARE: 0.85,
    Category.MEDICARE_ENROLMENT: 0.70,
    Category.CENTRELINK_CRN: 0.85,
    Category.PASSPORT: 0.85,
    Category.DRIVERS_LICENCE: 0.80,
    Category.CREDIT_CARD: 0.80,
    Category.BANK_BSB: 0.75,
    Category.IBAN: 0.85,
    Category.EMAIL: 0.85,
    Category.PHONE: 0.85,
    Category.NAME: 0.75,
    Category.STREET_ADDRESS: 0.70,
    Category.ORGANISATION_NAME: 0.78,
    Category.IMPLICATION: 0.65,
}


@dataclass(frozen=True, slots=True)
class Policy:
    """Configuration governing what to detect, how to route uncertainty, and how to rewrite."""

    categories: frozenset[Category] | None = None
    generalise_categories: frozenset[Category] = frozenset()
    styles: StyleMap = field(default_factory=dict)
    nlp_enabled: bool = False
    #: "deberta-v3-small", "mdeberta-v3-base", "modernbert-base", "gazetteer", or "spacy"
    nlp_backend: str = "gazetteer"
    spacy_model: str = "en_core_web_lg"
    nlp_min_confidence: float = 0.55
    min_confidence: float = 0.0
    #: Floor down to which uncertain findings are captured for the review queue
    review_floor: float = 0.40
    #: Class-specific routing thresholds (above -> auto-redact, below -> review queue)
    class_thresholds: Mapping[Category, float] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_THRESHOLDS)
    )
    #: "longest" (exact span winner) or "union" (union-of-spans for overlapping detections)
    overlap_mode: str = "longest"
    #: Enable OCR-tolerant checksum fallback (retains failed-checksum candidates for review)
    ocr_mode: bool = False
    require_context: bool = True
    salt: bytes | None = None
    default_style: MaskStyle = MaskStyle.MASK
    mask_char: str = "*"
    partial_keep_prefix: int = 0
    partial_keep_suffix: int = 4
    strict: bool = False

    def style_for(self, category: Category) -> MaskStyle:
        style = self.styles.get(category, self.styles.get("default"))
        if style is None:
            return self.default_style
        return style if isinstance(style, MaskStyle) else MaskStyle(str(style))

    def enabled(self, category: Category) -> bool:
        return self.categories is None or category in self.categories

    def threshold_for(self, category: Category) -> float:
        return float(self.class_thresholds.get(category, 0.70))


@dataclass(frozen=True)
class RedactionResult:
    """Outcome of a single redaction pass."""

    text: str
    findings: tuple[Finding, ...]
    uncertain_findings: tuple[Finding, ...] = ()

    @cached_property
    def counts(self) -> Counter[str]:
        return Counter(f.category.value for f in self.findings)

    @cached_property
    def stage_counts(self) -> Counter[str]:
        return Counter(f.stage.value for f in self.findings)

    @property
    def redacted_categories(self) -> frozenset[Category]:
        return frozenset(f.category for f in self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self) -> Iterator[Finding]:
        return iter(self.findings)

    def summary(self) -> str:
        parts = ", ".join(f"{name}={count}" for name, count in sorted(self.counts.items()))
        deterministic = self.stage_counts.get(Stage.DETERMINISTIC.value, 0)
        nlp_count = self.stage_counts.get(Stage.NLP.value, 0)
        return f"{len(self.findings)} findings ({deterministic} deterministic, {nlp_count} nlp): {parts}"


DEFAULT_DETECTORS: tuple[type[Detector], ...] = (
    au_ids.TFNDetector,
    au_ids.ABNDetector,
    au_ids.ACNDetector,
    au_ids.MedicareDetector,
    au_ids.MedicareEnrolmentDetector,
    au_ids.CentrelinkCRNDetector,
    au_ids.EmployeeReferenceDetector,
    au_ids.ClaimReferenceDetector,
    au_ids.CreditCardDetector,
    au_ids.IBANDetector,
    au_ids.BSBDetector,
    au_ids.DriversLicenceDetector,
    au_ids.PassportDetector,
    generic.URLCredentialDetector,
    generic.EmailDetector,
    generic.PhoneDetector,
    generic.IPv4Detector,
    generic.IPv6Detector,
    generic.MACDetector,
    generic.StatePostcodeDetector,
    generic.PostcodeDetector,
    generic.StreetAddressDetector,
    generic.DateDetector,
)


class Redactor:
    """Applies a :class:`Policy` and detector pipeline to text."""

    def __init__(
        self,
        policy: Policy | None = None,
        detectors: Sequence[Detector] | None = None,
    ) -> None:
        self.policy = policy or Policy()
        if detectors is not None:
            self._detectors: tuple[Detector, ...] = tuple(detectors)
        else:
            built: list[Detector] = []
            for cls in DEFAULT_DETECTORS:
                if issubclass(cls, au_ids.ChecksumDetector):
                    built.append(cls(ocr_mode=self.policy.ocr_mode))
                else:
                    built.append(cls())  # type: ignore[call-arg]
            self._detectors = tuple(built)
        self._order = {d.name: index for index, d in enumerate(self._detectors)}
        self._nlp_backend = self._build_nlp_backend()

    def _build_nlp_backend(self):
        if not self.policy.nlp_enabled:
            return None
        key = (self.policy.nlp_backend.strip().lower(), self.policy.spacy_model)
        backend = _BACKEND_CACHE.get(key)
        if backend is None:
            backend = nlp.build_nlp_backend(
                self.policy.nlp_backend,
                spacy_model=self.policy.spacy_model,
            )
            _BACKEND_CACHE[key] = backend
        return backend

    def with_policy(self, **changes: Any) -> Redactor:
        new_policy = replace(self.policy, **changes)
        detectors = self._detectors
        if "ocr_mode" in changes:
            detectors = tuple(
                type(d)(ocr_mode=new_policy.ocr_mode)
                if isinstance(d, au_ids.ChecksumDetector)
                else d
                for d in self._detectors
            )
        return Redactor(policy=new_policy, detectors=detectors)

    def scan(
        self,
        text: str,
        *,
        extra_detectors: Sequence[Detector] = (),
        question_context: str | None = None,
    ) -> list[Finding]:
        """Detect PII without altering ``text``."""
        findings, _ = self._resolve_with_uncertainty(
            text,
            extra_detectors=extra_detectors,
            question_context=question_context,
        )
        return findings

    def redact(
        self,
        text: str,
        *,
        extra_detectors: Sequence[Detector] = (),
        question_context: str | None = None,
        hard_constraints: Sequence[Finding] = (),
        suppressed_spans: Sequence[Span] = (),
    ) -> RedactionResult:
        """Redact ``text`` and return redacted string, applied findings, and uncertain findings."""
        if not isinstance(text, str):
            raise RedactionError(f"expected str, got {type(text).__name__}")
        findings, uncertain = self._resolve_with_uncertainty(
            text,
            extra_detectors=extra_detectors,
            question_context=question_context,
            hard_constraints=hard_constraints,
            suppressed_spans=suppressed_spans,
        )
        return RedactionResult(
            text=self._apply(text, findings),
            findings=tuple(findings),
            uncertain_findings=tuple(uncertain),
        )

    def redact_text(
        self,
        text: str,
        *,
        extra_detectors: Sequence[Detector] = (),
        question_context: str | None = None,
    ) -> str:
        return self.redact(
            text,
            extra_detectors=extra_detectors,
            question_context=question_context,
        ).text

    def _resolve_with_uncertainty(
        self,
        text: str,
        *,
        extra_detectors: Sequence[Detector] = (),
        question_context: str | None = None,
        hard_constraints: Sequence[Finding] = (),
        suppressed_spans: Sequence[Span] = (),
    ) -> tuple[list[Finding], list[Finding]]:
        already = _placeholder_regions(text)
        deterministic, det_uncertain = self._run_deterministic(
            text,
            extra_detectors=extra_detectors,
        )
        deterministic = _clamp_findings(deterministic, len(text))
        det_uncertain = _clamp_findings(det_uncertain, len(text))

        # Apply human-reviewer suppressed spans (explicit REMOVE decisions)
        if suppressed_spans:
            deterministic = [
                f for f in deterministic if not any(f.span.overlaps(s) for s in suppressed_spans)
            ]
            det_uncertain = [
                f for f in det_uncertain if not any(f.span.overlaps(s) for s in suppressed_spans)
            ]

        # Lock in human-reviewer hard constraints first (spans are clamped to the
        # text so a stale/incorrect reviewer decision can never truncate output).
        locked = _clamp_findings(hard_constraints, len(text))
        claimed = [f.span for f in locked] + [f.span for f in deterministic]

        nlp_findings, nlp_uncertain = self._run_nlp(text, question_context=question_context)
        nlp_findings = _clamp_findings(nlp_findings, len(text))
        nlp_uncertain = _clamp_findings(nlp_uncertain, len(text))
        if suppressed_spans:
            nlp_findings = [
                f for f in nlp_findings if not any(f.span.overlaps(s) for s in suppressed_spans)
            ]
            nlp_uncertain = [
                f for f in nlp_uncertain if not any(f.span.overlaps(s) for s in suppressed_spans)
            ]

        surviving = self._filter_nlp(nlp_findings, claimed, text)
        if already:
            deterministic = [f for f in deterministic if not _inside_any(f.span, already)]
            surviving = [f for f in surviving if not _inside_any(f.span, already)]
            det_uncertain = [f for f in det_uncertain if not _inside_any(f.span, already)]
            nlp_uncertain = [f for f in nlp_uncertain if not _inside_any(f.span, already)]

        combined = [*locked, *deterministic, *surviving]
        if locked or self.policy.overlap_mode == "union":
            combined = resolve_overlaps(
                combined,
                self._order,
                mode=self.policy.overlap_mode,
                text=text,
            )

        uncertain_all = [
            u
            for u in [*det_uncertain, *nlp_uncertain]
            if not any(u.span.overlaps(f.span) for f in combined)
        ]
        return combined, uncertain_all

    def _run_deterministic(
        self,
        text: str,
        *,
        extra_detectors: Sequence[Detector] = (),
    ) -> tuple[list[Finding], list[Finding]]:
        collected: list[Finding] = []
        uncertain: list[Finding] = []
        all_detectors = (*extra_detectors, *self._detectors)
        for detector in all_detectors:
            for finding in detector.detect(text):
                if not self.policy.enabled(finding.category):
                    continue
                if finding.meta().get("checksum") == "failed_ocr_candidate":
                    # Uncertainty must never leak: an OCR-corrupted identifier is
                    # retained for safe masking *and* routed to review even when the
                    # global confidence floor would otherwise discard it.
                    if finding.confidence >= self.policy.review_floor:
                        collected.append(finding)
                        uncertain.append(finding)
                    continue
                if finding.confidence < self.policy.min_confidence:
                    if finding.confidence >= self.policy.review_floor:
                        uncertain.append(finding)
                    continue
                collected.append(finding)
        resolved = resolve_overlaps(
            collected,
            self._order,
            mode=self.policy.overlap_mode,
            text=text,
        )
        return resolved, uncertain

    def _run_nlp(
        self,
        text: str,
        *,
        question_context: str | None = None,
    ) -> tuple[list[Finding], list[Finding]]:
        backend = self._nlp_backend
        if backend is None:
            return [], []
        collected: list[Finding] = []
        uncertain: list[Finding] = []
        detect_fn = backend.detect
        try:
            iterator = detect_fn(text, question_context=question_context)
        except TypeError:
            iterator = detect_fn(text)
        for finding in iterator:
            if not self.policy.enabled(finding.category):
                continue
            if finding.confidence < self.policy.nlp_min_confidence:
                if finding.confidence >= self.policy.review_floor:
                    uncertain.append(finding)
                continue
            if finding.confidence < self.policy.min_confidence:
                if finding.confidence >= self.policy.review_floor:
                    uncertain.append(finding)
                continue
            if finding.confidence < self.policy.threshold_for(finding.category):
                uncertain.append(finding)
            collected.append(finding)
        return collected, uncertain

    def _filter_nlp(
        self,
        findings: list[Finding],
        claimed: list[Span],
        text: str | None = None,
    ) -> list[Finding]:
        if not claimed:
            return resolve_overlaps(findings, {}, mode=self.policy.overlap_mode, text=text)
        surviving = [
            f for f in findings if not any(f.span.overlaps(s) for s in claimed)
        ]
        return resolve_overlaps(surviving, {}, mode=self.policy.overlap_mode, text=text)

    def _apply(self, text: str, findings: Sequence[Finding]) -> str:
        ordered = sorted(
            _clamp_findings(findings, len(text)),
            key=lambda f: f.span.start,
            reverse=True,
        )
        out = text
        for finding in ordered:
            raw = finding.value if finding.value is not None else finding.span.extract(text)
            replacement = self._replacement(finding, raw)
            if self.policy.strict and not replacement and finding.span.length:
                raise RedactionError(
                    f"strict mode produced an empty replacement for {finding.category.value}"
                )
            out = out[: finding.span.start] + replacement + out[finding.span.end :]
        return out

    def _replacement(self, finding: Finding, raw: str) -> str:
        if finding.category in self.policy.generalise_categories:
            return generalise(finding.category, raw)
        style = self.policy.style_for(finding.category)
        masker = build_masker(
            style,
            finding.category.value,
            self.policy.salt,
            self.policy.mask_char,
            self.policy.partial_keep_prefix,
            self.policy.partial_keep_suffix,
        )
        return masker(raw)


_PLACEHOLDER_RE = re.compile(
    rf"{masking.LABEL_PATTERN}|\b{masking.TOKEN_PATTERN}\b"
)


def _placeholder_regions(text: str) -> list[Span]:
    if not text or ("[" not in text and "_" not in text):
        return []
    return [Span(m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(text)]


def _inside_any(span: Span, regions: list[Span]) -> bool:
    return any(span.start >= r.start and span.end <= r.end for r in regions)


def _clamp_findings(findings: Iterable[Finding], text_length: int) -> list[Finding]:
    """Trim findings to valid character offsets and drop empty spans.

    Detectors always emit in-range spans, but externally supplied findings
    (reviewer decisions, serialised batches, callers of :func:`resolve_overlaps`)
    may reference stale offsets. Applying those unclamped would silently delete
    text outside the intended span, so they are trimmed here instead.
    """
    clamped: list[Finding] = []
    for finding in findings:
        start = min(max(0, finding.span.start), text_length)
        end = min(max(start, finding.span.end), text_length)
        if end <= start:
            continue
        if start == finding.span.start and end == finding.span.end:
            clamped.append(finding)
        else:
            clamped.append(replace(finding, span=Span(start, end)))
    return clamped


# Categories eligible for union-of-spans boundary merging when two detectors
# overlap on a name, organisation, address, or implication clause. Checksummed
# digit identifiers (TFN, ABN, ACN, Medicare) are never expanded by unioning
# so their exact validated digit boundaries stay intact.
_UNION_ELIGIBLE = frozenset(
    {
        Category.NAME,
        Category.ORGANISATION,
        Category.ORGANISATION_NAME,
        Category.STREET_ADDRESS,
        Category.IMPLICATION,
        Category.LOCATION,
    }
)


def _merge_union_span(
    kept: list[Finding],
    index: int,
    candidate: Finding,
    text: str | None,
) -> None:
    """Extend ``kept[index]`` with ``candidate`` and any kept span it then touches.

    A short bridging span (e.g. an NLP prediction straddling two deterministic
    spans) can widen a kept span into its neighbour; the neighbour is absorbed
    too so no two kept spans overlap. Reviewer hard constraints are never
    expanded or swallowed.
    """
    base = kept[index]
    start = min(base.span.start, candidate.span.start)
    end = max(base.span.end, candidate.span.end)
    absorbed = {index}
    changed = True
    while changed:
        changed = False
        for other_idx, other in enumerate(kept):
            if other_idx in absorbed or other.meta().get("hard_constraint"):
                continue
            if other.category not in _UNION_ELIGIBLE:
                continue
            if other.span.start < end and start < other.span.end:
                start = min(start, other.span.start)
                end = max(end, other.span.end)
                absorbed.add(other_idx)
                changed = True

    if start == base.span.start and end == base.span.end:
        return

    # A merged span must never touch a locked human-review decision: if it
    # would, drop the candidate and keep the hard constraint intact.
    for other_idx, other in enumerate(kept):
        if other_idx == index or not other.meta().get("hard_constraint"):
            continue
        if other.span.start < end and start < other.span.end:
            return

    value = base.value
    if text is not None and start >= 0 and end <= len(text):
        value = text[start:end]
    kept[index] = replace(base, span=Span(start, end), value=value)
    for other_idx in sorted(absorbed - {index}, reverse=True):
        kept.pop(other_idx)


def resolve_overlaps(
    findings: Iterable[Finding],
    order: dict[str, int],
    *,
    mode: str = "longest",
    text: str | None = None,
) -> list[Finding]:
    """Resolve overlapping findings using either ``longest`` or ``union`` semantics.

    * ``longest``: Longest span wins, then deterministic over NLP, then higher
      confidence, then earlier detector in ``order``.
    * ``union``: Uses the same priority ranking to choose the winning category
      and detector, but when overlapping spans are union-eligible (e.g., one
      detector matched ``"John"`` and another matched ``"John Smith"`` with
      overlapping boundaries), merges their character boundaries to
      ``Span(min_start, max_end)`` so partial tokens are never leaked. Merging is
      transitive: if the widened span reaches a further kept span, that span is
      absorbed as well, so the returned spans never overlap.
    """
    ranked = sorted(
        findings,
        key=lambda f: (
            0 if f.meta().get("hard_constraint") else 1,
            -f.span.length,
            0 if f.stage is Stage.DETERMINISTIC else 1,
            -f.confidence,
            order.get(f.detector, len(order)),
            f.span.start,
        ),
    )
    kept: list[Finding] = []
    for candidate in ranked:
        if candidate.span.length <= 0:
            continue
        overlapping_idx = next(
            (i for i, k in enumerate(kept) if candidate.span.overlaps(k.span)),
            None,
        )
        if overlapping_idx is None:
            kept.append(candidate)
            continue

        if mode == "union":
            winner = kept[overlapping_idx]
            if (
                not winner.meta().get("hard_constraint")
                and winner.category in _UNION_ELIGIBLE
                and candidate.category in _UNION_ELIGIBLE
            ):
                _merge_union_span(kept, overlapping_idx, candidate, text)

    kept.sort(key=lambda f: f.span.start)
    return kept


def redact(text: str, policy: Policy | None = None) -> str:
    """Module-level convenience wrapper."""
    return Redactor(policy).redact_text(text)
