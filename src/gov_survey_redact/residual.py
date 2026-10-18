"""Residual disclosure control and post-redaction deterministic validation.

Combines:
1. **Deterministic residual validation** (re-running direct-identifier rules and
   cross-field known-record values on the redacted text to ensure zero mandatory
   PII escape before Gold release).
2. **Statistical disclosure / k-anonymity cohort estimation** (measuring surviving
   person, workplace, temporal, numeric, organisation, and narrative attributes
   against workforce ``population`` and escalating via ``PASS``, ``GENERALISE``,
   or ``SUPPRESS``).
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .detectors.base import Detector
from .masking import PLACEHOLDER_PATTERN
from .types import Finding
from .vocabulary import OCCUPATION_FAMILY


class ResidualKind(str, enum.Enum):
    """Class of information a surviving value discloses."""

    PERSON = "person"
    WORKPLACE = "workplace"
    TEMPORAL = "temporal"
    NUMERIC = "numeric"
    ORGANISATION = "organisation"
    NARRATIVE = "narrative"

    def __str__(self) -> str:
        return self.value


SELECTIVITY: dict[ResidualKind, float] = {
    ResidualKind.PERSON: 8.0,
    ResidualKind.WORKPLACE: 12.0,
    ResidualKind.TEMPORAL: 40.0,
    ResidualKind.NUMERIC: 25.0,
    ResidualKind.ORGANISATION: 60.0,
    ResidualKind.NARRATIVE: 200.0,
}

_LONG_DIGITS = re.compile(r"(?<![\w-])\d{6,}(?![\w-])")
_GROUPED_DIGITS = re.compile(r"(?<![\w-])\d{2,4}(?:[ -]\d{2,4}){1,4}(?![\w-])")
_REFERENCE = re.compile(r"(?<![\w-])[A-Z]{2,4}-?\d{3,}(?:-\d{2,})*(?![\w-])")
_NAMED_SYSTEM = re.compile(
    r"(?<![.!?]\s)(?P<who>[A-Z][A-Za-z'\u2019-]{2,})\s+"
    r"(?:platform|system|portal|register|workflow|dashboard|tool|suite|"
    r"network|intranet|repository|spreadsheet|helpdesk|ticketing)\b"
)
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_EXACT_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_BAND = re.compile(
    r"\b\d{2,3}\s*-\s*\d{2,3}\b|\b\d{2,3}\+|"
    r"\bunder\s+\d{1,3}\b|\b\d{2,3}\s+years\b|\b\d{2,3}\+?\s+people\b"
)
_SMALL_COUNT = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d{1,3})"
    r"\s+(?:of\s+us\s+)?(?:incidents?|complaints?|reports?|"
    r"people|staff|workers|members)\b",
    re.I,
)
_PROPER_NOUN = re.compile(
    r"(?<![.!?]\s)\b[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3}\b"
)
_AGE = re.compile(r"\b\d{1,3}\s*(?:years?\s*old|yrs?)\b", re.I)
_TENURE = re.compile(r"\b\d{1,2}\+?\s*(?:years?|yrs?)\b", re.I)
_OCCUPATION = re.compile(rf"(?<![.!?]\s)\b(?:{OCCUPATION_FAMILY})\b", re.IGNORECASE)
_NARRATIVE = re.compile(
    r"(?i)\b(?:bully\w*|harass\w*|abuse\w*|aggress\w*|threat\w*|humiliat\w*|"
    r"grievance|complaint|incident|retaliat\w*|excluded|ostraci[sz]ed|"
    r"anxiety|depression|ptsd|psychological|burnout|unsafe|danger\w*|"
    r"injur\w*|discriminat\w*|bullied|"
    r"singl(?:ed|ing)\s+(?:me\s+)?out|pick(?:ed|ing)\s+(?:me\s+)?on|"
    r"target(?:ed|ing|s)?|"
    r"shout(?:ed|ing)|embarrass\w*|undermin\w*|"
    r"intimidat\w*|mock(?:ed|ing)|barrage|excluded\s+me|left\s+out|"
    r"set\s+me\s+up|put\s+me\s+on|against\s+me|"
    r"did\s+not\s+listen|would\s+not\s+listen|nobody\s+(?:cared|acted)|"
    r"nothing\s+happened|no\s+action)\b"
)
_PLACEHOLDER = re.compile(PLACEHOLDER_PATTERN)

_NOT_A_NAME = frozenset(
    {
        "My",
        "Our",
        "The",
        "This",
        "That",
        "These",
        "Those",
        "His",
        "Her",
        "Its",
        "Their",
        "Your",
        "A",
        "An",
        "I",
        "We",
        "They",
        "He",
        "She",
        "It",
        "No",
        "Not",
        "Yes",
        "But",
        "And",
        "So",
        "When",
        "After",
        "Before",
        "Since",
        "If",
        "Then",
        "There",
        "Here",
        "Now",
        "One",
        "Two",
        "Three",
        "Each",
        "Every",
        "New",
        "Old",
        "Both",
        "All",
        "Some",
        "Any",
        "Most",
        "Many",
        "Few",
        "More",
        "Less",
        "Just",
        "Still",
        "Very",
        "Really",
        "Quite",
        "Also",
        "Even",
        "Only",
        "TFN",
        "ABN",
        "ACN",
        "DOB",
        "BSB",
        "EAP",
        "PA",
        "ID",
        "HR",
        "APS",
    }
)


@dataclass(frozen=True, slots=True)
class ResidualAttribute:
    """One piece of information that survived redaction."""

    kind: ResidualKind
    value: str
    selectivity: float
    span: tuple[int, int] | None = None

    @property
    def is_high_selectivity(self) -> bool:
        return self.selectivity >= HIGH_SELECTIVITY

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "value": self.value[:60],
            "selectivity": round(self.selectivity, 2),
            "span": list(self.span) if self.span else None,
        }


HIGH_SELECTIVITY = 1000.0
_SURVIVING_PROPER_NOUN_SELECTIVITY = 2000.0


@dataclass(frozen=True)
class ResidualProfile:
    """Inventory of attributes still disclosed by a redacted record."""

    attributes: tuple[ResidualAttribute, ...] = ()
    text: str = ""

    @property
    def selectivity_product(self) -> float:
        product = 1.0
        for attribute in self.attributes:
            product *= attribute.selectivity
        return product

    @property
    def classes(self) -> frozenset[ResidualKind]:
        return frozenset(a.kind for a in self.attributes)

    @property
    def high_selectivity(self) -> tuple[ResidualAttribute, ...]:
        return tuple(a for a in self.attributes if a.is_high_selectivity)

    @property
    def identifying(self) -> bool:
        return any(a.is_high_selectivity for a in self.attributes)

    @property
    def has_reducible(self) -> bool:
        return any(
            a.span is not None and a.kind is not ResidualKind.NARRATIVE
            for a in self.high_selectivity
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "selectivity_product": round(self.selectivity_product, 1),
            "identifying": self.identifying,
            "classes": sorted(k.value for k in self.classes),
            "attributes": [a.as_dict() for a in self.attributes],
        }


class DisclosureAction(str, enum.Enum):
    """Action taken when residual disclosure risk exceeds policy."""

    PASS = "pass"
    GENERALISE = "generalise"
    SUPPRESS = "suppress"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class DisclosurePolicy:
    """Population and k-anonymity cohort thresholds for residual disclosure control."""

    population: int = 1000
    min_cohort: float = 5.0
    attribute_independence: float = 1.0
    narrative_selectivity: float = SELECTIVITY[ResidualKind.NARRATIVE]

    def expected_matches(self, profile: ResidualProfile) -> float:
        return self.population / profile.selectivity_product * self.attribute_independence


WITHHELD_TEXT = "[SENSITIVE RESPONSE WITHHELD FOR DISCLOSURE CONTROL]"


@dataclass(frozen=True)
class DisclosureDecision:
    """Outcome of residual risk assessment with pre/post cohort estimates."""

    action: DisclosureAction
    profile: ResidualProfile
    cohort: float
    reason: str
    drivers: tuple[str, ...] = ()
    post_cohort: float | None = None

    @property
    def escalated(self) -> bool:
        return self.action is not DisclosureAction.PASS

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "expected_matches": round(self.cohort, 4),
            "expected_matches_after": (
                None if self.post_cohort is None else round(self.post_cohort, 4)
            ),
            "reason": self.reason,
            "drivers": list(self.drivers),
            "residual": self.profile.as_dict(),
        }


def measure(text: str, narrative_selectivity: float | None = None) -> ResidualProfile:
    """Inventory what survives in already-redacted ``text``."""
    if not text or not text.strip():
        return ResidualProfile()

    attributes: list[ResidualAttribute] = []
    claimed: list[tuple[int, int]] = []
    markers = [m.span() for m in _PLACEHOLDER.finditer(text)]

    def add(
        kind: ResidualKind,
        match: re.Match[str],
        selectivity: float,
        group: int | str = 0,
    ) -> None:
        span = match.span(group)
        if any(span[0] < end and start < span[1] for start, end in markers):
            return
        if any(span[0] < end and start < span[1] for start, end in claimed):
            return
        claimed.append(span)
        attributes.append(ResidualAttribute(kind, match.group(group), selectivity, span))

    for match in _LONG_DIGITS.finditer(text):
        add(ResidualKind.NUMERIC, match, 1e9)

    for match in _GROUPED_DIGITS.finditer(text):
        if sum(len(part) for part in match.group().split()) >= 7:
            add(ResidualKind.NUMERIC, match, 1e8)

    for match in _REFERENCE.finditer(text):
        add(ResidualKind.ORGANISATION, match, 5e6)

    for match in _NAMED_SYSTEM.finditer(text):
        add(ResidualKind.ORGANISATION, match, _SURVIVING_PROPER_NOUN_SELECTIVITY)

    for match in _EXACT_DATE.finditer(text):
        add(ResidualKind.TEMPORAL, match, SELECTIVITY[ResidualKind.TEMPORAL] * 0.1)

    for match in _AGE.finditer(text):
        add(ResidualKind.PERSON, match, SELECTIVITY[ResidualKind.PERSON] * 3)

    for match in _OCCUPATION.finditer(text):
        add(ResidualKind.PERSON, match, SELECTIVITY[ResidualKind.PERSON] * 4)

    for match in _TENURE.finditer(text):
        add(ResidualKind.PERSON, match, SELECTIVITY[ResidualKind.PERSON] * 2)

    for match in _BAND.finditer(text):
        add(ResidualKind.PERSON, match, SELECTIVITY[ResidualKind.PERSON])

    for match in _SMALL_COUNT.finditer(text):
        add(ResidualKind.NUMERIC, match, SELECTIVITY[ResidualKind.NUMERIC] * 0.5)

    for match in _YEAR.finditer(text):
        add(ResidualKind.TEMPORAL, match, SELECTIVITY[ResidualKind.TEMPORAL] * 0.5)

    for match in _PROPER_NOUN.finditer(text):
        words = match.group(0).split()
        if not words or words[0] in _NOT_A_NAME:
            continue
        add(ResidualKind.ORGANISATION, match, _SURVIVING_PROPER_NOUN_SELECTIVITY)

    if _NARRATIVE.search(text):
        attributes.append(
            ResidualAttribute(
                ResidualKind.NARRATIVE,
                "incident narrative",
                narrative_selectivity or SELECTIVITY[ResidualKind.NARRATIVE],
            )
        )

    return ResidualProfile(tuple(attributes), text)


def decide(
    profile: ResidualProfile,
    policy: DisclosurePolicy | None = None,
) -> DisclosureDecision:
    """Decide whether a redacted record may be released as it stands."""
    policy = policy or DisclosurePolicy()
    expected = policy.expected_matches(profile)
    drivers = tuple(sorted({a.value for a in profile.high_selectivity}))

    if not profile.attributes:
        return DisclosureDecision(
            action=DisclosureAction.PASS,
            profile=profile,
            cohort=expected,
            reason="nothing survived redaction, so nothing is disclosed",
        )

    if expected < policy.min_cohort:
        if not profile.has_reducible:
            return DisclosureDecision(
                action=DisclosureAction.SUPPRESS,
                profile=profile,
                cohort=expected,
                reason=(
                    f"the combination matches about {expected:.2f} of a "
                    f"population of {policy.population} and cannot be "
                    f"generalised without discarding the response"
                ),
                drivers=drivers,
            )
        return DisclosureDecision(
            action=DisclosureAction.GENERALISE,
            profile=profile,
            cohort=expected,
            reason=(
                f"the combination matches about {expected:.2f} of a population "
                f"of {policy.population}, below the threshold of "
                f"{policy.min_cohort}"
            ),
            drivers=drivers,
        )

    return DisclosureDecision(
        action=DisclosureAction.PASS,
        profile=profile,
        cohort=expected,
        reason=(
            f"the combination matches about {expected:.1f} of a population of "
            f"{policy.population}, which is large enough to be unremarkable"
        ),
    )


def generalise_residual(text: str, profile: ResidualProfile) -> str:
    """Replace surviving high-selectivity literals with a category marker."""
    out = text
    for attribute in sorted(
        profile.high_selectivity,
        key=lambda a: a.span[0] if a.span else 0,
        reverse=True,
    ):
        if attribute.span is None or attribute.kind is ResidualKind.NARRATIVE:
            continue
        start, end = attribute.span
        out = out[:start] + f"[{attribute.kind.value.upper()} REDACTED]" + out[end:]
    return out


def protect(
    text: str,
    policy: DisclosurePolicy | None = None,
) -> tuple[str, DisclosureDecision]:
    """Measure and act on a redacted record. Returns ``(text, decision)``."""
    policy = policy or DisclosurePolicy()
    profile = measure(text, policy.narrative_selectivity)
    decision = decide(profile, policy)

    if decision.action is DisclosureAction.SUPPRESS:
        return WITHHELD_TEXT, decision
    if decision.action is DisclosureAction.GENERALISE:
        cleaned = generalise_residual(text, profile)
        after = measure(cleaned, policy.narrative_selectivity)
        return cleaned, replace(decision, post_cohort=policy.expected_matches(after))
    return text, decision


def validate_residual_deterministic(
    redacted_text: str,
    detectors: Sequence[Detector],
) -> list[Finding]:
    """Re-run deterministic direct-identifier and known-value recognisers on ``redacted_text``.

    Any surviving direct identifier found outside a placeholder marker triggers
    immediate supervisor escalation rather than silent release to Gold.
    """
    if not redacted_text or not redacted_text.strip():
        return []
    markers = [m.span() for m in _PLACEHOLDER.finditer(redacted_text)]
    leaks: list[Finding] = []
    for detector in detectors:
        for finding in detector.detect(redacted_text):
            if not (finding.category.is_direct_identifier or finding.category.is_organisation_identifier):
                continue
            if any(
                finding.span.start >= m_start and finding.span.end <= m_end
                for m_start, m_end in markers
            ):
                continue
            leaks.append(finding)
    return leaks


def summarise(decisions: Iterable[DisclosureDecision]) -> dict[str, Any]:
    decisions = list(decisions)
    total = len(decisions)
    counts = {action.value: 0 for action in DisclosureAction}
    for decision in decisions:
        counts[decision.action.value] += 1
    escalated = (
        counts[DisclosureAction.GENERALISE.value] + counts[DisclosureAction.SUPPRESS.value]
    )
    return {
        "records": total,
        "actions": counts,
        "escalation_rate": round(escalated / total, 4) if total else 0.0,
        "mean_cohort": round(sum(d.cohort for d in decisions) / total, 1) if total else 0.0,
    }


__all__ = [
    "HIGH_SELECTIVITY",
    "SELECTIVITY",
    "WITHHELD_TEXT",
    "DisclosureAction",
    "DisclosureDecision",
    "DisclosurePolicy",
    "ResidualAttribute",
    "ResidualKind",
    "ResidualProfile",
    "decide",
    "generalise_residual",
    "measure",
    "protect",
    "summarise",
    "validate_residual_deterministic",
]
