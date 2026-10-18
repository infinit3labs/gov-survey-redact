"""Implication rules: clauses and word sequences that identify a person or workplace.

Handles indirect re-identification vectors in Australian Government (APS/State)
and corporate survey responses:
* Uniqueness claims ("the only woman on the night shift", "one of two apprentices")
* Public-sector units ("the digital services branch", "in Services division")
* APS/EL/SES officer classifications ("one of three EL2s in the Migration directorate")
* Government processes ("a machinery of government review", "a FOI request")
* Security clearances ("security clearance at the secret level", "NV1", "PV")
* Cohorts and intakes ("the 2021 intake", "the graduate cohort")
* Named workplace sites ("Kwinana depot", "Box Hill store", with 1-edit typo tolerance)
* Third-party colleague/supervisor naming ("my manager Fiona", "my caseworker was Nguyễn Thị Hương")
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ..normalise import normalise_aligned
from ..types import Category, Finding, Span, Stage
from ..vocabulary import (
    APS_LEVEL,
    CLEARANCE_LEVEL,
    CLEARANCE_MARKER,
    COHORT_MARKER,
    CORPORATE_MARKER,
    DEPARTMENT_STRUCTURE,
    DEPARTMENT_STRUCTURE_QUALIFIED,
    GOVERNMENT_PROCESS,
    OCCUPATION_FAMILY,
    PUBLIC_SECTOR_ROLE,
    SENIOR_ROLE,
    SERVICE_FUNCTION,
    WORKPLACE_TYPE,
    WORKPLACE_TYPE_STRONG,
    alt,
)


def _typo(word: str, *, min_variant_length: int = 0) -> str:
    """Regex fragment matching ``word`` within one edit (drop, insert, sub, swap).

    ``min_variant_length`` suppresses *deletion* variants shorter than the given
    length, which stops short site nouns from matching unrelated shorter English
    words (e.g. dropping the "t" from ``plant`` must not match ``plan``).
    """
    alts = {word}
    for index in range(len(word)):
        dropped = word[:index] + word[index + 1 :]
        if len(dropped) >= max(1, min_variant_length):
            alts.add(dropped)
        alts.add(word[:index] + "[a-z]" + word[index + 1 :])
        alts.add(word[:index] + "[a-z]" + word[index:])
    for index in range(len(word) - 1):
        alts.add(word[:index] + word[index + 1] + word[index] + word[index + 2 :])
    return alt(*sorted(alts))


_ROSTER = alt(
    "team",
    "unit",
    "ward",
    "crew",
    "shift",
    "roster",
    "rota",
    "department",
    "store",
    "depot",
    "yard",
    "class",
    "section",
    "branch",
    "site",
    "workshop",
    "floor",
    "line",
    "cell",
)
_SHIFT = alt(
    "shift",
    "roster",
    "rota",
    "day",
    "night",
    "weekend",
    "early",
    "late",
    "daytime",
    "overnight",
    "backfill",
)
_ROLE = alt(
    *(
        _typo(w) if len(w) >= 5 else w
        for w in (
            "nurse",
            "teacher",
            "carer",
            "cleaner",
            "driver",
            "technician",
            "officer",
            "cook",
            "chef",
            "barista",
            "clerk",
            "labourer",
            "electrician",
            "plumber",
            "warden",
            "midwife",
            "allocator",
            "caseworker",
            "auditor",
            "pharmacist",
        )
    )
)
_AGE_PHASE = alt(
    *(_typo(w) if len(w) >= 4 else w for w in ("youngest", "oldest", "newest", "new", "senior", "junior", "only"))
)
_SMALL_NUMBER = alt(
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    r"\d{1,2}",
)
_ARTICLE = alt("the", "my", "our", "his", "her", "its")
_UNIQUE_KIND = alt(
    "woman",
    "man",
    "person",
    "casual",
    "temp",
    "migrant",
    "indigenous",
    "apprentice",
    "parent",
    "student",
    "trainee",
    "graduate",
    "lone",
)
_SELF = alt(
    r"i\s+am",
    r"i'?m",
    r"i\s+was",
    r"being\s+the",
    r"it\s+is\s+that\s+i\s+am",
    r"i\b",
)
_SITE_NOUN = alt(
    *(
        word if len(word) < 5 else _typo(word, min_variant_length=5)
        for word in (
            "depot",
            "store",
            "site",
            "branch",
            "plant",
            "facility",
            "terminal",
            "warehouse",
            "hub",
            "campus",
            "centre",
            "center",
            "workshop",
            "yard",
            "floor",
            "building",
            "lab",
            "refinery",
            "kiln",
            "mine",
            "exchange",
        )
    )
)

_NOT_SERVICE_FUNCTION = rf"(?!(?:{SERVICE_FUNCTION})\b)"

_GENERIC_MODIFIER = (
    "Main",
    "Big",
    "Small",
    "Large",
    "Central",
    "Upper",
    "Lower",
    "Inner",
    "Outer",
    "Front",
    "Rear",
    "Other",
    "Each",
    "Every",
    "Same",
    "Whole",
    "Entire",
    "New",
    "Old",
)

_NOT_A_NAME = alt(
    "My",
    "The",
    "Our",
    "His",
    "Her",
    "Its",
    "This",
    "That",
    "A",
    "An",
    "Their",
    "Your",
    *_GENERIC_MODIFIER,
)

_NOT_A_MODIFIER_CI = alt(
    "my",
    "our",
    "the",
    "his",
    "her",
    "their",
    *(m.lower() for m in _GENERIC_MODIFIER),
)

_WORK_EVENT = alt(
    "closure",
    r"shut\s?down",
    _typo("merger"),
    rf"take-?{_typo('over')}",
    _typo("inspection"),
    _typo("audit"),
    _typo("investigation"),
    _typo("reopening"),
    _typo("relocation"),
    "sale",
    _typo("commissioning"),
    _typo("handover"),
    _typo("demolition"),
    _typo("walkout"),
    _typo("lockout"),
)

_INTERNAL_THING = alt(
    "intranet",
    "portal",
    "dashboard",
    "database",
    "register",
    "workflow",
    "helpdesk",
    "timesheet",
    "rostering",
    "reporting tool",
    "incident system",
    "reporting system",
    "hr system",
    "claims system",
    "payroll system",
)

_PERSON_RELATION = alt(
    "manager",
    "supervisor",
    "boss",
    r"team\s*lead",
    "coordinator",
    "colleague",
    r"co-?worker",
    "caseworker",
    r"case\s*manager",
    r"case\s*officer",
    "consultant",
    "adjuster",
    r"claims\s*officer",
    r"rtw\s*coordinator",
    r"rehabilitation\s*consultant",
    r"union\s*rep",
    "organiser",
    "delegate",
    "mate",
    "partner",
    "friend",
    "brother",
    "sister",
    "ceo",
    "director",
    "gm",
    r"line\s*manager",
)

_NOT_A_MODIFIER = (
    rf"(?!(?:{_PERSON_RELATION}|"
    r"keeps?|kept|sends?|sent|uses?|used|fills?|filled|opens?|opened|"
    r"updates?|updated|ignores?|ignored|reports?|reported|asks?|asked|"
    r"tells?|told|says?|said|needs?|needed|wants?|wanted|makes?|made|"
    r"runs?|ran|owns?|owned|manages?|managed)\b)"
)


@dataclass(frozen=True, slots=True)
class ImplicationRule:
    """One pattern that identifies an implicating person or workplace sequence."""

    name: str
    pattern: re.Pattern[str]
    label: str
    confidence: float = 0.8
    implicates: str = "person"

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pattern": self.pattern.pattern,
            "label": self.label,
            "confidence": self.confidence,
            "implicates": self.implicates,
        }


def _rule(
    name: str,
    pattern: str,
    label: str,
    *,
    confidence: float = 0.8,
    implicates: str = "person",
) -> ImplicationRule:
    return ImplicationRule(
        name=name,
        pattern=re.compile(pattern, re.IGNORECASE),
        label=label,
        confidence=confidence,
        implicates=implicates,
    )


RULES: tuple[ImplicationRule, ...] = (
    _rule(
        "only_one_on_team",
        rf"(?P<clause>(?:the\s+)?only\s+(?:\w+\s+){{0,2}}?{_UNIQUE_KIND}"
        rf"(?:\s+(?:{_ROLE}))?\s+(?:in|on|of|at)\s+"
        rf"(?:{_ARTICLE}\s+)?(?:\w+\s+){{0,2}}{_ROSTER})",
        "claims to be the only person of their kind in a named group",
    ),
    _rule(
        "one_of_n",
        rf"(?P<clause>one\s+of\s+(?:just\s+|only\s+)?{_SMALL_NUMBER}\s+"
        rf"(?:\w+\s+){{0,2}}(?:on|in|of|at)\s+(?:{_ARTICLE}\s+)?{_ROSTER})",
        "states membership of a very small named group",
    ),
    _rule(
        "sole_holder",
        rf"(?P<clause>(?:i\s+am|i'?m|the)\s+(?:sole|only|lone)\s+"
        rf"(?:\w+\s+){{0,2}}?(?:{_ROSTER}|{_ROLE}))",
        "describes sole responsibility for a group or role",
    ),
    _rule(
        "no_one_else",
        r"(?P<clause>(?:no\s+one\s+else|not\s+anyone\s+else|nobody\s+else)"
        r"(?:\s+\w+){0,3})",
        "claims no one else shares this duty",
    ),
    _rule(
        "unique_attribute",
        rf"(?P<clause>(?:{_SELF}\s+)?the\s+{_AGE_PHASE}\s+(?:\w+\s+){{0,2}}"
        r"(?:here|(?:on|at|in)\s+the(?:\s+\w+)?))",
        "identifies by comparative age or seniority within a group",
    ),
    _rule(
        "only_one_does",
        r"(?P<clause>(?:the\s+)?only\s+one\s+(?:who\s+|that\s+|to\s+)?"
        r"(?:\w+\s+){0,3}?(?:does|do|did|handles|covers|operates|runs|starts)"
        r"(?:\s+the\s+[\w'’-]+(?:\s+[\w'’-]+){0,2})?)",
        "claims a unique duty",
    ),
    _rule(
        "shift_group",
        rf"(?P<clause>(?:the\s+)?(?<![\d-])(?:\d{{1,2}}|{_SMALL_NUMBER})\s+"
        r"(?:people|staff|workers|employees|members|of\s+us)\s+"
        rf"(?:on|across|covering)\s+(?:{_ARTICLE}\s+)?(?:\w+\s+){{0,2}}{_SHIFT})",
        "describes the size of a named shift group",
        confidence=0.75,
    ),
    _rule(
        "covers_sites",
        rf"(?P<clause>{_SELF}\s+(?:the\s+)?(?:sole|only|entire)?\s*\w*\s*"
        r"(?:cover|covering|covers)\s+(?:all\s+)?"
        rf"(?:{_ARTICLE}\s+)?(?:\w+\s+){{0,2}}?"
        r"(?:sites|depots|stores|branches|locations|facilities))",
        "describes sole coverage of several workplaces",
        implicates="workplace",
    ),
    _rule(
        "unique_training",
        r"(?P<clause>(?:the\s+)?only\s+one\s+(?:here|on\s+the\s+\w+|at\s+the\s+\w+)\s+"
        r"(?:who|that|to)\s+(?:\w+\s+){0,3}?(?:trained|certified|qualified|ticketed))",
        "states a unique qualification in a group",
    ),
    _rule(
        "post_event",
        rf"(?P<clause>(?:after|since|when|following|post)\s+(?:the|our)\s+"
        rf"(?:\w+\s+){{0,2}}{_WORK_EVENT})",
        "anchors the response to a workplace event",
        confidence=0.75,
        implicates="workplace",
    ),
    _rule(
        "named_site",
        rf"(?P<workplace>(?!(?i:{_NOT_A_NAME})\s)"
        rf"(?-i:[A-Z])[A-Za-z'’-]+(?:\s+[A-Za-z'’-]+){{0,2}}\s+{_SITE_NOUN}\b)",
        "names a workplace site",
        confidence=0.7,
        implicates="workplace",
    ),
    _rule(
        "site_change",
        r"(?P<clause>(?:when|after|since)\s+(?:they|we|the\s+company|it|the\s+\w+)\s+"
        r"(?:was|were|is|are|got|got\s+)?\s*"
        r"(?:closed|moved|sold|merged|renamed|demolished|reopened|shut\s?down))",
        "describes a change to a specific site",
        implicates="workplace",
    ),
    _rule(
        "named_site_lowercase",
        rf"(?P<workplace>(?:work\s+out\s+of|based\s+(?:in|at)|site\s+is|"
        rf"located\s+(?:at|in)|posted\s+to)\s+(?:the\s+|our\s+|my\s+)?"
        rf"[a-z][a-z'\u2019-]+(?:\s+[a-z][a-z'\u2019-]+){{0,2}}\s+{_SITE_NOUN}\b)",
        "names a workplace site without capitalisation",
        confidence=0.65,
        implicates="workplace",
    ),
    _rule(
        "internal_thing",
        rf"(?P<clause>\b(?:the|our|my|your|his|her|its|their)\b\s+"
        rf"(?:{_NOT_A_MODIFIER}\w+\s+){{0,2}}?{_INTERNAL_THING})",
        "describes an internal system or process",
        confidence=0.65,
        implicates="workplace",
    ),
    _rule(
        "names_third_party",
        rf"(?P<clause>(?:my|our|his|her|their|the)\s+(?:\w+\s+){{0,2}}"
        rf"{_PERSON_RELATION}(?:\s+(?:was|is))?\s+"
        r"(?P<who>(?-i:[A-Z])[A-Za-z'’\-]+(?:\s+(?-i:[A-Z])[A-Za-z'’\-]+){0,2}))",
        "names a colleague by role and name",
        confidence=0.9,
    ),
    _rule(
        "reported_by",
        r"(?P<clause>(?:told|said|reported|mentioned)\s+(?:it\s+)?(?:to|by)\s*"
        rf"(?:{_ARTICLE}\s+)?(?:\w+\s+){{0,2}}{_PERSON_RELATION}"
        r"(?:\s+(?P<who>(?-i:[A-Z])[A-Za-z'’\-]+))?)",
        "attributes a statement to a specific person",
        confidence=0.7,
    ),
    _rule(
        "classification",
        r"(?P<clause>(?:pay\s*)?(?:grade|band|step|category|classification)\s*[:#]?\s*"
        r"[a-z]?\d{1,2}\b(?:\s+(?:increment|point|salary|rate))?)",
        "states a pay grade or classification, unique within an employer",
        confidence=0.85,
    ),
    _rule(
        "department_structure",
        rf"(?P<clause>\b(?:my|our|the|his|her)\b\s+(?!(?:{_NOT_A_MODIFIER_CI})\s)(?:"
        rf"(?:\w+\s+){{0,3}}?{DEPARTMENT_STRUCTURE}"
        rf"|{_NOT_SERVICE_FUNCTION}{_NOT_A_MODIFIER}(?:\w{{3,}}\s+){{1,2}}"
        rf"{DEPARTMENT_STRUCTURE_QUALIFIED})\b"
        rf"(?:\s+(?:of|for)\s+(?:the\s+)?\w+){{0,2}}?)",
        "identifies the respondent by their department or branch",
        confidence=0.7,
        implicates="workplace",
    ),
    _rule(
        "unit_by_preposition",
        rf"(?P<clause>\b(?:in|at|within|across|into)\s+(?:"
        rf"(?:the\s+|a\s+|an\s+|my\s+|our\s+)?(?:\w+\s+){{0,3}}?{DEPARTMENT_STRUCTURE}"
        rf"|(?:the\s+|a\s+|an\s+|my\s+|our\s+){_NOT_SERVICE_FUNCTION}"
        rf"{_NOT_A_MODIFIER}(?:\w{{3,}}\s+){{1,2}}{DEPARTMENT_STRUCTURE_QUALIFIED})\b)",
        "identifies an administrative unit by preposition",
        confidence=0.65,
        implicates="workplace",
    ),
    _rule(
        "government_process",
        rf"(?P<clause>\b(?:the|our|my|a|an)\b\s+(?:\w+\s+){{0,2}}?{GOVERNMENT_PROCESS})",
        "describes an administrative act that only happens in government",
        confidence=0.7,
        implicates="workplace",
    ),
    _rule(
        "workplace_type",
        rf"(?P<clause>(?:\b(?:our|my|their|its)\b\s+{WORKPLACE_TYPE}\b"
        rf"|\b(?:the|a|an)\b\s+{WORKPLACE_TYPE_STRONG}\b)"
        rf"(?:\s+(?:in|at|near)\s+(?:the\s+)?\w+){{0,2}}?)",
        "identifies the type of workplace without naming it",
        confidence=0.65,
        implicates="workplace",
    ),
    _rule(
        "senior_role",
        rf"(?P<clause>\b(?:i\s+am|i'?m|as)\b\s+(?:the\s+|a\s+|an\s+)?{SENIOR_ROLE}\b"
        rf"(?:\s+of\s+(?:the\s+)?\w+){{0,2}}?)",
        "names a role that has very few holders in any organisation",
        confidence=0.8,
    ),
    _rule(
        "occupation_family",
        rf"(?P<clause>\b(?:i\s+am|i'?m|as|works?\s+as\s+an?|my\s+(?:role|title|job|"
        rf"position)\s+is)\b\s+(?:a\s+|an\s+|the\s+)?{OCCUPATION_FAMILY}\b)",
        "names a specific occupation family",
        confidence=0.7,
    ),
    _rule(
        "occupation_with_structure",
        rf"(?P<clause>(?:a|an|the)\s+{PUBLIC_SECTOR_ROLE}\s+(?:in|at|with)\s+"
        rf"(?:the\s+|my\s+)?(?:\w+\s+){{0,2}}{DEPARTMENT_STRUCTURE})",
        "an occupation inside a named unit identifies one person",
        confidence=0.85,
    ),
    _rule(
        "aps_level",
        rf"(?P<clause>(?:a|an|as\s+an?)\s+{APS_LEVEL}\b"
        rf"(?:\s+(?:in|at|of)\s+(?:the\s+|my\s+)?(?:\w+\s+){{0,2}}"
        rf"{DEPARTMENT_STRUCTURE})?)",
        "states an APS officer level, usually unique within a branch",
        confidence=0.85,
    ),
    _rule(
        "aps_level_alone",
        rf"(?P<clause>\b{APS_LEVEL}\b(?:[,.;:]\s*|\s+(?:officer|role|classification|position)?))",
        "states an APS officer level",
        confidence=0.7,
    ),
    _rule(
        "seniority_in_unit",
        rf"(?P<clause>(?:one\s+of|there\s+are|only)\s+(?:just\s+|only\s+)?"
        rf"{_SMALL_NUMBER}\s+"
        rf"(?:of\s+us\s+|of\s+them\s+|people\s+|of\s+|left\s+|remaining\s+)*"
        rf"(?:\w+\s+){{0,2}}?(?:{APS_LEVEL}s?\b|(?:{OCCUPATION_FAMILY})s?\b"
        rf"|senior|junior|acting|permanent|provisional|others?|ones?|us)?"
        rf"(?:\s+(?:in|of|on)\s+(?:the\s+|my\s+)?(?:"
        rf"(?:\w+\s+){{0,3}}?{DEPARTMENT_STRUCTURE}"
        rf"|(?:\w+\s+){{0,2}}{DEPARTMENT_STRUCTURE_QUALIFIED}"
        rf"|(?:\w+\s+){{0,2}}{_ROSTER}))?)",
        "states seniority within a small named unit",
        confidence=0.85,
    ),
    _rule(
        "cohort_marker",
        rf"(?P<clause>\b(?:i\s+am|i'?m|came\s+through|am)\b\s+"
        rf"(?:part\s+of\s+|through\s+|in\s+|on\s+)?{COHORT_MARKER}\b)",
        "places the respondent in a small named cohort",
        confidence=0.75,
    ),
    _rule(
        "clearance",
        rf"(?P<clause>\b(?:i\s+(?:am|'m|hold|have)|my|(?:to\s+)?get|hold)\b\s+"
        rf"(?:a\s+|an\s+|the\s+)?{CLEARANCE_MARKER}\s*(?:{CLEARANCE_LEVEL}\b)?)",
        "names a security clearance level",
        confidence=0.9,
    ),
    _rule(
        "corporate_marker",
        rf"(?P<clause>\b(?:our|my|the|a|an)\b\s+(?:\w+\s+){{0,2}}?{CORPORATE_MARKER})",
        "identifies a corporate setting",
        confidence=0.65,
        implicates="workplace",
    ),
)


class ImplicationDetector:
    """Finds clauses and word sequences that implicate a person or workplace."""

    name = "implication"
    stage = Stage.DETERMINISTIC

    def __init__(
        self,
        rules: Sequence[ImplicationRule] | None = None,
        *,
        enable_person: bool = True,
        enable_workplace: bool = True,
    ) -> None:
        self.rules = tuple(rules) if rules is not None else RULES
        self.enable_person = enable_person
        self.enable_workplace = enable_workplace

    def detect(self, text: str) -> Iterator[Finding]:
        if not text or not text.strip():
            return
        aligned = normalise_aligned(text)
        claimed: list[tuple[int, int]] = []
        for rule in sorted(self.rules, key=lambda r: -r.confidence):
            if rule.implicates == "person" and not self.enable_person:
                continue
            if rule.implicates == "workplace" and not self.enable_workplace:
                continue
            for match in rule.pattern.finditer(aligned):
                start, end = self._span_for(match, aligned)
                if start is None or end is None:
                    continue
                if any(start < c_end and c_start < end for c_start, c_end in claimed):
                    continue
                claimed.append((start, end))
                yield Finding(
                    span=Span(start, end),
                    category=Category.IMPLICATION,
                    detector=self.name,
                    stage=self.stage,
                    confidence=rule.confidence,
                    value=text[start:end],
                    metadata=(
                        ("rule", rule.name),
                        ("label", rule.label),
                        ("implicates", rule.implicates),
                        ("recognizer", f"IMPLICATION_{rule.name.upper()}_V1"),
                    ),
                )

    @staticmethod
    def _span_for(match: re.Match[str], text: str) -> tuple[int | None, int | None]:
        for name in ("clause", "workplace"):
            if name in match.re.groupindex and match.group(name) is not None:
                start, end = match.span(name)
                break
        else:
            start, end = match.span()
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end


def default_implication_detector() -> ImplicationDetector:
    return ImplicationDetector()


__all__ = [
    "RULES",
    "ImplicationDetector",
    "ImplicationRule",
    "alt",
    "default_implication_detector",
]
