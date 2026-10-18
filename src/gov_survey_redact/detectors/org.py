"""Organisation, government agency, internal-term, and award detectors."""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..lexicon import Lexicon, LexiconMatch, load_builtin_lexicon
from ..normalise import normalise_aligned, token_key
from ..types import Category, Finding, Span, Stage

_ENTITY_SUFFIX = (
    r"Pty(?:\s+Ltd)?|Ltd|Limited|Inc|Corp(?:oration)?|Company|Co|Group|Holdings|"
    r"Solutions|Technologies|Partners|Associates|"
    r"Foundation|Trust|Society|Association|Union|Council|"
    r"Academy|College|University|School|"
    r"Hospital|Medical\s+Centre|Clinic|Superclinic|Family\s+Practice|Medical\s+Practice|"
    r"Church|Charity|Institute|"
    r"Health|Healthcare|Care|Caring|Services|Resources|Industries|"
    r"Organisation|Organization|Management|Consulting|Rehabilitation|Adjusting"
)

_EMPLOYMENT_CONTEXT = (
    r"work(?:ed|ing)?\s+(?:at|for)|employ(?:ed|ment)?\s+(?:by|at)|"
    r"join(?:ed|ing)|joined|at\s+my|workplace|employer|"
    r"my\s+company|our\s+company|contract(?:ed|or)?\s+(?:to|by)|"
    r"this\s+is\s+about|it\s+is\s+about|specifically\s+about|"
    r"employed\s+with|works?\s+with"
)

_TRIGGER_VERBS = frozenset(
    {
        "joined",
        "join",
        "joining",
        "worked",
        "work",
        "working",
        "employed",
        "employment",
        "contracted",
    }
)

_ORG_NAME_RE = re.compile(
    rf"\b(?P<name>(?:[A-Z][\w&'’.\-]*\s+){{1,4}}(?:{_ENTITY_SUFFIX}))\b"
)
_GOV_AGENCY_NAME_RE = re.compile(
    r"\b(?P<name>(?:Department|Ministry|Commission|Office|Bureau|Authority)\s+of\s+"
    r"[A-Z][\w&'’.\-]*(?:\s+(?:and\s+)?[A-Z][\w&'’.\-]*){0,4})\b"
)
_EMPLOYMENT_ORG_RE = re.compile(
    rf"(?i:{_EMPLOYMENT_CONTEXT})\s+(?:the\s+)?"
    r"(?P<name>[A-Z][\w&'’.\-]*(?:\s+[A-Z][\w&'’.\-]*){0,3})"
)
_AU_ACN_RE = re.compile(
    r"\b(?:ACN|ABN)\s*:?\s*\d{1,4}(?:[ -]?\d{1,4}){1,4}\b",
    re.IGNORECASE,
)

_GENERIC_ORG_NAMES = frozenset(
    {
        "safetycommittee",
        "healthandsafetycommittee",
        "hsccommittee",
        "safetycommitteeteam",
        "wesafetycommittee",
        "humanresources",
        "hrteam",
        "hrdepartment",
        "peopleandculture",
        "thecompany",
        "ourcompany",
        "mycompany",
        "theorganisation",
        "ourorganisation",
        "myorganisation",
        "myemployer",
        "ouremployer",
        "thesite",
        "mysite",
        "oursite",
        "headoffice",
        "ouroffice",
        "theoffice",
        "mymanager",
        "mysupervisor",
        "myteam",
        "ourteam",
        "theteam",
        "seniormanagement",
        "uppermmanagement",
        "middlemanagement",
        "themanagementteam",
        "managementteam",
        "leadershipteam",
        "everyone",
        "everybody",
        "nobody",
        "noone",
        "noleone",
        "workplace",
        "theworkplace",
        "myworkplace",
        "ourworkplace",
        "incident",
        "theincident",
        "grievance",
        "thegrievance",
        "complaint",
        "award",
        "theaward",
        "union",
        "theunion",
        "safety",
        "health",
        "trainingteam",
        "recruitmentteam",
        "payrollteam",
        "helpdesk",
        "people",
        "staff",
        "employees",
        "workers",
        "customers",
        "client",
        "clients",
        "thepolice",
        "thecouncil",
        "thegovernment",
        "freedomofinformation",
    }
)

_GENERIC_WORDS = frozenset(
    {
        "the",
        "our",
        "my",
        "a",
        "an",
        "safety",
        "health",
        "support",
        "management",
        "team",
        "staff",
        "workers",
        "employee",
        "employees",
        "workplace",
        "site",
        "office",
        "head",
        "senior",
        "upper",
        "middle",
        "leadership",
        "human",
        "resources",
        "incident",
        "grievance",
        "complaint",
        "award",
        "union",
        "company",
        "organisation",
        "organization",
        "group",
        "everyone",
        "nobody",
        "no",
        "one",
        "people",
        "committee",
        "department",
        "division",
        "training",
        "recruitment",
        "payroll",
        "help",
        "desk",
        "police",
        "council",
        "government",
        "system",
        "systems",
        "member",
        "customers",
        "client",
        "clients",
        "customer",
        "abn",
        "acn",
        "tfn",
        "crn",
        "bsb",
    }
)

_NAME_FUNCTION_WORDS = frozenset(
    {
        "from",
        "to",
        "for",
        "with",
        "and",
        "or",
        "of",
        "in",
        "at",
        "on",
        "by",
        "as",
        "this",
        "that",
        "these",
        "those",
        "their",
        "our",
        "your",
        "my",
        "but",
        "if",
        "so",
        "then",
        "than",
        "they",
        "them",
        "we",
        "us",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "not",
    }
)


class OrganisationNameDetector:
    """Detects named employers, agencies, clinics, and corporate entities."""

    name = "organisation_name"
    rule_id = "ORG_AU_NAMED_ENTITY_V1"
    stage = Stage.DETERMINISTIC

    def __init__(self, *, enable_named_entities: bool = True) -> None:
        self.enable_named_entities = enable_named_entities

    def detect(self, text: str) -> Iterator[Finding]:
        if not text or not text.strip():
            return
        aligned = normalise_aligned(text)
        claimed: list[tuple[int, int]] = []
        for regex, confidence, reason in (
            (_ORG_NAME_RE, 0.95, "entity suffix"),
            (_GOV_AGENCY_NAME_RE, 0.92, "government agency"),
            (_EMPLOYMENT_ORG_RE, 0.80, "employment context"),
        ):
            for match in regex.finditer(aligned):
                start = match.start("name")
                full = aligned[match.start("name") : match.end("name")]
                raw = _trim_name(full, allow_of=(reason == "government agency"))
                if raw is None:
                    continue
                offset = full.find(raw)
                if offset < 0:
                    continue
                start += offset
                end = start + len(raw)
                if any(start < c_end and c_start < end for c_start, c_end in claimed):
                    continue
                claimed.append((start, end))
                yield Finding(
                    span=Span(start, end),
                    category=Category.ORGANISATION_NAME,
                    detector=self.name,
                    stage=self.stage,
                    confidence=confidence,
                    value=text[start:end],
                    metadata=(
                        ("entity", raw),
                        ("reason", reason),
                        ("recognizer", self.rule_id),
                    ),
                )
        for match in _AU_ACN_RE.finditer(aligned):
            if not re.match(r"(?i)abn", match.group()):
                continue
            yield Finding(
                span=Span(*match.span()),
                category=Category.ABN,
                detector=self.name,
                stage=self.stage,
                confidence=0.95,
                value=match.group(),
                metadata=(
                    ("reason", "employer identifier"),
                    ("recognizer", "ORG_AU_LABELLED_ABN_V1"),
                ),
            )


class LexiconTermDetector:
    """Exposes a :class:`Lexicon` as a deterministic engine detector."""

    name = "lexicon_term"
    stage = Stage.DETERMINISTIC

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        self.lexicon = lexicon if lexicon is not None else Lexicon()

    def detect(self, text: str) -> Iterator[Finding]:
        for match in self.lexicon.match(text):
            yield _lexicon_finding(match, self.name)


class AwardCodeDetector:
    """Modern award identifiers such as ``MA000019``."""

    name = "award_code"
    rule_id = "ORG_AU_AWARD_CODE_V1"
    stage = Stage.DETERMINISTIC

    _CODE_RE = re.compile(r"\bMA\d{6}\b", re.IGNORECASE)

    def detect(self, text: str) -> Iterator[Finding]:
        for match in self._CODE_RE.finditer(text):
            yield Finding(
                span=Span(*match.span()),
                category=Category.AWARD,
                detector=self.name,
                stage=self.stage,
                confidence=1.0,
                value=match.group(),
                metadata=(
                    ("code", match.group().upper()),
                    ("recognizer", self.rule_id),
                ),
            )


def _trim_name(raw: str, *, allow_of: bool = False) -> str | None:
    name = raw.strip(" \t.,;:")
    if not name:
        return None

    words = name.split()
    while len(words) > 1 and words[0].casefold() in _TRIGGER_VERBS:
        words = words[1:]
    name = " ".join(words).strip()
    if not name:
        return None

    folded = token_key(name)
    if folded in _GENERIC_ORG_NAMES:
        return None
    lowered = {w.casefold().strip(".,") for w in words}
    if lowered and lowered <= _GENERIC_WORDS:
        return None
    blocked_fn = _NAME_FUNCTION_WORDS - ({"of", "and"} if allow_of else set())
    if lowered & blocked_fn:
        return None
    return name


def _lexicon_finding(match: LexiconMatch, detector: str) -> Finding:
    metadata: dict[str, object] = {
        "canonical": match.canonical,
        "label": match.label,
        "recognizer": "LEXICON_TERM_V1",
    }
    if match.fuzzy:
        metadata["fuzzy"] = True
        metadata["distance"] = match.distance
    return Finding(
        span=Span(*match.span),
        category=match.category,
        detector=detector,
        stage=Stage.DETERMINISTIC,
        confidence=0.85 if match.fuzzy else 1.0,
        value=match.matched_text,
        metadata=tuple(sorted(metadata.items())),
    )


def default_organisation_detectors(
    lexicon: Lexicon | None = None,
) -> tuple[OrganisationNameDetector, LexiconTermDetector, AwardCodeDetector]:
    resolved = lexicon if lexicon is not None else load_builtin_lexicon()
    return (
        OrganisationNameDetector(),
        LexiconTermDetector(resolved),
        AwardCodeDetector(),
    )


__all__ = [
    "AwardCodeDetector",
    "LexiconTermDetector",
    "OrganisationNameDetector",
    "default_organisation_detectors",
]
