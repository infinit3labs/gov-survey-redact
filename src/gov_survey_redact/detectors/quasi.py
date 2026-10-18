"""Quasi-identifier and sensitive-attribute detectors for survey free text.

Per the hybrid architecture specification:
* Quasi-identifiers (job title, pay grade, tenure, team size, employment status)
  are detected so they can be generalised into analytical bands or scored for
  combination disclosure risk.
* Substantive sensitive attributes (e.g. ``HEALTH_DETAIL``) are detected with
  strict guards against validated protective survey phrasing ("I am clear on
  what my responsibilities are", "I have support to work safely") and governed
  by release policy rather than blindly erased.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..generalise import is_banded
from ..normalise import normalise_aligned
from ..types import Category, Finding, Span, Stage
from ..vocabulary import APS_LEVEL

_AU_OCCUPATIONS = (
    r"nurse|midwife|doctor|physician|surgeon|dentist|pharmacist|"
    r"teacher|lecturer|principal|"
    r"police officer|detective|constable|firefighter|paramedic|ambulance officer|"
    r"electrician|plumber|carpenter|welder|builder|labourer|"
    r"bus driver|truck driver|forklift driver|taxi driver|delivery driver|"
    r"cleaner|housekeeper|janitor|custodian|"
    r"chef|cook|baker|barista|"
    r"care worker|carer|support worker|"
    r"secretary|receptionist|accountant|auditor|"
    r"cashier|storeperson|"
    r"trainee|apprentice|intern|"
    r"librarian|curator|"
    r"security officer|"
    r"registered nurse|enrolled nurse"
)

_JOB_TITLE_LABEL_RE = re.compile(
    rf"(?i)\b(?:my\s+(?:job\s*title|role|position|occupation)|"
    rf"job\s*title|position\s*title|designation|"
    rf"i\s+(?:am|work|worked)\s+as\s+an?)\s*:?\s*"
    rf"(?P<value>(?:(?:[a-z]+\s+){{0,2}})(?:{_AU_OCCUPATIONS}))\b"
)
_JOB_TITLE_BARE_RE = re.compile(
    rf"\b(?P<value>(?:registered|enrolled|senior|junior|trained)?\s*(?:{_AU_OCCUPATIONS}))\b",
    re.IGNORECASE,
)

_GRADE_RE = re.compile(
    r"(?i)\b(?:grade|band|class|category|step|pay\s*point)\s*[:#]?\s*"
    r"(?P<value>[a-z]{1,2}\s?\d{1,2}|\d{1,2}[a-z]?)\b"
)

_TENURE_AFTER = (
    r"(?:"
    r"\s+of\s+service|"
    r"\s+(?:with|at)\s+(?:the|my|our|this)\s+\w+|"
    r"\s+in\s+(?:the|my|our|this)\s+\w+|"
    r"\s+there|"
    r"\s+at\s+(?:the|my|our)\s+(?:company|organisation|employer|workplace)"
    r")"
)
_TENURE_BEFORE = (
    r"(?:"
    r"(?:i'?ve\s+)?(?:been|worked|working)\s+"
    r"(?:here|there|for\s+them|with\s+(?:them|the\s+company))\s+"
    r"(?:now\s+|already\s+)?"
    r")"
)
_TENURE_RE = re.compile(
    rf"(?i)\b(?:{_TENURE_BEFORE})?(?P<value>\d{{1,2}}(?:\.\d)?\s*\+?\s*years?{_TENURE_AFTER})"
    rf"|\b{_TENURE_BEFORE}(?P<value2>\d{{1,2}}(?:\.\d)?\s*\+?\s*years?)"
)
_TENURE_SINCE_RE = re.compile(
    r"(?i)\b(?:since|from|started\s+there\s+in|been\s+(?:here|with\s+\w+)\s+since|"
    r"i\s+(?:joined|started)\s+(?:in|here\s+in))\s+(?P<value>(?:19|20)\d{2})\b"
)

_TEAM_SIZE_NOUN = r"team|department|dept|unit|ward|crew|roster|school|store|classroom|section"
_NUMBER = (
    r"\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"a\s+few|several"
)
_TEAM_SIZE_RE = re.compile(
    rf"(?i)\b(?:{_TEAM_SIZE_NOUN})\b\s*"
    r"(?:(?:of|with|has|had|was|is|comprising|consisting\s+of)\s*)?"
    r"(?:just|only|about|around|approximately|roughly)?\s*"
    rf"(?P<value>{_NUMBER})\s*(?:\+\s*)?"
    r"(?:\s*(?:people|staff|workers|employees|members|persons|of\s+us|strong))?"
    r"(?!\s*(?:year|month|week|hour|day|minute))"
    r"(?![\d:/\-])"
)

_STATUS_VALUE = (
    r"casual|part[\s-]?time|full[\s-]?time|temporary|temp|contract(?:or)?|"
    r"agency|fixed[\s-]?term|relief|shift\s+worker|rotating\s+shift|night\s+shift|"
    r"subcontractor|apprentice|trainee|intern|volunteer"
)
_EMPLOYMENT_STATUS_RE = re.compile(
    rf"(?i)\b(?:i\s+am|i'?m|as)\s+(?:(?:a|an|currently|still|now)\s+)*"
    rf"(?P<value>{_STATUS_VALUE})\b"
)
_EMPLOYMENT_STATUS_LIST_RE = re.compile(
    rf"(?i)(?:,\s*|^)(?P<value>{_STATUS_VALUE})\s*(?:,|$|\.)",
)

_HEALTH_DETAIL_RE = re.compile(
    r"(?i)\b(?:i\s+(?:am|'m|have|suffer\s+from|developed)|my\s+(?:condition|illness|injury|disease|"
    r"disorder|symptom)s?|diagnosed\s+with|struggling\s+with)\s+"
    r"(?:(?:is|was|are|were)\s+)?"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9\-']*(?:\s+[A-Za-z0-9][A-Za-z0-9\-']*){0,4})"
)
_MENTAL_HEALTH_RE = re.compile(
    r"(?i)\b(?:anxiety|depression|ptsd|psychological\s+distress|"
    r"mental\s+health|panic\s+attacks?|burnout|psychosis|"
    r"psychological\s+safety|mental\s+illness|chronic\s+fatigue)\b"
)

_CONDITION_ED = frozenset(
    {
        "depressed",
        "undiagnosed",
        "undiagnosed?",
        "bedridden",
        "longterm",
        "overworked",
        "overtired",
        "understaffed",
        "overstretched",
    }
)

_NOT_HEALTH_VERBS = frozenset(
    {
        "put",
        "got",
        "gotten",
        "taken",
        "done",
        "seen",
        "noticed",
        "heard",
        "read",
        "written",
        "said",
        "told",
        "asked",
        "found",
        "used",
        "left",
        "been",
        "going",
        "being",
        "doing",
        "working",
        "trying",
        "hoping",
        "already",
        "also",
        "just",
        "only",
        "still",
        "even",
        "never",
        "always",
        "twice",
        "thrice",
        "since",
        "before",
        "after",
        "again",
        "twice,",
        "raised",
        "worked",
        "let",
        "set",
        "kept",
        "gave",
        "made",
        "rung",
        "brought",
        "held",
        "spent",
        "sent",
        "paid",
    }
)

_NOT_HEALTH_NOUNS = frozenset(
    {
        "family",
        "families",
        "kids",
        "twins",
        "mortgage",
        "transport",
        "passport",
        "licence",
        "license",
        "visa",
        "loan",
        "degree",
        "certificate",
        "keys",
        "phone",
        "laptop",
        "car",
        "house",
        "home",
        "job",
        "holiday",
        "vacation",
        "screenshots",
        "screenshot",
        "receipts",
        "receipt",
        "paperwork",
        "references",
        "reference",
        "questions",
        "concerns",
        "notes",
        "records",
        "documents",
        "forms",
        "photos",
        "copies",
        "invoices",
        "invoice",
        "bookings",
        "password",
        "reminders",
        "appointments",
        "files",
        "statements",
        "passwords",
        "spare",
        "spares",
    }
)

_OCCUPATION_WORDS = frozenset(
    {
        "nurse",
        "midwife",
        "teacher",
        "doctor",
        "driver",
        "cleaner",
        "electrician",
        "plumber",
        "barista",
        "chef",
        "cook",
        "warden",
        "carer",
        "care",
        "worker",
        "labourer",
        "technician",
        "officer",
        "clerk",
        "accountant",
        "librarian",
        "pharmacist",
    }
)

_NOT_HEALTH = frozenset(
    {
        "the",
        "a",
        "an",
        "my",
        "in",
        "of",
        "at",
        "to",
        "for",
        "on",
        "and",
        "or",
        "only",
        "woman",
        "man",
        "guy",
        "lady",
        "girl",
        "person",
        "casual",
        "temp",
        "part",
        "time",
        "full",
        "contract",
        "contractor",
        "apprentice",
        "trainee",
        "intern",
        "volunteer",
        "team",
        "unit",
        "department",
        "shift",
        "grade",
        "new",
        "old",
        "sure",
        "not",
        "so",
        "very",
        "really",
        "just",
        "still",
        "always",
        "never",
        "here",
        "there",
        "back",
        "doing",
        "working",
        "been",
        "one",
        "two",
        "three",
        "no",
        "yes",
        "about",
        "after",
        "before",
        "since",
        "lot",
        "bit",
        "kind",
        "sort",
        "same",
        "other",
        "another",
        "each",
        "is",
        "was",
        "are",
        "were",
        "be",
        "being",
        "am",
        "years",
        "year",
        "yrs",
        "yr",
        "months",
        "month",
        "weeks",
        "week",
        "days",
        "day",
        "hours",
        "hour",
        "aged",
        "every",
        "all",
        "most",
        "some",
        "any",
        "many",
        "few",
        "last",
        "next",
        "past",
        "recent",
        "recently",
        "lastly",
        "finally",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
        "with",
        "from",
        "this",
        "that",
        "these",
        "those",
        "have",
        "has",
        "had",
        "clear",
        "torn",
        "support",
        "confused",
        "lucky",
        "grateful",
        "confident",
        "happy",
        "proud",
        "aware",
    }
)

_HEALTH_ALLOWED_CONNECTORS = frozenset({"and", "or", "of", "in", "to", "with"})
_MAX_HEALTH_WORDS = 4


def _value_group(regex: re.Pattern[str], match: re.Match[str]) -> str | int | None:
    named = [name for name in regex.groupindex if name.startswith("value")]
    for name in named:
        if match.group(name) is not None:
            return name
    return 0 if not named else None


class _RegexCategoryDetector:
    name = "quasi"
    stage = Stage.DETERMINISTIC

    def patterns(self) -> tuple[tuple[re.Pattern[str], Category, float], ...]:
        raise NotImplementedError

    def clean_value(self, raw: str) -> str | None:
        return raw

    def detect(self, text: str) -> Iterator[Finding]:
        if not text or not text.strip():
            return
        aligned = normalise_aligned(text)
        for regex, category, confidence in self.patterns():
            for match in regex.finditer(aligned):
                group = _value_group(regex, match)
                if group is None:
                    continue
                captured = match.group(group)
                raw = captured.strip()
                cleaned = self.clean_value(raw)
                if cleaned is None:
                    continue
                offset = match.start(group) + (len(captured) - len(captured.lstrip()))
                if cleaned != raw:
                    index = aligned.find(cleaned, offset)
                    if index < 0:
                        continue
                    offset = index
                start, end = offset, offset + len(cleaned)
                yield Finding(
                    span=Span(start, end),
                    category=category,
                    detector=self.name,
                    stage=self.stage,
                    confidence=confidence,
                    value=text[start:end],
                    metadata=(
                        ("cue", cleaned.lower()),
                        ("recognizer", f"QUASI_{self.name.upper()}_V1"),
                    ),
                )


_CLASSIFICATION_ONLY_RE = re.compile(
    rf"(?i)^(?:{APS_LEVEL}|ses|so[0-3]|grade\s*\d{{1,2}}|band\s*[a-z]?\d{{1,2}}|"
    r"step\s*\d|class\s*\d|category\s*\d)$"
)


class JobTitleDetector(_RegexCategoryDetector):
    name = "job_title"

    def patterns(self):
        return (
            (_JOB_TITLE_LABEL_RE, Category.JOB_TITLE, 0.95),
            (_GRADE_RE, Category.JOB_TITLE, 0.9),
            (_JOB_TITLE_BARE_RE, Category.JOB_TITLE, 0.55),
        )


class TenureDetector(_RegexCategoryDetector):
    name = "tenure"

    def patterns(self):
        return (
            (_TENURE_SINCE_RE, Category.TENURE, 0.75),
            (_TENURE_RE, Category.TENURE, 0.85),
        )


class TeamSizeDetector(_RegexCategoryDetector):
    name = "team_size"

    def patterns(self):
        return ((_TEAM_SIZE_RE, Category.TEAM_SIZE, 0.85),)


class EmploymentStatusDetector(_RegexCategoryDetector):
    name = "employment_status"

    def patterns(self):
        return (
            (_EMPLOYMENT_STATUS_RE, Category.EMPLOYMENT_STATUS, 0.85),
            (_EMPLOYMENT_STATUS_LIST_RE, Category.EMPLOYMENT_STATUS, 0.6),
        )


class HealthDetailDetector(_RegexCategoryDetector):
    """Health conditions disclosed by the respondent."""

    name = "health_detail"

    def patterns(self):
        return (
            (_MENTAL_HEALTH_RE, Category.HEALTH_DETAIL, 0.8),
            (_HEALTH_DETAIL_RE, Category.HEALTH_DETAIL, 0.7),
        )

    def clean_value(self, raw: str) -> str | None:
        words = raw.split()
        if not words:
            return None
        if _CLASSIFICATION_ONLY_RE.match(raw):
            return None
        if words[0].lower().endswith("ing"):
            return None
        if words[0].lower().endswith("ed") and words[0].lower() not in _CONDITION_ED:
            return None
        kept: list[str] = []
        for index, word in enumerate(words):
            lowered = word.lower().strip(".,;:!?")
            if not lowered:
                break
            if (
                lowered in _NOT_HEALTH
                or lowered in _OCCUPATION_WORDS
                or lowered in _NOT_HEALTH_VERBS
                or lowered in _NOT_HEALTH_NOUNS
            ):
                break
            if index and lowered in _HEALTH_ALLOWED_CONNECTORS and kept:
                kept.append(word)
                continue
            kept.append(word)
            if len(kept) >= _MAX_HEALTH_WORDS:
                break
        while kept and kept[-1].lower() in _HEALTH_ALLOWED_CONNECTORS:
            kept.pop()
        if not kept:
            return None
        if all(word.isdigit() for word in kept):
            return None
        if is_banded(raw):
            return None
        return " ".join(kept)


QUASI_DETECTORS: tuple[type[_RegexCategoryDetector], ...] = (
    JobTitleDetector,
    TenureDetector,
    TeamSizeDetector,
    EmploymentStatusDetector,
    HealthDetailDetector,
)


def default_quasi_detectors() -> tuple[_RegexCategoryDetector, ...]:
    return tuple(cls() for cls in QUASI_DETECTORS)


__all__ = [
    "EmploymentStatusDetector",
    "HealthDetailDetector",
    "JobTitleDetector",
    "TeamSizeDetector",
    "TenureDetector",
    "default_quasi_detectors",
]
