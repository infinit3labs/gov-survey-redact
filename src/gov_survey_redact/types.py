"""Core types shared across the government survey redaction pipeline."""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class Category(str, enum.Enum):
    """Detection category for a PII or organisational attribution finding."""

    # --- Direct person & contact identifiers ---------------------------------
    EMAIL = "email"
    PHONE = "phone"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    MAC = "mac"
    URL_CREDENTIAL = "url_credential"
    TFN = "tfn"
    ABN = "abn"
    ACN = "acn"
    MEDICARE = "medicare"
    MEDICARE_ENROLMENT = "medicare_enrolment"
    CENTRELINK_CRN = "centrelink_crn"
    PASSPORT = "passport"
    DRIVERS_LICENCE = "drivers_licence"
    CREDIT_CARD = "credit_card"
    BANK_BSB = "bsb"
    BANK_ACCOUNT = "bank_account"
    REFERENCE_ID = "reference_id"
    POSTCODE = "postcode"
    STREET_ADDRESS = "street_address"
    STATE_POSTCODE = "state_postcode"
    IBAN = "iban"
    DATE = "date"
    DATE_AU = "date_au"
    DATE_ISO = "date_iso"
    NAME = "name"
    ORGANISATION = "organisation"
    LOCATION = "location"
    AGE = "age"
    GENDER = "gender"

    # --- Organisation & employer attribution ---------------------------------
    ORGANISATION_NAME = "organisation_name"
    INTERNAL_TERM = "internal_term"
    AWARD = "award"

    # --- Quasi-identifiers & implicating sequences ---------------------------
    JOB_TITLE = "job_title"
    TENURE = "tenure"
    TEAM_SIZE = "team_size"
    EMPLOYMENT_STATUS = "employment_status"
    HEALTH_DETAIL = "health_detail"
    IMPLICATION = "implication"

    @property
    def is_quasi_identifier(self) -> bool:
        """True for attributes that identify a respondent in combination."""
        return self in _QUASI_IDENTIFIERS

    @property
    def is_direct_identifier(self) -> bool:
        """True for attributes that directly identify a person or account."""
        return self in _DIRECT_IDENTIFIERS

    @property
    def is_organisation_identifier(self) -> bool:
        """True for attributes that identify or narrow an organisation/employer."""
        return self in _ORGANISATION_IDENTIFIERS

    def __str__(self) -> str:
        return self.value


_QUASI_IDENTIFIERS = frozenset(
    {
        Category.JOB_TITLE,
        Category.TENURE,
        Category.TEAM_SIZE,
        Category.EMPLOYMENT_STATUS,
        Category.HEALTH_DETAIL,
        Category.IMPLICATION,
        Category.AGE,
        Category.GENDER,
        Category.DATE_AU,
        Category.DATE_ISO,
        Category.LOCATION,
    }
)

_DIRECT_IDENTIFIERS = frozenset(
    {
        Category.EMAIL,
        Category.PHONE,
        Category.TFN,
        Category.MEDICARE,
        Category.MEDICARE_ENROLMENT,
        Category.CENTRELINK_CRN,
        Category.PASSPORT,
        Category.DRIVERS_LICENCE,
        Category.CREDIT_CARD,
        Category.IBAN,
        Category.STREET_ADDRESS,
        Category.NAME,
        Category.IPV4,
        Category.IPV6,
        Category.BANK_BSB,
        Category.BANK_ACCOUNT,
        Category.REFERENCE_ID,
    }
)

_ORGANISATION_IDENTIFIERS = frozenset(
    {
        Category.ABN,
        Category.ACN,
        Category.ORGANISATION,
        Category.ORGANISATION_NAME,
        Category.INTERNAL_TERM,
        Category.AWARD,
    }
)


class Stage(str, enum.Enum):
    """Pipeline stage that produced a finding."""

    DETERMINISTIC = "deterministic"
    NLP = "nlp"

    def __str__(self) -> str:
        return self.value


class MaskStyle(str, enum.Enum):
    """Replacement strategy for a detected span."""

    MASK = "mask"
    REMOVE = "remove"
    PARTIAL = "partial"
    LABEL = "label"
    HASH = "hash"
    TOKEN = "token"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Span:
    """Half-open character range ``[start, end)`` in the source text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span: [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def extract(self, text: str) -> str:
        return text[self.start : self.end]


@dataclass(frozen=True, slots=True)
class Finding:
    """Detected PII or attribution span with provenance and confidence."""

    span: Span
    category: Category
    detector: str
    stage: Stage = Stage.DETERMINISTIC
    confidence: float = 1.0
    value: str | None = None
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @property
    def priority(self) -> tuple[int, float]:
        """Sort key: deterministic findings outrank NLP ones, then confidence."""
        base = 0 if self.stage is Stage.DETERMINISTIC else -1
        return (base, round(self.confidence, 6))

    def meta(self) -> dict[str, Any]:
        return dict(self.metadata)


class RedactionError(ValueError):
    """Raised when input cannot be redacted."""


def total_overlap(spans: Iterable[Span]) -> int:
    return sum(s.length for s in spans)
