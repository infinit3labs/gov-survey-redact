"""Cross-field deterministic propagation for structured-to-narrative PII matching.

As specified in the hybrid architecture:
When a structured respondent table already records that the respondent is
``Jane Smith``, lives at ``12 Example Street``, works at ``Northline Services``,
or has household members ``John Smith``, we construct an ephemeral per-record
dictionary from those structured columns and scan all open-text answers for
them *before* generic NER.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from .normalise import normalise_aligned
from .types import Category, Finding, Span, Stage

# Common words that must never be propagated as bare single-token given/family names
_COMMON_WORD_STOP_SET = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "have",
        "will",
        "work",
        "care",
        "health",
        "safe",
        "safety",
        "team",
        "ward",
        "hill",
        "west",
        "north",
        "south",
        "east",
        "park",
        "street",
        "road",
        "lane",
        "court",
        "place",
        "young",
        "long",
        "short",
        "clear",
        "good",
        "best",
        "well",
        "back",
        "just",
        "only",
        "may",
        "june",
        "march",
        "august",
        "monday",
        "friday",
        "yes",
        "no",
        "none",
        "null",
        "n/a",
    }
)

DEFAULT_STRUCTURED_FIELD_MAP: dict[str, Category] = {
    "respondent_name": Category.NAME,
    "full_name": Category.NAME,
    "name": Category.NAME,
    "given_name": Category.NAME,
    "first_name": Category.NAME,
    "family_name": Category.NAME,
    "surname": Category.NAME,
    "last_name": Category.NAME,
    "household_members": Category.NAME,
    "relative_name": Category.NAME,
    "spouse_name": Category.NAME,
    "supervisor_name": Category.NAME,
    "manager_name": Category.NAME,
    "respondent_email": Category.EMAIL,
    "email": Category.EMAIL,
    "respondent_phone": Category.PHONE,
    "phone": Category.PHONE,
    "mobile": Category.PHONE,
    "street_address": Category.STREET_ADDRESS,
    "residential_address": Category.STREET_ADDRESS,
    "address": Category.STREET_ADDRESS,
    "employer": Category.ORGANISATION_NAME,
    "employer_name": Category.ORGANISATION_NAME,
    "organisation": Category.ORGANISATION_NAME,
    "facility_name": Category.ORGANISATION_NAME,
    "school_name": Category.ORGANISATION_NAME,
    "tfn": Category.TFN,
    "medicare": Category.MEDICARE,
    "crn": Category.CENTRELINK_CRN,
    "claim_number": Category.REFERENCE_ID,
    "claim_id": Category.REFERENCE_ID,
    "claim_reference": Category.REFERENCE_ID,
}


@dataclass(frozen=True, slots=True)
class KnownValueEntry:
    """One known identifier extracted from a record's structured columns."""

    value: str
    category: Category
    source_field: str
    confidence: float = 0.99


class CrossFieldDictionary:
    """Ephemeral per-record dictionary built from structured survey columns."""

    def __init__(self, entries: Sequence[KnownValueEntry]) -> None:
        # Sort longest value first so full names ("Jane Smith") claim spans before
        # single tokens ("Jane" or "Smith").
        self.entries: tuple[KnownValueEntry, ...] = tuple(
            sorted(entries, key=lambda e: (-len(e.value), e.source_field))
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        field_map: Mapping[str, Category] | None = None,
    ) -> CrossFieldDictionary:
        """Extract known person, address, contact, and employer values from ``record``."""
        mapping = {k.lower(): v for k, v in (field_map or DEFAULT_STRUCTURED_FIELD_MAP).items()}
        entries: list[KnownValueEntry] = []
        seen: set[tuple[str, Category]] = set()

        def _add(val: str, cat: Category, field_name: str, conf: float = 0.99) -> None:
            cleaned = val.strip()
            if len(cleaned) < 3:
                return
            if cleaned.lower() in _COMMON_WORD_STOP_SET:
                return
            key = (cleaned.lower(), cat)
            if key in seen:
                return
            seen.add(key)
            entries.append(KnownValueEntry(value=cleaned, category=cat, source_field=field_name, confidence=conf))

        for raw_field, raw_val in record.items():
            if raw_val is None:
                continue
            category = mapping.get(raw_field.lower())
            if category is None:
                continue

            values: list[str]
            if isinstance(raw_val, (list, tuple)):
                values = [str(item) for item in raw_val if item]
            else:
                text_val = str(raw_val).strip()
                if not text_val:
                    continue
                # Split semicolon/comma lists for household rosters
                if category is Category.NAME and (";" in text_val or "," in text_val):
                    values = [part.strip() for part in re.split(r"[;,]", text_val) if part.strip()]
                else:
                    values = [text_val]

            for val in values:
                _add(val, category, raw_field, 0.99)
                # For multi-word person names ("Jane Smith"), also register the
                # individual given name and family surname so "my husband John Smith"
                # or "call Jane" in narrative prose is caught deterministically.
                if category is Category.NAME:
                    parts = [p.strip(".,;:()") for p in val.split() if len(p.strip(".,;:()")) >= 3]
                    if len(parts) >= 2:
                        for part in parts:
                            if part.lower() not in {"mr", "mrs", "ms", "miss", "mx", "dr", "prof"}:
                                _add(part, Category.NAME, f"{raw_field}:part", 0.95)

        return cls(entries)

    def __bool__(self) -> bool:
        return bool(self.entries)


class KnownRecordDetector:
    """Scans free-text fields for exact/aligned matches from a ``CrossFieldDictionary``."""

    name = "cross_field_dictionary"
    rule_id = "KNOWN_RECORD_VALUE_V1"
    stage = Stage.DETERMINISTIC

    def __init__(self, dictionary: CrossFieldDictionary) -> None:
        self.dictionary = dictionary
        self._compiled: list[tuple[re.Pattern[str], KnownValueEntry]] = [
            (
                re.compile(rf"(?<![\w]){re.escape(normalise_aligned(entry.value))}(?![\w])", re.IGNORECASE),
                entry,
            )
            for entry in dictionary.entries
        ]

    def detect(self, text: str) -> Iterator[Finding]:
        if not text or not self._compiled:
            return
        aligned = normalise_aligned(text)
        claimed: list[tuple[int, int]] = []
        for pattern, entry in self._compiled:
            for match in pattern.finditer(aligned):
                start, end = match.span()
                if any(start < c_end and c_start < end for c_start, c_end in claimed):
                    continue
                claimed.append((start, end))
                yield Finding(
                    span=Span(start, end),
                    category=entry.category,
                    detector=self.name,
                    stage=self.stage,
                    confidence=entry.confidence,
                    value=text[start:end],
                    metadata=(
                        ("source_field", entry.source_field),
                        ("recognizer", self.rule_id),
                    ),
                )


__all__ = [
    "DEFAULT_STRUCTURED_FIELD_MAP",
    "CrossFieldDictionary",
    "KnownRecordDetector",
    "KnownValueEntry",
]
