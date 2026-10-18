"""Schema-aware field pruning and column disposition policy for government surveys.

Per the hybrid architecture specification:
1. **Schema pruning comes first**:
   * Whole columns whose disposition is already known (``respondent_id``,
     ``household_id``, ``respondent_email``, ``dob``) are tokenised, masked, or
     generalised without invoking NER.
   * Numerical/categorical survey measures (e.g. Likert 1–5 items, timestamps,
     instrument flags) pass through untouched.
   * Only unresolved open-text fields proceed to deterministic + compact NER.
2. **Question context preservation**:
   * Open-text fields retain their question prompt (``[Question: ...]``) for
     contextual NER without concatenating the entire questionnaire into one blob.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field

from .generalise import generalise_date
from .masking import build_masker
from .types import Category, MaskStyle


class FieldDisposition(str, enum.Enum):
    """How a survey column is governed before free-text redaction."""

    PASSTHROUGH_METADATA = "passthrough_metadata"
    PASSTHROUGH_NUMERIC = "passthrough_numeric"
    WHOLE_COLUMN_REDACT = "whole_column_redact"
    WHOLE_COLUMN_TOKENISE = "whole_column_tokenise"
    WHOLE_COLUMN_REMOVE = "whole_column_remove"
    WHOLE_COLUMN_GENERALISE_DOB = "whole_column_generalise_dob"
    FREE_TEXT_HYBRID = "free_text_hybrid"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class FieldPolicy:
    """Governance rule for a single survey column."""

    column: str
    disposition: FieldDisposition
    category: Category | None = None
    question_text: str | None = None
    propagate_to_free_text: bool = False


DEFAULT_METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "ResponseID",
        "study_id",
        "survey_id",
        "template_id",
        "StartDate",
        "EndDate",
        "Instrument",
        "Consent",
        "language",
        "ocr_confidence",
    }
)

DEFAULT_TOKENISE_COLUMNS: dict[str, Category] = {
    "respondent_id": Category.REFERENCE_ID,
    "household_id": Category.REFERENCE_ID,
    "member_id": Category.REFERENCE_ID,
    "claim_number": Category.REFERENCE_ID,
    "claim_id": Category.REFERENCE_ID,
    "claim_reference": Category.REFERENCE_ID,
}

DEFAULT_REDACT_COLUMNS: dict[str, Category] = {
    "respondent_name": Category.NAME,
    "full_name": Category.NAME,
    "given_name": Category.NAME,
    "family_name": Category.NAME,
    "surname": Category.NAME,
    "respondent_email": Category.EMAIL,
    "respondent_phone": Category.PHONE,
    "street_address": Category.STREET_ADDRESS,
    "residential_address": Category.STREET_ADDRESS,
    "employer_name": Category.ORGANISATION_NAME,
    "IPAddress": Category.IPV4,
    "ip_address": Category.IPV4,
}

DEFAULT_DOB_COLUMNS: frozenset[str] = frozenset(
    {"dob", "date_of_birth", "birth_date", "respondent_dob"}
)


@dataclass(frozen=True, slots=True)
class SurveySchema:
    """Schema-aware column governance and question-context registry."""

    fields: Mapping[str, FieldPolicy] = field(default_factory=dict)
    question_prompts: Mapping[str, str] = field(default_factory=dict)
    redact_metadata_ip: bool = False

    def policy_for(self, column: str, sample_value: object = None) -> FieldPolicy:
        """Resolve the governance policy for ``column``."""
        if column in self.fields:
            return self.fields[column]

        col_lower = column.lower()
        if column in DEFAULT_METADATA_COLUMNS or col_lower in {c.lower() for c in DEFAULT_METADATA_COLUMNS}:
            if self.redact_metadata_ip and col_lower in {"ipaddress", "ip_address"}:
                return FieldPolicy(
                    column=column,
                    disposition=FieldDisposition.WHOLE_COLUMN_REDACT,
                    category=Category.IPV4,
                )
            return FieldPolicy(column=column, disposition=FieldDisposition.PASSTHROUGH_METADATA)

        if col_lower in DEFAULT_DOB_COLUMNS:
            return FieldPolicy(
                column=column,
                disposition=FieldDisposition.WHOLE_COLUMN_GENERALISE_DOB,
                category=Category.DATE_AU,
            )

        if col_lower in DEFAULT_TOKENISE_COLUMNS:
            return FieldPolicy(
                column=column,
                disposition=FieldDisposition.WHOLE_COLUMN_TOKENISE,
                category=DEFAULT_TOKENISE_COLUMNS[col_lower],
            )

        for key, cat in DEFAULT_REDACT_COLUMNS.items():
            if col_lower == key.lower():
                return FieldPolicy(
                    column=column,
                    disposition=FieldDisposition.WHOLE_COLUMN_REDACT,
                    category=cat,
                    propagate_to_free_text=True,
                )

        if isinstance(sample_value, (int, float, bool)):
            return FieldPolicy(column=column, disposition=FieldDisposition.PASSTHROUGH_NUMERIC)

        return FieldPolicy(
            column=column,
            disposition=FieldDisposition.FREE_TEXT_HYBRID,
            question_text=self.question_prompts.get(column),
        )

    def apply_whole_column(
        self,
        policy: FieldPolicy,
        value: object,
        *,
        salt: bytes | None = None,
        style: MaskStyle = MaskStyle.LABEL,
    ) -> object:
        """Apply deterministic whole-column action without running free-text detectors."""
        if value is None:
            return None
        if policy.disposition in (
            FieldDisposition.PASSTHROUGH_METADATA,
            FieldDisposition.PASSTHROUGH_NUMERIC,
        ):
            return value
        text = str(value)
        if not text.strip():
            return text
        if policy.disposition is FieldDisposition.WHOLE_COLUMN_REMOVE:
            return ""
        if policy.disposition is FieldDisposition.WHOLE_COLUMN_GENERALISE_DOB:
            return generalise_date(text)
        cat = (policy.category or Category.REFERENCE_ID).value
        if policy.disposition is FieldDisposition.WHOLE_COLUMN_TOKENISE:
            masker = build_masker(MaskStyle.TOKEN, cat, salt=salt)
            return masker(text)
        if policy.disposition is FieldDisposition.WHOLE_COLUMN_REDACT:
            masker = build_masker(style, cat, salt=salt)
            return masker(text)
        return value


__all__ = [
    "DEFAULT_DOB_COLUMNS",
    "DEFAULT_METADATA_COLUMNS",
    "DEFAULT_REDACT_COLUMNS",
    "DEFAULT_TOKENISE_COLUMNS",
    "FieldDisposition",
    "FieldPolicy",
    "SurveySchema",
]
