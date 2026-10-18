"""Australian Government and financial identifier detectors.

Implements a Presidio-style two-tier recogniser design:
1. Narrow candidate regexes (with optional label context and OCR character
   substitution tolerance such as ``O↔0``, ``I↔1``, ``S↔5``).
2. Checksum validation (TFN mod-11, ABN mod-89, ACN mod-10, Medicare mod-10,
   Credit Card Luhn, IBAN mod-97).

Crucially for scanned/handwritten government surveys:
* In standard digital mode, unlabelled checksum failures are rejected so
  arbitrary 9–11 digit numbers do not trigger false positives.
* When an explicit identifier label is present (e.g. ``"TFN: 123 456 783"``) or
  when ``ocr_mode=True`` (or OCR character substitutions were repaired), a
  checksum failure is **retained at reduced confidence** (``0.55``) with
  ``checksum="failed_ocr_candidate"`` so the record is routed to the human
  review queue rather than silently leaked!
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from .. import checksums as ck
from ..normalise import normalise_aligned
from ..types import Category, Finding, Span
from .base import RegexDetector

# OCR digit repair table used only when an identifier label precedes a candidate
# or when ocr_mode is active.
_OCR_DIGIT_MAP = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "S": "5",
        "B": "8",
    }
)


def repair_ocr_digits(raw: str) -> tuple[str, bool]:
    """Repair common OCR letter-for-digit substitutions and report if changed."""
    repaired = raw.translate(_OCR_DIGIT_MAP)
    return repaired, repaired != raw


class ChecksumDetector(RegexDetector):
    """Scans with a candidate regex and validates with a checksum."""

    candidate_pattern: str
    category: Category
    flags = 0
    require_checksum = True
    label_pattern: re.Pattern[str] | None = None

    def __init__(self, *, ocr_mode: bool = False) -> None:
        self.ocr_mode = ocr_mode

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (re.compile(self.candidate_pattern, self.flags),)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def _has_nearby_label(self, text: str, start: int) -> bool:
        if self.label_pattern is None:
            return False
        window = text[max(0, start - 32) : start]
        return bool(self.label_pattern.search(window))

    def detect(self, text: str) -> Iterator[Finding]:
        aligned = normalise_aligned(text)
        for match in self.patterns()[0].finditer(aligned):
            start, end = match.span()
            raw = aligned[start:end]
            has_label = self._has_nearby_label(aligned, start)
            repaired, was_ocr_repaired = (
                repair_ocr_digits(raw) if (self.ocr_mode or has_label) else (raw, False)
            )
            # Only allow OCR repair if the candidate actually contains mostly digits
            digit_count = sum(c.isdigit() for c in raw)
            if was_ocr_repaired and digit_count < 6:
                continue

            valid = self.validator(repaired if was_ocr_repaired else raw)
            if not valid:
                # Per architectural requirement: pattern match with nearby label
                # or in OCR mode survives checksum failure as an uncertain finding
                # so the confidence gate routes it to the review queue.
                if self.require_checksum and not (self.ocr_mode and has_label):
                    continue
                meta = self.metadata_for(match)
                meta["checksum"] = "failed_ocr_candidate" if (self.ocr_mode or has_label) else "failed"
                if self.rule_id:
                    meta["recognizer"] = self.rule_id
                yield Finding(
                    span=Span(start, end),
                    category=self.category,
                    detector=self.name,
                    stage=self.stage,
                    confidence=0.58 if (self.ocr_mode or has_label) else 0.4,
                    value=text[start:end],
                    metadata=tuple(meta.items()),
                )
                continue

            meta = self.metadata_for(match)
            if was_ocr_repaired:
                meta["ocr_repaired"] = True
                meta["normalised"] = ck.digits_only(repaired)
            if self.rule_id:
                meta["recognizer"] = self.rule_id
            base_conf = self.confidence_for(match)
            confidence = 0.72 if was_ocr_repaired else base_conf
            yield Finding(
                span=Span(start, end),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=confidence,
                value=text[start:end],
                metadata=tuple(meta.items()),
            )

    def validator(self, raw: str) -> bool:
        raise NotImplementedError


class TFNDetector(ChecksumDetector):
    """Australian Tax File Number (9 digits, ATO mod-11)."""

    name = "tfn"
    rule_id = "GOV_ID_AU_TFN_V1"
    category = Category.TFN
    candidate_pattern = r"\b[\dOIl]{3}[- ]?[\dOIl]{3}[- ]?[\dOIl]{3}\b"
    label_pattern = re.compile(r"(?i)\b(?:tfn|tax\s*file\s*(?:no|number|#)?)\b\s*[:=-]?\s*$")

    def validator(self, raw: str) -> bool:
        return ck.is_tfn(raw)

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        return {"normalised": ck.digits_only(match.group()), "recognizer": self.rule_id}


class ABNDetector(ChecksumDetector):
    """Australian Business Number (11 digits, ABR mod-89)."""

    name = "abn"
    rule_id = "GOV_ID_AU_ABN_V1"
    category = Category.ABN
    candidate_pattern = r"\b[\dOIl]{2}[- ]?[\dOIl]{3}[- ]?[\dOIl]{3}[- ]?[\dOIl]{3}\b"
    label_pattern = re.compile(r"(?i)\b(?:abn|australian\s*business\s*(?:no|number|#)?)\b\s*[:=-]?\s*$")

    def validator(self, raw: str) -> bool:
        return ck.is_abn(raw)

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        return {"normalised": ck.digits_only(match.group()), "recognizer": self.rule_id}


class ACNDetector(ChecksumDetector):
    """Australian Company Number (9 digits, ASIC mod-10)."""

    name = "acn"
    rule_id = "GOV_ID_AU_ACN_V1"
    category = Category.ACN
    candidate_pattern = r"\b[\dOIl]{3}[- ]?[\dOIl]{3}[- ]?[\dOIl]{3}\b"
    label_pattern = re.compile(r"(?i)\b(?:acn|australian\s*company\s*(?:no|number|#)?)\b\s*[:=-]?\s*$")

    def validator(self, raw: str) -> bool:
        return ck.is_acn(raw)

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        return {"normalised": ck.digits_only(match.group()), "recognizer": self.rule_id}


class MedicareDetector(ChecksumDetector):
    """Australian Medicare card number (10 digits, weighted check digit)."""

    name = "medicare"
    rule_id = "GOV_ID_AU_MEDICARE_V1"
    category = Category.MEDICARE
    candidate_pattern = r"(?<![\d\-])[1-9](?:[\s-]?[\dOIl]){9}(?![\d\-])"
    label_pattern = re.compile(r"(?i)\b(?:medicare(?:\s*(?:no|number|card|#))?)\b\s*[:=-]?\s*$")

    def validator(self, raw: str) -> bool:
        return ck.is_medicare(raw)

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        return {"normalised": ck.digits_only(match.group()), "recognizer": self.rule_id}


class MedicareEnrolmentDetector(ChecksumDetector):
    """Medicare enrolment reference (shape-validated, reduced confidence)."""

    name = "medicare_enrolment"
    rule_id = "GOV_ID_AU_MEDICARE_ENROLMENT_V1"
    category = Category.MEDICARE_ENROLMENT
    candidate_pattern = r"\b[1-9]\s\d{4}\s\d{4}\s\d\b"
    require_checksum = False
    confidence = 0.6

    def validator(self, raw: str) -> bool:
        return ck.is_medicare_enrolment(raw)

    def confidence_for(self, match: re.Match[str]) -> float:
        return self.confidence if self.validator(match.group()) else 0.4


class CentrelinkCRNDetector(RegexDetector):
    """Services Australia / Centrelink Customer Reference Number (CRN)."""

    name = "centrelink_crn"
    rule_id = "GOV_ID_AU_CRN_V1"
    category = Category.CENTRELINK_CRN

    _CANDIDATE = re.compile(
        r"\b(?P<crn>[1-9]\d{2}[-\s]?\d{3}[-\s]?\d{3}[-\s]?[ABCHJKLSTVXZ])\b",
        re.IGNORECASE,
    )
    _LABEL = re.compile(r"(?i)\b(?:crn|centrelink|customer\s*reference(?:\s*number|\s*no)?)\b")

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (self._CANDIDATE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        aligned = normalise_aligned(text)
        for match in self._CANDIDATE.finditer(aligned):
            raw = match.group("crn")
            if not ck.is_centrelink_crn(raw):
                continue
            start, end = match.span("crn")
            window = aligned[max(0, start - 40) : min(len(aligned), end + 20)]
            has_label = bool(self._LABEL.search(window))
            # Require either a nearby CRN/Centrelink label or canonical 3-3-3A spacing
            if not has_label and " " not in raw and "-" not in raw:
                continue
            yield Finding(
                span=Span(start, end),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=0.95 if has_label else 0.80,
                value=text[start:end],
                metadata=(
                    ("normalised", re.sub(r"[\s-]", "", raw).upper()),
                    ("recognizer", self.rule_id),
                ),
            )


class CreditCardDetector(ChecksumDetector):
    """Payment card numbers (Visa, Mastercard, Amex, eftpos, UnionPay + Luhn)."""

    name = "credit_card"
    rule_id = "FIN_AU_CARD_V1"
    category = Category.CREDIT_CARD
    candidate_pattern = (
        r"(?<![\d\-])(?:"
        r"4\d{3}"
        r"|5[1-5]\d{2}"
        r"|6(?:011|5\d{2})"
        r"|3[47]\d{2}"
        r"|2(?:2[2-9]\d|[3-6]\d{2}|7[01]\d|720)"
        r"|30[0-9]\d|36\d{2}|38\d{2}"
        r"|56(?:12|02)|63(?:04|59)"
        r"|979\d"
        r")"
        r"(?:[\s-]?\d){8,16}"
        r"(?![\d\-])"
    )
    require_checksum = False

    def validator(self, raw: str) -> bool:
        return ck.is_credit_card(raw)

    def confidence_for(self, match: re.Match[str]) -> float:
        digits = ck.digits_only(match.group())
        brand = ck.card_brand(digits)
        if brand is None:
            return 0.3
        return 1.0 if ck.luhn_check(digits) else 0.5

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        digits = ck.digits_only(match.group())
        return {
            "brand": ck.card_brand(digits) or "unknown",
            "length": len(digits),
            "luhn": ck.luhn_check(digits),
            "recognizer": self.rule_id,
        }


class PassportDetector(RegexDetector):
    """Australian passport number (1 or 2 letters followed by 7 digits)."""

    name = "passport"
    rule_id = "GOV_ID_AU_PASSPORT_V1"
    category = Category.PASSPORT
    pattern = r"\b(?:(?![EIO])[A-Z]\d{7}|P(?![OI])[A-Z]\d{7})\b"

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (re.compile(self.pattern),)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def confidence_for(self, match: re.Match[str]) -> float:
        return 1.0 if ck.is_passport(match.group()) else 0.5


class BSBDetector(RegexDetector):
    """Bank State Branch (6 digits), gated on routing label or following account number."""

    name = "bsb"
    rule_id = "FIN_AU_BSB_V1"
    category = Category.BANK_BSB

    #: Only the immediately preceding characters can supply a routing label; a
    #: "BSB" mention earlier in the sentence must not capture an unrelated
    #: six-digit number elsewhere in the clause.
    _LABEL_WINDOW = 18

    _CONTEXT = re.compile(
        r"(?i)\b(?:bsb|routing|rtn|bank\s*state\s*branch|sb-?no|branch)\b"
        r"|\b(?:acc(?:ount)?|a/c)\s*(?:no\.?|number|#)?"
    )
    _ACCOUNT_AFTER = re.compile(r"^\s?[- ]?\s?\d{5,10}")

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (ck._BSB_RE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        aligned = normalise_aligned(text)
        for match in ck._BSB_RE.finditer(aligned):
            raw = match.group()
            if not ck.is_bsb(raw):
                continue
            start, end = match.span()
            window_before = aligned[max(0, start - self._LABEL_WINDOW) : start]
            window_after = aligned[end : end + 20]
            has_label = bool(self._CONTEXT.search(window_before))
            has_account = bool(self._ACCOUNT_AFTER.match(window_after))
            if not (has_label or has_account):
                continue
            confidence = 1.0 if has_label else 0.75
            yield Finding(
                span=Span(start, end),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=confidence,
                value=text[start:end],
                metadata=(
                    ("context", "label" if has_label else "account"),
                    ("recognizer", self.rule_id),
                ),
            )


class DriversLicenceDetector(RegexDetector):
    """Australian state/territory driver licence numbers (requires nearby label)."""

    name = "drivers_licence"
    rule_id = "GOV_ID_AU_DL_V1"
    category = Category.DRIVERS_LICENCE

    _LABEL = re.compile(
        r"(?i)\b(?:driver'?s?\s*licen[cs]e|drivers\s*licence|licence\s*(?:no|number|#)|"
        r"dl|lic(?:ence)?)\b\.?\s*(?:no\.?|number|#)?\s*:?\s*"
    )
    _VALUE = re.compile(r"(?i)\b([A-Z0-9]{5,14})\b")
    _SHAPES: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"^\d{8}$"), "nsw"),
        (re.compile(r"^[A-Z]\d{6}[A-Z]$"), "qld"),
        (re.compile(r"^[A-Z]\d{6}$"), "wa"),
        (re.compile(r"^\d{9}$"), "sa"),
        (re.compile(r"^[A-Z]\d{7}$"), "vic"),
        (re.compile(r"^[A-Z]{2}\d{5}$"), "tas"),
        (re.compile(r"^\d{6}[A-Z]$"), "act"),
        (re.compile(r"^\d{7}$"), "nt"),
    )

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (self._LABEL,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        aligned = normalise_aligned(text)
        for label in self._LABEL.finditer(aligned):
            tail_start = label.end()
            value_match = self._VALUE.match(aligned, tail_start)
            if value_match is None:
                continue
            raw = value_match.group(1)
            if raw.lower() in {"and", "the", "for", "exp", "expiry", "class", "was", "is"}:
                continue
            state = next((s for pattern, s in self._SHAPES if pattern.match(raw)), None)
            start, end = value_match.span(1)
            yield Finding(
                span=Span(start, end),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=1.0 if state else 0.7,
                value=text[start:end],
                metadata=(
                    ("state_format", state or "unknown"),
                    ("recognizer", self.rule_id),
                ),
            )


class IBANDetector(RegexDetector):
    """International Bank Account Number with ISO 7064 mod-97 check."""

    name = "iban"
    rule_id = "FIN_IBAN_V1"
    category = Category.IBAN

    _CANDIDATE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (self._CANDIDATE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def confidence_for(self, match: re.Match[str]) -> float:
        return 1.0 if ck.iban_check(match.group()) else 0.4


class EmployeeReferenceDetector(RegexDetector):
    """APS / Government employee numbers (AGS number, payroll ID, staff ID)."""

    name = "reference_id"
    rule_id = "GOV_ID_AU_EMPLOYEE_REF_V1"
    category = Category.REFERENCE_ID

    _PATTERN = re.compile(
        r"(?i)\b(?:ags\s*(?:no\.?|number|#)?|employee\s*(?:id|no\.?|number|#)|"
        r"payroll\s*(?:id|no\.?|number|#)|staff\s*(?:id|no\.?|number|#)|"
        r"personnel\s*(?:id|no\.?|number|#))\s*[:=-]?\s*"
        r"(?P<span>[A-Z0-9]{2,4}[-\s]?\d{4,9}|\d{6,10})\b"
    )

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (self._PATTERN,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category


class ClaimReferenceDetector(RegexDetector):
    """Workers' compensation / insurance claim or incident references.

    Covers labelled references ("claim number WC-2026-448112", "insurer case
    reference 44811290") and the common prefixed shapes (``WC-``, ``CLM-``,
    ``INC-``) used by Australian compensation schemes.
    """

    name = "claim_reference"
    rule_id = "GOV_ID_AU_CLAIM_REF_V1"
    category = Category.REFERENCE_ID

    _SHAPE = r"[A-Z]{2,4}[- ]?\d{4,8}(?:[- ]\d{2,8})*"
    _LABEL_RE = re.compile(
        rf"(?i)\b(?:claim|incident|case|insurer|employer)\s*"
        rf"(?:ref(?:erence)?|no\.?|number|id|#)\s*[:=-]?\s*"
        rf"(?P<span>{_SHAPE}|\d{{6,12}})\b"
    )
    _PREFIX_RE = re.compile(r"\b(?P<span>(?:WC|CLM|INC|REHAB|COMP)[- ]?\d{4}[- ]?\d{4,8})\b")

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (self._LABEL_RE, self._PREFIX_RE)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def confidence_for(self, match: re.Match[str]) -> float:
        return 1.0 if match.re is self._PREFIX_RE else 0.95
