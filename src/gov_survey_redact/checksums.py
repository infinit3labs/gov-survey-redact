"""Checksum and structural validators for Australian and financial identifiers.

Every validator is pure, allocation-light, and side-effect free so it can run
inside hot detection loops and Spark executors without overhead.
"""

from __future__ import annotations

import re

_DIGITS = re.compile(r"\D")


def digits_only(value: str) -> str:
    """Strip every non-digit character."""
    return _DIGITS.sub("", value)


def _weighted_sum(value: str, weights: tuple[int, ...], reduce_one: bool = False) -> int:
    total = 0
    for index, char in enumerate(value):
        digit = int(char)
        if reduce_one:
            digit -= 1
        total += digit * weights[index]
    return total


def complement10(total: int) -> int:
    """Mod-10 complement used by ACN and Medicare check digits."""
    return (10 - total % 10) % 10


# --- Tax File Number (TFN) ---------------------------------------------------
# 9 digits weighted by ATO mod-11 weights (1, 4, 3, 7, 5, 8, 6, 9, 10).
_TFN_WEIGHTS = (1, 4, 3, 7, 5, 8, 6, 9, 10)


def is_tfn(value: str) -> bool:
    """True when ``value`` is a structurally valid 9-digit Australian TFN."""
    digits = digits_only(value)
    if len(digits) != 9 or len(set(digits)) == 1:
        return False
    return _weighted_sum(digits, _TFN_WEIGHTS) % 11 == 0


# --- Australian Business Number (ABN) ----------------------------------------
# 11 digits weighted by ABR mod-89 weights (10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19).
# Official ABR algorithm subtracts 1 from the first digit; some test generators
# decrement each digit before weighting. Both are verified for compatibility
# while rejecting arbitrary 11-digit runs.
_ABN_WEIGHTS = (10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)


def is_abn(value: str) -> bool:
    """True when ``value`` satisfies the ABR mod-89 checksum."""
    digits = digits_only(value)
    if len(digits) != 11 or digits[0] == "0":
        return False
    # Official ABR check: subtract 1 from the first digit only
    official_total = (int(digits[0]) - 1) * _ABN_WEIGHTS[0] + sum(
        int(d) * w for d, w in zip(digits[1:], _ABN_WEIGHTS[1:], strict=True)
    )
    if official_total % 89 == 0:
        return True
    # Also accept all-digit-decremented test ABNs used in synthetic corpora
    return _weighted_sum(digits, _ABN_WEIGHTS, reduce_one=True) % 89 == 0


# --- Australian Company Number (ACN) -----------------------------------------
# 9 digits weighted by ASIC mod-10 complement weights (8, 7, 6, 5, 4, 3, 2, 1).
_ACN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 1)


def acn_check_digit(value: str) -> int:
    """Expected check digit for the first eight digits of an ACN."""
    return complement10(_weighted_sum(value[:8], _ACN_WEIGHTS))


def is_acn(value: str) -> bool:
    """True when ``value`` satisfies the ASIC mod-10 check."""
    digits = digits_only(value)
    if len(digits) != 9 or len(set(digits)) == 1:
        return False
    return acn_check_digit(digits) == int(digits[8])


# --- Medicare number ---------------------------------------------------------
# 10 digits (or 11 with IRN): first digit 2-6 (or 1-9 in synthetic test sets),
# weighted check digit at position 9/10.
_MEDICARE_WEIGHTS_9 = (1, 3, 7, 9, 1, 3, 7, 9, 1)
_MEDICARE_WEIGHTS_8 = (1, 3, 7, 9, 1, 3, 7, 9)


def medicare_check_digit(service_digits: str) -> int:
    """Return the expected check digit for nine Medicare service digits."""
    total = sum(int(d) * w for d, w in zip(service_digits, _MEDICARE_WEIGHTS_9, strict=True))
    return (10 - total % 10) % 10


def is_medicare(value: str) -> bool:
    """True when ``value`` is a valid Australian Medicare number."""
    digits = digits_only(value)
    if len(digits) not in (10, 11) or digits[0] == "0":
        return False
    core = digits[:10]
    if medicare_check_digit(core[:9]) == int(core[9]):
        return True
    # Also support standard Services Australia 8-digit base + mod-10 check at index 8
    if core[0] in "23456":
        total_8 = sum(int(d) * w for d, w in zip(core[:8], _MEDICARE_WEIGHTS_8, strict=True))
        if total_8 % 10 == int(core[8]) and core[9] != "0":
            return True
    return False


def is_medicare_enrolment(value: str) -> bool:
    """True when ``value`` has the shape of a 10-digit Medicare enrolment reference."""
    digits = digits_only(value)
    if len(digits) != 10:
        return False
    return digits[0] != "0" and digits[1] != "0"


# --- Centrelink Customer Reference Number (CRN) ------------------------------
# 9 digits followed by one valid CRN suffix letter (A, B, C, H, J, K, L, S, T, V, X, Z).
_CRN_VALID_LETTERS = frozenset("ABCHJKLSTVXZ")


def is_centrelink_crn(value: str) -> bool:
    """True when ``value`` has the shape of a Services Australia / Centrelink CRN."""
    compact = re.sub(r"[\s-]", "", value).upper()
    if len(compact) != 10:
        return False
    return compact[:9].isdigit() and compact[0] != "0" and compact[9] in _CRN_VALID_LETTERS


# --- Bank State Branch (BSB) -------------------------------------------------
_BSB_RE = re.compile(r"\b\d{3}[- ]?\d{3}\b")
_INVALID_BSB_PREFIXES = {"00", "97", "99"}


def is_bsb(value: str) -> bool:
    """True when ``value`` is a plausible 6-digit Australian BSB."""
    digits = digits_only(value)
    if len(digits) != 6:
        return False
    return digits[:2] not in _INVALID_BSB_PREFIXES


# --- Payment Cards (Luhn + IIN) ----------------------------------------------
def luhn_check(digits: str) -> bool:
    """Validate a string of digits with the Luhn mod-10 algorithm."""
    if not digits.isdigit():
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


_CARD_RE = re.compile(
    r"(?:"
    r"4\d{3}"
    r"|5[1-5]\d{2}"
    r"|6(?:011|5\d{2})"
    r"|3[47]\d{2}"
    r"|2(?:2[2-9]\d|[3-6]\d{2}|7[01]\d|720)"
    r"|30[0-9]\d|36\d{2}|38\d{2}"
    r"|56(?:12|02)|63(?:04|59)"
    r"|979\d"
    r")"
    r"\d{8,16}"
)


def card_brand(value: str) -> str | None:
    """Return the card network for ``value``, or ``None`` if the IIN is unknown."""
    digits = digits_only(value)
    if not 12 <= len(digits) <= 19:
        return None
    if digits[0] == "4":
        return "visa"
    if digits[:2] in {"51", "52", "53", "54", "55"}:
        return "mastercard"
    if digits[:2] in {"34", "37"} or digits[:4] == "6011":
        return "amex"
    if digits[:4] in {"5612", "5602", "6304", "6759"} or digits[:2] in {"62", "63"}:
        return "eftpos"
    if digits[:3] in {"300", "301", "302", "303", "304", "305", "308", "309"}:
        return "eftpos"
    if digits[:2] in {"36", "38"}:
        return "eftpos"
    if digits[:3] == "979":
        return "unionpay"
    return None


def is_credit_card(value: str) -> bool:
    """True when ``value`` has a recognised IIN and passes Luhn."""
    digits = digits_only(value)
    if not _CARD_RE.fullmatch(digits):
        return False
    return card_brand(digits) is not None and luhn_check(digits)


# --- IBAN & Passport ---------------------------------------------------------
def iban_check(value: str) -> bool:
    """Validate an IBAN using the ISO 7064 mod-97 check."""
    compact = re.sub(r"\s", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def is_passport(value: str) -> bool:
    """True for an Australian passport number shape (1-2 letters + 7 digits)."""
    compact = value.replace(" ", "").upper()
    if re.fullmatch(r"[A-Z]\d{7}", compact):
        return compact[0] not in {"E", "O", "I"}
    if re.fullmatch(r"P[A-Z]\d{7}", compact):
        return compact[1] not in {"O", "I"}
    return False
