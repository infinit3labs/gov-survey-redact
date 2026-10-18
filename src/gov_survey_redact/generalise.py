"""Generalisation of quasi-identifiers to preserve survey utility while reducing re-identification precision.

Widens exact values into analytical bands:
* Exact dates (``"12/03/1985"``) -> year only (``"1985"``)
* Exact ages (``"45"``) -> 10-year age band (``"45-54"``)
* Tenure (``"18 years of service"``) -> service band (``"10-19 years of service"``)
* Team size (``"three"``) -> headcount band (``"2-5 people"``)
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .types import Category

_AGE_BANDS: tuple[tuple[int, int, str], ...] = (
    (0, 17, "under 18"),
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 64, "55-64"),
    (65, 74, "65-74"),
    (75, 120, "75+"),
)

_TENURE_BANDS: tuple[tuple[int, str], ...] = (
    (2, "under 2 years"),
    (5, "2-4 years"),
    (10, "5-9 years"),
    (20, "10-19 years"),
    (30, "20-29 years"),
    (40, "30-39 years"),
)

_TEAM_BANDS: tuple[tuple[int, str], ...] = (
    (1, "1 person"),
    (5, "2-5 people"),
    (10, "6-10 people"),
    (20, "11-20 people"),
    (50, "21-50 people"),
    (10**9, "50+ people"),
)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BAND_LABEL_RE = re.compile(
    r"^(?:under\s+\d+|\d+\+|\d+\s*-\s*\d+|\d+\+?\s+years?|\d+\+?\s+people?)$"
)


def is_banded(value: str) -> bool:
    """True when ``value`` is already a band emitted by this module."""
    first = value.strip().split(",", 1)[0].split(" ", 1)[0] if value.strip() else ""
    return bool(_BAND_LABEL_RE.fullmatch(first.strip()))


_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
}
_WORD_NUMBER_RE = re.compile(r"\b(" + "|".join(_WORD_NUMBERS) + r")\b", re.IGNORECASE)


def parse_number(value: str) -> float | None:
    """First number in ``value`` as digits or an English number word."""
    match = _NUMBER_RE.search(value)
    if match is not None:
        try:
            return float(match.group())
        except ValueError:
            return None
    match = _WORD_NUMBER_RE.search(value)
    if match is not None:
        return float(_WORD_NUMBERS[match.group(1).lower()])
    return None


def _is_bare_number(value: str) -> bool:
    tokens = value.strip().split()
    if not tokens:
        return False
    return parse_number(tokens[0]) is not None


def generalise_age(value: str) -> str:
    """Widen an age into a 10-year age band."""
    if is_banded(value) or not _is_bare_number(value):
        return value
    number = parse_number(value)
    if number is None:
        return value
    age = int(number)
    for low, high, label in _AGE_BANDS:
        if low <= age <= high:
            return label
    return value


_TRAILING_UNIT_RE = re.compile(
    r"^\s*\+?\s*(?:years?|yrs?|months?|people|staff|workers|employees|members)\b",
    re.IGNORECASE,
)


def _replace_number(value: str, number: float, band_label: str) -> str:
    match = _NUMBER_RE.search(value)
    if match is not None:
        trailing = value[match.end() :]
    else:
        word = _WORD_NUMBER_RE.search(value)
        if word is None:
            return value
        trailing = value[word.end() :]
    if _TRAILING_UNIT_RE.match(trailing):
        trailing = ""
    return f"{band_label}{trailing}"


def generalise_tenure(value: str) -> str:
    """Widen length of service into a service band."""
    if is_banded(value) or not _is_bare_number(value):
        return value
    number = parse_number(value)
    if number is None:
        return value
    if 1900 <= number <= 2099 and re.fullmatch(r"\s*(?:19|20)\d{2}\s*", value):
        return value.strip()
    years = int(number)
    band = next((label for ceiling, label in _TENURE_BANDS if years < ceiling), "40+ years")
    return _replace_number(value, years, band)


def generalise_team_size(value: str) -> str:
    """Widen a team headcount into a size band."""
    if is_banded(value) or not _is_bare_number(value):
        return value
    number = parse_number(value)
    if number is None:
        return value
    count = int(number)
    band = next((label for ceiling, label in _TEAM_BANDS if count <= ceiling), "50+ people")
    return _replace_number(value, count, band)


def generalise_date(value: str) -> str:
    """Extract only the 4-digit year from a calendar date."""
    years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", value)
    if years:
        return years[0]
    return value


def generalise_year(value: str) -> str:
    return generalise_date(value)


GENERALISERS: dict[Category, Callable[[str], str]] = {
    Category.AGE: generalise_age,
    Category.TENURE: generalise_tenure,
    Category.TEAM_SIZE: generalise_team_size,
    Category.DATE_AU: generalise_date,
    Category.DATE_ISO: generalise_date,
    Category.DATE: generalise_date,
}


def generalise(category: Category, value: str) -> str:
    """Widen ``value`` for ``category`` or return it unchanged."""
    handler = GENERALISERS.get(category)
    return handler(value) if handler is not None else value


__all__ = [
    "GENERALISERS",
    "generalise",
    "generalise_age",
    "generalise_date",
    "generalise_team_size",
    "generalise_tenure",
    "generalise_year",
    "is_banded",
    "parse_number",
]
