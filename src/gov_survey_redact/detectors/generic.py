"""Deterministic detectors for contact details, network identifiers, locations, and dates.

Australian-first rules:
* Phone numbers validate against the ACMA Australian numbering plan (04 mobiles,
  02/03/07/08 landlines, 1300/1800 service numbers, +61 international), rejecting
  dates, version numbers, and currency figures.
* Emails support both standard syntax and character-spaced obfuscation
  (``j a n e @ e x a m p l e . c o m``).
* Postcodes validate against real Australia Post state allocations.
* Calendar dates validate real month/day bounds in Australian day-first and ISO orders.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..normalise import normalise_aligned
from ..types import Category, Finding, Span, Stage
from .base import CompiledPatternDetector, RegexDetector

# --- Email -------------------------------------------------------------------
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}\b"
)

_LOCAL_U = r"(?:[A-Za-z0-9!#$%&'*+/=?^_`{|}~.\-]\s)"
_LABEL_U = r"(?:[A-Za-z0-9\-]\s)"
_TLD_U = r"(?:[A-Za-z]\s)"
_EMAIL_SPACED_RE = re.compile(
    r"(?<![\w.])"
    rf"{_LOCAL_U}{{3,64}}"
    r"@\s?"
    rf"{_LABEL_U}{{1,63}}"
    rf"(?:\.\s?{_LABEL_U}{{1,63}}(?=\s?\.))*"
    rf"\.\s?{_TLD_U}{{1,62}}[A-Za-z]",
    re.IGNORECASE,
)


class EmailDetector(RegexDetector):
    """Email addresses, including NFKC fullwidth and character-spaced forms."""

    name = "email"
    rule_id = "CONTACT_EMAIL_V1"
    stage = Stage.DETERMINISTIC

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (EMAIL_RE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return Category.EMAIL

    def detect(self, text: str) -> Iterator[Finding]:
        aligned = normalise_aligned(text)
        for match in EMAIL_RE.finditer(aligned):
            start, end = match.span()
            yield Finding(
                span=Span(start, end),
                category=Category.EMAIL,
                detector=self.name,
                stage=self.stage,
                confidence=1.0,
                value=text[start:end],
                metadata=(("recognizer", self.rule_id),),
            )
        for match in _EMAIL_SPACED_RE.finditer(aligned):
            start, end = match.span()
            while end > start and aligned[end - 1].isspace():
                end -= 1
            yield Finding(
                span=Span(start, end),
                category=Category.EMAIL,
                detector=self.name + "_spaced",
                stage=self.stage,
                confidence=0.9,
                value=text[start:end],
                metadata=(
                    ("note", "characters separated by spaces"),
                    ("recognizer", self.rule_id + "_SPACED"),
                ),
            )


# --- Australian phone numbers ------------------------------------------------
_PHONE_CANDIDATE = re.compile(
    r"(?<![\w\d.+])"
    r"(?P<international>\+?61[\s.-]?)?"
    r"(?:"
    r"\(\d{2,4}\)"
    r"|\d{1,4}"
    r")"
    r"(?:[\s.-]?\d){5,10}"
    r"(?![\d])"
)

_AU_LANDLINE_LEADING_DIGITS = frozenset("2378")


def _is_au_nsn(body: str) -> bool:
    if len(body) != 9:
        return False
    return body[0] == "4" or body[0] in _AU_LANDLINE_LEADING_DIGITS


class PhoneDetector(RegexDetector):
    """Australian phone numbers validated against the ACMA numbering plan."""

    name = "phone"
    rule_id = "CONTACT_AU_PHONE_V1"
    category = Category.PHONE

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (_PHONE_CANDIDATE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        aligned = normalise_aligned(text)
        for match in _PHONE_CANDIDATE.finditer(aligned):
            start, end = match.span()
            while end > start and aligned[end - 1] in ".,;:":
                end -= 1
            digits = re.sub(r"\D", "", aligned[start:end])
            normalised = self._normalise(digits)
            if normalised is None:
                continue
            yield Finding(
                span=Span(start, end),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=self._confidence(normalised),
                value=text[start:end],
                metadata=(
                    ("normalised", normalised),
                    ("recognizer", self.rule_id),
                ),
            )

    @staticmethod
    def _normalise(digits: str) -> str | None:
        if digits.startswith("61"):
            body = digits[2:]
            return f"+61{body}" if _is_au_nsn(body) else None
        if re.fullmatch(r"(?:1300|1800)\d{6}", digits):
            return "0" + digits
        if not digits.startswith("0"):
            return None
        body = digits[1:]
        if body.startswith("4") and len(body) == 9:
            return "04" + body[1:]
        if len(body) == 9 and body[0] in _AU_LANDLINE_LEADING_DIGITS:
            return "0" + body
        return None

    @staticmethod
    def _confidence(normalised: str) -> float:
        if normalised.startswith(("04", "01300", "01800")):
            return 1.0
        return 0.95


# --- Network identifiers (IPv4, IPv6, MAC, URL credentials) ------------------
IPV4_RE = re.compile(
    r"(?<![\w.])"
    r"(?<!release\s)(?<!version\s)(?<!build\s)(?<!v)\b"
    r"(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
    r"(?![\w.])",
    re.IGNORECASE,
)
IPV6_RE = re.compile(
    r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,7}:\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,6}:[A-Fa-f0-9]{1,4}\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,5}(?::[A-Fa-f0-9]{1,4}){1,2}\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,4}(?::[A-Fa-f0-9]{1,4}){1,3}\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,3}(?::[A-Fa-f0-9]{1,4}){1,4}\b"
    r"|\b(?:[A-Fa-f0-9]{1,4}:){1,2}(?::[A-Fa-f0-9]{1,4}){1,5}\b"
    r"|\b[A-Fa-f0-9]{1,4}:(?::[A-Fa-f0-9]{1,4}){1,6}\b"
    r"|::(?:[A-Fa-f0-9]{1,4}:){0,6}[A-Fa-f0-9]{1,4}\b"
)
MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")


class IPv4Detector(CompiledPatternDetector):
    name = "ipv4"
    rule_id = "NET_IPV4_V1"
    regex = IPV4_RE
    category = Category.IPV4


class IPv6Detector(CompiledPatternDetector):
    name = "ipv6"
    rule_id = "NET_IPV6_V1"
    regex = IPV6_RE
    category = Category.IPV6


class MACDetector(CompiledPatternDetector):
    name = "mac"
    rule_id = "NET_MAC_V1"
    regex = MAC_RE
    category = Category.MAC


URL_CREDENTIAL_RE = re.compile(
    r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@]+(?::[^\s/@]*)?@|"
    r"\b(?:api[_-]?key|apikey|token|access[_-]?token|secret|password|passwd|pwd)"
    r"[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-./+=]{8,}",
    re.IGNORECASE,
)


class URLCredentialDetector(RegexDetector):
    name = "url_credential"
    rule_id = "NET_URL_CREDENTIAL_V1"
    category = Category.URL_CREDENTIAL

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (URL_CREDENTIAL_RE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category


# --- Calendar Dates ----------------------------------------------------------
_DAYS = (
    "mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|monday|tuesday|wednesday|thursday"
    "|friday|saturday|sunday"
)
_MONTHS = (
    "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    "|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

DATE_AU_RE = re.compile(
    r"\b(?:(?P<d1>\d{1,2})[/-](?P<m1>\d{1,2})[/-](?P<y1>\d{2,4})"
    r"|(?P<y2>\d{4})[/-](?P<m2>\d{1,2})[/-](?P<d2>\d{1,2}))\b"
)
_MONTH_NUMBER = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

DATE_NAMED_RE = re.compile(
    rf"\b(?:(?P<d1>\d{{1,2}})\s+(?P<m1>{_MONTHS})\s+(?P<y1>\d{{4}})"
    rf"|(?P<d2>\d{{1,2}})\s+(?P<m2>{_MONTHS})"
    rf"|(?P<m3>{_MONTHS})\s+(?P<d3>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y3>\d{{4}})"
    rf"|(?:{_DAYS})\s+(?P<d4>\d{{1,2}})\s+(?:of\s+)?(?P<m4>{_MONTHS})\s+(?P<y4>\d{{4}}))\b",
    re.IGNORECASE,
)

_DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


class DateDetector(RegexDetector):
    """Calendar dates in Australian (day-first) and ISO orders."""

    name = "date"
    rule_id = "TEMPORAL_DATE_V1"
    stage = Stage.DETERMINISTIC

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (DATE_AU_RE, DATE_NAMED_RE)

    def category_for(self, match: re.Match[str]) -> Category:
        parsed = self._parse(match)
        if parsed is None:
            return Category.DATE_AU
        return Category.DATE_ISO if parsed[3] == "iso" else Category.DATE_AU

    def detect(self, text: str) -> Iterator[Finding]:
        aligned = normalise_aligned(text)
        for pattern in (DATE_AU_RE, DATE_NAMED_RE):
            for match in pattern.finditer(aligned):
                parsed = self._parse(match)
                if parsed is None:
                    continue
                day, month, year, order = parsed
                start, end = match.span()
                iso = f"{year:04d}-{month:02d}-{day:02d}" if year else f"{month:02d}-{day:02d}"
                yield Finding(
                    span=Span(start, end),
                    category=Category.DATE_ISO if order == "iso" else Category.DATE_AU,
                    detector=self.name,
                    stage=self.stage,
                    confidence=1.0 if year else 0.8,
                    value=text[start:end],
                    metadata=(
                        ("iso", iso),
                        ("order", order),
                        ("recognizer", self.rule_id),
                    ),
                )

    @staticmethod
    def _parse(match: re.Match[str]) -> tuple[int, int, int | None, str] | None:
        g = {k: (v.lower() if isinstance(v, str) else v) for k, v in match.groupdict().items()}
        if g.get("d1") and g.get("y1") and g.get("m1") not in _MONTH_NUMBER:
            day, month, year = int(g["d1"]), int(g["m1"]), int(g["y1"])
            order = "day_first"
        elif g.get("d1") and g.get("m1") in _MONTH_NUMBER and g.get("y1"):
            day, month, year = int(g["d1"]), _MONTH_NUMBER[g["m1"]], int(g["y1"])
            order = "day_first"
        elif g.get("y2") and g.get("m2") and g.get("d2"):
            day, month, year = int(g["d2"]), int(g["m2"]), int(g["y2"])
            order = "iso"
        elif g.get("d3") and g.get("m3") and g.get("y3"):
            day, month, year = int(g["d3"]), _MONTH_NUMBER[g["m3"]], int(g["y3"])
            order = "month_first"
        elif g.get("d4") and g.get("m4") and g.get("y4"):
            day, month, year = int(g["d4"]), _MONTH_NUMBER[g["m4"]], int(g["y4"])
            order = "day_first"
        elif g.get("d2") and g.get("m2") in _MONTH_NUMBER:
            day, month, year = int(g["d2"]), _MONTH_NUMBER[g["m2"]], None
            order = "day_first"
        else:
            return None

        if not 1 <= month <= 12:
            return None
        if not 1 <= day <= _DAYS_IN_MONTH[month - 1]:
            return None
        if year is not None:
            if year < 100:
                year += 2000 if year <= 69 else 1900
            if not 1900 <= year <= 2099:
                return None
        return day, month, year, order


# --- Postcodes and Street Addresses ------------------------------------------
_STATE_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "NSW": ((1000, 1999), (2000, 2599), (2620, 2899), (2900, 2999)),
    "ACT": ((2600, 2618), (2620, 2631), (2900, 2920)),
    "VIC": ((3000, 3999),),
    "QLD": ((4000, 4499), (4500, 4911), (4920, 4929), (4940, 4950)),
    "SA": ((5000, 5999),),
    "WA": ((6000, 6799), (6800, 6999)),
    "NT": ((800, 899),),
    "TAS": ((7000, 7999),),
    "QLD_OTHER": ((4920, 4999),),
}


def valid_au_postcode(code: str) -> bool:
    if not re.fullmatch(r"\d{4}", code):
        return False
    value = int(code)
    if value == 0:
        return False
    return any(
        lo <= value <= hi
        for ranges in _STATE_RANGES.values()
        for lo, hi in ranges
    )


STATE_CODES = ("NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT")

_STATE_POSTCODE_RE = re.compile(
    rf"\b(?P<state>{'|'.join(STATE_CODES)})\s+(?P<code>\d{{4}})\b"
    rf"|\b(?P<code2>\d{{4}})\s+(?P<state2>{'|'.join(STATE_CODES)})\b"
)

POSTCODE_CONTEXT_RE = re.compile(
    r"(?i)\b(?:post\s*code|postcode|zip|suburb|suburb/town|city)\b"
)

_STREET_SUFFIX = (
    "street|st|road|rd|avenue|ave|parade|pde|highway|hwy|place|pl|square|sq|court|ct"
    "|crescent|cres|close|cl|drive|dr|lane|ln|terrace|tce|boulevard|bvd|way|walk|circle|cir"
    "|esplanade|esp|ter|view|vd|rise|walkway|wy|loop|grove|gr|bay|cove|cv|mews|parkway|pkwy"
    "|triangle|plaza|quay|qy|alley|aly|promenade|prom|bend|glade|haven|vdge"
)
_UNIT_RE = r"(?:unit|u|flat|apt|apartment|suite|ste)\s*\d+[a-z]?(?:/\d+[a-z]?)?"
_STREET_RE = re.compile(
    rf"\b\d{{1,5}}\s+"
    rf"(?:(?:{_UNIT_RE})\s+)?"
    rf"(?:[A-Z][A-Za-z'\u2019\-]+\s+){{1,4}}"
    rf"(?:{_STREET_SUFFIX})\b",
    re.IGNORECASE,
)
_PO_BOX_RE = re.compile(r"\bPO\s+Box\s+\d+\b", re.IGNORECASE)


class StatePostcodeDetector(RegexDetector):
    """State abbreviation + 4-digit Australia Post postcode."""

    name = "state_postcode"
    rule_id = "LOC_AU_STATE_POSTCODE_V1"
    category = Category.POSTCODE

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (_STATE_POSTCODE_RE,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        for match in _STATE_POSTCODE_RE.finditer(text):
            code = match.group("code") or match.group("code2")
            state = match.group("state") or match.group("state2")
            if not code or not valid_au_postcode(code):
                continue
            yield Finding(
                span=Span(*match.span()),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=1.0,
                value=match.group(),
                metadata=(
                    ("postcode", code),
                    ("state", state),
                    ("recognizer", self.rule_id),
                ),
            )


class PostcodeDetector(RegexDetector):
    """Bare 4-digit postcode when accompanied by postal context."""

    name = "postcode"
    rule_id = "LOC_AU_POSTCODE_V1"
    category = Category.POSTCODE

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (re.compile(r"\b\d{4}\b"),)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def detect(self, text: str) -> Iterator[Finding]:
        claimed = [f.span for f in StatePostcodeDetector().detect(text)]
        for match in re.finditer(r"\b\d{4}\b", text):
            start, end = match.span()
            if any(s.start < end and start < s.end for s in claimed):
                continue
            code = match.group()
            if not valid_au_postcode(code):
                continue
            window = text[max(0, start - 30) : end + 30]
            if not POSTCODE_CONTEXT_RE.search(window):
                continue
            yield Finding(
                span=Span(start, end),
                category=self.category,
                detector=self.name,
                stage=self.stage,
                confidence=0.75,
                value=code,
                metadata=(
                    ("context", "label"),
                    ("recognizer", self.rule_id),
                ),
            )


class StreetAddressDetector(RegexDetector):
    """Street addresses, unit/street pairs, and PO boxes."""

    name = "street_address"
    rule_id = "LOC_AU_STREET_V1"
    category = Category.STREET_ADDRESS

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (_STREET_RE, _PO_BOX_RE)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category

    def confidence_for(self, match: re.Match[str]) -> float:
        return 1.0 if _PO_BOX_RE.fullmatch(match.group()) else 0.85
