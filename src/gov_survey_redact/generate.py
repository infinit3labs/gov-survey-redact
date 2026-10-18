"""Stratified synthetic government-survey generator with ground-truth character spans.

Implements all 8 evaluation strata from the hybrid benchmark specification:
1. ``structured`` (30%): Name, contact fields, DOB, respondent ID, household roster
2. ``short_open_text`` (25%): Names, phone/email, addresses, employers, facilities
3. ``difficult_contextual`` (15%): Event dates vs DOBs, public bodies, common-word names
4. ``long_narrative`` (10%): Multi-paragraph 512+ word responses across window boundaries
5. ``household_grid`` (5%): Multiple household members + cross-field propagation into prose
6. ``ocr_form`` (5%): OCR character substitutions (``O↔0``, ``I↔1``), low OCR confidence
7. ``handwriting_scan`` (5%): Simulated handwritten scans with OCR confidence variation
8. ``adversarial`` (5%): Typos, fullwidth Unicode, leetspeak, spaced emails/phones

Supports template- and household-separated train/test splits so synthetic
identities never leak across splits.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .checksums import acn_check_digit, is_tfn, medicare_check_digit

STRATA_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("structured", 0.30),
    ("short_open_text", 0.25),
    ("difficult_contextual", 0.15),
    ("long_narrative", 0.10),
    ("household_grid", 0.05),
    ("ocr_form", 0.05),
    ("handwriting_scan", 0.05),
    ("adversarial", 0.05),
)

AGENCY_UNITS = [
    "Migration directorate",
    "Home Affairs branch",
    "Services division",
    "the digital services branch",
    "the integrity unit",
    "the policy centre",
    "the enforcement directorate",
    "the complaints branch",
    "the grants secretariat",
    "the regional office",
    "the state office",
]
APS_LEVELS = ["EL1", "EL2", "EL3", "EL4", "EL6", "SES", "APS 4", "APS 6"]
PUBLIC_ROLES = [
    "policy officer",
    "assistant director",
    "branch manager",
    "eligibility specialist",
    "case officer",
    "contact centre agent",
    "grants administrator",
    "contracts officer",
    "investigator",
    "compliance officer",
    "statistician",
    "economist",
    "archivist",
    "policy adviser",
    "program manager",
]
CORPORATE_ROLES = [
    "site supervisor",
    "depot manager",
    "shift supervisor",
    "store manager",
    "senior trader",
    "relationship manager",
    "fund manager",
    "mine supervisor",
]
GOVERNMENT_ACTS = [
    "a machinery of government review",
    "a parliamentary estimates submission",
    "a freedom of information request",
    "an outcome roadmap",
    "a gateway review",
    "a cabinet submission",
    "a ministerial submission",
]
WORKPLACE_TYPES = [
    "a residential aged care facility",
    "a correctional centre",
    "our distribution centre",
    "our depot",
    "a water authority",
    "a community health centre",
    "an open cut mine",
    "our freight terminal",
    "a disability accommodation service",
    "a waste facility",
]
COHORTS = [
    "the graduate cohort",
    "the 2021 intake",
    "the pilot cohort",
    "the reference site",
    "the review team",
]
CLEARANCES = [
    "a security clearance",
    "clearance at the secret level",
    "national security vetting",
    "an NVL clearance",
]

SITES = [
    "Kwinana depot",
    "Altona distribution centre",
    "Fremantle yard",
    "Geelong workshop",
    "Parramatta hub",
    "Box Hill store",
    "Newcastle terminal",
    "Townsville depot",
    "Ballarat site",
    "Cairns branch",
    "Sunshine warehouse",
    "Dandenong store",
]
SUBURBS = [
    "Melbourne",
    "Sydney",
    "Brisbane",
    "Perth",
    "Adelaide",
    "Hobart",
    "Darwin",
    "Surry Hills",
    "Newtown",
    "Fremantle",
    "Kwinana",
    "Altona",
]
ROLES = [
    "registered nurse",
    "teacher",
    "care worker",
    "cleaner",
    "truck driver",
    "electrician",
    "barista",
    "labourer",
    "warden",
    "caseworker",
]
TEAMS = ["night shift", "day shift", "ward team", "yard team", "store team", "roster"]
GRADES = ["Grade 3", "Grade 4", "Band 6", "Step 3", "Class 2", "Level 4"]
HEALTH = ["anxiety", "chronic fatigue", "depression", "migraines", "a shoulder injury"]
SURNAMES = [
    "Whitlam",
    "Nguyen",
    "Okafor",
    "Brennan",
    "Kowalski",
    "Ferreira",
    "Nakamura",
    "Abadi",
    "Sinclair",
    "Halvorsen",
    "Petrov",
    "Mbeki",
]
GIVEN = [
    "Fiona",
    "Jane",
    "Ahmed",
    "Priya",
    "Tomas",
    "Grace",
    "Daniel",
    "Marisol",
    "Kiran",
    "Aoife",
    "Hassan",
    "Lena",
]
ORGS = [
    "Acme Pty Ltd",
    "Reticule Health",
    "Northline Services",
    "Corvus Group",
    "Halden Holdings",
    "Pinnacle Corporation",
]
FIRST_NAMES_PHRASES = [
    "the only woman on the night shift",
    "the only one here who does the handover",
    "one of two apprentices on the yard team",
    "the youngest on the team here",
    "the sole nurse in the unit",
    "the only casual on the roster",
    "nobody else does the late round",
    "the only one who does the 6am start",
    "I cover all three depots on my own",
]
SECOND_NAMES_PHRASES = [
    "my manager Fiona",
    "my team lead Daniel",
    "told my supervisor Grace",
    "my supervisor Ahmed",
    "my manager Priya",
]
WORK_EVENTS_PHRASES = [
    "after the depot closure",
    "since the take-over",
    "after the merger announcement",
    "when they closed the Box Hill store",
    "after the safety inspection",
    "since the site was sold",
]

HARMFUL = [
    "constantly undermined in front of the team",
    "targeted me in meetings until I cried",
    "threatened me when I raised a safety issue",
    "excluded me from the roster after I complained",
    "humiliated me in front of the new starters",
    "retaliated against me the week I reported it",
    "shouted at me in the break room",
    "made comments about my age whenever I pushed back",
]
POSITIVE = [
    "my team has always had my back",
    "the training here is genuinely good",
    "my supervisor listens and acts on it",
    "I feel safe raising concerns here",
    "the flexible roster makes a real difference",
]
HEDGES = [
    "I would rather not say too much in writing.",
    "I am not the only one who feels this way.",
    "Please do not share this with my team.",
    "I do not want my employer to find out I said this.",
    "This is anonymous but the detail is real.",
]
SERVICE_GOOD = [
    "the claim was paid without argument",
    "the referral was approved the same day",
    "whoever answers the phone was polite and knew my name",
    "telehealth saved me a two hour drive each way",
    "the physiotherapist was thorough and did not rush me",
    "everything has been paid on time for two years",
]
SERVICE_BAD = [
    "three months and still chasing a claim of four hundred dollars",
    "nobody could tell me where the file had gone",
    "the pre-approval took four months and was never needed",
    "I was put on hold for an hour and then cut off",
    "the refund has not landed and nobody owns the delay",
    "the decision does not match what the specialist wrote",
]
HAZARDS = [
    "we are short three people and everyone is picking up the rest",
    "the workload has doubled since the restructure",
    "roles are unclear and everyone has three managers",
    "the roster changes are announced the night before",
    "I do two twelve hour shifts back to back",
    "working from home means nobody sees how bad it has got",
    "the consultation happened after the decision was made",
    "I have been off sick twice this quarter and neither was discussed",
]
REGISTERS = [
    "THIS IS UNACCEPTABLE AND I WANT SOMEONE TO OWN IT.",
    "no complaints",
    "same as last year",
    "fine i guess",
    "The whole thing has been a joke from the start.",
    "honestly, who is reading these?",
]

_OPENERS = [
    "To be honest,",
    "Long story short,",
    "I have thought about this for weeks.",
    "For what it is worth,",
    "I probably should not have said this, but",
]
_TAILS = [
    "It has been going on for months.",
    "I have raised it twice now.",
    "Nobody has done anything about it.",
    "I just needed to put it somewhere.",
    "",
    "I am done pretending it is fine.",
]


@dataclass(frozen=True, slots=True)
class Secret:
    """One planted ground-truth identifier in a generated record."""

    kind: str
    literal: str
    expect_category: str | None = None
    checksummed: bool = False
    degradation: str = "clean"
    mandatory: bool = True
    field: str | None = None


@dataclass
class GeneratedRecord:
    """One synthetic survey response with ground truth, stratum, and OCR/language metadata."""

    response_id: str
    answers: dict[str, str]
    secrets: list[Secret] = field(default_factory=list)
    must_survive: list[str] = field(default_factory=list)
    template: str = ""
    stratum: str = "short_open_text"
    language: str = "en"
    ocr_confidence: float | None = None
    structured_fields: dict[str, str] = field(default_factory=dict)
    household_id: str = "HH0000"

    def to_json(self) -> dict[str, object]:
        record: dict[str, object] = {
            "ResponseID": self.response_id,
            "stratum": self.stratum,
            "language": self.language,
            "household_id": self.household_id,
        }
        if self.ocr_confidence is not None:
            record["ocr_confidence"] = self.ocr_confidence
        record.update(self.structured_fields)
        record.update(self.answers)
        return record


_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"}


def misspell(text: str, rng: random.Random) -> str:
    if len(text) < 5:
        return text
    index = rng.randrange(1, len(text) - 1)
    kind = rng.choice(("drop", "double", "swap", "substitute"))
    if kind == "drop":
        return text[:index] + text[index + 1 :]
    if kind == "double":
        return text[:index] + text[index] + text[index:]
    if kind == "swap" and index + 1 < len(text):
        return text[:index] + text[index + 1] + text[index] + text[index + 2 :]
    return text[:index] + rng.choice("abcdefghijklmnopqrstuvwxyz") + text[index + 1 :]


def to_leet(text: str) -> str:
    return "".join(_LEET.get(c, c) if c.isdigit() else c for c in text)


def to_fullwidth(text: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if "0" <= c <= "9" else c for c in text)


def to_ocr_corrupted(text: str) -> str:
    """Simulate OCR digit-to-letter corruption (0 -> O, 1 -> I) on an identifier."""
    out: list[str] = []
    replaced = False
    for ch in text:
        if ch == "0" and not replaced:
            out.append("O")
            replaced = True
        elif ch == "1" and not replaced:
            out.append("I")
            replaced = True
        else:
            out.append(ch)
    if not replaced and out:
        # Force at least one OCR character substitution
        for idx, ch in enumerate(out):
            if ch.isdigit():
                out[idx] = "O"
                break
    return "".join(out)


def space_out(text: str) -> str:
    return " ".join(text)


def degrade(text: str, rng: random.Random) -> tuple[str, str]:
    roll = rng.random()
    if roll < 0.35:
        return misspell(text, rng), "misspelled"
    if roll < 0.45:
        return to_leet(text), "leet"
    if roll < 0.55:
        return to_fullwidth(text), "fullwidth"
    if roll < 0.60:
        return space_out(text), "spaced"
    if roll < 0.68:
        return text.lower(), "lower"
    return text, "clean"


def valid_tfn(rng: random.Random) -> str:
    while True:
        candidate = rng.randrange(100_000_000, 999_999_999)
        if is_tfn(str(candidate)):
            digits = str(candidate)
            return f"{digits[:3]} {digits[3:6]} {digits[6:]}"


def valid_medicare(rng: random.Random) -> str:
    base = rng.choice((2, 3, 4, 5, 6, 9)) * 100_000_000 + rng.randrange(10_000_000, 99_999_999)
    digits = f"{base}{medicare_check_digit(str(base))}"
    return f"{digits[0]} {digits[1:5]} {digits[5:9]} {digits[9]}"


def valid_acn(rng: random.Random) -> str:
    base = f"{rng.randrange(10_000_000, 99_999_999):08d}"
    acn = base + str(acn_check_digit(base))
    return f"{acn[:3]} {acn[3:6]} {acn[6:]}"


def valid_phone(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return f"04{rng.randrange(10, 99)} {rng.randrange(100, 999)} {rng.randrange(100, 999)}"
    area = rng.choice(("02", "03", "07", "08"))
    return f"{area}9{rng.randrange(100, 999)} {rng.randrange(1000, 9999)}"


def valid_email(rng: random.Random) -> str:
    return (
        f"{rng.choice(GIVEN).lower()}.{rng.choice(SURNAMES).lower()}@"
        f"{rng.choice(('mail', 'corp', 'group', 'connect')).lower()}"
        f".{rng.choice(('com.au', 'com.au', 'net.au'))}"
    )


def valid_card(rng: random.Random) -> str:
    prefix = rng.choice(("4111", "555534", "378282"))
    return f"{prefix} {rng.randrange(1000, 9999)} {rng.randrange(1000, 9999)} {rng.randrange(1000, 9999)}"


def valid_dob(rng: random.Random) -> str:
    return f"{rng.randrange(1, 29):02d}/{rng.randrange(1, 13):02d}/{rng.randrange(1955, 2006)}"


def valid_bsb(rng: random.Random) -> str:
    return f"{rng.randrange(10, 97)}{rng.randrange(1000, 9999)}"


def valid_passport(rng: random.Random) -> str:
    # Australian passport shape: one letter (not E/O/I) + 7 digits.
    return f"{rng.choice('ABCDFGHJKMNPQRSTUVWXYZ')}{rng.randrange(1_000_000, 9_999_999)}"


def valid_drivers_licence(rng: random.Random) -> str:
    # Eight-digit NSW-shaped licence number.
    return str(rng.randrange(10_000_000, 99_999_999))


def valid_crn(rng: random.Random) -> str:
    return (
        f"{rng.randrange(100, 999)} {rng.randrange(100, 999)} "
        f"{rng.randrange(100, 999)}{rng.choice('ABCHJKLSTVXZ')}"
    )


def valid_employee_ref(rng: random.Random) -> str:
    return str(rng.randrange(10_000_000, 99_999_999))


def valid_medicare_enrolment(rng: random.Random) -> str:
    return (
        f"{rng.randrange(1, 10)} {rng.randrange(1000, 9999)} "
        f"{rng.randrange(1000, 9999)} {rng.randrange(0, 10)}"
    )


def valid_iban(rng: random.Random) -> str:
    """Build a GB IBAN with a correct ISO 7064 mod-97 check pair."""
    bank = "WEST"
    account = f"{rng.randrange(10**11, 10**12):012d}"
    rearranged = bank + account + "GB00"
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    check = 98 - (int(numeric) % 97)
    return f"GB{check:02d}{bank}{account}"


def valid_ipv6(rng: random.Random) -> str:
    return "2001:0db8:" + ":".join(f"{rng.randrange(0x1000, 0xFFFF):04x}" for _ in range(6))


def valid_mac(rng: random.Random) -> str:
    return ":".join(f"{rng.randrange(0, 256):02X}" for _ in range(6))


def valid_street(rng: random.Random) -> str:
    suburb = rng.choice(
        ("Collins", "Macquarie", "Elizabeth", "George", "Flinders", "Bourke", "Swanston")
    )
    suffix = rng.choice(("Street", "Road", "Avenue", "Parade"))
    return f"{rng.randrange(1, 220)} {suburb} {suffix}"


@dataclass(frozen=True, slots=True)
class Injector:
    kind: str
    expect_category: str | None
    checksummed: bool = False
    sentences: tuple[str, ...] = ()
    builders: tuple = ()
    employment_only: bool = False
    exact: bool = False
    mandatory: bool = True


INJECTORS: tuple[Injector, ...] = (
    Injector(
        "phone",
        "phone",
        exact=True,
        sentences=(
            "You can reach me on {value} if that helps.",
            "My mobile is {value}, though I cannot always answer.",
        ),
        builders=(valid_phone,),
    ),
    Injector(
        "email",
        "email",
        exact=True,
        sentences=(
            "Email me at {value} if you need more detail.",
            "I have put my address {value} in the other box.",
        ),
        builders=(valid_email,),
    ),
    Injector(
        "tfn",
        "tfn",
        checksummed=True,
        exact=True,
        sentences=("My TFN is {value} if you need to look me up.",),
        builders=(valid_tfn,),
    ),
    Injector(
        "medicare",
        "medicare",
        checksummed=True,
        exact=True,
        sentences=("My Medicare number is {value}.",),
        builders=(valid_medicare,),
    ),
    Injector(
        "acn",
        "acn",
        checksummed=True,
        exact=True,
        sentences=("The employer ACN is {value} if that is useful.",),
        builders=(valid_acn,),
    ),
    Injector(
        "credit_card",
        "credit_card",
        checksummed=True,
        exact=True,
        sentences=("Pay me on {value} if it is easier.",),
        builders=(valid_card,),
    ),
    Injector(
        "dob",
        "date_au",
        exact=True,
        sentences=("DOB {value}, just so the age gap is clear.",),
        builders=(valid_dob,),
    ),
    Injector(
        "bsb",
        "bsb",
        exact=True,
        employment_only=True,
        sentences=("BSB {value} if you need to run a report.",),
        builders=(valid_bsb,),
    ),
    Injector(
        "organisation",
        "organisation_name",
        exact=True,
        employment_only=True,
        sentences=("For context I work at {value}.", "This is about {value} specifically."),
        builders=(lambda rng: rng.choice(ORGS),),
    ),
    Injector(
        "site",
        "implication",
        employment_only=True,
        mandatory=False,
        sentences=("I work out of {value} most weeks.", "My site is {value}."),
        builders=(lambda rng: rng.choice(SITES),),
    ),
    Injector(
        "implication_person",
        "implication",
        employment_only=True,
        mandatory=False,
        sentences=(
            "Realistically I am {value}.",
            "To put it plainly, {value}.",
        ),
        builders=(lambda rng: rng.choice(FIRST_NAMES_PHRASES),),
    ),
    Injector(
        "implication_workplace",
        "implication",
        employment_only=True,
        mandatory=False,
        sentences=("It all happened {value}.", "Things changed {value}."),
        builders=(lambda rng: rng.choice(WORK_EVENTS_PHRASES),),
    ),
    Injector(
        "third_party",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("It was {value} who signed it off.",),
        builders=(lambda rng: rng.choice(SECOND_NAMES_PHRASES),),
    ),
    Injector(
        "suburb",
        "location",
        employment_only=True,
        exact=True,
        sentences=("I am based in {value} these days.",),
        builders=(lambda rng: rng.choice(SUBURBS),),
    ),
    Injector(
        "aps_unit",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I work in {value}.", "My role sits in {value}."),
        builders=(lambda rng: rng.choice(AGENCY_UNITS),),
    ),
    Injector(
        "aps_level",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I am {value} and that makes me one of very few.",),
        builders=(lambda rng: rng.choice(APS_LEVELS),),
    ),
    Injector(
        "public_role",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I work as a {value}.", "My title is {value}."),
        builders=(lambda rng: rng.choice(PUBLIC_ROLES + CORPORATE_ROLES),),
    ),
    Injector(
        "seniority_in_unit",
        "implication",
        employment_only=True,
        exact=True,
        sentences=(
            "I am one of three of us in {value}.",
            "There are two of us left in {value}.",
        ),
        builders=(lambda rng: rng.choice(AGENCY_UNITS),),
    ),
    Injector(
        "government_act",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("Last year we handled {value}.", "It went through {value}."),
        builders=(lambda rng: rng.choice(GOVERNMENT_ACTS),),
    ),
    Injector(
        "workplace_type",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I have worked in {value} for years.", "I am based in {value}."),
        builders=(lambda rng: rng.choice(WORKPLACE_TYPES),),
    ),
    Injector(
        "cohort",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I am part of {value}.", "I came through {value}."),
        builders=(lambda rng: rng.choice(COHORTS),),
    ),
    Injector(
        "clearance",
        "implication",
        employment_only=True,
        exact=True,
        sentences=("I hold {value}.", "It took years to get {value}."),
        builders=(lambda rng: rng.choice(CLEARANCES),),
    ),
    Injector(
        "phone_obfuscated",
        "phone",
        sentences=("Reach me on {value}.", "Best number is {value}."),
        builders=(valid_phone,),
    ),
    Injector(
        "email_spaced",
        "email",
        sentences=("My address is {value}.",),
        builders=(valid_email,),
    ),
    Injector(
        "passport",
        "passport",
        exact=True,
        sentences=("My passport number is {value} if that helps.",),
        builders=(valid_passport,),
    ),
    Injector(
        "drivers_licence",
        "drivers_licence",
        exact=True,
        sentences=("Driver licence: {value}.",),
        builders=(valid_drivers_licence,),
    ),
    Injector(
        "centrelink_crn",
        "centrelink_crn",
        exact=True,
        sentences=("My CRN is {value} from an old claim.",),
        builders=(valid_crn,),
    ),
    Injector(
        "employee_ref",
        "reference_id",
        exact=True,
        sentences=("My AGS number: {value}.",),
        builders=(valid_employee_ref,),
    ),
    Injector(
        "medicare_enrolment",
        "medicare_enrolment",
        exact=True,
        sentences=("The enrolment reference is {value}.",),
        builders=(valid_medicare_enrolment,),
    ),
    Injector(
        "iban",
        "iban",
        exact=True,
        sentences=("An overseas account uses IBAN {value}.",),
        builders=(valid_iban,),
    ),
    Injector(
        "ipv6",
        "ipv6",
        exact=True,
        sentences=("The gateway logs show {value} for my session.",),
        builders=(valid_ipv6,),
    ),
    Injector(
        "mac_address",
        "mac",
        exact=True,
        sentences=("The laptop MAC {value} was recorded.",),
        builders=(valid_mac,),
    ),
    Injector(
        "street_address",
        "street_address",
        exact=True,
        sentences=("I live at {value}.",),
        builders=(valid_street,),
    ),
)


def _join(text: str, sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return text
    if not sentence[0].isupper():
        sentence = sentence[0].upper() + sentence[1:]
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return f"{text} {sentence}"


def _degraded(value: str, injector: Injector, rng: random.Random) -> tuple[str, str]:
    if injector.kind == "phone_obfuscated":
        return to_fullwidth(value), "fullwidth"
    if injector.kind == "email_spaced":
        return space_out(value), "spaced"
    if injector.exact or injector.checksummed:
        return value, "clean"
    return degrade(value, rng)


class SurveyGenerator:
    """Composes synthetic government survey responses across all 8 benchmark strata."""

    def __init__(
        self,
        seed: int = 20260308,
        *,
        injection_rate: float = 0.85,
        employment_rate: float = 0.7,
    ) -> None:
        self.rng = random.Random(seed)
        self.injection_rate = injection_rate
        self.employment_rate = employment_rate

    def _pick_stratum(self) -> str:
        roll = self.rng.random()
        cumulative = 0.0
        for name, weight in STRATA_WEIGHTS:
            cumulative += weight
            if roll <= cumulative:
                return name
        return "short_open_text"

    def record(self, index: int, *, stratum: str | None = None) -> GeneratedRecord:
        rng = self.rng
        chosen_stratum = stratum or self._pick_stratum()
        template = rng.choice(
            ("harm", "positive", "harm", "harm", "health", "service", "hazard", "terse")
        )
        employment = rng.random() < self.employment_rate

        body = self._body(template, employment)
        secrets: list[Secret] = []
        must_survive: list[str] = []
        structured_fields: dict[str, str] = {}
        ocr_confidence: float | None = None
        language = "en"
        household_id = f"HH{index // 3:04d}"

        if chosen_stratum == "household_grid":
            # Populate structured respondent/household columns and repeat in narrative
            resp_given = rng.choice(GIVEN)
            resp_surname = rng.choice(SURNAMES)
            spouse_given = rng.choice([g for g in GIVEN if g != resp_given])
            full_name = f"{resp_given} {resp_surname}"
            spouse_name = f"{spouse_given} {resp_surname}"
            structured_fields["respondent_name"] = full_name
            structured_fields["household_members"] = f"{full_name}; {spouse_name}"
            body = _join(
                body,
                f"My partner {spouse_name} and I ({full_name}) both experienced this.",
            )
            secrets.append(
                Secret(
                    kind="cross_field_person",
                    literal=spouse_name,
                    expect_category="name",
                    degradation="clean",
                )
            )
            secrets.append(
                Secret(
                    kind="cross_field_person",
                    literal=full_name,
                    expect_category="name",
                    degradation="clean",
                )
            )
        elif chosen_stratum in ("ocr_form", "handwriting_scan"):
            ocr_confidence = round(rng.uniform(0.55, 0.76), 2)
            raw_tfn = valid_tfn(rng)
            corrupted_tfn = to_ocr_corrupted(raw_tfn)
            body = _join(body, f"TFN: {corrupted_tfn}.")
            secrets.append(
                Secret(
                    kind="ocr_tfn",
                    literal=corrupted_tfn,
                    expect_category="tfn",
                    checksummed=True,
                    degradation="ocr_corrupted",
                )
            )
        elif chosen_stratum == "long_narrative":
            # Pad with realistic non-identifying policy/service paragraphs so
            # length exceeds 512 tokens while planting a secret near the end.
            filler = (
                "The workload expectations across the quarter were discussed at length "
                "during the general staff forum, and many colleagues noted that "
                "standard operating procedures had been updated several times without "
                "adequate training time allocated on the roster. "
            )
            body = (filler * 16) + " " + body
            planted_email = valid_email(rng)
            body = _join(body, f"Email me at {planted_email} if you need more detail.")
            secrets.append(
                Secret(
                    kind="email",
                    literal=planted_email,
                    expect_category="email",
                    degradation="clean",
                )
            )
        elif rng.random() < self.injection_rate:
            for injector in self._choose_injectors(employment):
                built = injector.builders[rng.randrange(len(injector.builders))](rng)
                value, how = _degraded(built, injector, rng)
                sentence = injector.sentences[rng.randrange(len(injector.sentences))]
                body = _join(body, sentence.format(value=value))
                secrets.append(
                    Secret(
                        kind=injector.kind,
                        literal=value,
                        expect_category=injector.expect_category,
                        checksummed=injector.checksummed,
                        degradation=how,
                        mandatory=injector.mandatory,
                    )
                )

        must_survive.extend(self._stable_fragments(template, body))
        answers = self._split_answers(body)

        return GeneratedRecord(
            response_id=f"G{index:05d}",
            answers=answers,
            secrets=secrets,
            must_survive=must_survive,
            template=template,
            stratum=chosen_stratum,
            language=language,
            ocr_confidence=ocr_confidence,
            structured_fields=structured_fields,
            household_id=household_id,
        )

    def corpus(self, count: int) -> list[GeneratedRecord]:
        return [self.record(i) for i in range(count)]

    def train_test_split(
        self,
        records: Sequence[GeneratedRecord],
        *,
        test_ratio: float = 0.2,
    ) -> tuple[list[GeneratedRecord], list[GeneratedRecord]]:
        """Split corpus by ``(household_id, template)`` so no household leaks across splits."""
        households = sorted({r.household_id for r in records})
        cutoff = max(1, int(len(households) * (1.0 - test_ratio)))
        train_hh = set(households[:cutoff])
        train = [r for r in records if r.household_id in train_hh]
        test = [r for r in records if r.household_id not in train_hh]
        return train, test

    def _choose_injectors(self, employment: bool) -> list[Injector]:
        pool = [i for i in INJECTORS if employment or not i.employment_only]
        rng = self.rng
        chosen = [i for i in pool if rng.random() < 0.3]
        if len(chosen) <= 3:
            return chosen
        # Randomly sample (rather than truncate) so injectors later in the catalogue
        # are exercised as often as the first few.
        return sorted(rng.sample(chosen, 3), key=INJECTORS.index)

    def _body(self, template: str, employment: bool) -> str:
        rng = self.rng
        parts: list[str] = []
        if rng.random() < 0.5:
            parts.append(_OPENERS[rng.randrange(len(_OPENERS))])

        if template == "harm":
            parts.append(f"My manager {HARMFUL[rng.randrange(len(HARMFUL))]}.")
        elif template == "positive":
            parts.append(f"{POSITIVE[rng.randrange(len(POSITIVE))]}.")
            parts.append(f"{SERVICE_GOOD[rng.randrange(len(SERVICE_GOOD))]}.")
        elif template == "service":
            pool = SERVICE_GOOD if rng.random() < 0.4 else SERVICE_BAD
            parts.append(f"Cover wise, {pool[rng.randrange(len(pool))]}.")
        elif template == "hazard":
            parts.append(f"{HAZARDS[rng.randrange(len(HAZARDS))]}.")
            parts.append(f"{SERVICE_BAD[rng.randrange(len(SERVICE_BAD))]}.")
        elif template == "terse":
            parts.append(REGISTERS[rng.randrange(len(REGISTERS))])
        else:
            parts.append(
                "I have been struggling since this started and it has left me with "
                f"{HEALTH[rng.randrange(len(HEALTH))]}."
            )

        if rng.random() < 0.6:
            condition = rng.choice(HEDGES)
            parts.append(condition)
            if not employment and (
                "do not want my employer" in condition or "Please do not share" in condition
            ):
                parts.append(f"I work at {rng.choice(ORGS)} as a {rng.choice(ROLES)}.")

        if employment and rng.random() < 0.5:
            if rng.random() < 0.55:
                parts.append(
                    f"I am a {rng.choice(PUBLIC_ROLES)}, {rng.choice(APS_LEVELS)}, "
                    f"in {rng.choice(AGENCY_UNITS)}, with "
                    f"{rng.randrange(2, 30)} years of service."
                )
            else:
                parts.append(
                    f"I am a {rng.choice(ROLES + CORPORATE_ROLES)}, "
                    f"{rng.choice(GRADES)}, {rng.randrange(2, 30)} years of "
                    f"service, on the {rng.choice(TEAMS)} of a team of "
                    f"{rng.randrange(2, 15)}."
                )

        tail = _TAILS[rng.randrange(len(_TAILS))]
        if tail:
            parts.append(tail)
        return " ".join(parts)

    @staticmethod
    def _stable_fragments(template: str, body: str) -> list[str]:
        fragments: list[str] = []
        for pool in (HARMFUL, POSITIVE, HAZARDS, SERVICE_GOOD, SERVICE_BAD, REGISTERS):
            for phrase in pool:
                if phrase in body:
                    fragments.append(phrase.split()[0] + " " + phrase.split()[1])
                    break
        return fragments

    def _split_answers(self, body: str) -> dict[str, str]:
        rng = self.rng
        # Never split inside character-spaced obfuscation such as
        # "s a r a h . c h e n @ a g e n c y . g o v . a u": a split would make the
        # token invisible to per-field detectors while hiding it from joined checks.
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+(?![a-z]\s)", body)
            if s.strip()
        ]
        if len(sentences) < 2:
            return {"Q1": body}
        cut = rng.randrange(1, len(sentences))
        return {"Q1": " ".join(sentences[:cut]), "Q2": " ".join(sentences[cut:])}


def write_corpus(
    records: Sequence[GeneratedRecord],
    ndjson_path: Path,
    manifest_path: Path,
) -> None:
    ndjson_path.write_text(
        "\n".join(json.dumps(r.to_json(), ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "count": len(records),
        "records": [
            {
                "id": r.response_id,
                "template": r.template,
                "stratum": r.stratum,
                "language": r.language,
                "ocr_confidence": r.ocr_confidence,
                "secrets": [
                    {
                        "kind": s.kind,
                        "literal": s.literal,
                        "expect_category": s.expect_category,
                        "checksummed": s.checksummed,
                        "degradation": s.degradation,
                        "mandatory": s.mandatory,
                        "field": s.field,
                    }
                    for s in r.secrets
                ],
                "must_survive": r.must_survive,
            }
            for r in records
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "INJECTORS",
    "STRATA_WEIGHTS",
    "GeneratedRecord",
    "Secret",
    "SurveyGenerator",
    "degrade",
    "misspell",
    "space_out",
    "to_fullwidth",
    "to_leet",
    "to_ocr_corrupted",
    "valid_acn",
    "valid_bsb",
    "valid_card",
    "valid_crn",
    "valid_dob",
    "valid_drivers_licence",
    "valid_email",
    "valid_employee_ref",
    "valid_iban",
    "valid_ipv6",
    "valid_mac",
    "valid_medicare",
    "valid_medicare_enrolment",
    "valid_passport",
    "valid_phone",
    "valid_street",
    "valid_tfn",
    "write_corpus",
]
