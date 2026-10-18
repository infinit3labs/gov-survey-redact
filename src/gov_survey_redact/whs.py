"""WHS claims-experience survey: Qualtrics instrument and synthetic batch generator.

The instrument replicates the themes and published item wording of Safe Work
Australia's **National Return to Work Survey** claims-experience module
(Collie et al. 2019, *BMC Public Health* 19:927): the five 5-point agreement
items (open and honest process, good communication, protecting best interests,
treated fairly, helped with recovery) plus insurer claims-experience items for
case-manager respect/contact, return-to-work plans, suitable duties, employer
contact, payment timeliness and adequacy, recovery confidence, and injury,
jurisdiction, time-to-lodge and employer-type classifiers.

Responses carry Qualtrics metadata columns, embedded claims-system data
(``claim_number``, ``respondent_name`` …), closed-question recode values, and
free-text answers that realistically mention case managers, employers, worksites,
treating practitioners, claim references and contact details — the identifying
content the redaction pipeline must catch before the dataset is released.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .generate import (
    GIVEN,
    ORGS,
    SITES,
    SURNAMES,
    Secret,
    misspell,
    space_out,
    to_fullwidth,
    to_leet,
    valid_bsb,
    valid_crn,
    valid_email,
    valid_employee_ref,
    valid_medicare,
    valid_phone,
    valid_street,
    valid_tfn,
)
from .qualtrics import (
    EmbeddedField,
    QualtricsChoice,
    QualtricsInstrument,
    QualtricsQuestion,
    QualtricsResponse,
)
from .types import Category

LIKERT_CHOICES = (
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
)

_PHYSICIAN_TITLES = ("Dr", "Dr", "Dr", "A/Prof")
_CLINICS = (
    "Altona Medical Centre",
    "Kwinana Health Hub",
    "Surry Hills Family Practice",
    "Bendigo Community Clinic",
    "Fremantle Occupational Health",
    "Newcastle GP Superclinic",
)
_UNION_ROLES = ("union rep", "delegate", "organiser")
_REHAB_PROVIDERS = (
    "Reticule Health",
    "Northline Rehabilitation",
    "Corvus Injury Management",
    "Pinnacle Recovery Services",
)


def _choices(*texts: str) -> tuple[QualtricsChoice, ...]:
    return tuple(
        QualtricsChoice(text=text, recode=index, choice_id=index)
        for index, text in enumerate(texts, start=1)
    )


def build_whs_claims_instrument() -> QualtricsInstrument:
    """Build the WHS claims-experience Qualtrics instrument."""
    return QualtricsInstrument(
        survey_id="SV_whsClaimsExperience2026",
        name="WHS Claims Experience Survey 2026",
        questions=(
            QualtricsQuestion(
                "QID1", "claim_lodgement", "How did you lodge your workers' compensation claim?",
                qtype="MC", selector="SAVR",
                choices=_choices(
                    "Online", "By phone", "Through my employer", "Through my union", "Other"
                ),
            ),
            QualtricsQuestion(
                "QID2", "time_to_first_contact",
                "How long after you lodged your claim did the insurer first contact you?",
                qtype="MC", selector="SAVR",
                choices=_choices(
                    "Within 2 business days", "3 to 7 business days", "1 to 2 weeks",
                    "More than 2 weeks", "I was never contacted",
                ),
            ),
            QualtricsQuestion(
                "QID3", "process_open_honest", "The claims process was open and honest.",
                qtype="Likert", choices=_choices(*LIKERT_CHOICES),
            ),
            QualtricsQuestion(
                "QID4", "good_communication",
                "There was good communication between the people and organisations involved in my claim.",
                qtype="Likert", choices=_choices(*LIKERT_CHOICES),
            ),
            QualtricsQuestion(
                "QID5", "protecting_interests",
                "I felt like the system was working to protect my best interests.",
                qtype="Likert", choices=_choices(*LIKERT_CHOICES),
            ),
            QualtricsQuestion(
                "QID6", "treated_fairly", "I believe the system treated me fairly.",
                qtype="Likert", choices=_choices(*LIKERT_CHOICES),
            ),
            QualtricsQuestion(
                "QID7", "helped_recovery", "I feel that the system helped me with my recovery.",
                qtype="Likert", choices=_choices(*LIKERT_CHOICES),
            ),
            QualtricsQuestion(
                "QID8", "case_manager_respect",
                "My case manager treated me with respect and sensitivity.",
                qtype="Likert", choices=_choices(*LIKERT_CHOICES),
            ),
            QualtricsQuestion(
                "QID9", "case_manager_contact",
                "How often did your case manager keep you informed about your claim?",
                qtype="MC", selector="SAVR",
                choices=_choices(
                    "At least weekly", "Every 2 to 3 weeks", "Monthly", "Less often",
                    "I was not kept informed",
                ),
            ),
            QualtricsQuestion(
                "QID10", "rtw_plan_discussed",
                "Was a return-to-work plan discussed with you?",
                qtype="MC", selector="SAVR",
                choices=_choices(
                    "Yes", "No", "I'm not sure", "It had not been developed before I returned"
                ),
            ),
            QualtricsQuestion(
                "QID11", "suitable_duties",
                "Were suitable duties available when you were ready to return to work?",
                qtype="MC", selector="SAVR",
                choices=_choices(
                    "Yes", "No", "Partly", "Not applicable - I have not returned to work"
                ),
            ),
            QualtricsQuestion(
                "QID12", "employer_contact",
                "Did your employer stay in contact with you while you were off work?",
                qtype="MC", selector="SAVR",
                choices=_choices(
                    "Yes, regularly", "Yes, occasionally", "No",
                    "I am self-employed or no longer employed there",
                ),
            ),
            QualtricsQuestion(
                "QID13", "payment_timeliness",
                "Were your income support payments made on time?",
                qtype="MC", selector="SAVR",
                choices=_choices(
                    "Always", "Mostly", "Sometimes", "Rarely or never",
                    "I did not receive income support",
                ),
            ),
            QualtricsQuestion(
                "QID14", "compensation_adequacy",
                "The financial support I received was adequate for my needs.",
                qtype="Likert", choices=_choices(*LIKERT_CHOICES),
            ),
            QualtricsQuestion(
                "QID15", "recovery_confidence",
                "I am confident about my recovery and return to work.",
                qtype="Likert", choices=_choices(*LIKERT_CHOICES),
            ),
            QualtricsQuestion(
                "QID16", "injury_type", "What kind of condition is your claim for?",
                qtype="MC", selector="SAVR",
                choices=_choices(
                    "Musculoskeletal condition", "Fracture", "Mental health condition",
                    "Neurological injury or disease", "Other traumatic injury", "Other disease",
                ),
            ),
            QualtricsQuestion(
                "QID17", "jurisdiction", "Which workers' compensation scheme is your claim with?",
                qtype="MC", selector="SAVR",
                choices=_choices(
                    "Queensland", "New South Wales", "Victoria", "Western Australia",
                    "South Australia", "Tasmania", "Northern Territory", "ACT",
                    "Comcare", "Seacare",
                ),
            ),
            QualtricsQuestion(
                "QID18", "time_to_lodge",
                "How long after your injury or illness did you lodge your claim?",
                qtype="MC", selector="SAVR",
                choices=_choices(
                    "Less than 7 days", "7 to 13 days", "14 to 27 days", "28 to 55 days",
                    "56 to 83 days", "84 to 180 days", "More than 180 days",
                ),
            ),
            QualtricsQuestion(
                "QID19", "employer_type", "Which best describes your employer?",
                qtype="MC", selector="SAVR",
                choices=_choices("Premium-paying employer", "Self-insured employer"),
            ),
            QualtricsQuestion(
                "QID20", "current_work_status", "Which best describes your situation now?",
                qtype="MC", selector="SAVR",
                choices=_choices(
                    "Working - same employer", "Working - different employer", "Not working",
                    "Retired",
                ),
            ),
            QualtricsQuestion(
                "QID21", "service_experience",
                "Thinking about your entire claims experience, what worked well and what could "
                "be improved?",
            ),
            QualtricsQuestion(
                "QID22", "case_manager_experience",
                "Tell us about your experience with your case manager, including anything they "
                "did that helped or made things harder.",
            ),
            QualtricsQuestion(
                "QID23", "rtw_experience",
                "What has helped, or gotten in the way of, your return to work?",
            ),
            QualtricsQuestion(
                "QID24", "other_comments",
                "Is there anything else you would like us to know about your claim or your "
                "recovery?",
            ),
            QualtricsQuestion(
                "QID25", "callback_contact", "May we contact you about your experience?",
                qtype="MC", selector="SAVR", choices=_choices("Yes", "No"),
            ),
            QualtricsQuestion(
                "QID26", "contact_name", "Thank you. What name should we use when we contact you?"
            ),
            QualtricsQuestion(
                "QID27", "contact_email", "What is the best email address to reach you on?"
            ),
            QualtricsQuestion(
                "QID28", "contact_phone", "What is the best phone number to reach you on?"
            ),
        ),
        embedded_fields=(
            EmbeddedField(
                "claim_number", "Claims system claim reference",
                category=Category.REFERENCE_ID, tokenise=True,
            ),
            EmbeddedField("respondent_name", "Sampled respondent name", category=Category.NAME),
            EmbeddedField("respondent_email", "Sampled respondent email", category=Category.EMAIL),
            EmbeddedField("respondent_phone", "Sampled respondent phone", category=Category.PHONE),
            EmbeddedField(
                "employer_name", "Employer on the claim", category=Category.ORGANISATION_NAME
            ),
        ),
    )


WHS_CLAIMS_INSTRUMENT = build_whs_claims_instrument()


# --------------------------------------------------------------------------- #
# Response content pools (grounded in NRTWS item themes)
# --------------------------------------------------------------------------- #

_SERVICE_POSITIVE = (
    "The claim was approved quickly and my case manager rang me the same week to explain the "
    "next steps.",
    "The online form was easy to complete and I received a decision within two weeks.",
    "Communication was clear throughout and I always knew what was happening with my claim.",
    "Payments arrived on time and the amounts matched what I was told at the start.",
    "My rehabilitation provider was thorough and the treatment plan actually helped.",
)
_SERVICE_NEGATIVE = (
    "It took nearly three months to approve my claim and nobody explained the delay.",
    "I had to repeat the same information to four different people.",
    "The decision letter did not explain why my hours were cut and no one would discuss it.",
    "My payments stopped without notice and I had to chase the insurer three times.",
    "I felt like the system was working against me from the first phone call.",
)
_SERVICE_NEUTRAL = (
    "The claim was processed but I would not say anyone went out of their way.",
    "Some parts were straightforward and other parts took much longer than I expected.",
)
_CASE_MANAGER_POSITIVE = (
    "My case manager kept in touch every fortnight and explained each step of the process.",
    "She rang my employer to set up suitable duties before I even asked.",
    "He always returned my calls the same day and never made me feel like a burden.",
)
_CASE_MANAGER_NEGATIVE = (
    "My case manager changed four times and each one asked me to start my story again.",
    "He never returned my calls and closed my file without telling me.",
    "She spoke to my employer before she had even spoken to me about my injury.",
)
_CASE_MANAGER_NEUTRAL = (
    "My case manager was polite but very hard to reach.",
    "The case manager did the paperwork but not much else.",
)
_RTW_POSITIVE = (
    "My employer offered light duties on reduced hours and that made coming back manageable.",
    "The return-to-work plan was agreed before I came back and my supervisor checked in weekly.",
    "The occupational therapist visited my worksite and adjusted my workstation.",
)
_RTW_NEGATIVE = (
    "When I came back there was no suitable duties program and I was put straight onto full "
    "duties.",
    "My supervisor made it clear that my injury was inconvenient for the team.",
    "The rehabilitation provider cancelled appointments and I lost three weeks of progress.",
)
_RTW_NEUTRAL = (
    "I came back on reduced hours for a month and then returned to my normal roster.",
    "The return to work happened slowly and mostly because I pushed for it.",
)
_OTHER_COMMENTS = (
    "I would like someone independent to review how my claim was handled.",
    "The medical certificates were hard to get and each one cost me money up front.",
    "Thank you to the staff member who answered the phone late at night.",
    "Please make it easier for people in regional areas to see a specialist.",
)
_HEDGES = (
    "I would rather not say too much in writing.",
    "I am not the only person at my workplace this has happened to.",
    "Please do not share this with my employer.",
    "This is anonymous but every detail in it is real.",
)


def _pii_sentences(rng: random.Random) -> tuple[list[tuple[str, list[Secret]]], list[str]]:
    """Build candidate PII-bearing sentences with their planted secrets.

    Returns ``(sentence_and_secrets, plain_sentences)``; callers inject a subset
    and keep the rest as non-identifying narrative.
    """
    case_manager = f"{rng.choice(GIVEN)} {rng.choice(SURNAMES)}"
    supervisor = f"{rng.choice(GIVEN)} {rng.choice(SURNAMES)}"
    union_rep = f"{rng.choice(GIVEN)} {rng.choice(SURNAMES)}"
    doctor = f"{_PHYSICIAN_TITLES[rng.randrange(len(_PHYSICIAN_TITLES))]} {rng.choice(GIVEN)} " \
             f"{rng.choice(SURNAMES)}"
    clinic = _CLINICS[rng.randrange(len(_CLINICS))]
    org = ORGS[rng.randrange(len(ORGS))]
    site = SITES[rng.randrange(len(SITES))]
    rehab = _REHAB_PROVIDERS[rng.randrange(len(_REHAB_PROVIDERS))]
    claim_ref = f"WC-2026-{rng.randrange(100000, 999999)}"
    staff_id = valid_employee_ref(rng)
    phone = valid_phone(rng)
    email = valid_email(rng)
    street = valid_street(rng)
    bsb = valid_bsb(rng)
    account = str(rng.randrange(10**7, 10**8))
    union_role = _UNION_ROLES[rng.randrange(len(_UNION_ROLES))]

    def secret(
        kind: str,
        literal: str,
        category: Category | str | None,
        *,
        mandatory: bool = True,
        checksummed: bool = False,
    ) -> Secret:
        return Secret(
            kind=kind,
            literal=literal,
            expect_category=category.value if isinstance(category, Category) else category,
            checksummed=checksummed,
            mandatory=mandatory,
        )

    candidates: list[tuple[str, list[Secret]]] = [
        (
            f"My case manager {case_manager} was the only person who listened.",
            [secret("person_name", case_manager, "name", mandatory=False)],
        ),
        (
            f"The consultant {supervisor} from {org} closed my file while I was still off work.",
            [
                secret("person_name", supervisor, "name", mandatory=False),
                secret("organisation", org, "organisation_name"),
            ],
        ),
        (
            f"I work at {org} in the {site} and nobody passed my messages on.",
            [
                secret("organisation", org, "organisation_name"),
                secret("site", site, "implication", mandatory=False),
            ],
        ),
        (
            f"My {union_role} {union_rep} helped me fill in the forms.",
            [secret("person_name", union_rep, "name", mandatory=False)],
        ),
        (
            f"{doctor} at {clinic} wrote the certificate and said I was not ready for full duties.",
            [
                secret("person_name", doctor, "name", mandatory=False),
                secret("organisation", clinic, "organisation_name"),
            ],
        ),
        (
            f"The rehabilitation provider {rehab} cancelled two appointments in a row.",
            [secret("organisation", rehab, "organisation_name")],
        ),
        (
            f"If you need the details, call me on {phone} after 5pm.",
            [secret("phone", phone, "phone")],
        ),
        (
            f"My email is {email} if someone wants to follow up properly.",
            [secret("email", email, "email")],
        ),
        (
            f"My claim number is {claim_ref} in case that helps someone look it up.",
            [secret("claim_reference", claim_ref, "reference_id")],
        ),
        (
            f"My staff number is {staff_id} and the roster is still wrong.",
            [secret("employee_id", staff_id, "reference_id")],
        ),
        (
            f"I still have not been reimbursed for travel costs, and my address is {street}.",
            [secret("address", street, "street_address")],
        ),
        (
            f"Please send the reimbursement to BSB {bsb} account {account}.",
            [secret("bsb", bsb, "bsb")],
        ),
    ]
    # Degraded variants occasionally appear (people obfuscate details in web forms).
    if rng.random() < 0.35:
        corrupted = valid_phone(rng)
        candidates.append(
            (
                f"my mobile pasted from my record is {to_fullwidth(corrupted)}",
                [secret("phone", to_fullwidth(corrupted), "phone")],
            )
        )
    if rng.random() < 0.25:
        spaced = space_out(valid_email(rng))
        candidates.append(
            (
                f"i typed my address with spaces {spaced} so the filter would not block it",
                [secret("email", spaced, "email")],
            )
        )
    if rng.random() < 0.15:
        tfn = valid_tfn(rng)
        candidates.append(
            (
                f"my TFN is {tfn} if you need to find my file",
                [secret("tfn", tfn, "tfn", checksummed=True)],
            )
        )
    if rng.random() < 0.12:
        medicare = valid_medicare(rng)
        crn = valid_crn(rng)
        candidates.append(
            (
                f"Medicare {medicare}, Centrelink CRN {crn}, in case that links my records.",
                [
                    secret("medicare", medicare, "medicare", checksummed=True),
                    secret("centrelink_crn", crn, "centrelink_crn"),
                ],
            )
        )
    plain = [
        "The whole process took longer than the injury itself.",
        "I just want to get back to the job I was doing before.",
        "Nobody should have to fight this hard while they are recovering.",
    ]
    return candidates, plain


def _degrade(text: str, rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.12:
        return misspell(text, rng)
    if roll < 0.16:
        return to_leet(text)
    if roll < 0.20:
        return text.lower()
    return text


def _likert_value(profile: str, rng: random.Random) -> int:
    if profile == "positive":
        return rng.choice((4, 5, 5))
    if profile == "negative":
        return rng.choice((1, 1, 2))
    return 3


def _mc_value(profile: str, qid: str, rng: random.Random) -> int:
    """Return a plausible recode value for a closed question under ``profile``."""
    if qid == "QID1":
        return rng.choice((1, 2, 3, 3, 5))
    if qid == "QID2":
        return rng.choice((1, 2, 2, 3, 4)) if profile != "negative" else rng.choice((3, 4, 5, 5))
    if qid == "QID9":
        return rng.choice((1, 1, 2, 2)) if profile != "negative" else rng.choice((3, 4, 5, 5))
    if qid == "QID10":
        return rng.choice((1, 1, 1)) if profile != "negative" else rng.choice((2, 2, 3, 4))
    if qid == "QID11":
        return rng.choice((1, 1, 3)) if profile != "negative" else rng.choice((2, 2, 4))
    if qid == "QID12":
        return rng.choice((1, 1, 2)) if profile != "negative" else rng.choice((3, 3, 4))
    if qid == "QID13":
        return rng.choice((1, 1, 2)) if profile != "negative" else rng.choice((3, 3, 4, 4, 5))
    if qid == "QID16":
        return rng.choices((1, 2, 3, 4, 5, 6), weights=(55, 12, 8, 4, 15, 6))[0]
    if qid == "QID17":
        return rng.choices(range(1, 11), weights=(20, 16, 17, 12, 10, 5, 4, 2, 9, 5))[0]
    if qid == "QID18":
        return rng.choices(range(1, 8), weights=(30, 20, 18, 12, 6, 8, 6))[0]
    if qid == "QID19":
        return 1 if rng.random() < 0.85 else 2
    if qid == "QID20":
        if profile == "positive":
            return rng.choice((1, 1, 1, 2))
        if profile == "negative":
            return rng.choice((3, 1, 2, 4))
        return rng.choice((1, 2, 3))
    if qid == "QID25":
        return 1 if rng.random() < 0.42 else 2
    return rng.randint(1, 4)


@dataclass(frozen=True, slots=True)
class WHSGroundTruth:
    """Ground truth for one generated Qualtrics response."""

    response_id: str
    secrets: tuple[Secret, ...] = ()
    embedded_secrets: tuple[Secret, ...] = ()
    must_survive: tuple[str, ...] = ()
    answered_fields: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WHSGroundTruth:
        """Rebuild ground truth from its serialised form (sidecar files)."""

        def secret(data: dict[str, Any]) -> Secret:
            return Secret(
                kind=str(data["kind"]),
                literal=str(data["literal"]),
                expect_category=data.get("expect_category"),
                checksummed=bool(data.get("checksummed", False)),
                degradation=str(data.get("degradation", "clean")),
                mandatory=bool(data.get("mandatory", True)),
                field=data.get("field"),
            )

        return cls(
            response_id=str(payload["ResponseId"]),
            secrets=tuple(secret(s) for s in payload.get("secrets", ())),
            embedded_secrets=tuple(secret(s) for s in payload.get("embedded_secrets", ())),
            must_survive=tuple(str(f) for f in payload.get("must_survive", ())),
            answered_fields=tuple(str(f) for f in payload.get("answered_fields", ())),
        )

    def as_dict(self) -> dict[str, Any]:
        def secret_dict(secret: Secret) -> dict[str, Any]:
            return {
                "field": secret.field,
                "kind": secret.kind,
                "literal": secret.literal,
                "expect_category": secret.expect_category,
                "checksummed": secret.checksummed,
                "degradation": secret.degradation,
                "mandatory": secret.mandatory,
            }

        return {
            "ResponseId": self.response_id,
            "secrets": [secret_dict(s) for s in self.secrets],
            "embedded_secrets": [secret_dict(s) for s in self.embedded_secrets],
            "must_survive": list(self.must_survive),
            "answered_fields": list(self.answered_fields),
        }


@dataclass
class WHSBatch:
    """One batch of generated Qualtrics responses with ground truth."""

    batch_index: int
    responses: list[QualtricsResponse] = field(default_factory=list)
    ground_truth: dict[str, WHSGroundTruth] = field(default_factory=dict)

    @property
    def partial_count(self) -> int:
        return sum(1 for response in self.responses if response.partial)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "response_count": len(self.responses),
            "partial_count": self.partial_count,
            "responses": [
                {"ResponseId": r.response_id, "values": r.values, "displayed_fields": r.displayed_fields}
                for r in self.responses
            ],
            "ground_truth": [self.ground_truth[r.response_id].as_dict() for r in self.responses],
        }


class WHSBatchGenerator:
    """Generates reproducible batches of Qualtrics-shaped WHS responses."""

    def __init__(
        self,
        seed: int = 20261018,
        *,
        instrument: QualtricsInstrument | None = None,
        negative_rate: float = 0.30,
        pii_rate: float = 0.75,
        partial_rate: float = 0.10,
        language_weights: tuple[float, float, float, float] = (80.0, 8.0, 6.0, 6.0),
    ) -> None:
        self.instrument = instrument or WHS_CLAIMS_INSTRUMENT
        self.rng = random.Random(seed)
        self.seed = seed
        self.negative_rate = negative_rate
        self.pii_rate = pii_rate
        self.partial_rate = partial_rate
        #: Relative weights for (EN-AU, EN-GB, VI, AR) response languages.
        self.language_weights = language_weights
        self.start_date = datetime(2026, 8, 3, 8, 30, tzinfo=timezone.utc)

    # -- public ---------------------------------------------------------------

    def batch(self, count: int, batch_index: int = 0) -> WHSBatch:
        batch = WHSBatch(batch_index=batch_index)
        for index in range(count):
            response, truth = self.response(batch_index=batch_index, index=index)
            batch.responses.append(response)
            batch.ground_truth[response.response_id] = truth
        return batch

    def batches(self, batch_count: int, per_batch: int) -> list[WHSBatch]:
        return [self.batch(per_batch, batch_index=index) for index in range(batch_count)]

    # -- per-response composition --------------------------------------------

    def response(
        self, *, batch_index: int, index: int
    ) -> tuple[QualtricsResponse, WHSGroundTruth]:
        rng = self.rng
        roll = rng.random()
        profile = "positive" if roll > self.negative_rate else "negative"
        if 0.0 < roll <= 0.06 and self.negative_rate < 0.5:
            profile = "neutral"

        partial = rng.random() < self.partial_rate
        progress = rng.randrange(25, 85) if partial else 100
        advance = rng.randrange(3, 40)
        start = self.start_date + timedelta(
            days=batch_index * 7, minutes=index * 19 + rng.randrange(0, 9)
        )
        end = start + timedelta(minutes=advance)
        response_id = f"R_{batch_index:02d}{index:04d}{rng.randrange(0x1000, 0xFFFF):04X}"

        values: dict[str, Any] = {
            "StartDate": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "EndDate": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Status": 0,
            "IPAddress": (
                f"203.0.113.{rng.randrange(2, 254)}"
                if rng.random() < 0.9
                else f"2001:0db8:85a3::{rng.randrange(1, 0xFFFF):04x}"
            ),
            "Progress": progress,
            "Duration (in seconds)": int((end - start).total_seconds()),
            "Finished": 0 if partial else 1,
            "RecordedDate": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "ResponseId": response_id,
            "DistributionChannel": rng.choices(
                ("email", "anonymous", "qr"), weights=(65, 25, 10)
            )[0],
            "UserLanguage": rng.choices(
                ("EN-AU", "EN-GB", "VI", "AR"), weights=self.language_weights
            )[0],
        }

        # Embedded claims-system data (sampled at invitation time).
        values["claim_number"] = f"WC-2026-{rng.randrange(100000, 999999)}"
        values["respondent_name"] = f"{rng.choice(GIVEN)} {rng.choice(SURNAMES)}"
        values["respondent_email"] = valid_email(rng)
        if rng.random() < 0.8:
            values["respondent_phone"] = valid_phone(rng)
        values["employer_name"] = ORGS[rng.randrange(len(ORGS))]

        # Closed questions (all answered unless the response is partial).
        closed_qids = [q.qid for q in self.instrument.questions if q.is_closed]
        answered_qids = closed_qids if not partial else closed_qids[: rng.randrange(8, 16)]
        for qid in answered_qids:
            if qid in ("QID3", "QID4", "QID5", "QID6", "QID7", "QID8", "QID14", "QID15"):
                values[qid] = _likert_value(profile, rng)
            else:
                values[qid] = _mc_value(profile, qid, rng)

        # Free text: compose sentences per answerable open question.
        secrets: list[Secret] = []
        must_survive: list[str] = []
        pii_candidates, plain_sentences = _pii_sentences(rng)
        use_pii = rng.random() < self.pii_rate
        answered_fields: list[str] = list(answered_qids)

        open_questions = [q for q in self.instrument.questions if q.is_text_entry]
        skipped_tail = rng.randrange(1, 3) if partial else 0
        for order_index, question in enumerate(open_questions):
            if partial and order_index >= len(open_questions) - skipped_tail:
                break  # abandoned before the final blocks
            if question.qid in ("QID26", "QID27", "QID28"):
                if values.get("QID25") != 1:
                    continue
                if question.qid == "QID26":
                    body = str(values["respondent_name"])
                    contact_secret = Secret(
                        kind="contact_name", literal=body, expect_category="name"
                    )
                elif question.qid == "QID27":
                    body = str(values["respondent_email"])
                    contact_secret = Secret(
                        kind="contact_email", literal=body, expect_category="email"
                    )
                else:
                    if "respondent_phone" not in values:
                        continue
                    body = str(values["respondent_phone"])
                    contact_secret = Secret(
                        kind="contact_phone", literal=body, expect_category="phone"
                    )
                values[question.qid] = body
                secrets.append(_with_field(contact_secret, question.qid))
                answered_fields.append(question.qid)
                continue

            # Degradation applies to the non-identifying narrative only, so the
            # ground-truth literals always match the generated text verbatim.
            body = _degrade(
                self._compose_open_text(question.qid, profile, rng, plain_sentences), rng
            )
            if body and rng.random() < 0.6:
                words = list(re.finditer(r"\S+", body))
                if len(words) >= 3:
                    # Slice the body verbatim so doubled spaces from degradation
                    # stay part of the fragment (split() would collapse them).
                    fragment = body[: words[min(4, len(words) - 1)].end()]
                    must_survive.append(fragment)
            if use_pii and rng.random() < 0.55:
                sentence, planted = pii_candidates[rng.randrange(len(pii_candidates))]
                body = f"{body} {sentence}".strip()
                for planted_secret in planted:
                    secrets.append(_with_field(planted_secret, question.qid))
            values[question.qid] = body
            answered_fields.append(question.qid)

        embedded_secrets = [
            _with_field(
                Secret(kind="claim_number", literal=values["claim_number"], expect_category="reference_id"),
                "claim_number",
            ),
            _with_field(
                Secret(kind="respondent_name", literal=values["respondent_name"], expect_category="name"),
                "respondent_name",
            ),
            _with_field(
                Secret(kind="respondent_email", literal=values["respondent_email"], expect_category="email"),
                "respondent_email",
            ),
        ]
        if "respondent_phone" in values:
            embedded_secrets.append(
                _with_field(
                    Secret(
                        kind="respondent_phone",
                        literal=values["respondent_phone"],
                        expect_category="phone",
                    ),
                    "respondent_phone",
                )
            )

        response = QualtricsResponse(response_id=response_id, values=values)
        truth = WHSGroundTruth(
            response_id=response_id,
            secrets=tuple(secrets),
            embedded_secrets=tuple(embedded_secrets),
            must_survive=tuple(must_survive),
            answered_fields=tuple(answered_fields),
        )
        return response, truth

    def _compose_open_text(
        self,
        qid: str,
        profile: str,
        rng: random.Random,
        plain_sentences: list[str],
    ) -> str:
        pools: dict[str, dict[str, tuple[str, ...]]] = {
            "QID21": {
                "positive": _SERVICE_POSITIVE,
                "negative": _SERVICE_NEGATIVE,
                "neutral": _SERVICE_NEUTRAL,
            },
            "QID22": {
                "positive": _CASE_MANAGER_POSITIVE,
                "negative": _CASE_MANAGER_NEGATIVE,
                "neutral": _CASE_MANAGER_NEUTRAL,
            },
            "QID23": {
                "positive": _RTW_POSITIVE,
                "negative": _RTW_NEGATIVE,
                "neutral": _RTW_NEUTRAL,
            },
            "QID24": {
                "positive": _OTHER_COMMENTS,
                "negative": _OTHER_COMMENTS,
                "neutral": _OTHER_COMMENTS,
            },
        }
        pool = pools.get(qid, {}).get(profile, _OTHER_COMMENTS)
        parts = [pool[rng.randrange(len(pool))]]
        if rng.random() < 0.35:
            parts.append(plain_sentences[rng.randrange(len(plain_sentences))])
        if rng.random() < 0.18:
            parts.append(_HEDGES[rng.randrange(len(_HEDGES))])
        return " ".join(parts)

    # -- contact block helpers --------------------------------------------


def _with_field(secret: Secret, field_id: str) -> Secret:
    return Secret(
        kind=secret.kind,
        literal=secret.literal,
        expect_category=secret.expect_category,
        checksummed=secret.checksummed,
        degradation=secret.degradation,
        mandatory=secret.mandatory,
        field=field_id,
    )


def write_batch_files(
    directory: str,
    batch: WHSBatch,
    instrument: QualtricsInstrument | None = None,
    *,
    formats: tuple[str, ...] = ("api", "csv", "tsv"),
) -> list[str]:
    """Write one batch in the requested Qualtrics export formats."""
    from pathlib import Path

    from .qualtrics import write_api_json, write_csv, write_ndjson, write_tsv

    instrument = instrument or WHS_CLAIMS_INSTRUMENT
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    stem = f"batch_{batch.batch_index + 1:02d}"
    if "api" in formats:
        path = write_api_json(base / f"{stem}.api.json", batch.responses, instrument)
        written.append(str(path))
    if "ndjson" in formats:
        path = write_ndjson(base / f"{stem}.ndjson", batch.responses, instrument)
        written.append(str(path))
    if "csv" in formats:
        path = write_csv(base / f"{stem}.responses.csv", batch.responses, instrument)
        written.append(str(path))
    if "tsv" in formats:
        path = write_tsv(base / f"{stem}.responses.tsv", batch.responses, instrument)
        written.append(str(path))
    truth_path = base / f"{stem}.ground_truth.json"
    truth_path.write_text(
        json_dumps([batch.ground_truth[r.response_id].as_dict() for r in batch.responses]),
        encoding="utf-8",
    )
    written.append(str(truth_path))
    return written


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


__all__ = [
    "LIKERT_CHOICES",
    "WHS_CLAIMS_INSTRUMENT",
    "WHSBatch",
    "WHSBatchGenerator",
    "WHSGroundTruth",
    "build_whs_claims_instrument",
    "write_batch_files",
]
