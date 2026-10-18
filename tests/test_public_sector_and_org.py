"""Tests for Australian public-sector vocabulary, organisation attribution, and implicating sequences."""

from __future__ import annotations

import pytest

from gov_survey_redact import Category, Lexicon, SurveyRedactor, Term, survey_redactor, vocabulary
from gov_survey_redact.detectors.implication import ImplicationDetector


@pytest.fixture
def detector() -> ImplicationDetector:
    return ImplicationDetector()


def spans(detector: ImplicationDetector, text: str) -> list[str]:
    return [f.span.extract(text) for f in detector.detect(text)]


def rules_fired(detector: ImplicationDetector, text: str) -> list[str]:
    return [f.meta()["rule"] for f in detector.detect(text)]


class TestPublicSectorAndImplicationRules:
    @pytest.mark.parametrize(
        "name",
        [
            "DEPARTMENT_STRUCTURE",
            "DEPARTMENT_STRUCTURE_QUALIFIED",
            "GOVERNMENT_PROCESS",
            "APS_LEVEL",
            "SENIOR_ROLE",
            "OCCUPATION_FAMILY",
            "PUBLIC_SECTOR_ROLE",
            "WORKPLACE_TYPE",
            "WORKPLACE_TYPE_STRONG",
            "CORPORATE_MARKER",
            "COHORT_MARKER",
            "CLEARANCE_MARKER",
            "CLEARANCE_LEVEL",
            "NON_CONDITION_VERBS",
        ],
    )
    def test_vocabulary_is_grouped(self, name: str) -> None:
        value = getattr(vocabulary, name)
        assert value.startswith("(?:") and value.endswith(")")

    @pytest.mark.parametrize(
        ("text", "rule"),
        [
            ("I work in the digital services branch.", "department_structure"),
            ("My role sits in the integrity unit.", "department_structure"),
            ("I work in Services division.", "unit_by_preposition"),
            ("Last year we handled a machinery of government review.", "government_process"),
            ("Our agency ran a freedom of information request.", "government_process"),
            ("I am an assistant director in the Home Affairs branch.", "occupation_with_structure"),
            ("I am an eligibility specialist in the Services division.", "occupation_with_structure"),
            ("My role is policy adviser.", "occupation_family"),
            ("I am one of three EL2s in the Migration directorate.", "seniority_in_unit"),
            ("There are two of us left in the integrity unit.", "seniority_in_unit"),
            ("I am part of the pilot cohort.", "cohort_marker"),
            ("I came through the 2021 intake.", "cohort_marker"),
            ("I hold a security clearance.", "clearance"),
            ("I hold clearance at the secret level.", "clearance"),
            ("I am the branch manager.", "senior_role"),
            ("My caseworker was Nguyễn Thị Hương.", "names_third_party"),
            ("It was my manager Fiona who signed it.", "names_third_party"),
        ],
    )
    def test_public_sector_and_third_party_rules_fire(
        self, detector: ImplicationDetector, text: str, rule: str
    ) -> None:
        assert rule in rules_fired(detector, text), (text, rules_fired(detector, text))

    def test_typo_tolerance_on_sites_and_lexicon(self) -> None:
        lex = Lexicon.from_terms(
            [
                Term(
                    "Project Kestrel",
                    Category.INTERNAL_TERM,
                    aliases=["kestrel"],
                    label="internal project",
                )
            ]
        )
        s = SurveyRedactor(survey_redactor(lexicon=lex))
        rec = s.redact_record(
            "After the takc-over at the Box Hill storre, the Kestral timsheet failed."
        )
        assert "Kestral" not in rec.text
        assert "timsheet" not in rec.text
        assert "Box Hill storre" not in rec.text

    @pytest.mark.parametrize(
        "benign",
        [
            "My team has always had my back.",
            "The training here is genuinely good.",
            "my supervisor listens and acts on it",
            "I feel safe raising concerns here.",
            "I work in a team of nine.",
            "My office is cold.",
            "I am clear on what my responsibilities are.",
            "I have support to work safely.",
            "I am torn between two competing priorities.",
            "The main branch closes early on Fridays.",
            "We have support from the other team.",
            "Nothing has changed since the announcement.",
            "DO NOT BUY WORKPLACE COVER FROM THESE PEOPLE.",
            "Nobody in the claims team has answered.",
        ],
    )
    def test_benign_and_protective_items_are_not_over_redacted(self, benign: str) -> None:
        s = SurveyRedactor(survey_redactor())
        rec = s.redact_record(benign)
        if "team of nine" in benign:
            assert "6-10 people" in rec.text
        else:
            assert rec.text == benign, (benign, rec.text, rec.findings)
