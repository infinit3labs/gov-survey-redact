"""Regression tests over the JSON survey test datasets in ``data/``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gov_survey_redact import (
    BoundingPolygon,
    Category,
    GeneratedRecord,
    Lexicon,
    OCRFieldLayout,
    OCRWord,
    ReviewAction,
    ReviewerDecision,
    ReviewState,
    Secret,
    Span,
    SurveyRedactor,
    Term,
    evaluate_corpus,
    survey_redactor,
)
from gov_survey_redact.qualtrics import QualtricsInstrument, row_values_from_mapping

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

#: Embedded Qualtrics fields that were previously carried as ``structured_fields``.
_EMBEDDED_STRUCTURED = (
    "respondent_name",
    "household_members",
    "street_address",
    "employer_name",
    "respondent_email",
)


def _load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _instrument(dataset: dict) -> QualtricsInstrument:
    return QualtricsInstrument.from_dict(dataset["qualtrics"])


def _language(record: dict) -> str:
    return str(record.get("UserLanguage", "EN")).split("-")[0].lower()


def _embedded_context(record: dict) -> dict | None:
    context = {key: record[key] for key in _EMBEDDED_STRUCTURED if key in record}
    return context or None


def _record_from_manifest(entry: dict, instrument: QualtricsInstrument) -> GeneratedRecord:
    """Rebuild a :class:`GeneratedRecord` from the checked-in Qualtrics benchmark manifest."""
    answers, embedded = row_values_from_mapping(entry, instrument)
    return GeneratedRecord(
        response_id=entry["ResponseId"],
        answers=answers,
        secrets=[
            Secret(
                kind=str(s["kind"]),
                literal=str(s["literal"]),
                expect_category=s.get("expect_category"),
                checksummed=bool(s.get("checksummed", False)),
                degradation=str(s.get("degradation", "clean")),
                mandatory=bool(s.get("mandatory", True)),
                field=s.get("field"),
            )
            for s in entry.get("secrets", [])
        ],
        must_survive=list(entry.get("must_survive", [])),
        template=str(entry.get("template", "")),
        stratum=str(entry.get("stratum", "short_open_text")),
        language=_language(entry),
        ocr_confidence=entry.get("ocr_confidence"),
        structured_fields={
            key: value
            for key, value in embedded.items()
            if key in _EMBEDDED_STRUCTURED
        },
        household_id=str(entry.get("household_id", "HH0000")),
    )


def _layout_from_spec(spec: dict, text: str, field_confidence: float | None) -> OCRFieldLayout:
    """Build an ``OCRFieldLayout``; words may omit polygon geometry (confidence-only)."""
    words = []
    for word in spec["words"]:
        polygon = None
        if word.get("polygon") is not None:
            polygon = BoundingPolygon(
                page=word["polygon"]["page"],
                points=tuple(tuple(pt) for pt in word["polygon"]["points"]),
            )
        words.append(
            OCRWord(
                text=word["text"],
                span=Span(word["start"], word["end"]),
                confidence=word["confidence"],
                polygon=polygon,
            )
        )
    return OCRFieldLayout(text=text, words=tuple(words), field_confidence=field_confidence)


class TestAPSWorkplaceCensusDirtyDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> dict:
        return _load_json("aps_workplace_census_dirty.json")

    @pytest.fixture(scope="class")
    def redactor(self, dataset: dict) -> SurveyRedactor:
        terms = [
            Term(
                term=t["term"],
                category=Category(t["category"]),
                aliases=t.get("aliases", ()),
                label=t.get("label", "internal term"),
            )
            for t in dataset.get("lexicon_terms", [])
        ]
        lexicon = Lexicon.from_terms(terms)
        return SurveyRedactor(survey_redactor(lexicon=lexicon, salt=b"0" * 32))

    def test_all_aps_records_meet_redaction_and_survival_expectations(
        self, dataset: dict, redactor: SurveyRedactor
    ) -> None:
        instrument = _instrument(dataset)
        for record in dataset["records"]:
            rec_id = record["ResponseId"]
            expectations = record["expectations"]
            answers, _embedded = row_values_from_mapping(record, instrument)
            redacted_texts: list[str] = []
            found_categories: set[str] = set()
            states: set[ReviewState] = set()

            for field_id, answer in answers.items():
                res = redactor.redact_record(
                    answer,
                    record_id=rec_id,
                    field_id=field_id,
                    language=_language(record),
                )
                redacted_texts.append(res.text)
                found_categories.update(c.value for c in res.redacted_categories)
                states.add(res.review_state)

            joined = "\n".join(redacted_texts)
            for forbidden in expectations["must_not_appear"]:
                assert forbidden not in joined, f"{rec_id}: {forbidden!r} survived in {joined!r}"
            for required in expectations["must_appear"]:
                assert required in joined, f"{rec_id}: {required!r} was over-redacted in {joined!r}"
            for cat in expectations["expected_categories"]:
                assert cat in found_categories, f"{rec_id}: missing category {cat!r} (got {found_categories})"

            expected_state = ReviewState(expectations["expected_review_state"])
            if expected_state is ReviewState.PENDING_REVIEW:
                assert ReviewState.PENDING_REVIEW in states, f"{rec_id}: expected PENDING_REVIEW"
            else:
                assert states == {ReviewState.AUTO_ACCEPTED}, f"{rec_id}: expected AUTO_ACCEPTED, got {states}"


class TestCitizenServicesClaimsAndOCRDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> dict:
        return _load_json("citizen_services_claims_and_ocr.json")

    @pytest.fixture(scope="class")
    def redactor(self) -> SurveyRedactor:
        return SurveyRedactor(survey_redactor(salt=b"0" * 32))

    def test_all_claims_and_ocr_records_meet_expectations(
        self, dataset: dict, redactor: SurveyRedactor
    ) -> None:
        instrument = _instrument(dataset)
        for record in dataset["records"]:
            rec_id = record["ResponseId"]
            expectations = record["expectations"]
            structured = _embedded_context(record)
            ocr_conf = record.get("ocr_confidence")
            ocr_layout_spec = record.get("ocr_layout")
            answers, _embedded = row_values_from_mapping(record, instrument)

            redacted_texts: list[str] = []
            found_categories: set[str] = set()
            states: set[ReviewState] = set()
            visual_boxes_count = 0
            pages_seen: set[int] = set()

            for field_id, answer in answers.items():
                layout = None
                if ocr_layout_spec and ocr_layout_spec.get("field") == field_id:
                    layout = _layout_from_spec(ocr_layout_spec, answer, ocr_conf)

                res = redactor.redact_record(
                    answer,
                    record_id=rec_id,
                    field_id=field_id,
                    structured_context=structured,
                    ocr_layout=layout,
                    ocr_confidence=ocr_conf,
                    language=_language(record),
                )
                redacted_texts.append(res.text)
                found_categories.update(c.value for c in res.redacted_categories)
                states.add(res.review_state)
                visual_boxes_count += len(res.visual_boxes)
                for box in res.visual_boxes:
                    pages_seen.update(polygon.page for polygon in box.polygons)

            joined = "\n".join(redacted_texts)
            for forbidden in expectations["must_not_appear"]:
                assert forbidden not in joined, f"{rec_id}: {forbidden!r} survived in {joined!r}"
            for required in expectations["must_appear"]:
                assert required in joined, f"{rec_id}: {required!r} was over-redacted in {joined!r}"
            for cat in expectations["expected_categories"]:
                assert cat in found_categories, f"{rec_id}: missing category {cat!r} (got {found_categories})"

            if ocr_layout_spec:
                assert visual_boxes_count >= 1, f"{rec_id}: expected visual redaction bounding boxes"
                expected_pages = set(ocr_layout_spec.get("expected_pages", ()))
                if expected_pages:
                    assert pages_seen == expected_pages, (
                        f"{rec_id}: expected polygons on pages {expected_pages}, saw {pages_seen}"
                    )

            expected_state = ReviewState(expectations["expected_review_state"])
            if expected_state is ReviewState.PENDING_REVIEW:
                assert ReviewState.PENDING_REVIEW in states, f"{rec_id}: expected PENDING_REVIEW"
            else:
                assert states == {ReviewState.AUTO_ACCEPTED}, f"{rec_id}: expected AUTO_ACCEPTED"


class TestCanonicalAnnotationsAndStratifiedJSON:
    def test_canonical_annotations_offset_integrity_and_detection(self) -> None:
        data = _load_json("canonical_annotations_and_edge_cases.json")
        s = SurveyRedactor(survey_redactor(salt=b"0" * 32))
        for item in data["annotations"]:
            text = item["text"]
            rec = s.redact_record(
                text,
                record_id=item["document_id"],
                field_id=item["field_id"],
                language=item["language"],
                ocr_confidence=item["ocr_confidence"],
            )
            for ent in item["entities"]:
                start, end = ent["start_char"], ent["end_char"]
                literal = text[start:end]
                assert 0 <= start < end <= len(text)
                if ent["mandatory_redact"]:
                    assert literal not in rec.text, (
                        f"{item['document_id']}: entity {literal!r} ({ent['entity_type']}) survived in {rec.text!r}"
                    )
            if item["language"] != "en":
                assert rec.review_state is ReviewState.PENDING_REVIEW
                assert any("non_english_language" in r for r in rec.review_reasons)
            if item.get("ocr_confidence") is not None and item["ocr_confidence"] < 0.8:
                assert rec.review_state is ReviewState.PENDING_REVIEW, (
                    f"{item['document_id']}: low OCR confidence must route to review"
                )
            expected_state = item.get("expect_review_state")
            if expected_state is not None:
                assert rec.review_state.value == expected_state

            reviewer_spec = item.get("reviewer_decision")
            if reviewer_spec is not None:
                decision = ReviewerDecision(
                    span=Span(reviewer_spec["start_char"], reviewer_spec["end_char"]),
                    action=ReviewAction(reviewer_spec["action"]),
                    category=Category(reviewer_spec["category"]),
                    reviewer_principal=reviewer_spec["reviewer_principal"],
                    reason_code=reviewer_spec["reason_code"],
                )
                reviewed, annotation = s.apply_review_decisions(
                    rec, [decision], field_id=item["field_id"], language=item["language"]
                )
                assert reviewed.review_state is ReviewState.REVIEWED
                assert annotation.entities, "reviewer decisions must emit canonical annotations"
                assert any(e.entity_type == "PERSON" for e in annotation.entities)

                second = s.run_second_pass(
                    reviewed, field_id=item["field_id"], language=item["language"]
                )
                assert (
                    second.review_state.value
                    == item["expect_review_state_after_second_pass"]
                ), f"{item['document_id']}: {second.review_state} (leaks: {second.residual_leaks})"
                for ent in item["entities"]:
                    if ent["mandatory_redact"]:
                        literal = text[ent["start_char"] : ent["end_char"]]
                        assert literal not in second.text

            expected_model = item.get("expect_second_pass_model")
            if expected_model is not None:
                second = s.run_second_pass(
                    rec, field_id=item["field_id"], language=item["language"]
                )
                assert second.second_pass_model_used == expected_model

    def test_stratified_benchmark_json_dataset(self) -> None:
        data = _load_json("synthetic_stratified_benchmark.json")
        assert data["count"] == 80
        assert len(data["records"]) == 80
        strata = {r["stratum"] for r in data["records"]}
        assert {
            "structured",
            "short_open_text",
            "difficult_contextual",
            "long_narrative",
            "household_grid",
            "ocr_form",
            "handwriting_scan",
            "adversarial",
        } <= strata
        kinds = {s["kind"] for r in data["records"] for s in r["secrets"]}
        assert {"tfn", "email", "phone", "cross_field_person"} <= kinds

    def test_checked_in_benchmark_passes_acceptance_gates(self) -> None:
        data = _load_json("synthetic_stratified_benchmark.json")
        instrument = _instrument(data)
        records = [_record_from_manifest(r, instrument) for r in data["records"]]
        assert any(r.ocr_confidence is not None for r in records)
        assert any(r.structured_fields for r in records)

        redactor = SurveyRedactor(survey_redactor(salt=b"0" * 32))
        metrics = evaluate_corpus(records, redactor)
        gates = metrics.verify_gates()
        assert gates["passed"] is True, metrics.as_dict()
        assert gates["mandatory_direct_id_recall"] >= 0.995
        assert metrics.overall_recall >= 0.96
        assert metrics.over_redaction_rate <= 0.005
