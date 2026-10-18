"""Tests for the asymmetric two-pass hybrid pipeline, cross-field propagation,
schema pruning, union-of-spans, reviewer hard constraints, OCR bounding boxes,
MLOps audit logs, and Databricks helpers."""

from __future__ import annotations

import json

from gov_survey_redact import (
    BoundingPolygon,
    Category,
    DisclosurePolicy,
    FieldDisposition,
    Finding,
    OCRFieldLayout,
    OCRWord,
    ReviewAction,
    ReviewerDecision,
    ReviewState,
    Span,
    Stage,
    SurveyGenerator,
    SurveyRedactor,
    SurveySchema,
    evaluate_corpus,
    resolve_overlaps,
    survey_redactor,
)
from gov_survey_redact.spark import (
    RedactorSpec,
    estimate_databricks_cost,
    filter_cdf_reviewed_records,
    select_gold_columns,
    summarise_batch,
)


class TestTwoPassHybridWorkflow:
    def test_cross_field_deterministic_propagation(self) -> None:
        """Structured respondent/household columns propagate to free-text answers before NER."""
        structured = {
            "respondent_name": "Marisol Halvorsen",
            "household_members": "Marisol Halvorsen; Kiran Halvorsen",
            "street_address": "42 Macquarie Street",
            "employer_name": "Northline Services",
        }
        s = SurveyRedactor(survey_redactor())
        text = "Please call Marisol or my partner Kiran Halvorsen at 42 Macquarie Street regarding Northline Services."
        rec = s.redact_record(text, structured_context=structured)
        assert "Marisol" not in rec.text
        assert "Kiran Halvorsen" not in rec.text
        assert "42 Macquarie Street" not in rec.text
        assert "Northline Services" not in rec.text
        detectors = {f.detector for f in rec.findings}
        assert "cross_field_dictionary" in detectors

    def test_schema_pruning_and_whole_column_governance(self) -> None:
        schema = SurveySchema(
            question_prompts={"Q1": "Please describe your workplace experience."}
        )
        assert schema.policy_for("ResponseID").disposition is FieldDisposition.PASSTHROUGH_METADATA
        assert schema.policy_for("likert_score", 4).disposition is FieldDisposition.PASSTHROUGH_NUMERIC

        dob_policy = schema.policy_for("date_of_birth")
        assert dob_policy.disposition is FieldDisposition.WHOLE_COLUMN_GENERALISE_DOB
        assert schema.apply_whole_column(dob_policy, "14/08/1982") == "1982"

        id_policy = schema.policy_for("respondent_id")
        assert id_policy.disposition is FieldDisposition.WHOLE_COLUMN_TOKENISE
        tok1 = schema.apply_whole_column(id_policy, "RESP-9912", salt=b"0" * 32)
        tok2 = schema.apply_whole_column(id_policy, "RESP-9912", salt=b"0" * 32)
        assert tok1 == tok2 and str(tok1).startswith("reference_id_")

        q1_policy = schema.policy_for("Q1", "Great workplace")
        assert q1_policy.disposition is FieldDisposition.FREE_TEXT_HYBRID
        assert q1_policy.question_text == "Please describe your workplace experience."

    def test_union_of_spans_prevents_partial_name_leak(self) -> None:
        text = "Spoke with John Smith yesterday."
        f1 = Finding(
            span=Span(11, 15),
            category=Category.NAME,
            detector="det_a",
            stage=Stage.DETERMINISTIC,
            confidence=0.9,
            value="John",
        )
        f2 = Finding(
            span=Span(11, 21),
            category=Category.NAME,
            detector="nlp_b",
            stage=Stage.NLP,
            confidence=0.8,
            value="John Smith",
        )
        f3 = Finding(
            span=Span(16, 21),
            category=Category.NAME,
            detector="det_c",
            stage=Stage.DETERMINISTIC,
            confidence=0.85,
            value="Smith",
        )
        # Even if two partial spans overlap ("John " and " Smith"), union mode merges them
        f_left = Finding(Span(11, 17), Category.NAME, "d1", Stage.DETERMINISTIC, 0.9, "John S")
        f_right = Finding(Span(16, 21), Category.NAME, "d2", Stage.NLP, 0.85, "Smith")
        merged = resolve_overlaps([f_left, f_right], {"d1": 0, "d2": 1}, mode="union", text=text)
        assert len(merged) == 1
        assert merged[0].span == Span(11, 21)
        assert merged[0].value == "John Smith"

        # And longest span wins between f1, f2, f3
        resolved = resolve_overlaps([f1, f2, f3], {}, mode="union", text=text)
        assert len(resolved) == 1
        assert resolved[0].span == Span(11, 21)

    def test_reviewer_hard_constraints_and_second_pass_state_machine(self) -> None:
        """Pass 1 -> PENDING_REVIEW -> apply_review_decisions -> Pass 2 -> RELEASED."""
        s = SurveyRedactor(survey_redactor(salt=b"0" * 32))
        text = "My supervisor Priyadarshini told me not to report the incident at Acme Pty Ltd."
        first = s.redact_record(text, record_id="R-101", field_id="Q3")
        # High risk + named org + third party routes to PENDING_REVIEW
        assert first.review_state is ReviewState.PENDING_REVIEW

        # Reviewer explicitly extends/approves a custom span and removes any false positive
        name_start = text.index("Priyadarshini")
        name_end = name_start + len("Priyadarshini")
        decisions = [
            ReviewerDecision(
                span=Span(name_start, name_end),
                action=ReviewAction.APPROVE,
                category=Category.NAME,
                reviewer_principal="alice.reviewer@aps.gov.au",
                reason_code="CONFIRMED_SUPERVISOR_NAME",
            )
        ]
        reviewed, annotation = s.apply_review_decisions(first, decisions, field_id="Q3")
        assert reviewed.review_state is ReviewState.REVIEWED
        assert annotation.adjudication_status == "FINAL"
        assert any(
            e.start_char == name_start and e.end_char == name_end and e.entity_type == "PERSON"
            for e in annotation.entities
        )

        # Run Pass 2: must preserve the human-approved span as a hard constraint,
        # select mdeberta-v3-base (or modernbert-base for >512 words), and transition to RELEASED
        second = s.run_second_pass(reviewed, field_id="Q3")
        assert second.review_state is ReviewState.RELEASED
        assert second.second_pass_model_used == "mdeberta-v3-base"
        assert "Priyadarshini" not in second.text
        assert "Acme Pty Ltd" not in second.text

        # Verify privacy-safe audit log never contains raw PII text
        audit_blob = json.dumps([e.as_dict() for e in second.audit_entries])
        assert "Priyadarshini" not in audit_blob
        assert "Acme Pty Ltd" not in audit_blob
        assert "alice.reviewer@aps.gov.au" in audit_blob

    def test_long_narrative_routes_second_pass_to_modernbert(self) -> None:
        s = SurveyRedactor(survey_redactor(), long_narrative_tokens=50)
        long_text = ("The general staff forum discussed the roster changes at length. " * 10) + " Email: jane.doe@example.com.au"
        first = s.redact_record(long_text, record_id="R-LONG")
        second = s.run_second_pass(first, language="en")
        assert second.second_pass_model_used == "modernbert-base"
        assert "jane.doe@example.com.au" not in second.text

    def test_ocr_geometry_maps_character_spans_to_bounding_polygons(self) -> None:
        text = "TFN 123 456 782"
        poly = BoundingPolygon(page=1, points=((10.0, 20.0), (80.0, 20.0), (80.0, 35.0), (10.0, 35.0)))
        layout = OCRFieldLayout(
            text=text,
            words=(
                OCRWord(text="TFN", span=Span(0, 3), confidence=0.99),
                OCRWord(text="123 456 782", span=Span(4, 15), confidence=0.72, polygon=poly),
            ),
        )
        s = SurveyRedactor(survey_redactor())
        rec = s.redact_record(text, ocr_layout=layout)
        assert rec.review_state is ReviewState.PENDING_REVIEW
        assert len(rec.visual_boxes) == 1
        assert rec.visual_boxes[0].polygons[0].bounding_box == (10.0, 20.0, 80.0, 35.0)
        assert rec.visual_boxes[0].min_ocr_confidence == 0.72

    def test_residual_disclosure_control_and_idempotence(self) -> None:
        s = SurveyRedactor(
            survey_redactor(salt=b"0" * 32),
            residual_policy=DisclosurePolicy(population=3000),
        )
        rec = s.redact_record("My employee number is 0041827.")
        assert "0041827" not in rec.text

        # Idempotence: second pass over already-redacted text changes nothing
        spec = RedactorSpec.build(salt="0" * 32)
        sample = [
            "I am a Grade 4 nurse at Acme Pty Ltd with 18 years of service, aged 45, TFN 123 456 782.",
            "I am one of three EL2s in the Migration directorate.",
        ]
        pass1 = summarise_batch(spec, sample)["redacted"]
        pass2 = summarise_batch(spec, pass1)["redacted"]
        assert pass1 == pass2

    def test_databricks_cost_model_and_cdf_and_gold_helpers(self) -> None:
        cpu_est = estimate_databricks_cost(sku="Standard_D8as_v5", nodes=1, hours=40.0)
        gpu_est = estimate_databricks_cost(sku="Standard_NC4as_T4_v3", nodes=1, hours=40.0)
        assert round(cpu_est.total_cost_usd, 2) == 29.28
        assert round(gpu_est.total_cost_usd, 2) == 39.36

        cdf_rows = [
            {"ResponseID": "1", "_change_type": "update_postimage", "review_state": "REVIEWED"},
            {"ResponseID": "2", "_change_type": "update_preimage", "review_state": "REVIEWED"},
            {"ResponseID": "3", "_change_type": "insert", "review_state": "PENDING_REVIEW"},
        ]
        filtered = filter_cdf_reviewed_records(cdf_rows)
        assert [r["ResponseID"] for r in filtered] == ["1"]

        gold_cols = select_gold_columns(["ResponseID", "StartDate"], ["Q1", "Q2"])
        assert "Q1_redacted" in gold_cols
        assert "Q1_findings" not in gold_cols
        assert "Q1" not in gold_cols

    def test_stratified_synthetic_benchmark_and_gates(self) -> None:
        gen = SurveyGenerator(seed=42)
        corpus = gen.corpus(120)
        train, test = gen.train_test_split(corpus, test_ratio=0.25)
        assert not ({r.household_id for r in train} & {r.household_id for r in test})

        s = SurveyRedactor(survey_redactor(salt=b"0" * 32))
        metrics = evaluate_corpus(corpus, s)
        gates = metrics.verify_gates()
        assert gates["mandatory_direct_id_recall"] >= 0.995
        assert gates["over_redaction_rate"] == 0.0
        assert metrics.overall_recall >= 0.96
