"""Edge-case robustness tests: stale reviewer spans, union bridging, non-string
cells, negative masking parameters, out-of-range folded spans, OCR uncertainty
floors, and backend reuse."""

from __future__ import annotations

import random

import pytest

from gov_survey_redact import (
    BoundingPolygon,
    Category,
    Finding,
    FoldedText,
    OCRFieldLayout,
    OCRWord,
    Policy,
    RedactionError,
    Redactor,
    ReviewAction,
    ReviewerDecision,
    ReviewState,
    Span,
    Stage,
    SurveyRedactor,
    clamp_span,
    iban_check,
    is_centrelink_crn,
    is_passport,
    partial_mask,
    resolve_overlaps,
    survey_redactor,
)
from gov_survey_redact.generate import (
    valid_crn,
    valid_drivers_licence,
    valid_iban,
    valid_ipv6,
    valid_mac,
    valid_medicare_enrolment,
    valid_passport,
    valid_street,
)
from gov_survey_redact.spark import select_gold_columns


def _finding(start: int, end: int, detector: str, *, stage: Stage = Stage.DETERMINISTIC,
             category: Category = Category.NAME, confidence: float = 0.9,
             hard: bool = False) -> Finding:
    return Finding(
        span=Span(start, end),
        category=category,
        detector=detector,
        stage=stage,
        confidence=confidence,
        metadata=(("hard_constraint", True),) if hard else (),
    )


class TestOverlapResolutionEdges:
    def test_union_bridging_candidate_merges_all_touched_spans(self) -> None:
        """A short span straddling two kept spans must not leave overlapping output."""
        text = "abcdefghijklmnopqrstuvwxyz"
        left = _finding(0, 10, "left")
        right = _finding(12, 22, "right")
        bridge = _finding(8, 14, "bridge", stage=Stage.NLP, confidence=0.8)
        merged = resolve_overlaps([left, right, bridge], {"left": 0, "right": 1}, mode="union", text=text)
        spans = [(f.span.start, f.span.end) for f in merged]
        assert len(merged) == 1
        assert spans[0] == (0, 22)
        assert merged[0].value == text[0:22]

    def test_union_merge_never_expands_across_hard_constraints(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz"
        left = _finding(0, 10, "left")
        locked = _finding(12, 22, "human", hard=True)
        bridge = _finding(8, 14, "bridge", stage=Stage.NLP, confidence=0.8)
        merged = resolve_overlaps(
            [left, locked, bridge], {"left": 0, "human": 1}, mode="union", text=text
        )
        for i, first in enumerate(merged):
            for second in merged[i + 1 :]:
                assert not first.span.overlaps(second.span)
        hard = [f for f in merged if f.meta().get("hard_constraint")]
        assert [f.span for f in hard] == [Span(12, 22)]

    def test_zero_length_findings_are_discarded(self) -> None:
        text = "hello"
        empty = _finding(2, 2, "empty")
        kept = resolve_overlaps([empty], {}, mode="union", text=text)
        assert kept == []

    def test_union_mode_is_deterministic_under_reordering(self) -> None:
        findings = [
            _finding(0, 6, "a", category=Category.NAME),
            _finding(4, 12, "b", category=Category.ORGANISATION_NAME),
            _finding(10, 16, "c", stage=Stage.NLP, category=Category.LOCATION),
        ]
        forward = resolve_overlaps(findings, {"a": 0, "b": 1}, mode="union", text="x" * 20)
        backward = resolve_overlaps(list(reversed(findings)), {"a": 0, "b": 1}, mode="union", text="x" * 20)
        assert [(f.span.start, f.span.end) for f in forward] == [
            (f.span.start, f.span.end) for f in backward
        ]


class TestReviewerSpanEdges:
    def test_clamp_span_bounds(self) -> None:
        assert clamp_span(Span(5, 20), 10) == Span(5, 10)
        assert clamp_span(Span(0, 3), 10) == Span(0, 3)
        assert clamp_span(Span(50, 60), 10) == Span(10, 10)
        assert clamp_span(Span(5, 20), 10).length == 5
        # A span is itself validated at construction time.
        with pytest.raises(ValueError):
            Span(8, 2)

    def test_out_of_range_approval_does_not_destroy_text(self) -> None:
        s = SurveyRedactor(survey_redactor(salt=b"0" * 32))
        original = "My manager Fiona was there."
        rec = s.redact_record(original, record_id="R-EDGE")
        decision = ReviewerDecision(
            span=Span(5, 5000), action=ReviewAction.APPROVE, category=Category.NAME
        )
        reviewed, _ = s.apply_review_decisions(rec, [decision])
        second = s.run_second_pass(reviewed)
        # Fail-safe direction: over-masking is acceptable, silent truncation is not.
        assert len(second.text) >= len(original)
        assert second.text.count(" ") == original.count(" ")
        assert second.text.startswith("My ")

    def test_zero_length_and_reversed_spans_are_ignored(self) -> None:
        s = SurveyRedactor(survey_redactor(salt=b"0" * 32))
        text = "The Migration directorate is busy."
        rec = s.redact_record(text, record_id="R-ZERO")
        decisions = [
            ReviewerDecision(Span(4, 4), ReviewAction.APPROVE, Category.ORGANISATION_NAME),
            ReviewerDecision(Span(len(text), len(text)), ReviewAction.APPROVE, Category.NAME),
            ReviewerDecision(Span(0, 0), ReviewAction.REMOVE, Category.NAME),
        ]
        reviewed, annotation = s.apply_review_decisions(rec, decisions)
        assert reviewed.review_state is ReviewState.REVIEWED
        assert all(e.end_char > e.start_char for e in annotation.entities)
        second = s.run_second_pass(reviewed)
        assert "[ORGANISATION NAME]Migration" not in second.text

    def test_remove_decision_with_stale_span_is_ignored(self) -> None:
        s = SurveyRedactor(survey_redactor(salt=b"0" * 32))
        rec = s.redact_record("Contact jane.doe@example.com.au for details.", record_id="R-REM")
        stale = ReviewerDecision(Span(900, 999), ReviewAction.REMOVE, Category.EMAIL)
        reviewed, _ = s.apply_review_decisions(rec, [stale])
        second = s.run_second_pass(reviewed)
        # The email was detected independently and must stay redacted.
        assert "jane.doe@example.com.au" not in second.text


class TestInputCoercionEdges:
    def test_redact_record_accepts_non_string_cells(self) -> None:
        s = SurveyRedactor(survey_redactor())
        assert s.redact_record(12345).text == "12345"
        assert s.redact_record(None).text == ""
        assert s.redact_record("   ").review_state is ReviewState.AUTO_ACCEPTED

    def test_redactor_still_rejects_non_strings(self) -> None:
        with pytest.raises(RedactionError):
            Redactor(Policy()).redact(5)  # type: ignore[arg-type]

    def test_redact_many_pads_missing_ids(self) -> None:
        s = SurveyRedactor(survey_redactor())
        out = s.redact_many(["first text", "second text"], record_ids=["R-1"])
        assert [r.record_id for r in out] == ["R-1", None]
        assert len(out) == 2
        assert out[1].text == "second text"


class TestMaskingEdges:
    def test_partial_mask_negative_keep_counts_are_clamped(self) -> None:
        assert partial_mask("0412345678", keep_prefix=-2, keep_suffix=-1) == "**********"
        assert partial_mask("0412345678", keep_prefix=0, keep_suffix=4) == "******5678"
        assert partial_mask("123", keep_prefix=2, keep_suffix=2) == "***"

    def test_folded_text_out_of_range_spans(self) -> None:
        folded = FoldedText.build("hello world")
        assert folded.original_span(0, 5) == (0, 5)
        assert folded.original_span(100, 200) == (100, 200)
        assert folded.original_span(5, 4) == (5, 4)
        empty = FoldedText.build("!!!")
        assert empty.original_span(0, 2) == (0, 2)


class TestOcrUncertaintyFloor:
    def test_failed_ocr_candidate_never_leaks_even_above_min_confidence(self) -> None:
        s = SurveyRedactor(survey_redactor(ocr_mode=True, min_confidence=0.95))
        rec = s.redact_record("My TFN: 123 456 78O", ocr_confidence=0.5)
        assert "123 456 78O" not in rec.text
        assert rec.review_state is ReviewState.PENDING_REVIEW
        assert "ocr_checksum_failure" in rec.review_reasons

    def test_second_pass_preserves_ocr_tolerant_profile(self) -> None:
        """A low-confidence OCR record must not be un-masked by the second pass."""
        s = SurveyRedactor(survey_redactor(salt=b"0" * 32))
        rec = s.redact_record(
            "TFN: 123 456 78O and second scanned form TFN: 123 456 783.",
            ocr_confidence=0.64,
        )
        assert rec.used_ocr_mode is True
        assert "123 456 783" not in rec.text
        second = s.run_second_pass(rec)
        assert "123 456 783" not in second.text
        assert "123 456 78O" not in second.text
        assert second.residual_leaks == ()

    def test_second_pass_without_ocr_mode_keeps_standard_profile(self) -> None:
        s = SurveyRedactor(survey_redactor(salt=b"0" * 32))
        rec = s.redact_record("My TFN is 123 456 782.", ocr_confidence=0.99)
        assert rec.used_ocr_mode is False
        second = s.run_second_pass(rec)
        assert "123 456 782" not in second.text


class TestBackendReuse:
    def test_nlp_backend_is_shared_across_policy_variants(self) -> None:
        base = survey_redactor(nlp_backend="gazetteer")
        ocr_variant = base.with_policy(ocr_mode=True)
        # A second-pass specialist variant reuses the cached backend registry.
        _ = base.with_policy(
            nlp_enabled=True, nlp_backend="mdeberta-v3-base", overlap_mode="union"
        )
        assert base._nlp_backend is ocr_variant._nlp_backend

    def test_second_pass_redactor_is_cached(self) -> None:
        s = SurveyRedactor(survey_redactor(salt=b"0" * 32))
        rec = s.redact_record("My manager Fiona was there.")
        first = s.run_second_pass(rec)
        second = s.run_second_pass(rec)
        assert first.second_pass_model_used == second.second_pass_model_used
        assert first.text == second.text


class TestSparkHelpers:
    def test_select_gold_columns_deduplicates(self) -> None:
        columns = select_gold_columns(["Q1_risk"], ["Q1"])
        assert columns.count("Q1_risk") == 1

    def test_redactor_spec_empty_generalisation_is_honoured(self) -> None:
        from gov_survey_redact.spark import RedactorSpec, build_worker_redactor

        text = "I am 45 years old and work in the integrity unit."
        default_rec = build_worker_redactor(
            RedactorSpec.build(nlp_backend="gazetteer")
        ).redact_record(text)
        assert "45-54" in default_rec.text

        no_generalise = build_worker_redactor(
            RedactorSpec.build(
                nlp_backend="gazetteer", generalise_categories=frozenset()
            )
        ).redact_record(text)
        assert "45-54" not in no_generalise.text


class TestCliAndLayoutEdges:
    def test_cli_ndjson_format_emits_one_json_object_per_input(self, monkeypatch, capsys):
        import io

        from gov_survey_redact.cli import main

        monkeypatch.setattr("sys.stdin", io.StringIO("My TFN is 123 456 782."))
        exit_code = main(["--format", "ndjson"])
        assert exit_code == 0
        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line.strip()]
        assert len(lines) == 1
        import json as _json

        payload = _json.loads(lines[0])
        assert "123 456 782" not in payload["text"]
        assert payload["findings"]

    def test_stale_ocr_layout_is_not_mapped_to_wrong_offsets(self) -> None:
        stale_text = "TFN 123 456 782"
        layout = OCRFieldLayout(
            text=stale_text,
            words=(
                OCRWord(
                    text="123 456 782",
                    span=Span(4, 15),
                    confidence=0.6,
                    polygon=BoundingPolygon(page=1, points=((0.0, 0.0), (1.0, 1.0))),
                ),
            ),
        )
        s = SurveyRedactor(survey_redactor(salt=b"0" * 32))
        rec = s.redact_record("A completely different version of this field.", ocr_layout=layout)
        assert rec.visual_boxes == ()


class TestGeneratorBuilders:
    """The synthetic builders must produce values their detectors accept."""

    def test_identifier_builders_are_valid(self) -> None:
        rng = random.Random(7)
        assert iban_check(valid_iban(rng))
        assert is_passport(valid_passport(rng))
        assert is_centrelink_crn(valid_crn(rng))
        assert len(valid_drivers_licence(rng)) == 8
        assert valid_medicare_enrolment(rng).count(" ") == 3
        mac = valid_mac(rng)
        assert len(mac.split(":")) == 6
        ipv6 = valid_ipv6(rng)
        assert ipv6.startswith("2001:0db8:")
        street = valid_street(rng)
        assert street.split()[1][0].isupper()

    def test_generated_corpus_covers_new_identifier_kinds(self) -> None:
        from gov_survey_redact import SurveyGenerator

        corpus = SurveyGenerator(seed=99).corpus(300)
        kinds = {s.kind for record in corpus for s in record.secrets}
        assert {"passport", "drivers_licence", "centrelink_crn", "iban", "ipv6", "mac_address"} <= kinds
