"""Tests for Australian identifiers, checksums, and OCR-tolerant checksum routing."""

from __future__ import annotations

from gov_survey_redact import (
    Category,
    Policy,
    Redactor,
    ReviewState,
    SurveyRedactor,
    iban_check,
    is_abn,
    is_acn,
    is_bsb,
    is_centrelink_crn,
    is_credit_card,
    is_medicare,
    is_passport,
    is_tfn,
    survey_redactor,
)


class TestChecksumsAndDirectIdentifiers:
    def test_tfn_validation_and_redaction(self) -> None:
        assert is_tfn("123 456 782")
        assert not is_tfn("123 456 783")
        r = Redactor(Policy())
        assert r.redact_text("My TFN is 123 456 782.") == "My TFN is *** *** ***."
        # Unlabelled invalid TFN in digital mode is ignored
        assert r.redact_text("Order 123 456 783 arrived.") == "Order 123 456 783 arrived."

    def test_abn_and_acn_validation(self) -> None:
        assert is_abn("51 824 753 556")
        assert not is_abn("53 000 001 000")
        assert is_acn("004 085 616")
        assert not is_acn("004 085 617")
        r = Redactor(Policy())
        assert "51 824 753 556" not in r.redact_text("ABN 51 824 753 556")
        assert "004 085 616" not in r.redact_text("ACN 004 085 616")

    def test_medicare_and_crn_and_employee_id(self) -> None:
        assert is_medicare("2123 45670 1") or is_medicare("2 1234 5678 2")
        assert is_centrelink_crn("234 567 890A")
        assert not is_centrelink_crn("234 567 890E")
        r = Redactor(Policy())
        redacted = r.redact_text("CRN: 234 567 890A and AGS number 12345678")
        assert "234 567 890A" not in redacted
        assert "12345678" not in redacted

    def test_bsb_card_iban_passport_and_licence(self) -> None:
        assert is_bsb("062-000")
        assert not is_bsb("002-000")
        assert is_credit_card("4111 1111 1111 1111")
        assert iban_check("GB82 WEST 1234 5698 7654 32")
        assert is_passport("N1234567")
        assert is_passport("PA1234567")
        assert not is_passport("E1234567")
        r = Redactor(Policy())
        text = "BSB 062-000, Passport PA1234567, Driver licence: 12345678"
        out = r.redact_text(text)
        assert "062-000" not in out
        assert "PA1234567" not in out
        assert "12345678" not in out

    def test_email_spaced_and_fullwidth_phone_and_ipv4_guards(self) -> None:
        r = Redactor(Policy())
        spaced = "Contact me at j a n e . d o e @ e x a m p l e . c o m . a u today."
        assert "j a n e" not in r.redact_text(spaced)

        fullwidth_phone = "Call ０４１２ ３４５ ６７８ urgently."
        assert "０４１２" not in r.redact_text(fullwidth_phone)

        # Version strings and amounts must not be falsely redacted as IPv4 or phone
        benign = "Release 2.4.0.1 and v1.2.3.4 cost $1,250.00 on 12/03/2024."
        findings = r.scan(benign)
        cats = {f.category for f in findings}
        assert Category.IPV4 not in cats
        assert Category.PHONE not in cats

    def test_ocr_corrupted_identifier_survives_checksum_and_routes_to_review(self) -> None:
        """Per spec: an OCR-corrupted government ID must not be treated as safe when its checksum fails."""
        s = SurveyRedactor(survey_redactor(ocr_mode=True))
        # Valid TFN "123 456 782" corrupted by OCR to "123 456 78O" or "123 456 783" with low OCR confidence
        rec1 = s.redact_record("My TFN: 123 456 78O", ocr_confidence=0.68)
        assert "123 456 78O" not in rec1.text
        assert rec1.review_state is ReviewState.PENDING_REVIEW
        assert any("ocr" in reason for reason in rec1.review_reasons)

        rec2 = s.redact_record("TFN: 123 456 783", ocr_confidence=0.65)
        assert "123 456 783" not in rec2.text
        assert rec2.review_state is ReviewState.PENDING_REVIEW
        assert "ocr_checksum_failure" in rec2.review_reasons
