"""Canonical character-offset annotation schema and privacy-safe MLflow/Unity Catalog audit trail.

Implements:
1. Canonical character-offset training/adjudication records matching the
   specification's ontology (``PERSON``, ``FACILITY``, ``EMPLOYER``, ``GOV_ID``, etc.).
2. Privacy-safe ``AuditLogEntry`` records that capture ``job_run_id``,
   ``document_id``, ``field_id``, ``model_name``, ``model_version``,
   ``rule_pack_version``, ``threshold_set_version``, ``entity_type``, ``start``,
   ``end``, ``span_sha256``, ``confidence``, ``detector_source``, ``action``,
   ``review_status``, ``reviewer_principal``, ``review_reason``,
   ``second_pass_model``, ``final_disposition``, and ``timestamp`` **without**
   copying raw PII text into logs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .types import Category, Finding

# Mapping from internal Category to the canonical ontology labels in the spec
ONTOLOGY_LABEL_MAP: dict[Category, tuple[str, str, bool]] = {
    Category.NAME: ("PERSON", "HIGH", True),
    Category.EMAIL: ("EMAIL", "HIGH", True),
    Category.PHONE: ("PHONE", "HIGH", True),
    Category.STREET_ADDRESS: ("STREET_ADDRESS", "HIGH", True),
    Category.POSTCODE: ("POSTCODE", "MODERATE", True),
    Category.STATE_POSTCODE: ("POSTCODE", "MODERATE", True),
    Category.LOCATION: ("GEO_FINE", "POLICY_DEPENDENT", False),
    Category.TFN: ("TAX_ID", "HIGH", True),
    Category.ABN: ("GOV_ID", "HIGH", True),
    Category.ACN: ("GOV_ID", "HIGH", True),
    Category.MEDICARE: ("HEALTH_ID", "HIGH", True),
    Category.MEDICARE_ENROLMENT: ("HEALTH_ID", "HIGH", True),
    Category.CENTRELINK_CRN: ("GOV_ID", "HIGH", True),
    Category.PASSPORT: ("PASSPORT", "HIGH", True),
    Category.DRIVERS_LICENCE: ("DRIVER_LICENCE", "HIGH", True),
    Category.REFERENCE_ID: ("GOV_ID", "HIGH", True),
    Category.CREDIT_CARD: ("CREDIT_CARD", "HIGH", True),
    Category.BANK_BSB: ("BANK_ACCOUNT", "HIGH", True),
    Category.BANK_ACCOUNT: ("BANK_ACCOUNT", "HIGH", True),
    Category.IBAN: ("IBAN", "HIGH", True),
    Category.IPV4: ("IP_ADDRESS", "MODERATE", True),
    Category.IPV6: ("IP_ADDRESS", "MODERATE", True),
    Category.MAC: ("MAC_ADDRESS", "MODERATE", True),
    Category.URL_CREDENTIAL: ("URL_PERSONAL", "HIGH", True),
    Category.DATE_AU: ("DOB", "POLICY_DEPENDENT", False),
    Category.DATE_ISO: ("DOB", "POLICY_DEPENDENT", False),
    Category.DATE: ("DOB", "POLICY_DEPENDENT", False),
    Category.ORGANISATION_NAME: ("EMPLOYER", "POLICY_DEPENDENT", True),
    Category.ORGANISATION: ("ORGANISATION", "POLICY_DEPENDENT", True),
    Category.INTERNAL_TERM: ("EMPLOYER", "POLICY_DEPENDENT", True),
    Category.AWARD: ("EMPLOYER", "POLICY_DEPENDENT", True),
    Category.AGE: ("EXACT_AGE", "POLICY_DEPENDENT", False),
    Category.GENDER: ("RARE_ATTRIBUTE", "POLICY_DEPENDENT", False),
    Category.JOB_TITLE: ("OCCUPATION", "POLICY_DEPENDENT", False),
    Category.TENURE: ("RARE_ATTRIBUTE", "POLICY_DEPENDENT", False),
    Category.TEAM_SIZE: ("RARE_ATTRIBUTE", "POLICY_DEPENDENT", False),
    Category.EMPLOYMENT_STATUS: ("RARE_ATTRIBUTE", "POLICY_DEPENDENT", False),
    Category.IMPLICATION: ("RARE_ATTRIBUTE", "HIGH", True),
    Category.HEALTH_DETAIL: ("HEALTH_CONDITION", "POLICY_DEPENDENT", False),
}


@dataclass(frozen=True, slots=True)
class AnnotatedEntity:
    """One canonical character-offset entity annotation."""

    start_char: int
    end_char: int
    entity_type: str
    risk_tier: str = "HIGH"
    mandatory_redact: bool = True
    annotation_source: str = "HUMAN_ADJUDICATED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
            "entity_type": self.entity_type,
            "risk_tier": self.risk_tier,
            "mandatory_redact": self.mandatory_redact,
            "annotation_source": self.annotation_source,
        }


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """Canonical training and adjudication record storing character offsets."""

    document_id: str
    field_id: str
    language: str
    text: str
    entities: tuple[AnnotatedEntity, ...]
    ocr_confidence: float | None = None
    template_id: str = "gov-survey-v1"
    adjudication_status: str = "FINAL"

    @classmethod
    def from_findings(
        cls,
        *,
        document_id: str,
        field_id: str,
        text: str,
        findings: Sequence[Finding],
        language: str = "en",
        ocr_confidence: float | None = None,
        template_id: str = "gov-survey-v1",
        adjudication_status: str = "PROPOSED",
        annotation_source: str = "MODEL_PROPOSED",
    ) -> AnnotationRecord:
        entities: list[AnnotatedEntity] = []
        for f in findings:
            ent_type, risk_tier, mandatory = ONTOLOGY_LABEL_MAP.get(
                f.category, (f.category.value.upper(), "HIGH", True)
            )
            entities.append(
                AnnotatedEntity(
                    start_char=f.span.start,
                    end_char=f.span.end,
                    entity_type=ent_type,
                    risk_tier=risk_tier,
                    mandatory_redact=mandatory,
                    annotation_source=annotation_source,
                )
            )
        return cls(
            document_id=document_id,
            field_id=field_id,
            language=language,
            text=text,
            entities=tuple(entities),
            ocr_confidence=ocr_confidence,
            template_id=template_id,
            adjudication_status=adjudication_status,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "field_id": self.field_id,
            "language": self.language,
            "text": self.text,
            "entities": [e.as_dict() for e in self.entities],
            "ocr_confidence": self.ocr_confidence,
            "template_id": self.template_id,
            "adjudication_status": self.adjudication_status,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class LineageVersionSet:
    """Frozen MLOps release chain metadata for reproducible redaction runs."""

    data_ontology_version: str = "gov-pii-ontology-v1.0"
    annotation_dataset_version: str = "au-gov-survey-synth-v1.0"
    model_name: str = "microsoft/deberta-v3-small"
    model_version: str = "v1.0.0"
    second_pass_model_name: str = "microsoft/mdeberta-v3-base"
    second_pass_model_version: str = "v1.0.0"
    rule_pack_version: str = "AU_GOV_RULES_V1.0"
    threshold_set_version: str = "THRESHOLDS_V1.0"

    def as_dict(self) -> dict[str, str]:
        return {
            "data_ontology_version": self.data_ontology_version,
            "annotation_dataset_version": self.annotation_dataset_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "second_pass_model_name": self.second_pass_model_name,
            "second_pass_model_version": self.second_pass_model_version,
            "rule_pack_version": self.rule_pack_version,
            "threshold_set_version": self.threshold_set_version,
        }


@dataclass(frozen=True, slots=True)
class AuditLogEntry:
    """Privacy-safe inference and review audit event (never stores raw span text)."""

    job_run_id: str
    document_id: str
    field_id: str
    model_name: str
    model_version: str
    rule_pack_version: str
    threshold_set_version: str
    entity_type: str
    start: int
    end: int
    span_sha256: str
    confidence: float
    detector_source: str
    action: str
    review_status: str
    reviewer_principal: str | None = None
    review_reason: str | None = None
    second_pass_model: str | None = None
    final_disposition: str = "REDACTED"
    timestamp: str = ""

    @classmethod
    def from_finding(
        cls,
        finding: Finding,
        *,
        job_run_id: str,
        document_id: str,
        field_id: str,
        lineage: LineageVersionSet,
        action: str,
        review_status: str,
        salt: bytes | None = None,
        reviewer_principal: str | None = None,
        review_reason: str | None = None,
        second_pass_model: str | None = None,
        final_disposition: str = "REDACTED",
    ) -> AuditLogEntry:
        raw_val = (finding.value or "").encode("utf-8")
        if salt:
            digest = hmac.new(salt, raw_val, hashlib.sha256).hexdigest()[:16]
        else:
            digest = hashlib.sha256(raw_val).hexdigest()[:16]
        ent_type, _, _ = ONTOLOGY_LABEL_MAP.get(
            finding.category, (finding.category.value.upper(), "HIGH", True)
        )
        recognizer = finding.meta().get("recognizer", finding.detector)
        return cls(
            job_run_id=job_run_id,
            document_id=document_id,
            field_id=field_id,
            model_name=lineage.model_name,
            model_version=lineage.model_version,
            rule_pack_version=lineage.rule_pack_version,
            threshold_set_version=lineage.threshold_set_version,
            entity_type=ent_type,
            start=finding.span.start,
            end=finding.span.end,
            span_sha256=digest,
            confidence=round(finding.confidence, 4),
            detector_source=str(recognizer),
            action=action,
            review_status=review_status,
            reviewer_principal=reviewer_principal,
            review_reason=review_reason,
            second_pass_model=second_pass_model,
            final_disposition=final_disposition,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_run_id": self.job_run_id,
            "document_id": self.document_id,
            "field_id": self.field_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "rule_pack_version": self.rule_pack_version,
            "threshold_set_version": self.threshold_set_version,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "span_sha256": self.span_sha256,
            "confidence": self.confidence,
            "detector_source": self.detector_source,
            "action": self.action,
            "review_status": self.review_status,
            "reviewer_principal": self.reviewer_principal,
            "review_reason": self.review_reason,
            "second_pass_model": self.second_pass_model,
            "final_disposition": self.final_disposition,
            "timestamp": self.timestamp,
        }


__all__ = [
    "ONTOLOGY_LABEL_MAP",
    "AnnotatedEntity",
    "AnnotationRecord",
    "AuditLogEntry",
    "LineageVersionSet",
]
