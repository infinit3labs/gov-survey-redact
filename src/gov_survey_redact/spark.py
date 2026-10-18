"""Azure Databricks & PySpark integration for the two-pass hybrid redaction pipeline.

Supports:
* Frozen, JSON-serialisable :class:`RedactorSpec` with per-executor caching
  (:data:`_WORKER_CACHE`) so regexes, lexicons, and encoder models load once per
  Spark worker/partition rather than per row.
* Dynamic length bucketing inside :func:`summarise_batch` so short survey answers
  are grouped together.
* ``pandas_udf`` and ``mapInPandas`` batch execution (:func:`redact_frame`,
  :func:`redact_dataframe`).
* Bronze -> Delta Review Queue -> Second Pass (via Delta Change Data Feed) ->
  Gold column pruning (:func:`split_review_queue`, :func:`filter_cdf_for_second_pass`,
  :func:`select_gold_columns`).
* Transparent Azure Databricks VM + DBU + OCR cost model
  (:func:`estimate_databricks_cost`, covering ``Standard_D8as_v5``,
  ``Standard_NC4as_T4_v3``, and ``Standard_NC8as_T4_v3``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .detectors.nlp import bucket_by_length
from .lexicon import Lexicon
from .records import ReviewState, SurveyRedactor
from .risk import Audience, RiskBand
from .survey import DEFAULT_GENERALISED
from .types import Category

REDACTED_SUFFIX = "_redacted"
FINDINGS_SUFFIX = "_findings"
RISK_SUFFIX = "_risk"
RISK_BAND_SUFFIX = "_risk_band"
REVIEW_SUFFIX = "_needs_review"
REVIEW_STATE_SUFFIX = "_review_state"

_STRUCT_FIELDS = {
    "redacted": REDACTED_SUFFIX,
    "findings": FINDINGS_SUFFIX,
    "risk": RISK_SUFFIX,
    "risk_band": RISK_BAND_SUFFIX,
    "needs_review": REVIEW_SUFFIX,
    "review_state": REVIEW_STATE_SUFFIX,
}

_WORKER_CACHE: dict[str, SurveyRedactor] = {}


@dataclass(frozen=True, slots=True)
class DatabricksSKU:
    """Hardware and illustrative Australia East pricing for a Databricks compute SKU."""

    name: str
    vcpus: int
    ram_gb: int
    gpu_name: str | None
    gpu_vram_gb: int
    vm_hourly_usd: float
    illustrative_dbu_per_hour: float


DATABRICKS_SKUS: dict[str, DatabricksSKU] = {
    "Standard_D8as_v5": DatabricksSKU(
        name="Standard_D8as_v5",
        vcpus=8,
        ram_gb=32,
        gpu_name=None,
        gpu_vram_gb=0,
        vm_hourly_usd=0.432,
        illustrative_dbu_per_hour=1.0,
    ),
    "Standard_NC4as_T4_v3": DatabricksSKU(
        name="Standard_NC4as_T4_v3",
        vcpus=4,
        ram_gb=28,
        gpu_name="NVIDIA T4",
        gpu_vram_gb=16,
        vm_hourly_usd=0.684,
        illustrative_dbu_per_hour=1.0,
    ),
    "Standard_NC8as_T4_v3": DatabricksSKU(
        name="Standard_NC8as_T4_v3",
        vcpus=8,
        ram_gb=56,
        gpu_name="NVIDIA T4",
        gpu_vram_gb=16,
        vm_hourly_usd=1.368,
        illustrative_dbu_per_hour=2.0,
    ),
}


@dataclass(frozen=True, slots=True)
class DatabricksCostEstimate:
    """Transparent cost breakdown separating text-field compute from scanned-page OCR."""

    sku: str
    nodes: int
    node_hours: float
    vm_cost_usd: float
    dbu_cost_usd: float
    ocr_cost_usd: float
    total_cost_usd: float
    cost_per_million_fields_usd: float
    cost_per_thousand_ocr_pages_usd: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "nodes": self.nodes,
            "node_hours": round(self.node_hours, 3),
            "vm_cost_usd": round(self.vm_cost_usd, 2),
            "dbu_cost_usd": round(self.dbu_cost_usd, 2),
            "ocr_cost_usd": round(self.ocr_cost_usd, 2),
            "total_cost_usd": round(self.total_cost_usd, 2),
            "cost_per_million_fields_usd": round(self.cost_per_million_fields_usd, 2),
            "cost_per_thousand_ocr_pages_usd": round(self.cost_per_thousand_ocr_pages_usd, 2),
        }


def estimate_databricks_cost(
    *,
    sku: str = "Standard_NC4as_T4_v3",
    nodes: int = 1,
    hours: float = 1.0,
    dbu_rate_usd: float = 0.30,
    actual_dbus: float | None = None,
    text_fields_processed: int = 1_000_000,
    scanned_pages_processed: int = 0,
    ocr_per_thousand_pages_usd: float = 1.50,
) -> DatabricksCostEstimate:
    """Compute the job cost model:

    ``job_cost = VM_hourly_cost * node_hours + actual_DBUs * contract_DBU_price + OCR_cost``
    """
    spec = DATABRICKS_SKUS.get(sku, DATABRICKS_SKUS["Standard_NC4as_T4_v3"])
    node_hours = nodes * hours
    vm_cost = spec.vm_hourly_usd * node_hours
    dbus = actual_dbus if actual_dbus is not None else (spec.illustrative_dbu_per_hour * node_hours)
    dbu_cost = dbus * dbu_rate_usd
    ocr_cost = (scanned_pages_processed / 1000.0) * ocr_per_thousand_pages_usd
    compute_cost = vm_cost + dbu_cost
    total_cost = compute_cost + ocr_cost
    per_million_fields = (
        (compute_cost / (text_fields_processed / 1_000_000.0))
        if text_fields_processed > 0
        else 0.0
    )
    return DatabricksCostEstimate(
        sku=spec.name,
        nodes=nodes,
        node_hours=node_hours,
        vm_cost_usd=vm_cost,
        dbu_cost_usd=dbu_cost,
        ocr_cost_usd=ocr_cost,
        total_cost_usd=total_cost,
        cost_per_million_fields_usd=per_million_fields,
        cost_per_thousand_ocr_pages_usd=ocr_per_thousand_pages_usd if scanned_pages_processed > 0 else 0.0,
    )


@dataclass(frozen=True, slots=True)
class RedactorSpec:
    """Small, JSON-serialisable configuration shipped to Spark executors."""

    salt_hex: str | None = None
    nlp_enabled: bool = True
    nlp_backend: str = "deberta-v3-small"
    #: ``None`` means "use the survey default generalisation set"; an empty tuple
    #: deliberately disables generalisation.
    generalise_categories: tuple[str, ...] | None = None
    audience: str = Audience.INTERNAL.value
    review_threshold: str = RiskBand.HIGH.value
    lexicon_terms: tuple[dict[str, Any], ...] = ()
    policy_overrides: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def build(
        cls,
        *,
        salt: bytes | str | None = None,
        nlp_enabled: bool = True,
        nlp_backend: str = "deberta-v3-small",
        generalise_categories: frozenset[Category] | None = None,
        audience: Audience = Audience.INTERNAL,
        review_threshold: RiskBand = RiskBand.HIGH,
        lexicon: Lexicon | None = None,
        **policy_overrides: Any,
    ) -> RedactorSpec:
        return cls(
            salt_hex=_encode_salt(salt),
            nlp_enabled=nlp_enabled,
            nlp_backend=nlp_backend,
            generalise_categories=(
                tuple(sorted(c.value for c in generalise_categories))
                if generalise_categories is not None
                else tuple(sorted(c.value for c in DEFAULT_GENERALISED))
            ),
            audience=audience.value,
            review_threshold=review_threshold.value,
            lexicon_terms=tuple(lexicon.describe()) if lexicon is not None else (),
            policy_overrides=tuple(sorted(policy_overrides.items(), key=lambda kv: kv[0])),
        )

    def salt_bytes(self) -> bytes | None:
        return bytes.fromhex(self.salt_hex) if self.salt_hex else None

    def categories(self) -> frozenset[Category]:
        return frozenset(Category(v) for v in (self.generalise_categories or ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "salt_hex": self.salt_hex,
            "nlp_enabled": self.nlp_enabled,
            "nlp_backend": self.nlp_backend,
            "generalise_categories": (
                None
                if self.generalise_categories is None
                else list(self.generalise_categories)
            ),
            "audience": self.audience,
            "review_threshold": self.review_threshold,
            "lexicon_terms": [dict(t) for t in self.lexicon_terms],
            "policy_overrides": dict(self.policy_overrides),
        }

    def cache_key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)


def _encode_salt(salt: bytes | str | None) -> str | None:
    if isinstance(salt, bytes):
        return salt.hex()
    if isinstance(salt, str):
        stripped = salt.removeprefix("0x")
        if stripped and len(stripped) % 2 == 0 and _is_hex(stripped):
            return stripped.lower()
    return None


def _is_hex(value: str) -> bool:
    return all(c in "0123456789abcdefABCDEF" for c in value)


def build_worker_redactor(spec: RedactorSpec) -> SurveyRedactor:
    """Build or reuse the per-executor :class:`SurveyRedactor` for ``spec``."""
    key = spec.cache_key()
    cached = _WORKER_CACHE.get(key)
    if cached is not None:
        return cached

    from .engine import Redactor
    from .survey import survey_detectors, survey_policy

    lexicon = (
        Lexicon.from_spec({"terms": [dict(t) for t in spec.lexicon_terms]})
        if spec.lexicon_terms
        else None
    )
    overrides = dict(spec.policy_overrides)
    ocr_mode = bool(overrides.get("ocr_mode", False))
    generalise_categories = spec.categories()
    if spec.generalise_categories is None:
        generalise_categories = DEFAULT_GENERALISED
    policy = survey_policy(
        salt=spec.salt_bytes(),
        nlp_enabled=spec.nlp_enabled,
        nlp_backend=spec.nlp_backend,
        generalise_categories=generalise_categories,
        **overrides,
    )
    survey = SurveyRedactor(
        Redactor(policy, detectors=survey_detectors(lexicon, ocr_mode=ocr_mode)),
        audience=Audience(spec.audience),
        review_threshold=RiskBand(spec.review_threshold),
    )
    _WORKER_CACHE[key] = survey
    return survey


def redact_text(spec: RedactorSpec, text: str | None, record_id: str | None = None):
    """Redact a single text field using the worker-cached redactor."""
    return build_worker_redactor(spec).redact_record(text, record_id=record_id)


def summarise_batch(
    spec: RedactorSpec,
    texts: Sequence[str | None],
    *,
    question_context: str | None = None,
) -> dict[str, list[Any]]:
    """Redact a batch of texts using length-bucketed execution and restore input order."""
    survey = build_worker_redactor(spec)
    n = len(texts)
    redacted: list[Any] = [""] * n
    findings: list[Any] = ["[]"] * n
    risks: list[Any] = [0.0] * n
    bands: list[Any] = [RiskBand.LOW.value] * n
    reviews: list[Any] = [False] * n
    states: list[Any] = [ReviewState.AUTO_ACCEPTED.value] * n

    normalised_texts = [t or "" for t in texts]
    buckets = bucket_by_length(normalised_texts)
    for _bucket_max, items in buckets.items():
        for idx, raw_idx_text in items:
            original = texts[idx]
            record = survey.redact_record(
                original if original is not None else raw_idx_text,
                question_context=question_context,
            )
            redacted[idx] = record.text
            findings[idx] = record.findings_json()
            risks[idx] = record.risk.score
            bands[idx] = record.risk.band.value
            reviews[idx] = survey.needs_review(record)
            states[idx] = record.review_state.value

    return {
        "redacted": redacted,
        "findings": findings,
        "risk": risks,
        "risk_band": bands,
        "needs_review": reviews,
        "review_state": states,
    }


def redact_batch_udf(spec: RedactorSpec, *, question_context: str | None = None):
    """Vectorised Arrow ``pandas_udf`` for Spark DataFrame batch redaction."""
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        StringType,
        StructField,
        StructType,
    )

    schema = StructType(
        [
            StructField("redacted", StringType(), True),
            StructField("findings", StringType(), True),
            StructField("risk", DoubleType(), True),
            StructField("risk_band", StringType(), True),
            StructField("needs_review", BooleanType(), True),
            StructField("review_state", StringType(), True),
        ]
    )

    @pandas_udf(schema)
    def _apply(series):
        import pandas as pd

        out = summarise_batch(spec, series.tolist(), question_context=question_context)
        return pd.DataFrame(out, columns=list(_STRUCT_FIELDS))

    return _apply


def redact_column_udf(spec: RedactorSpec):
    """Scalar UDF fallback for small row counts."""
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    return udf(lambda text: redact_text(spec, text).text, StringType())


def redact_frame(df, columns: Sequence[str], spec: RedactorSpec):
    """Redact free-text ``columns`` on a Spark DataFrame."""
    from pyspark.sql.functions import col

    udf_fn = redact_batch_udf(spec)
    result = df
    for column in columns:
        expanded = udf_fn(col(column)).alias("_redaction")
        for field_name, suffix in _STRUCT_FIELDS.items():
            result = result.withColumn(f"{column}{suffix}", expanded[field_name])
        result = result.drop("_redaction")
    return result


def redact_dataframe(
    df,
    spec: RedactorSpec,
    *,
    columns: Sequence[str] | None = None,
    auto_detect: bool = True,
):
    """Redact a survey DataFrame, auto-detecting string columns when needed."""
    if columns is not None:
        return redact_frame(df, columns, spec)
    if not auto_detect:
        raise ValueError("provide columns= or enable auto_detect")
    string_columns = [
        field.name for field in df.schema.fields if field.dataType.typeName() == "StringType"
    ]
    return redact_frame(df, string_columns, spec)


def select_gold_columns(
    metadata_columns: Sequence[str],
    free_text_columns: Sequence[str],
    *,
    include_findings: bool = False,
) -> list[str]:
    """Return the safe column list for the Gold Delta table (drops raw PII and findings by default).

    Duplicate names (e.g. a metadata column that already carries a ``_risk``
    suffix) are collapsed so ``spark.createDataFrame``/Delta merges do not see
    the same column twice.
    """
    suffixes = [REDACTED_SUFFIX, RISK_SUFFIX, RISK_BAND_SUFFIX, REVIEW_SUFFIX, REVIEW_STATE_SUFFIX]
    if include_findings:
        suffixes.append(FINDINGS_SUFFIX)
    out = list(metadata_columns)
    for col_name in free_text_columns:
        for suffix in suffixes:
            out.append(f"{col_name}{suffix}")
    return list(dict.fromkeys(out))


def filter_cdf_reviewed_records(
    cdf_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter Delta Change Data Feed (CDF) rows for incremental Pass 2 processing.

    Selects only rows where ``_change_type`` is ``insert`` or ``update_postimage``
    and ``review_status`` / ``review_state`` transitioned to ``REVIEWED``.
    """
    valid_changes = {"insert", "update_postimage"}
    return [
        row
        for row in cdf_rows
        if row.get("_change_type", "update_postimage") in valid_changes
        and (
            row.get("review_status") == ReviewState.REVIEWED.value
            or row.get("review_state") == ReviewState.REVIEWED.value
        )
    ]


__all__ = [
    "DATABRICKS_SKUS",
    "FINDINGS_SUFFIX",
    "REDACTED_SUFFIX",
    "REVIEW_STATE_SUFFIX",
    "REVIEW_SUFFIX",
    "RISK_BAND_SUFFIX",
    "RISK_SUFFIX",
    "DatabricksCostEstimate",
    "DatabricksSKU",
    "RedactorSpec",
    "build_worker_redactor",
    "estimate_databricks_cost",
    "filter_cdf_reviewed_records",
    "redact_batch_udf",
    "redact_column_udf",
    "redact_dataframe",
    "redact_frame",
    "redact_text",
    "select_gold_columns",
    "summarise_batch",
]
