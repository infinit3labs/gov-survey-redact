# Azure Databricks Deployment: Two-Pass Bronze → Review Queue → Gold Pipeline

This guide shows how to run `gov-survey-redact` on Azure Databricks Jobs compute (`Standard_NC4as_T4_v3` or `Standard_D8as_v5`) using Unity Catalog, Delta Change Data Feed (CDF), and MLflow lineage.

## 1. Build the serialisable `RedactorSpec` on the Driver

Compiled regular expressions, the SymSpell delete index, and encoder weights are never serialised into the Spark closure. Instead, build a frozen `RedactorSpec` on the driver; each executor JVM builds and caches its own `SurveyRedactor` once per worker process.

```python
from gov_survey_redact import Audience, Category, Lexicon, RiskBand, Term
from gov_survey_redact.spark import (
    RedactorSpec,
    estimate_databricks_cost,
    filter_cdf_reviewed_records,
    redact_frame,
    select_gold_columns,
)

salt = dbutils.secrets.get(scope="pii", key="redaction_salt")

agency_lexicon = Lexicon.from_terms([
    Term("Project Kestrel", Category.INTERNAL_TERM, aliases=["kestrel"], label="internal project"),
    Term("Beacon", Category.INTERNAL_TERM, label="internal program"),
])

spec = RedactorSpec.build(
    salt=salt,
    nlp_enabled=True,
    nlp_backend="deberta-v3-small",
    audience=Audience.INTERNAL,
    review_threshold=RiskBand.HIGH,
    lexicon=agency_lexicon,
)
```

## 2. First Pass on Restricted Bronze Delta

```python
bronze_df = spark.read.table("main.surveys_bronze.responses_raw")
free_text_cols = ["Q1", "Q2", "Q3", "Q4", "Q5"]

first_pass_df = redact_frame(bronze_df, free_text_cols, spec)

# Split into auto-accepted records and the governed Delta review queue
needs_review_expr = " OR ".join(f"{c}_needs_review = true" for c in free_text_cols)
review_queue_df = first_pass_df.filter(needs_review_expr)
auto_accepted_df = first_pass_df.filter(f"NOT ({needs_review_expr})")

(
    review_queue_df.write
    .format("delta")
    .mode("append")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable("main.surveys_silver.review_queue")
)
```

## 3. Incremental Second Pass via Delta Change Data Feed (CDF)

When reviewers approve, extend, remove, or relabel spans in the review UI and set `review_state = 'REVIEWED'`, read only the changed rows from the Delta Change Data Feed and run `SurveyRedactor.run_second_pass()` (which locks human decisions as hard constraints, routes long English responses to `ModernBERT-base` and multilingual/ambiguous responses to `mDeBERTa-v3-base`, and runs deterministic residual QA).

```python
cdf_df = (
    spark.read.format("delta")
    .option("readChangeFeed", "true")
    .option("startingVersion", last_processed_version)
    .table("main.surveys_silver.review_queue")
    .filter("_change_type IN ('insert', 'update_postimage') AND review_state = 'REVIEWED'")
)
```

## 4. Publish Redacted Gold Table Without Raw PII or Findings Offsets

```python
safe_cols = select_gold_columns(
    metadata_columns=["ResponseID", "StartDate", "EndDate", "Instrument", "Consent"],
    free_text_columns=free_text_cols,
    include_findings=False,
)

(
    auto_accepted_df.select(*safe_cols)
    .write.format("delta")
    .mode("append")
    .saveAsTable("main.surveys_gold.responses_redacted")
)
```

## 5. Compute Cost Model

```python
print(estimate_databricks_cost(sku="Standard_NC4as_T4_v3", nodes=2, hours=20.0).as_dict())
```
