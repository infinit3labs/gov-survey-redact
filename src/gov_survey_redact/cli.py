"""Command-line interface for local text, NDJSON, and CSV survey redaction & benchmarking."""

from __future__ import annotations

import argparse
import csv
import enum
import json
import sys
from pathlib import Path

from .benchmark import evaluate_corpus
from .engine import Policy, Redactor
from .generate import SurveyGenerator
from .qualtrics import (
    QualtricsInstrument,
    embedded_values,
    read_api_json,
    read_ndjson,
    read_tsv,
    text_answers,
)
from .records import SurveyRedactor
from .schema import SurveySchema
from .survey import survey_redactor
from .types import Category, Finding, MaskStyle


class OutputFormat(enum.Enum):
    TEXT = "text"
    JSON = "json"
    NDJSON = "ndjson"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gov-survey-redact",
        description=(
            "Hybrid two-pass PII & organisational attribution redaction pipeline "
            "for Australian Government survey data."
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="files to redact (text, .ndjson, or .csv); omit to read stdin",
    )
    parser.add_argument(
        "--survey",
        action="store_true",
        help="use the full government survey profile (organisation, implication, quasi-identifiers, generalisation, risk scoring)",
    )
    parser.add_argument(
        "--style",
        choices=[s.value for s in MaskStyle],
        default=MaskStyle.MASK.value,
        help="replacement style for direct identifiers (default: mask)",
    )
    parser.add_argument(
        "--keep-suffix",
        type=int,
        default=4,
        help="characters preserved at the end under --style partial (default: 4)",
    )
    parser.add_argument(
        "--keep-prefix",
        type=int,
        default=0,
        help="characters preserved at the start under --style partial (default: 0)",
    )
    parser.add_argument(
        "--mask-char",
        default="*",
        help="replacement character for --style mask and partial (default: *)",
    )
    parser.add_argument(
        "--categories",
        help="comma-separated categories to redact; default is all",
    )
    parser.add_argument(
        "--nlp",
        action="store_true",
        help="enable the NLP / encoder token-classification stage",
    )
    parser.add_argument(
        "--nlp-backend",
        choices=[
            "deberta-v3-small",
            "mdeberta-v3-base",
            "modernbert-base",
            "gazetteer",
            "spacy",
        ],
        default="deberta-v3-small",
        help="NLP backend to use (default: deberta-v3-small)",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_lg",
        help="spaCy model to load with --nlp-backend spacy",
    )
    parser.add_argument(
        "--format",
        choices=[f.value for f in OutputFormat],
        default=OutputFormat.TEXT.value,
        help="output format (default: text)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="report findings with offsets and metadata instead of redacting",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        metavar="N",
        help="run the 8-stratum synthetic government survey benchmark over N records and exit",
    )
    parser.add_argument(
        "--instrument",
        type=Path,
        default=None,
        help="Qualtrics instrument JSON for --survey exports (default: sibling instrument.json)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print a one-line summary to stderr after processing",
    )
    return parser


def _policy_from_args(args: argparse.Namespace) -> Policy:
    categories: frozenset[Category] | None = None
    if args.categories:
        wanted = {c.strip().upper() for c in args.categories.split(",") if c.strip()}
        known = {c.value.upper() for c in Category}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"unknown categories: {', '.join(sorted(unknown))}")
        categories = frozenset(c for c in Category if c.value.upper() in wanted)
    return Policy(
        categories=categories,
        styles={"default": MaskStyle(args.style)},
        nlp_enabled=args.nlp,
        nlp_backend=args.nlp_backend,
        spacy_model=args.spacy_model,
        mask_char=args.mask_char,
        partial_keep_prefix=args.keep_prefix,
        partial_keep_suffix=args.keep_suffix,
    )


def _finding_dict(f: Finding) -> dict[str, object]:
    return {
        "category": f.category.value,
        "start": f.span.start,
        "end": f.span.end,
        "detector": f.detector,
        "stage": f.stage.value,
        "confidence": round(f.confidence, 4),
        "metadata": f.meta(),
    }


def _render_findings(findings: list[Finding]) -> str:
    return json.dumps([_finding_dict(f) for f in findings], indent=2, default=str)


def _load_qualtrics_instrument(path: Path, instrument_path: Path | None) -> QualtricsInstrument | None:
    """Resolve the Qualtrics instrument for a response file.

    Order: explicit ``--instrument``, sibling ``instrument.json``, then an
    embedded ``qualtrics`` definition inside the payload itself.
    """
    candidates = []
    if instrument_path is not None:
        candidates.append(Path(instrument_path))
    candidates.append(path.parent / "instrument.json")
    for candidate in candidates:
        if candidate.exists():
            return QualtricsInstrument.from_dict(
                json.loads(candidate.read_text(encoding="utf-8"))
            )
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "qualtrics" in payload:
            return QualtricsInstrument.from_dict(payload["qualtrics"])
    return None


def _qualtrics_responses(path: Path):
    """Read Qualtrics responses from any supported export shape."""
    from .qualtrics import QualtricsResponse

    suffix = path.suffix.lower()
    if suffix == ".ndjson":
        return read_ndjson(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and ("responses" in payload or "result" in payload):
            return read_api_json(path)
        rows = payload.get("records", []) if isinstance(payload, dict) else []
        return [
            QualtricsResponse(
                response_id=str(row.get("ResponseId", "")), values=dict(row)
            )
            for row in rows
        ]
    return read_tsv(path)


def _process_qualtrics_file(
    path: Path,
    survey: SurveyRedactor,
    out,
    instrument_path: Path | None = None,
) -> int:
    """Redact a Qualtrics export file (API JSON, wide CSV, or three-line TSV).

    Emits one NDJSON summary per response with redacted open-text answers,
    schema-governed embedded PII, and risk/review routing metadata.
    """
    instrument = _load_qualtrics_instrument(path, instrument_path)
    if instrument is None:
        print(
            f"gov-survey-redact: could not find a Qualtrics instrument for {path} "
            "(pass --instrument or place instrument.json beside the export)",
            file=sys.stderr,
        )
        return 0
    schema = SurveySchema(redact_metadata_ip=True)
    salt = survey.redactor.policy.salt
    findings_total = 0
    for response in _qualtrics_responses(path):
        embedded = embedded_values(response, instrument)
        language = str(response.language).split("-")[0].lower()
        payload: dict[str, object] = {
            "ResponseId": response.response_id,
            "UserLanguage": response.language,
            "fields": {},
            "embedded": {},
        }
        queued = False
        worst_score = 0.0
        worst_band = "low"
        for qid, answer in text_answers(response, instrument).items():
            result = survey.redact_record(
                answer,
                record_id=response.response_id,
                field_id=qid,
                structured_context=embedded or None,
                language=language,
            )
            final = result.text
            if result.risk.score > worst_score:
                worst_score = result.risk.score
                worst_band = result.risk.band.value
            if result.review_state.value in ("PENDING_REVIEW", "ESCALATED"):
                queued = True
                second = survey.run_second_pass(
                    result,
                    field_id=qid,
                    structured_context=embedded or None,
                    language=language,
                )
                final = second.text
            fields = payload["fields"]
            assert isinstance(fields, dict)
            fields[qid] = {
                "redacted": final,
                "risk_band": result.risk.band.value,
                "review_state": result.review_state.value,
            }
            findings_total += result.finding_count
        governed_embedded: dict[str, object] = {}
        for name, value in embedded.items():
            policy = schema.policy_for(name, value)
            if policy.disposition.value.startswith("whole_column"):
                governed_embedded[name] = schema.apply_whole_column(policy, value, salt=salt)
            else:
                governed_embedded[name] = value
        payload["embedded"] = governed_embedded
        payload["risk_band"] = worst_band
        payload["review_state"] = "PENDING_REVIEW" if queued else "AUTO_ACCEPTED"
        out.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return findings_total


def _process_structured_file(path: Path, survey: SurveyRedactor, out) -> int:
    findings_total = 0
    if path.suffix.lower() == ".ndjson":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rec_id = str(row.get("ResponseID") or row.get("ResponseId") or "") or None
            for key, val in list(row.items()):
                if key.startswith("Q") and isinstance(val, str):
                    res = survey.redact_record(val, record_id=rec_id, field_id=key, structured_context=row)
                    row[f"{key}_redacted"] = res.text
                    row[f"{key}_risk_band"] = res.risk.band.value
                    row[f"{key}_review_state"] = res.review_state.value
                    findings_total += res.finding_count
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        q_cols = [c for c in fieldnames if c.startswith("Q")]
        extra_cols = [f"{c}_{s}" for c in q_cols for s in ("redacted", "risk_band", "review_state")]
        writer = csv.DictWriter(out, fieldnames=fieldnames + extra_cols)
        writer.writeheader()
        for row in rows:
            rec_id = str(row.get("ResponseID") or row.get("ResponseId") or "") or None
            for col in q_cols:
                val = row.get(col, "")
                res = survey.redact_record(val, record_id=rec_id, field_id=col, structured_context=row)
                row[f"{col}_redacted"] = res.text
                row[f"{col}_risk_band"] = res.risk.band.value
                row[f"{col}_review_state"] = res.review_state.value
                findings_total += res.finding_count
            writer.writerow(row)
    return findings_total


def _is_qualtrics_file(path: Path) -> bool:
    """True when the file looks like a Qualtrics structured export."""
    suffix = path.suffix.lower()
    if suffix == ".ndjson":
        try:
            with path.open("r", encoding="utf-8") as handle:
                first_line = next(
                    (line for line in handle if line.strip()), ""
                ).strip()
        except OSError:
            return False
        if not first_line:
            return False
        try:
            payload = json.loads(first_line)
        except ValueError:
            return False
        return isinstance(payload, dict) and (
            "responseId" in payload or "ResponseId" in payload or "values" in payload
        )
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(payload, dict) and (
            "qualtrics" in payload or "responses" in payload or "result" in payload
        )
    if suffix in {".csv", ".tsv"}:
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            return False
        return "ResponseId" in first_line or "QID" in first_line
    return False


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out = sys.stdout

    if args.benchmark > 0:
        gen = SurveyGenerator()
        corpus = gen.corpus(args.benchmark)
        s_redactor = SurveyRedactor(survey_redactor(nlp_backend=args.nlp_backend))
        metrics = evaluate_corpus(corpus, s_redactor)
        out.write(json.dumps(metrics.as_dict(), indent=2) + "\n")
        return 0

    redactor = (
        survey_redactor(nlp_backend=args.nlp_backend)
        if args.survey
        else Redactor(_policy_from_args(args))
    )
    s_redactor = SurveyRedactor(redactor) if args.survey else None

    paths = args.files or [None]
    findings_total = 0

    for path in paths:
        if path is not None and s_redactor is not None and _is_qualtrics_file(path):
            findings_total += _process_qualtrics_file(path, s_redactor, out, args.instrument)
            continue

        if (
            path is not None
            and s_redactor is not None
            and path.suffix.lower() in {".ndjson", ".csv"}
        ):
            findings_total += _process_structured_file(path, s_redactor, out)
            continue

        if path is None:
            text = sys.stdin.read()
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"gov-survey-redact: {exc}", file=sys.stderr)
                return 2

        result = redactor.redact(text)
        findings_total += len(result)

        if args.scan:
            out.write(_render_findings(list(result)) + "\n")
        elif args.format == OutputFormat.JSON.value:
            payload: dict[str, object] = {
                "text": result.text,
                "findings": [_finding_dict(f) for f in result],
            }
            if s_redactor is not None:
                rec = s_redactor.redact_record(text)
                payload["risk"] = rec.risk.as_dict()
                payload["review_state"] = rec.review_state.value
                payload["review_reasons"] = list(rec.review_reasons)
            out.write(json.dumps(payload, default=str) + "\n")
        elif args.format == OutputFormat.NDJSON.value:
            payload = {
                "text": result.text,
                "findings": [_finding_dict(f) for f in result],
            }
            out.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
        else:
            out.write(result.text)

        if args.summary:
            print(result.summary(), file=sys.stderr)

    if args.scan and findings_total == 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
