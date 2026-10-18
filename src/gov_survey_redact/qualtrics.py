"""Qualtrics-compatible survey structures, exports, and imports.

The WHS claims-experience programme runs on **Qualtrics**, so every survey
structure in this project replicates the Qualtrics conventions:

* **Wide export columns** — the standard metadata block
  (``StartDate``, ``EndDate``, ``Status``, ``IPAddress``, ``Progress``,
  ``Duration (in seconds)``, ``Finished``, ``RecordedDate``, ``ResponseId``,
  ``RecipientEmail``, ``DistributionChannel``, ``UserLanguage``, ...), then
  embedded-data fields, then one column per question ``QID``.
* **API JSON** (``GET /API/v3/surveys/{id}/responses`` shape) — flat ``values``
  keyed by API names (``startDate``, ``responseId``, ...); text-entry answers are
  keyed ``QIDn_TEXT`` while closed questions store recode values with display
  text in ``labels``.
* **Three-line-header TSV** — row 1 question text, row 2 ``{"ImportId": "QIDn"}``,
  row 3 answer reference text, followed by the response rows.

Closed questions are stored as **recode values** (1..n) internally, matching
Qualtrics exports; choice text is only materialised on demand.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import Category

#: Standard Qualtrics wide-export metadata columns, in export order.
QUALTRICS_METADATA_COLUMNS: tuple[str, ...] = (
    "StartDate",
    "EndDate",
    "Status",
    "IPAddress",
    "Progress",
    "Duration (in seconds)",
    "Finished",
    "RecordedDate",
    "ResponseId",
    "RecipientLastName",
    "RecipientFirstName",
    "RecipientEmail",
    "ExternalReference",
    "LocationLatitude",
    "LocationLongitude",
    "DistributionChannel",
    "UserLanguage",
)

#: Wide export column -> API JSON key (``values`` block of ``/responses``).
WIDE_TO_API_KEY: dict[str, str] = {
    "StartDate": "startDate",
    "EndDate": "endDate",
    "Status": "status",
    "IPAddress": "ipAddress",
    "Progress": "progress",
    "Duration (in seconds)": "duration",
    "Finished": "finished",
    "RecordedDate": "recordedDate",
    "ResponseId": "responseId",
    "RecipientLastName": "recipientLastName",
    "RecipientFirstName": "recipientFirstName",
    "RecipientEmail": "recipientEmail",
    "ExternalReference": "externalReference",
    "LocationLatitude": "locationLatitude",
    "LocationLongitude": "locationLongitude",
    "DistributionChannel": "distributionChannel",
    "UserLanguage": "userLanguage",
}
API_KEY_TO_WIDE: dict[str, str] = {api: wide for wide, api in WIDE_TO_API_KEY.items()}

#: Qualtrics question type codes used by this programme.
QUESTION_TEXT_ENTRY = "TE"
QUESTION_SINGLE_ANSWER = "MC"
QUESTION_LIKERT = "Likert"

_IMPORT_ID_KEYS = ("ImportId",)


@dataclass(frozen=True, slots=True)
class QualtricsChoice:
    """One answer option of a closed question (recode value + display text)."""

    text: str
    recode: int
    choice_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"text": self.text, "recode": self.recode}
        if self.choice_id is not None:
            data["choice_id"] = self.choice_id
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QualtricsChoice:
        return cls(
            text=str(data["text"]),
            recode=int(data["recode"]),
            choice_id=None if data.get("choice_id") is None else int(data["choice_id"]),
        )


@dataclass(frozen=True, slots=True)
class QualtricsQuestion:
    """One survey question (text entry, single answer, or Likert matrix row)."""

    qid: str
    export_tag: str
    question_text: str
    qtype: str = QUESTION_TEXT_ENTRY
    choices: tuple[QualtricsChoice, ...] = ()
    selector: str = ""
    force_response: bool = False

    @property
    def is_text_entry(self) -> bool:
        return self.qtype == QUESTION_TEXT_ENTRY

    @property
    def is_closed(self) -> bool:
        return not self.is_text_entry

    def choice(self, recode: int | str) -> QualtricsChoice | None:
        for choice in self.choices:
            if choice.recode == _coerce_int(recode):
                return choice
        return None

    def choice_text(self, recode: int | str) -> str | None:
        choice = self.choice(recode)
        return choice.text if choice is not None else None

    def recode_for_text(self, text: str) -> int | None:
        for choice in self.choices:
            if choice.text.casefold() == str(text).casefold():
                return choice.recode
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "export_tag": self.export_tag,
            "question_text": self.question_text,
            "qtype": self.qtype,
            "selector": self.selector,
            "force_response": self.force_response,
            "choices": [choice.to_dict() for choice in self.choices],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QualtricsQuestion:
        return cls(
            qid=str(data["qid"]),
            export_tag=str(data.get("export_tag", data["qid"])),
            question_text=str(data.get("question_text", "")),
            qtype=str(data.get("qtype", QUESTION_TEXT_ENTRY)),
            choices=tuple(QualtricsChoice.from_dict(c) for c in data.get("choices", ())),
            selector=str(data.get("selector", "")),
            force_response=bool(data.get("force_response", False)),
        )


@dataclass(frozen=True, slots=True)
class EmbeddedField:
    """A Qualtrics embedded-data field carried on every response.

    ``category`` marks fields that are themselves PII (governed by schema policy
    before free-text redaction); ``tokenise`` requests a stable pseudonym rather
    than a mask when whole-column governance is applied.
    """

    name: str
    label: str = ""
    category: Category | None = None
    tokenise: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "category": self.category.value if self.category else None,
            "tokenise": self.tokenise,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EmbeddedField:
        raw = data.get("category")
        return cls(
            name=str(data["name"]),
            label=str(data.get("label", "")),
            category=Category(str(raw)) if raw else None,
            tokenise=bool(data.get("tokenise", False)),
        )


@dataclass(frozen=True)
class QualtricsInstrument:
    """A Qualtrics survey definition: questions plus embedded-data fields."""

    survey_id: str
    name: str
    questions: tuple[QualtricsQuestion, ...]
    embedded_fields: tuple[EmbeddedField, ...] = ()
    metadata_columns: tuple[str, ...] = QUALTRICS_METADATA_COLUMNS

    @property
    def question_map(self) -> dict[str, QualtricsQuestion]:
        return {question.qid: question for question in self.questions}

    @property
    def text_entry_questions(self) -> tuple[QualtricsQuestion, ...]:
        return tuple(question for question in self.questions if question.is_text_entry)

    @property
    def embedded_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.embedded_fields)

    def question(self, qid: str) -> QualtricsQuestion | None:
        return self.question_map.get(qid)

    def pii_embedded_fields(self) -> tuple[EmbeddedField, ...]:
        return tuple(field_ for field_ in self.embedded_fields if field_.category is not None)

    def column_order(self) -> list[str]:
        """Canonical wide-export column order (metadata, embedded, questions)."""
        return [
            *self.metadata_columns,
            *self.embedded_names,
            *[question.qid for question in self.questions],
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "survey_id": self.survey_id,
            "name": self.name,
            "metadata_columns": list(self.metadata_columns),
            "embedded_fields": [field_.to_dict() for field_ in self.embedded_fields],
            "questions": [question.to_dict() for question in self.questions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QualtricsInstrument:
        return cls(
            survey_id=str(data.get("survey_id", "SV_unknown")),
            name=str(data.get("name", "survey")),
            questions=tuple(QualtricsQuestion.from_dict(q) for q in data.get("questions", ())),
            embedded_fields=tuple(
                EmbeddedField.from_dict(f_) for f_ in data.get("embedded_fields", ())
            ),
            metadata_columns=tuple(data.get("metadata_columns", QUALTRICS_METADATA_COLUMNS)),
        )


@dataclass
class QualtricsResponse:
    """One Qualtrics response row.

    ``values`` is keyed like the wide export: metadata column names
    (``ResponseId``, ``StartDate``, ...), embedded-data field names, and question
    QIDs (text-entry answers are stored under the plain ``QID``; the API writers
    add the ``_TEXT`` suffix that Qualtrics uses).
    """

    response_id: str
    values: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    displayed_fields: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.values.setdefault("ResponseId", self.response_id)

    @property
    def progress(self) -> int:
        return _coerce_int(self.values.get("Progress", 100)) or 0

    @property
    def finished(self) -> int:
        return _coerce_int(self.values.get("Finished", 1)) or 0

    @property
    def partial(self) -> bool:
        """True when the respondent abandoned the survey before the final block."""
        return self.finished == 0 or self.progress < 100

    @property
    def language(self) -> str:
        return str(self.values.get("UserLanguage", "EN-AU"))

    def answer(self, qid: str) -> Any:
        return self.values.get(qid)

    def text(self, qid: str) -> str:
        value = self.values.get(qid)
        return "" if value is None else str(value)

    def embedded(self, name: str) -> str | None:
        value = self.values.get(name)
        return None if value in (None, "") else str(value)

    def to_wide_row(
        self,
        instrument: QualtricsInstrument,
        *,
        use_choice_text: bool = False,
    ) -> dict[str, str]:
        """Format this response as a wide-export row (all columns present)."""
        row: dict[str, str] = {}
        for column in instrument.column_order():
            value = self.values.get(column)
            row[column] = format_cell(instrument, column, value, use_choice_text=use_choice_text)
        return row


def format_cell(
    instrument: QualtricsInstrument | None,
    column: str,
    value: Any,
    *,
    use_choice_text: bool = False,
) -> str:
    if value is None:
        return ""
    question = instrument.question(column) if instrument is not None else None
    if question is not None and question.is_closed and use_choice_text:
        text = question.choice_text(value)
        return text if text is not None else str(value)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# API JSON (GET /API/v3/surveys/{id}/responses shape)
# --------------------------------------------------------------------------- #


def to_api_json(
    responses: Iterable[QualtricsResponse],
    instrument: QualtricsInstrument,
) -> dict[str, Any]:
    """Serialise responses into the Qualtrics API export shape."""
    elements = []
    for response in responses:
        values: dict[str, Any] = {}
        labels: dict[str, str] = {}
        displayed: list[str] = []
        for column in instrument.column_order():
            if column not in response.values:
                continue
            value = response.values[column]
            question = instrument.question(column)
            if question is not None and question.is_text_entry:
                key = f"{column}_TEXT"
                values[key] = value
                displayed.append(key)
            else:
                key = WIDE_TO_API_KEY.get(column, column)
                values[key] = value
                if question is not None and question.is_closed:
                    text = question.choice_text(value)
                    if text is not None:
                        labels[key] = text
                        displayed.append(key)
        elements.append(
            {
                "responseId": response.response_id,
                "_recordId": response.response_id,
                "values": values,
                "labels": {**response.labels, **labels},
                "displayedFields": response.displayed_fields or displayed,
                "displayedValues": [],
            }
        )
    return {"responses": elements}


def write_api_json(
    path: str | Path,
    responses: Iterable[QualtricsResponse],
    instrument: QualtricsInstrument,
) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(to_api_json(responses, instrument), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def read_api_json(path: str | Path) -> list[QualtricsResponse]:
    """Read the API export shape (``responses`` or ``result.elements``)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    elements = data.get("responses")
    if elements is None:
        elements = data.get("result", {}).get("elements", [])
    return [_response_from_api_element(element) for element in elements]


def _response_from_api_element(payload: Mapping[str, Any]) -> QualtricsResponse:
    """Build a response from an API element (or a flat wide-style mapping)."""
    raw_values = payload.get("values")
    if not isinstance(raw_values, Mapping):
        meta_keys = {"labels", "displayedFields", "displayedValues"}
        raw_values = {key: value for key, value in payload.items() if key not in meta_keys}
    values: dict[str, Any] = {}
    for key, value in raw_values.items():
        if isinstance(key, str) and key.endswith("_TEXT"):
            values[key[: -len("_TEXT")]] = value
        else:
            values[API_KEY_TO_WIDE.get(key, key)] = value
    response_id = str(payload.get("responseId") or values.get("ResponseId") or "")
    return QualtricsResponse(
        response_id=response_id,
        values=values,
        labels=dict(payload.get("labels", {}) or {}),
        displayed_fields=list(payload.get("displayedFields", []) or []),
    )


def to_ndjson_lines(
    responses: Iterable[QualtricsResponse],
    instrument: QualtricsInstrument,
) -> list[str]:
    """One compact API-shaped JSON object per response (Qualtrics NDJSON export)."""
    elements = to_api_json(responses, instrument)["responses"]
    return [
        json.dumps(element, ensure_ascii=False, separators=(",", ":")) for element in elements
    ]


def write_ndjson(
    path: str | Path,
    responses: Iterable[QualtricsResponse],
    instrument: QualtricsInstrument,
) -> Path:
    path = Path(path)
    path.write_text("\n".join(to_ndjson_lines(responses, instrument)) + "\n", encoding="utf-8")
    return path


def read_ndjson(path: str | Path) -> list[QualtricsResponse]:
    """Read a Qualtrics NDJSON export (one response object per line).

    Accepts both the API element shape (``values`` / ``labels``) and flat
    wide-style rows keyed directly by column names.
    """
    records: list[QualtricsResponse] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        records.append(_response_from_api_element(payload))
    return records


# --------------------------------------------------------------------------- #
# Wide CSV / three-line-header TSV
# --------------------------------------------------------------------------- #


def to_wide_rows(
    instrument: QualtricsInstrument,
    responses: Iterable[QualtricsResponse],
    *,
    use_choice_text: bool = False,
) -> list[dict[str, str]]:
    return [
        response.to_wide_row(instrument, use_choice_text=use_choice_text)
        for response in responses
    ]


def write_csv(
    path: str | Path,
    responses: Iterable[QualtricsResponse],
    instrument: QualtricsInstrument,
    *,
    use_choice_text: bool = False,
) -> Path:
    """Write a single-header CSV in Qualtrics wide-export column order."""
    path = Path(path)
    columns = instrument.column_order()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for response in responses:
            writer.writerow(response.to_wide_row(instrument, use_choice_text=use_choice_text))
    return path


def write_tsv(
    path: str | Path,
    responses: Iterable[QualtricsResponse],
    instrument: QualtricsInstrument,
    *,
    use_choice_text: bool = False,
) -> Path:
    """Write a TSV with the Qualtrics three-line header (text, ImportId, answers)."""
    path = Path(path)
    columns = instrument.column_order()
    text_row: list[str] = []
    import_row: list[str] = []
    answer_row: list[str] = []
    for column in columns:
        question = instrument.question(column)
        if question is None:
            text_row.append(column)
            import_row.append(json.dumps({"ImportId": column}))
            answer_row.append("")
            continue
        text_row.append(question.question_text or question.export_tag)
        import_row.append(json.dumps({"ImportId": question.qid}))
        if use_choice_text and question.is_closed:
            answer_row.append("Answer")
        else:
            answer_row.append("")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(text_row) + "\n")
        handle.write("\t".join(import_row) + "\n")
        handle.write("\t".join(answer_row) + "\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for response in responses:
            row = response.to_wide_row(instrument, use_choice_text=use_choice_text)
            writer.writerow([row.get(column, "") for column in columns])
    return path


def read_tsv(path: str | Path) -> list[QualtricsResponse]:
    """Read CSV or TSV with either a single header row or the three-line header."""
    path = Path(path)
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    if not rows:
        return []
    header_index = 0
    headers = rows[0]
    if len(rows) > 1 and rows[1] and rows[1][0].strip().startswith("{"):
        # Qualtrics three-line header: row 2 carries {"ImportId": "QIDn"}.
        headers = [str(_import_id(cell) or cell) for cell in rows[1]]
        header_index = 2
    records = []
    for raw in rows[header_index + 1 :]:
        if not any(cell.strip() for cell in raw):
            continue
        values = {
            headers[index]: raw[index]
            for index in range(min(len(headers), len(raw)))
            if raw[index] != ""
        }
        response_id = str(values.get("ResponseId", ""))
        records.append(QualtricsResponse(response_id=response_id, values=values))
    return records


def read_csv(path: str | Path) -> list[QualtricsResponse]:
    return read_tsv(path)


def read_auto(path: str | Path) -> list[QualtricsResponse]:
    """Read a response file by suffix: ``.json`` API shape, ``.ndjson`` line-delimited,
    ``.csv``/``.tsv`` wide (including the three-line header)."""
    path = Path(path)
    if path.suffix.lower() == ".ndjson":
        return read_ndjson(path)
    if path.suffix.lower() == ".json":
        return read_api_json(path)
    return read_tsv(path)


def _import_id(cell: str) -> str | None:
    cell = cell.strip()
    if not cell.startswith("{"):
        return None
    try:
        payload = json.loads(cell)
    except json.JSONDecodeError:
        return None
    for key in _IMPORT_ID_KEYS:
        if key in payload:
            return str(payload[key])
    return None


# --------------------------------------------------------------------------- #
# Bridge to the redaction pipeline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RedactionTarget:
    """One column that the redaction pipeline should treat as sensitive."""

    column: str          # QID or embedded field name
    field_id: str        # export tag (questions) or embedded name
    kind: str            # "question" | "embedded"
    category: Category | None
    tokenise: bool = False
    label: str = ""


def redaction_targets(instrument: QualtricsInstrument) -> tuple[RedactionTarget, ...]:
    """Columns to redact: open-text questions plus PII embedded fields."""
    targets: list[RedactionTarget] = []
    for question in instrument.text_entry_questions:
        targets.append(
            RedactionTarget(
                column=question.qid,
                field_id=question.export_tag,
                kind="question",
                category=None,
                label=question.question_text,
            )
        )
    for field_ in instrument.pii_embedded_fields():
        targets.append(
            RedactionTarget(
                column=field_.name,
                field_id=field_.name,
                kind="embedded",
                category=field_.category,
                tokenise=field_.tokenise,
                label=field_.label or field_.name,
            )
        )
    return tuple(targets)


def text_answers(
    response: QualtricsResponse,
    instrument: QualtricsInstrument,
) -> dict[str, str]:
    """Non-empty text-entry answers keyed by QID (skips partial-response gaps)."""
    answers: dict[str, str] = {}
    for question in instrument.text_entry_questions:
        value = response.values.get(question.qid)
        if value not in (None, ""):
            answers[question.qid] = str(value)
    return answers


def embedded_values(
    response: QualtricsResponse,
    instrument: QualtricsInstrument,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for field_ in instrument.embedded_fields:
        value = response.embedded(field_.name)
        if value is not None:
            values[field_.name] = value
    return values


def row_values_from_mapping(
    row: Mapping[str, Any],
    instrument: QualtricsInstrument,
) -> tuple[dict[str, str], dict[str, str]]:
    """Split a serialised Qualtrics row into ``(text answers, embedded values)``."""
    answers = {
        question.qid: str(row[question.qid])
        for question in instrument.text_entry_questions
        if row.get(question.qid) not in (None, "")
    }
    embedded = {
        field_.name: str(row[field_.name])
        for field_ in instrument.embedded_fields
        if row.get(field_.name) not in (None, "")
    }
    return answers, embedded


__all__ = [
    "API_KEY_TO_WIDE",
    "QUALTRICS_METADATA_COLUMNS",
    "QUESTION_LIKERT",
    "QUESTION_SINGLE_ANSWER",
    "QUESTION_TEXT_ENTRY",
    "WIDE_TO_API_KEY",
    "EmbeddedField",
    "QualtricsChoice",
    "QualtricsInstrument",
    "QualtricsQuestion",
    "QualtricsResponse",
    "RedactionTarget",
    "embedded_values",
    "format_cell",
    "read_api_json",
    "read_auto",
    "read_csv",
    "read_ndjson",
    "read_tsv",
    "redaction_targets",
    "row_values_from_mapping",
    "text_answers",
    "to_api_json",
    "to_ndjson_lines",
    "to_wide_rows",
    "write_api_json",
    "write_csv",
    "write_ndjson",
    "write_tsv",
]
