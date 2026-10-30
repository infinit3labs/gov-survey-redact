"""Tests for Qualtrics structures, the WHS claims-experience generator, and batch checks."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from gov_survey_redact.qualtrics import (
    QUALTRICS_METADATA_COLUMNS,
    QualtricsInstrument,
    embedded_values,
    read_api_json,
    read_auto,
    read_csv,
    read_ndjson,
    read_tsv,
    redaction_targets,
    row_values_from_mapping,
    text_answers,
    to_api_json,
    to_ndjson_lines,
    write_api_json,
    write_csv,
    write_ndjson,
    write_tsv,
)
from gov_survey_redact.whs import (
    WHS_CLAIMS_INSTRUMENT,
    WHSBatchGenerator,
    write_batch_files,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _small_batch(seed: int = 5, count: int = 12):
    return WHSBatchGenerator(seed=seed, partial_rate=0.2).batch(count, batch_index=0)


class TestQualtricsInstrument:
    def test_instrument_round_trip(self) -> None:
        instrument = WHS_CLAIMS_INSTRUMENT
        restored = QualtricsInstrument.from_dict(
            json.loads(json.dumps(instrument.to_dict()))
        )
        assert restored == instrument
        assert restored.survey_id == "SV_whsClaimsExperience2026"

    def test_whs_instrument_covers_claims_experience_constructs(self) -> None:
        tags = {q.export_tag for q in WHS_CLAIMS_INSTRUMENT.questions}
        assert {
            "process_open_honest",
            "good_communication",
            "protecting_interests",
            "treated_fairly",
            "helped_recovery",
            "case_manager_respect",
            "case_manager_contact",
            "rtw_plan_discussed",
            "suitable_duties",
            "employer_contact",
            "payment_timeliness",
            "compensation_adequacy",
            "recovery_confidence",
            "injury_type",
            "jurisdiction",
            "time_to_lodge",
            "employer_type",
            "service_experience",
            "case_manager_experience",
            "rtw_experience",
            "other_comments",
        } <= tags

    def test_redaction_targets_cover_te_and_pii_embedded_only(self) -> None:
        targets = redaction_targets(WHS_CLAIMS_INSTRUMENT)
        columns = {t.column for t in targets}
        assert "QID21" in columns and "claim_number" in columns
        assert "QID3" not in columns  # Likert item, not free text
        assert "Instrument" not in columns


class TestQualtricsFormats:
    def test_api_json_uses_qualtrics_key_conventions(self, tmp_path: Path) -> None:
        batch = _small_batch()
        path = write_api_json(tmp_path / "b.api.json", batch.responses, WHS_CLAIMS_INSTRUMENT)
        payload = json.loads(path.read_text(encoding="utf-8"))
        element = payload["responses"][0]
        values = element["values"]
        assert "responseId" in values and "recordedDate" in values and "userLanguage" in values
        assert any(key.endswith("_TEXT") for key in values)
        assert "QID3" in values and "QID3" in element["labels"]
        # Text-entry answers live under QIDn_TEXT and strip back on read.
        restored = read_api_json(path)
        assert len(restored) == len(batch.responses)
        for original, back in zip(batch.responses, restored, strict=True):
            assert back.response_id == original.response_id
            assert text_answers(back, WHS_CLAIMS_INSTRUMENT) == text_answers(
                original, WHS_CLAIMS_INSTRUMENT
            )
            assert embedded_values(back, WHS_CLAIMS_INSTRUMENT) == embedded_values(
                original, WHS_CLAIMS_INSTRUMENT
            )

    def test_to_api_json_is_serialisable_without_files(self) -> None:
        batch = _small_batch(seed=6, count=2)
        payload = to_api_json(batch.responses, WHS_CLAIMS_INSTRUMENT)
        assert json.loads(json.dumps(payload))["responses"][0]["responseId"]

    def test_wide_csv_and_three_line_tsv_round_trip(self, tmp_path: Path) -> None:
        batch = _small_batch(seed=7)
        csv_path = write_csv(tmp_path / "b.csv", batch.responses, WHS_CLAIMS_INSTRUMENT)
        tsv_path = write_tsv(tmp_path / "b.tsv", batch.responses, WHS_CLAIMS_INSTRUMENT)

        header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert list(QUALTRICS_METADATA_COLUMNS) == header[: len(QUALTRICS_METADATA_COLUMNS)]

        lines = tsv_path.read_text(encoding="utf-8").splitlines()
        import_row = lines[1].split("\t")
        assert import_row[0] == '{"ImportId": "StartDate"}'
        qid_index = list(QUALTRICS_METADATA_COLUMNS).__len__() + len(
            WHS_CLAIMS_INSTRUMENT.embedded_names
        )
        assert import_row[qid_index] == '{"ImportId": "QID1"}'
        assert lines[0].split("\t")[qid_index] == WHS_CLAIMS_INSTRUMENT.questions[0].question_text

        for reader in (read_csv, read_tsv):
            rows = reader(csv_path if reader is read_csv else tsv_path)
            assert len(rows) == len(batch.responses)
            assert text_answers(rows[0], WHS_CLAIMS_INSTRUMENT) == text_answers(
                batch.responses[0], WHS_CLAIMS_INSTRUMENT
            )

    def test_row_values_from_mapping_splits_answers_and_embedded(self) -> None:
        batch = _small_batch(seed=8, count=3)
        response = batch.responses[0]
        answers, embedded = row_values_from_mapping(response.values, WHS_CLAIMS_INSTRUMENT)
        assert set(answers) <= {q.qid for q in WHS_CLAIMS_INSTRUMENT.text_entry_questions}
        assert "respondent_name" in embedded and "claim_number" in embedded

    def test_ndjson_export_is_api_shaped_and_round_trips(self, tmp_path: Path) -> None:
        batch = _small_batch(seed=9, count=5)
        path = write_ndjson(tmp_path / "wave.ndjson", batch.responses, WHS_CLAIMS_INSTRUMENT)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(batch.responses)
        first = json.loads(lines[0])
        assert "responseId" in first and "values" in first
        assert any(key.endswith("_TEXT") for key in first["values"])

        restored = read_ndjson(path)
        assert len(restored) == len(batch.responses)
        for original, back in zip(batch.responses, restored, strict=True):
            assert back.response_id == original.response_id
            assert back.values == original.values
        # read_auto dispatches .ndjson as well
        assert [r.response_id for r in read_auto(path)] == [
            r.response_id for r in batch.responses
        ]

    def test_ndjson_reader_accepts_flat_wide_rows(self, tmp_path: Path) -> None:
        batch = _small_batch(seed=10, count=2)
        path = tmp_path / "wide.ndjson"
        rows = [
            json.dumps({"ResponseId": r.response_id, **r.values}, ensure_ascii=False)
            for r in batch.responses
        ]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        restored = read_ndjson(path)
        assert [r.response_id for r in restored] == [r.response_id for r in batch.responses]
        assert text_answers(restored[0], WHS_CLAIMS_INSTRUMENT) == text_answers(
            batch.responses[0], WHS_CLAIMS_INSTRUMENT
        )

    def test_to_ndjson_lines_is_compact(self) -> None:
        batch = _small_batch(seed=12, count=2)
        lines = to_ndjson_lines(batch.responses, WHS_CLAIMS_INSTRUMENT)
        assert all("\n" not in line for line in lines)
        assert json.loads(lines[0])["responseId"].startswith("R_")


class TestWHSGenerator:
    def test_deterministic_for_same_seed(self) -> None:
        first = WHSBatchGenerator(seed=42).batch(10, batch_index=0)
        second = WHSBatchGenerator(seed=42).batch(10, batch_index=0)
        assert [r.values for r in first.responses] == [r.values for r in second.responses]

    def test_ground_truth_literals_are_present_in_generated_values(self) -> None:
        batch = WHSBatchGenerator(seed=11).batch(30, batch_index=0)
        text_secrets = 0
        embedded_secrets = 0
        for response in batch.responses:
            truth = batch.ground_truth[response.response_id]
            answers = text_answers(response, WHS_CLAIMS_INSTRUMENT)
            joined = "\n".join(answers.values())
            for secret in truth.secrets:
                assert secret.field is not None
                assert secret.literal in joined, secret.literal
                text_secrets += 1
            for secret in truth.embedded_secrets:
                assert secret.literal == response.values.get(secret.field or "")
                embedded_secrets += 1
        assert text_secrets > 10 and embedded_secrets >= 3 * len(batch.responses)

    def test_partial_responses_omit_tail_answers(self) -> None:
        batch = WHSBatchGenerator(seed=13, partial_rate=1.0).batch(6, batch_index=0)
        assert all(r.partial for r in batch.responses)
        assert all(r.progress < 100 and r.finished == 0 for r in batch.responses)
        complete = WHSBatchGenerator(seed=13, partial_rate=0.0).batch(6, batch_index=0)
        assert sum(len(text_answers(r, WHS_CLAIMS_INSTRUMENT)) for r in complete.responses) >= sum(
            len(text_answers(r, WHS_CLAIMS_INSTRUMENT)) for r in batch.responses
        )

    def test_batch_files_and_sidecars(self, tmp_path: Path) -> None:
        batch = _small_batch(seed=14, count=4)
        files = write_batch_files(str(tmp_path), batch, WHS_CLAIMS_INSTRUMENT)
        names = {Path(f).name for f in files}
        assert names == {
            "batch_01.api.json",
            "batch_01.responses.csv",
            "batch_01.responses.tsv",
            "batch_01.ground_truth.json",
        }
        truth = json.loads((tmp_path / "batch_01.ground_truth.json").read_text(encoding="utf-8"))
        assert len(truth) == 4 and "secrets" in truth[0]

    def test_response_metadata_matches_qualtrics_shape(self) -> None:
        response = WHSBatchGenerator(seed=15).batch(1, batch_index=2).responses[0]
        for column in ("StartDate", "EndDate", "RecordedDate", "Duration (in seconds)"):
            assert response.values[column]
        assert response.values["DistributionChannel"] in {"email", "anonymous", "qr"}
        assert response.language.split("-")[0] in {"EN", "VI", "AR"}
        assert response.response_id.startswith("R_02")


class TestBatchChecker:
    def test_batch_check_passes_gates_and_round_trips(self, tmp_path: Path) -> None:
        # scripts/ is not a package, so load it by path like the tests below do.
        module = _load_script_module("qualtrics_batch_check", "qualtrics_batch_check.py")
        BatchChecker = module.BatchChecker

        batch = WHSBatchGenerator(seed=21, partial_rate=0.15).batch(15, batch_index=0)
        checker = BatchChecker(WHS_CLAIMS_INSTRUMENT)
        metrics = checker.submit(
            batch, tmp_path, formats=("api", "csv", "tsv"), round_trip=True
        )
        assert metrics.round_trip_failures == 0
        assert metrics.idempotence_failures == 0
        assert metrics.planted > 0
        assert metrics.mandatory_recall == 1.0
        assert metrics.escape_rate == 0.0
        assert metrics.over_rate == 0.0
        assert metrics.embedded_caught == metrics.embedded_planted
        assert metrics.metadata_governed == metrics.metadata_checked


class TestMigratedFixturesUseQualtricsStructures:
    @pytest.mark.parametrize(
        "name",
        [
            "aps_workplace_census_dirty.json",
            "citizen_services_claims_and_ocr.json",
            "synthetic_stratified_benchmark.json",
        ],
    )
    def test_fixture_carries_instrument_and_metadata(self, name: str) -> None:
        payload = json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
        instrument = QualtricsInstrument.from_dict(payload["qualtrics"])
        assert instrument.survey_id.startswith("SV_")
        for record in payload["records"]:
            assert record["ResponseId"]
            for column in ("StartDate", "EndDate", "UserLanguage"):
                assert record[column]
        assert any(q.is_text_entry for q in instrument.questions)


class TestNdjsonDatasetScript:
    """The 10-dataset NDJSON generator must produce gradeable, leak-free waves."""

    def test_single_dataset_round_trip_and_grades(self, tmp_path: Path) -> None:
        module = _load_script_module("qualtrics_ndjson_datasets", "qualtrics_ndjson_datasets.py")
        spec = module.DatasetSpec(
            "test_wave",
            "tiny test wave",
            seed=9901,
            responses=10,
            partial_rate=0.2,
        )
        batch, files = module.generate_dataset(spec, tmp_path, batch_index=1)
        assert len(files) == 2
        assert (tmp_path / "test_wave.ndjson").exists()

        reloaded = module.load_dataset(spec, tmp_path, batch_index=1)
        assert module.verify_round_trip(batch, reloaded) == 0
        assert len(reloaded.responses) == 10

        from qualtrics_batch_check import BatchChecker

        checker = BatchChecker(WHS_CLAIMS_INSTRUMENT)
        metrics = checker.evaluate(reloaded)
        assert metrics.mandatory_recall == 1.0
        assert metrics.escape_rate == 0.0
        assert metrics.over_rate == 0.0

    def test_default_specs_are_ten_distinct_waves(self) -> None:
        module = _load_script_module("qualtrics_ndjson_datasets", "qualtrics_ndjson_datasets.py")
        specs = module.DEFAULT_SPECS
        assert len(specs) == 10
        assert len({spec.name for spec in specs}) == 10
        assert len({spec.seed for spec in specs}) == 10


class TestCliQualtricsIngest:
    def test_cli_ingests_ndjson_export_with_sibling_instrument(
        self, tmp_path: Path, capsys
    ) -> None:
        from gov_survey_redact.cli import main

        batch = WHSBatchGenerator(seed=31, partial_rate=0.3).batch(5, batch_index=1)
        write_ndjson(tmp_path / "responses.ndjson", batch.responses, WHS_CLAIMS_INSTRUMENT)
        (tmp_path / "instrument.json").write_text(
            json.dumps(WHS_CLAIMS_INSTRUMENT.to_dict()), encoding="utf-8"
        )
        exit_code = main(["--survey", str(tmp_path / "responses.ndjson")])
        assert exit_code == 0
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) == 5
        payload = json.loads(lines[0])
        assert payload["ResponseId"].startswith("R_01")
        assert payload["fields"], "open-text answers should be redacted"
        assert str(payload["embedded"].get("claim_number", "")).startswith("reference_id_")
        assert payload["embedded"].get("respondent_email") == "[EMAIL]"
