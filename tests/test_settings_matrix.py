"""Smoke and unit tests for the settings-matrix comparison script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_matrix_module():
    spec = importlib.util.spec_from_file_location(
        "run_settings_matrix", ROOT / "scripts" / "run_settings_matrix.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


matrix = _load_matrix_module()


class TestMatrixExpansion:
    def test_profile_run_counts(self) -> None:
        expected = {"quick": 24, "standard": 72, "full": 288}
        for profile, axes in matrix.PROFILE_AXES.items():
            runs = matrix.expand_runs(axes, matrix.RunConfig())
            assert len(runs) == expected[profile], profile
            # every combination is distinct
            assert len({run.label(axes) for run in runs}) == len(runs)

    def test_axis_override(self) -> None:
        runs = matrix.expand_runs(["nlp", "style"], matrix.RunConfig())
        assert len(runs) == 6
        assert {run.nlp for run in runs} == {True, False}
        assert {run.style for run in runs} == {"mask", "label", "partial"}

    def test_pinned_backend_flows_through(self) -> None:
        runs = matrix.expand_runs(["style"], matrix.RunConfig(nlp_backend="deberta-v3-small"))
        assert all(run.nlp_backend == "deberta-v3-small" for run in runs)


class TestTraceHeuristics:
    @pytest.mark.parametrize(
        ("literal", "expected"),
        [
            ("0412 345 678", True),
            ("N1234567", True),
            ("2 1234 5678 1", True),
            ("12/03/2024", False),
            ("18 years", False),
            ("WorkSafe Victoria", False),
            ("one of three EL2s in the Migration directorate", False),
            ("jane.doe@example.com.au", True),
        ],
    )
    def test_trace_candidate(self, literal: str, expected: bool) -> None:
        assert matrix.trace_candidate(literal) is expected

    def test_partial_mask_leak_is_traced(self) -> None:
        assert matrix.trace_leaked("4111 1111 1111 1111", "**** **** **** 1111")
        assert not matrix.trace_leaked("4111 1111 1111 1111", "**** **** **** ****")

    def test_benign_collision_is_not_traced(self) -> None:
        # "1234" of the masked enrolment number also appears inside a legitimate
        # account number that is expected to survive.
        assert matrix.trace_leaked("2 1234 5678 1", "* **** **** * account 12345678")
        assert not matrix.trace_leaked(
            "2 1234 5678 1", "* **** **** * account 12345678", "account 12345678"
        )


class TestMatrixReport:
    def test_end_to_end_smoke_report(self, capsys) -> None:
        exit_code = matrix.main(
            ["--axes", "style", "--datasets", "canonical", "--quiet", "--color", "never"]
        )
        assert exit_code == 0
        report = capsys.readouterr().out
        for heading in (
            "SETTINGS MATRIX REPORT",
            "LEADERBOARD",
            "FULL RUN MATRIX",
            "SENSITIVITY",
            "RECOMMENDED SETTINGS",
        ):
            assert heading in report
        assert "style=partial" in report

    def test_json_dump(self, tmp_path, capsys) -> None:
        out_path = tmp_path / "matrix.json"
        exit_code = matrix.main(
            [
                "--axes",
                "nlp",
                "--datasets",
                "canonical",
                "--quiet",
                "--color",
                "never",
                "--json",
                str(out_path),
            ]
        )
        assert exit_code == 0
        capsys.readouterr()
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["axes"] == ["nlp"]
        assert len(payload["runs"]) == 2
        run = payload["runs"][0]
        assert "config" in run and "score" in run and "benchmark" in run
        assert run["review_units"] > 0
