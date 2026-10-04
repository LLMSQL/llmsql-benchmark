"""Tests for saving evaluation results in the leaderboard ``run.yaml`` format."""

import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
from unittest.mock import MagicMock

import pytest
import yaml

import llmsql
from llmsql import evaluate
from llmsql._cli.llmsql_cli import ParserCLI
from llmsql.config.config import DEFAULT_LLMSQL_VERSION
from llmsql.utils.leaderboard_utils import (
    build_leaderboard_record,
    write_leaderboard_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD_DIR = REPO_ROOT / "leaderboard"


def _load_generate_leaderboard():
    """Import leaderboard/generate_leaderboard.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "generate_leaderboard", LEADERBOARD_DIR / "generate_leaderboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def benchmark_files(temp_dir, dummy_db_file):
    """Prepare fake questions/db in workdir and a matching outputs file."""
    (temp_dir / "questions.jsonl").write_text(
        json.dumps(
            {"question_id": 1, "table_id": 1, "question": "Q", "sql": "SELECT 1"}
        )
        + "\n"
        + json.dumps(
            {"question_id": 2, "table_id": 1, "question": "Q2", "sql": "SELECT 2"}
        )
    )
    shutil.copy(dummy_db_file, temp_dir / "sqlite_tables.db")

    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(
        json.dumps({"question_id": 1, "completion": "SELECT 1"})
        + "\n"
        + json.dumps({"question_id": 2, "completion": "SELECT 3"})
    )
    return outputs_path


def test_evaluate_saves_leaderboard_yaml(temp_dir, benchmark_files):
    yaml_path = temp_dir / "nested" / "run.yaml"
    report_path = temp_dir / "report.json"

    report = evaluate(
        outputs=str(benchmark_files),
        workdir_path=str(temp_dir),
        save_report=str(report_path),
        show_mismatches=False,
        model_name="org/my-model",
        save_leaderboard_yaml=str(yaml_path),
        run_metadata={
            "type": "open-source",
            "model": {"dtype": "bfloat16"},
            "inference": {
                "backend": "vllm",
                "arguments": {"num_fewshots": 5, "seed": 42},
            },
        },
    )

    assert report["model_name"] == "org/my-model"
    assert report["accuracy"] == 0.5

    saved_report = json.loads(report_path.read_text())
    assert saved_report["model_name"] == "org/my-model"
    assert saved_report["version"] == DEFAULT_LLMSQL_VERSION

    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text())

    # auto-filled fields
    assert data["model"]["name"] == "org/my-model"
    assert data["llmsql"]["version"] == llmsql.__version__
    assert data["version"] == DEFAULT_LLMSQL_VERSION
    assert data["python_version"] == platform.python_version()
    assert data["os_name"]
    assert data["date"] is not None
    assert data["results"]["execution_accuracy"] == 0.5
    assert data["results"]["num_samples"] == 2
    assert data["results"]["answers_path"] == str(benchmark_files)

    # merged metadata keeps the auto-filled siblings
    assert data["type"] == "open-source"
    assert data["model"]["dtype"] == "bfloat16"
    assert data["model"]["revision"] is None
    assert data["inference"]["backend"] == "vllm"
    assert data["inference"]["arguments"] == {"num_fewshots": 5, "seed": 42}

    # key order follows the leaderboard run.yaml layout
    assert list(data)[:4] == ["date", "model", "type", "llmsql"]
    assert list(data)[-1] == "results"


def test_evaluate_does_not_save_yaml_by_default(temp_dir, benchmark_files):
    report = evaluate(
        outputs=str(benchmark_files),
        workdir_path=str(temp_dir),
        save_report=str(temp_dir / "report.json"),
        show_mismatches=False,
    )

    assert report["model_name"] is None
    assert not list(temp_dir.rglob("*.yaml"))


def test_evaluate_dict_outputs_yaml_has_no_answers_path(temp_dir, benchmark_files):
    yaml_path = temp_dir / "run.yaml"
    evaluate(
        outputs=[{"question_id": 1, "completion": "SELECT 1"}],
        workdir_path=str(temp_dir),
        save_report=str(temp_dir / "report.json"),
        show_mismatches=False,
        save_leaderboard_yaml=str(yaml_path),
    )

    data = yaml.safe_load(yaml_path.read_text())
    assert data["model"]["name"] is None
    assert data["results"]["answers_path"] is None
    assert data["results"]["execution_accuracy"] == 1.0


def test_build_leaderboard_record_matches_existing_leaderboard_layout():
    record = build_leaderboard_record(accuracy=0.123456, total=10, version="2.0")
    assert record["results"]["execution_accuracy"] == 0.1235

    existing = next(LEADERBOARD_DIR.rglob("run.yaml"))
    existing_data = yaml.safe_load(existing.read_text())

    # every top-level / model / llmsql / inference / results key used in the
    # committed leaderboard files is present in the generated record
    assert set(existing_data) <= set(record)
    for section in ("model", "llmsql", "inference", "results"):
        assert set(existing_data[section]) <= set(record[section])


def test_build_leaderboard_record_does_not_mutate_metadata():
    metadata = {"inference": {"arguments": {"num_fewshots": 0}}}
    record = build_leaderboard_record(
        accuracy=1.0, total=1, version="2.0", run_metadata=metadata
    )
    record["inference"]["arguments"]["num_fewshots"] = 99
    assert metadata == {"inference": {"arguments": {"num_fewshots": 0}}}


def test_generated_yaml_is_consumable_by_generate_leaderboard(temp_dir):
    gen = _load_generate_leaderboard()

    full = build_leaderboard_record(
        accuracy=0.75,
        total=4,
        version="2.0",
        model_name="org/full",
        run_metadata={
            "type": "proprietary",
            "inference": {"backend": "api", "arguments": {"num_fewshots": 1}},
        },
    )
    minimal = build_leaderboard_record(
        accuracy=0.5, total=4, version="2.0", model_name="org/minimal"
    )
    write_leaderboard_yaml(temp_dir / "full" / "run.yaml", full)
    write_leaderboard_yaml(temp_dir / "minimal" / "run.yaml", minimal)
    (temp_dir / "broken").mkdir()
    (temp_dir / "broken" / "run.yaml").write_text("model:\n  name: x\n")

    rows = gen.collect_rows(temp_dir)

    assert [r["model"] for r in rows] == ["org/full", "org/minimal"]
    assert rows[0] == {
        "model": "org/full",
        "type": "proprietary",
        "fewshots": 1,
        "backend": "api",
        "accuracy": 0.75,
        "date": str(full["date"]),
    }
    assert rows[1]["fewshots"] is None
    assert rows[1]["backend"] == ""


def test_generate_leaderboard_still_reads_committed_runs():
    gen = _load_generate_leaderboard()
    rows = gen.collect_rows()
    assert rows
    assert all(r["model"] and r["accuracy"] is not None for r in rows)


def test_cli_passes_leaderboard_args(monkeypatch, temp_dir):
    mock_evaluate = MagicMock(return_value={})
    monkeypatch.setattr("llmsql._cli.evaluate.evaluate", mock_evaluate)

    metadata_path = temp_dir / "meta.yaml"
    metadata_path.write_text("type: open-source\ninference:\n  backend: vllm\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llmsql",
            "evaluate",
            "--outputs",
            "dummy.jsonl",
            "--model-name",
            "org/my-model",
            "--save-leaderboard-yaml",
            "run.yaml",
            "--run-metadata",
            str(metadata_path),
        ],
    )

    cli = ParserCLI()
    cli.execute(cli.parse_args())

    kwargs = mock_evaluate.call_args.kwargs
    assert kwargs["model_name"] == "org/my-model"
    assert kwargs["save_leaderboard_yaml"] == "run.yaml"
    assert kwargs["run_metadata"] == {
        "type": "open-source",
        "inference": {"backend": "vllm"},
    }


def test_cli_leaderboard_args_default_to_none(monkeypatch):
    mock_evaluate = MagicMock(return_value={})
    monkeypatch.setattr("llmsql._cli.evaluate.evaluate", mock_evaluate)
    monkeypatch.setattr(sys, "argv", ["llmsql", "evaluate", "--outputs", "dummy.jsonl"])

    cli = ParserCLI()
    cli.execute(cli.parse_args())

    kwargs = mock_evaluate.call_args.kwargs
    assert kwargs["model_name"] is None
    assert kwargs["save_leaderboard_yaml"] is None
    assert kwargs["run_metadata"] is None
