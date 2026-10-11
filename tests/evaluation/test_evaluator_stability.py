import json
from pathlib import Path
import shutil

import pytest

from llmsql import evaluate


@pytest.mark.asyncio
async def test_evaluate_with_mock(monkeypatch, temp_dir, dummy_db_file):
    # Fake questions.jsonl
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": 1,
                "table_id": 1,
                "question": "Sample quesiton",
                "sql": "SELECT 1",
            }
        )
    )

    shutil.copy(dummy_db_file, temp_dir / "sqlite_tables.db")

    # Fake outputs.jsonl
    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(json.dumps({"question_id": 1, "completion": "SELECT 1"}))

    # Monkeypatch dependencies
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.evaluate_sample",
        lambda *a, **k: (
            1,
            None,
            {"pred_none": 0, "gold_none": 0, "sql_error": 0, "exact_string_match": 1},
        ),
    )
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)

    report = evaluate(
        outputs=str(outputs_path),
        workdir_path=str(temp_dir),
        show_mismatches=False,
    )

    assert report["total"] == 1
    assert report["matches"] == 1
    assert report["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_evaluate_saves_report(monkeypatch, temp_dir, dummy_db_file):
    """Test that save_report parameter creates a JSON report file."""

    # Setup test files
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {"question_id": 1, "table_id": 1, "question": "Test", "sql": "SELECT 1"}
        )
    )
    shutil.copy(dummy_db_file, temp_dir / "sqlite_tables.db")

    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(json.dumps({"question_id": 1, "completion": "SELECT 1"}))

    report_path = temp_dir / "report.json"

    # Mock dependencies
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.evaluate_sample",
        lambda *a, **k: (
            1,
            None,
            {"pred_none": 0, "gold_none": 0, "sql_error": 0, "exact_string_match": 1},
        ),
    )
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)

    evaluate(
        outputs=str(outputs_path),
        workdir_path=str(temp_dir),
        save_report=str(report_path),
        show_mismatches=False,
    )

    # Verify report file was created
    assert report_path.exists()
    with open(report_path, encoding="utf-8") as f:
        saved_report = json.load(f)
    assert saved_report["total"] == 1
    assert saved_report["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_evaluate_with_jsonl_file(monkeypatch, temp_dir, dummy_db_file):
    # Create fake questions.jsonl
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {"question_id": 1, "table_id": 1, "question": "Sample", "sql": "SELECT 1"}
        )
    )
    shutil.copy(dummy_db_file, temp_dir / "sqlite_tables.db")

    # Create fake outputs.jsonl
    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(json.dumps({"question_id": 1, "completion": "SELECT 1"}))

    # Mock dependencies
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.evaluate_sample",
        lambda *a, **k: (
            1,
            None,
            {"pred_none": 0, "gold_none": 0, "sql_error": 0, "exact_string_match": 1},
        ),
    )
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.connect_sqlite", lambda db: dummy_db_file
    )
    monkeypatch.setattr(
        "llmsql.utils.utils.save_json_report", lambda path, report: None
    )

    report = evaluate(
        outputs=str(outputs_path),
        workdir_path=str(temp_dir),
        show_mismatches=False,
    )

    assert report["total"] == 1
    assert report["matches"] == 1
    assert report["accuracy"] == 1.0
    assert report["input_mode"] == "jsonl_path"


@pytest.mark.asyncio
async def test_evaluate_with_dict_list(monkeypatch, temp_dir, dummy_db_file):
    # Prepare fake questions dict
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {"question_id": 1, "table_id": 1, "question": "Sample", "sql": "SELECT 1"}
        )
    )
    shutil.copy(dummy_db_file, temp_dir / "sqlite_tables.db")

    # Output as a list of dicts
    outputs_list = [{"question_id": 1, "completion": "SELECT 1"}]

    # Mock dependencies
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.evaluate_sample",
        lambda *a, **k: (
            1,
            None,
            {"pred_none": 0, "gold_none": 0, "sql_error": 0, "exact_string_match": 1},
        ),
    )
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.connect_sqlite", lambda db: dummy_db_file
    )
    monkeypatch.setattr(
        "llmsql.utils.utils.load_jsonl_dict_by_key",
        lambda path, key: {1: {"question_id": 1}},
    )
    monkeypatch.setattr(
        "llmsql.utils.utils.save_json_report", lambda path, report: None
    )

    report = evaluate(
        outputs=outputs_list,
        workdir_path=str(temp_dir),
        show_mismatches=False,
    )

    assert report["total"] == 1
    assert report["matches"] == 1
    assert report["accuracy"] == 1.0
    assert report["input_mode"] == "dict_list"


def test_evaluate_with_list_outputs(mock_utils, mocker):
    outputs = [{"question_id": 1, "completion": "SELECT 1"}]

    report = evaluate(outputs, workdir_path=str(mock_utils))

    assert report["total"] == 1
    assert report["matches"] == 1
    assert report["accuracy"] == 1.0
    assert report["input_mode"] == "dict_list"


def test_evaluate_with_jsonl_path(mock_utils, mocker):
    # fake jsonl file
    jsonl_path = mock_utils / "preds.jsonl"
    jsonl_path.write_text("dummy", encoding="utf-8")

    report = evaluate(str(jsonl_path), workdir_path=str(mock_utils))

    assert report["total"] == 1
    assert report["input_mode"] == "jsonl_path"


def test_auto_temp_workdir_is_used_when_not_provided(mocker):
    resolve = mocker.patch(
        "llmsql.evaluation.evaluate.resolve_workdir_path",
        return_value=Path("/tmp/llmsql-test"),
    )

    evaluate([{"question_id": 1, "completion": "x"}], workdir_path=None)
    resolve.assert_called_once_with(None)


def test_download_occurs_if_files_missing(mock_utils, mocker):
    dl = mocker.patch("llmsql.evaluation.evaluate._maybe_download")
    dl.side_effect = [
        str(mock_utils / "questions.jsonl"),
        str(mock_utils / "sqlite_tables.db"),
    ]

    evaluate(
        [{"question_id": 1, "completion": "SELECT 1"}],
        workdir_path=str(mock_utils),
    )

    assert dl.call_count == 2  # questions + sqlite


def test_no_report_file_is_written_when_save_report_is_none(mock_utils, mocker):
    """save_report=None 时不能再往当前目录丢一个随机名的报告文件

    报告字典本来就会作为返回值交回调用方，没必要每次调用都落盘一份
    （里面带全部 mismatch 的模型输出，完整跑一次能到几十 MB）。
    """
    save = mocker.patch("llmsql.evaluation.evaluate.save_json_report")

    report = evaluate(
        [{"question_id": 1, "completion": "SELECT 1"}],
        workdir_path=str(mock_utils),
        save_report=None,
    )

    save.assert_not_called()
    assert report["total"] == 1


def test_report_is_written_to_the_given_path(mock_utils, mocker, tmp_path):
    """显式给了 save_report 时才落盘，且写到指定的路径"""
    save = mocker.patch("llmsql.evaluation.evaluate.save_json_report")
    report_path = tmp_path / "report.json"

    evaluate(
        [{"question_id": 1, "completion": "SELECT 1"}],
        workdir_path=str(mock_utils),
        save_report=str(report_path),
    )

    args, kwargs = save.call_args
    assert args[0] == str(report_path)


def test_mismatch_handling(mock_utils, mocker):
    """Test branch where a mismatch is returned."""
    mocker.patch(
        "llmsql.evaluation.evaluate.evaluate_sample",
        return_value=(
            0,
            {"info": "bad"},
            {"pred_none": 0, "gold_none": 0, "sql_error": 0, "exact_string_match": 1},
        ),
    )

    log_mis = mocker.patch("llmsql.evaluation.evaluate.log_mismatch")

    report = evaluate(
        [{"question_id": 1, "completion": "SELECT X"}],
        workdir_path=str(mock_utils),
        max_mismatches=3,
    )

    assert report["matches"] == 0
    assert len(report["mismatches"]) == 1
    log_mis.assert_called_once()
