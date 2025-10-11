import json
from pathlib import Path
import sqlite3

import pytest

from llmsql import LLMSQLEvaluator


@pytest.mark.asyncio
async def test_connect_and_close(dummy_db_file):
    evaluator = LLMSQLEvaluator()
    evaluator.connect(dummy_db_file)
    assert isinstance(evaluator.conn, sqlite3.Connection)
    evaluator.close()
    assert evaluator.conn is None


@pytest.mark.asyncio
async def test_download_file_is_called(monkeypatch, temp_dir):
    evaluator = LLMSQLEvaluator(workdir_path=temp_dir)

    def fake_download(*args, **kwargs):
        file_path = temp_dir / "fake_file.txt"
        file_path.write_text("content")
        return str(file_path)

    monkeypatch.setattr("llmsql.evaluation.evaluator.hf_hub_download", fake_download)

    path = evaluator._download_file("fake_file.txt")
    assert Path(path).exists()


@pytest.mark.asyncio
async def test_evaluate_with_mock(monkeypatch, temp_dir, dummy_db_file):
    evaluator = LLMSQLEvaluator(workdir_path=temp_dir)

    # Fake questions.jsonl
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps({"question_id": 1, "question": "Sample quesiton", "sql": "SELECT 1"})
    )

    # Fake outputs.jsonl
    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(json.dumps({"question_id": 1, "predicted": "SELECT 1"}))

    # Monkeypatch dependencies
    monkeypatch.setattr(
        "llmsql.evaluation.evaluator.evaluate_sample",
        lambda *a, **k: (1, None, {"pred_none": 0, "gold_none": 0, "sql_error": 0}),
    )
    monkeypatch.setattr("llmsql.evaluation.evaluator.log_mismatch", lambda **k: None)
    monkeypatch.setattr(
        "llmsql.evaluation.evaluator.print_summary", lambda *a, **k: None
    )

    report = evaluator.evaluate(
        outputs_path=str(outputs_path),
        questions_path=str(questions_path),
        db_path=dummy_db_file,
        show_mismatches=False,
    )

    assert report["total"] == 1
    assert report["matches"] == 1
    assert report["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_connect_with_nonexistent_db():
    """Test that connecting to non-existent database raises FileNotFoundError."""
    evaluator = LLMSQLEvaluator()
    with pytest.raises(FileNotFoundError, match="Database not found"):
        evaluator.connect("/nonexistent/path/to/database.db")


@pytest.mark.asyncio
async def test_evaluate_saves_report(monkeypatch, temp_dir, dummy_db_file):
    """Test that save_report parameter creates a JSON report file."""
    evaluator = LLMSQLEvaluator(workdir_path=temp_dir)

    # Setup test files
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps({"question_id": 1, "question": "Test", "sql": "SELECT 1"})
    )

    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(json.dumps({"question_id": 1, "completion": "SELECT 1"}))

    report_path = temp_dir / "report.json"

    # Mock dependencies
    monkeypatch.setattr(
        "llmsql.evaluation.evaluator.evaluate_sample",
        lambda *a, **k: (1, None, {"pred_none": 0, "gold_none": 0, "sql_error": 0}),
    )
    monkeypatch.setattr("llmsql.evaluation.evaluator.log_mismatch", lambda **k: None)
    monkeypatch.setattr(
        "llmsql.evaluation.evaluator.print_summary", lambda *a, **k: None
    )

    evaluator.evaluate(
        outputs_path=str(outputs_path),
        questions_path=str(questions_path),
        db_path=dummy_db_file,
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
async def test_evaluate_with_mismatches(monkeypatch, temp_dir, dummy_db_file):
    """Test that mismatches are logged when show_mismatches=True."""
    evaluator = LLMSQLEvaluator(workdir_path=temp_dir)

    # Setup test files
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps({"question_id": 1, "question": "Test", "sql": "SELECT 1"})
    )

    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(json.dumps({"question_id": 1, "completion": "SELECT 2"}))

    mismatch_logged = []

    def mock_log_mismatch(**kwargs):
        mismatch_logged.append(kwargs)

    # Mock dependencies - return mismatch
    monkeypatch.setattr(
        "llmsql.evaluation.evaluator.evaluate_sample",
        lambda *a, **k: (
            0,
            {
                "question_id": 1,
                "question": "Test",
                "gold_sql": "SELECT 1",
                "model_output": "SELECT 2",
                "gold_results": [(1,)],
                "prediction_results": [(2,)],
            },
            {"pred_none": 0, "gold_none": 0, "sql_error": 0},
        ),
    )
    monkeypatch.setattr("llmsql.evaluation.evaluator.log_mismatch", mock_log_mismatch)
    monkeypatch.setattr(
        "llmsql.evaluation.evaluator.print_summary", lambda *a, **k: None
    )

    report = evaluator.evaluate(
        outputs_path=str(outputs_path),
        questions_path=str(questions_path),
        db_path=dummy_db_file,
        show_mismatches=True,
        max_mismatches=5,
    )

    # Verify mismatch was logged
    assert len(mismatch_logged) == 1
    assert mismatch_logged[0]["question_id"] == 1
    assert report["matches"] == 0
    assert len(report["mismatches"]) == 1
