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
