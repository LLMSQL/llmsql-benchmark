import json
import os
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock

import pytest

import llmsql.inference.inference_vllm as inference_vllm


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def dummy_db_file(tmp_path):
    """Create a temporary SQLite DB file for testing, cleanup afterwards."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO test (name) VALUES ('Alice'), ('Bob')")
    conn.commit()
    conn.close()

    yield str(db_path)

    # cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def mock_llm(monkeypatch):
    """Mock vLLM LLM to avoid GPU/model loading."""

    class DummyOutput:
        def __init__(self, text="SELECT 1"):
            self.outputs = [type("Obj", (), {"text": text})()]

    class DummyLLM:
        def generate(self, prompts, sampling_params):
            return [DummyOutput(f"-- SQL for: {p}") for p in prompts]

    monkeypatch.setattr(inference_vllm, "LLM", lambda **_: DummyLLM())
    return DummyLLM()


@pytest.fixture
def fake_jsonl_files(tmp_path):
    """Create fake questions.jsonl and tables.jsonl."""
    qpath = tmp_path / "questions.jsonl"
    tpath = tmp_path / "tables.jsonl"

    questions = [
        {"question_id": "q1", "question": "How many users?", "table_id": "t1"},
        {"question_id": "q2", "question": "List names", "table_id": "t1"},
    ]
    tables = [
        {
            "table_id": "t1",
            "header": ["id", "name"],
            "types": ["int", "text"],
            "rows": [[1, "Alice"], [2, "Bob"]],
        }
    ]

    qpath.write_text("\n".join(json.dumps(q) for q in questions))
    tpath.write_text("\n".join(json.dumps(t) for t in tables))

    return str(qpath), str(tpath)


@pytest.fixture
def mock_utils(mocker, tmp_path):
    """Mock all underlying I/O + DB functions."""
    # load questions
    mocker.patch(
        "llmsql.evaluation.evaluate.load_jsonl_dict_by_key",
        return_value={1: {"question_id": 1, "gold": "SELECT 1"}},
    )

    # predictions loader
    mocker.patch(
        "llmsql.evaluation.evaluate.load_jsonl",
        return_value=[{"question_id": 1, "completion": "SELECT 1"}],
    )

    # DB connection
    fake_conn = MagicMock()
    mocker.patch("llmsql.evaluation.evaluate.connect_sqlite", return_value=fake_conn)

    # evaluate_sample → always correct prediction
    mocker.patch(
        "llmsql.evaluation.evaluate.evaluate_sample",
        return_value=(1, None, {"pred_none": 0, "gold_none": 0, "sql_error": 0}),
    )

    # rich logging
    mocker.patch("llmsql.evaluation.evaluate.log_mismatch")
    mocker.patch("llmsql.evaluation.evaluate.print_summary")

    # download files
    mocker.patch(
        "llmsql.evaluation.evaluate.download_benchmark_file",
        side_effect=lambda filename, wd: str(Path(wd) / filename),
    )

    # report writer
    mocker.patch("llmsql.evaluation.evaluate.save_json_report")

    return tmp_path
