import json
import os
import sqlite3

import pytest

import llmsql.inference.inference as inference


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

    monkeypatch.setattr(inference, "LLM", lambda **_: DummyLLM())
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
