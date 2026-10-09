import glob
import json
import os
from pathlib import Path
import shutil
import sqlite3
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_evaluation_results():
    """Remove evaluation_results* files produced during tests."""
    yield
    for path in glob.glob("evaluation_results*"):
        os.remove(path)


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
        return_value=(
            1,
            None,
            {"pred_none": 0, "gold_none": 0, "sql_error": 0, "exact_string_match": 0},
        ),
    )

    # rich logging
    mocker.patch("llmsql.evaluation.evaluate.log_mismatch")
    mocker.patch("llmsql.evaluation.evaluate.print_summary")

    # benchmark file resolver
    mocker.patch(
        "llmsql.evaluation.evaluate._maybe_download",
        side_effect=lambda repo_id, filename, workdir: str(Path(workdir) / filename),
    )

    # report writer
    mocker.patch("llmsql.evaluation.evaluate.save_json_report")

    return tmp_path


# --- LLMSQL 2.0 fixture -----------------------------------------------------
# 21 questions of the LLMSQL 2.0 release with their tables and a tiny SQLite DB,
# built by tests/fixtures/llmsql2/build_fixture.py.
LLMSQL2_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "llmsql2"
LLMSQL2_FILES = ("questions.jsonl", "tables.jsonl", "sqlite_tables.db")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def llmsql2_questions() -> list[dict]:
    return _read_jsonl(LLMSQL2_FIXTURE_DIR / "questions.jsonl")


@pytest.fixture
def llmsql2_tables() -> dict[str, dict]:
    return {t["table_id"]: t for t in _read_jsonl(LLMSQL2_FIXTURE_DIR / "tables.jsonl")}


@pytest.fixture
def llmsql2_workdir(tmp_path) -> Path:
    """A workdir pre-populated with the LLMSQL 2.0 fixture files, so that
    ``_maybe_download`` finds them cached and never touches the network."""
    workdir = tmp_path / "llmsql2_workdir"
    workdir.mkdir()
    for name in LLMSQL2_FILES:
        shutil.copy(LLMSQL2_FIXTURE_DIR / name, workdir / name)
    return workdir
