"""End-to-end evaluation of LLMSQL 2.0 predictions on the committed fixture."""

import json
import sqlite3

import pytest

from llmsql.evaluation.evaluate import evaluate
from llmsql.utils.evaluation_utils import evaluate_sample, execute_sql_with_timeout


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr("llmsql.evaluation.evaluate.log_mismatch", lambda **k: None)
    monkeypatch.setattr(
        "llmsql.evaluation.evaluate.print_summary", lambda *a, **k: None
    )


def _gold_completion(q):
    return f"Let me think.\n```sql\n{q['sql']}\n```"


def _run(outputs, workdir, tmp_path):
    return evaluate(
        outputs,
        version="2.0",
        workdir_path=str(workdir),
        save_report=str(tmp_path / "report.json"),
        show_mismatches=False,
    )


def test_gold_completions_score_100(llmsql2_questions, llmsql2_workdir, tmp_path):
    outputs = [
        {"question_id": q["question_id"], "completion": _gold_completion(q)}
        for q in llmsql2_questions
    ]
    report = _run(outputs, llmsql2_workdir, tmp_path)
    assert report["total"] == len(llmsql2_questions)
    assert report["accuracy"] == 1.0
    assert report["exact_string_matches"] == len(llmsql2_questions)
    assert report["sql_errors"] == 0
    assert report["mismatches"] == []
    cats = report["category_accuracy"]
    assert set(cats) == {"lookup", "convention", "text_quantity"}
    assert all(c["accuracy"] == 1.0 for c in cats.values())
    assert sum(c["total"] for c in cats.values()) == len(llmsql2_questions)


def test_answer_rows_as_literal_select_score_100(
    llmsql2_questions, llmsql2_workdir, tmp_path
):
    """A query that just returns the answer values (no table) is accepted."""

    def literal(v):
        return (
            "NULL"
            if v is None
            else repr(v)
            if not isinstance(v, str)
            else ("'" + v.replace("'", "''") + "'")
        )

    outputs = [
        {
            "question_id": q["question_id"],
            "completion": "```sql\nSELECT "
            + ", ".join(literal(v) for v in json.loads(q["answer"])[0])
            + ";\n```",
        }
        for q in llmsql2_questions
    ]
    report = _run(outputs, llmsql2_workdir, tmp_path)
    assert report["accuracy"] == 1.0


def test_wrong_completions_score_0(llmsql2_questions, llmsql2_workdir, tmp_path):
    outputs = [
        {"question_id": q["question_id"], "completion": "```sql\nSELECT 'nope';\n```"}
        for q in llmsql2_questions
    ]
    report = _run(outputs, llmsql2_workdir, tmp_path)
    assert report["accuracy"] == 0.0
    assert report["sql_errors"] == 0
    assert len(report["mismatches"]) == len(llmsql2_questions)
    m = report["mismatches"][0]
    assert m["prediction_results"] == [("nope",)]
    assert m["gold_results"] == json.loads(llmsql2_questions[0]["answer"])


def test_mixed_completions_and_counters(llmsql2_questions, llmsql2_workdir, tmp_path):
    q0, q1, q2, q3 = llmsql2_questions[:4]
    outputs = [
        {"question_id": q0["question_id"], "completion": _gold_completion(q0)},
        # no SQL at all -> counted as SQL error
        {"question_id": q1["question_id"], "completion": "I don't know."},
        # invalid SQL -> SQL error
        {
            "question_id": q2["question_id"],
            "completion": "```sql\nSELECT * FROM nope\n```",
        },
        # NULL result -> pred_none
        {"question_id": q3["question_id"], "completion": "```sql\nSELECT NULL\n```"},
    ]
    report = _run(outputs, llmsql2_workdir, tmp_path)
    assert report["matches"] == 1
    assert report["accuracy"] == 0.25
    assert report["sql_errors"] == 2
    assert report["pred_none"] == 1
    assert report["gold_none"] == 0


def test_evaluate_sample_v2_last_block_wins(llmsql2_questions, llmsql2_workdir):
    q = llmsql2_questions[0]
    conn = sqlite3.connect(llmsql2_workdir / "sqlite_tables.db")
    item = {
        "question_id": q["question_id"],
        "completion": "```sql\nSELECT 'draft'\n```\nFinal:\n" + _gold_completion(q),
    }
    is_match, mismatch, m = evaluate_sample(item, {q["question_id"]: q}, conn)
    assert is_match == 1 and mismatch is None
    item = {
        "question_id": q["question_id"],
        "completion": _gold_completion(q) + "\nFinal:\n```sql\nSELECT 'draft'\n```",
    }
    is_match, mismatch, m = evaluate_sample(item, {q["question_id"]: q}, conn)
    assert is_match == 0 and mismatch is not None
    conn.close()


def test_execute_sql_with_timeout_interrupts_long_queries():
    conn = sqlite3.connect(":memory:")
    endless = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
        "SELECT count(*) FROM c;"
    )
    assert execute_sql_with_timeout(conn, endless, timeout=0.2) is None
    # the progress handler is removed afterwards
    assert conn.execute("SELECT 1").fetchall() == [(1,)]
    assert execute_sql_with_timeout(conn, "SELECT 2", timeout=None) == [(2,)]
    conn.close()
