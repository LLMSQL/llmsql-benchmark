"""Tests for the LLMSQL 2.0 prompt (must reproduce the dataset's `prompt` field)."""

import json
import os
from pathlib import Path

import pytest

from llmsql.prompts.prompts import build_prompt_v2, render_table_schema_v2


def test_prompt_matches_dataset_field(llmsql2_questions, llmsql2_tables):
    assert any(len(q["tables"]) > 1 for q in llmsql2_questions)
    for q in llmsql2_questions:
        tables = [llmsql2_tables[t] for t in q["tables"]]
        assert build_prompt_v2(q["question"], tables) == q["prompt"], q["question_id"]


@pytest.mark.skipif(
    not os.environ.get("LLMSQL2_RELEASE_DIR"),
    reason="set LLMSQL2_RELEASE_DIR to a local copy of llmsql-bench/llmsql-2.0",
)
def test_prompt_matches_full_release():
    release = Path(os.environ["LLMSQL2_RELEASE_DIR"])
    with open(release / "tables.jsonl", encoding="utf-8") as f:
        tables = {t["table_id"]: t for t in map(json.loads, f)}
    with open(release / "questions.jsonl", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    assert len(questions) == 2000
    for q in questions:
        prompt = build_prompt_v2(q["question"], [tables[t] for t in q["tables"]])
        assert prompt == q["prompt"], q["question_id"]


TABLE = {
    "table_id": "1-1-1",
    "page_title": "Café season",
    "section_title": "Results",
    "header": ["Week", "Result", "Attendance"],
    "types": ["real", "text", "TEXT"],
    "rows": [[1, "W 21–7", "1,000"], [2, "L 0–3", "2,000"]],
}


def test_render_table_schema_v2_layout():
    assert render_table_schema_v2(TABLE, n_rows=1) == (
        'CREATE TABLE "1-1-1" ("Week" REAL, "Result" TEXT, "Attendance" TEXT);\n'
        "-- Wikipedia: Café season / Results\n"
        "-- first 1 of 2 rows:\n"
        '[1, "W 21–7", "1,000"]'
    )


@pytest.mark.parametrize("n_rows", [2, 3, None])
def test_render_table_schema_v2_all_rows(n_rows):
    out = render_table_schema_v2(TABLE, n_rows=n_rows)
    assert "-- all rows:\n" in out
    assert out.endswith('[2, "L 0–3", "2,000"]')


def test_build_prompt_v2_joins_tables_and_question():
    other = {**TABLE, "table_id": "1-1-2"}
    prompt = build_prompt_v2("How many?", [TABLE, other])
    assert prompt.startswith("You are an expert SQLite query writer.")
    assert "Return only the SQL in a ```sql block." in prompt
    assert prompt.index('CREATE TABLE "1-1-1"') < prompt.index('CREATE TABLE "1-1-2"')
    assert '"2,000"]\n\nCREATE TABLE "1-1-2"' in prompt
    assert prompt.endswith("\n\nQuestion: How many?")
