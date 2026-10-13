"""Tests for resuming an interrupted inference run (issue #122)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmsql.inference.inference_function import inference_function
from llmsql.utils.inference_utils import (
    prepare_benchmark,
    prepare_output_file,
    skip_completed,
)


class _Crash(BaseException):
    """Stands in for Ctrl-C / a killed process."""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _make_fixtures(tmp_path: Path, n_questions: int = 5) -> Path:
    """Populate tmp_path with the benchmark files so no download happens."""
    questions = [
        {"question_id": f"q{i}", "question": f"Q{i}?", "table_id": "t1"}
        for i in range(n_questions)
    ]
    tables = [{"table_id": "t1", "header": ["col"], "types": ["text"], "rows": [["x"]]}]
    _write_jsonl(tmp_path / "questions.jsonl", questions)
    _write_jsonl(tmp_path / "tables.jsonl", tables)
    return tmp_path / "out.jsonl"


def _read_ids(path: Path) -> list[str]:
    return [
        json.loads(line)["question_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_interrupted_run_then_resume_has_one_line_per_question(tmp_path):
    """
    Regression test for #122: an interrupted run followed by ``resume=True``
    must end up with exactly one line per question.

    With the old unconditional ``overwrite_jsonl`` the retry starts from an
    empty file and re-generates the questions that were already answered, so
    those ids appear twice.
    """
    out = _make_fixtures(tmp_path)
    calls = {"n": 0}

    async def crashing(prompt, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            raise _Crash
        return "SELECT 1"

    with pytest.raises(_Crash):
        inference_function(
            inference_function=crashing,
            max_concurrency=1,
            output_file=str(out),
            workdir_path=str(tmp_path),
        )

    assert len(_read_ids(out)) == 2, "the interrupted run should keep its progress"

    async def finishing(prompt, **kwargs):
        return "SELECT 2"

    results = inference_function(
        inference_function=finishing,
        max_concurrency=1,
        resume=True,
        output_file=str(out),
        workdir_path=str(tmp_path),
    )

    assert len(results) == 3, "only the missing questions should be generated"
    assert sorted(_read_ids(out)) == ["q0", "q1", "q2", "q3", "q4"]


def test_resume_skips_already_answered_questions(tmp_path):
    """``resume=True`` only generates the questions missing from the file."""
    out = _make_fixtures(tmp_path)
    _write_jsonl(
        out,
        [
            {"question_id": "q0", "completion": "SELECT 0"},
            {"question_id": "q2", "completion": "SELECT 2"},
        ],
    )

    seen: list[str] = []

    async def fake(prompt, **kwargs):
        seen.append(kwargs["question"]["question_id"])
        return "SELECT 1"

    inference_function(
        inference_function=fake,
        max_concurrency=1,
        resume=True,
        output_file=str(out),
        workdir_path=str(tmp_path),
    )

    assert sorted(seen) == ["q1", "q3", "q4"]
    assert sorted(_read_ids(out)) == ["q0", "q1", "q2", "q3", "q4"]


def test_without_resume_the_output_file_is_still_overwritten(tmp_path):
    """The default (``resume=False``) keeps overwriting, as before."""
    out = _make_fixtures(tmp_path)
    _write_jsonl(out, [{"question_id": "stale", "completion": "OLD"}] * 3)

    async def fake(prompt, **kwargs):
        return "SELECT 1"

    results = inference_function(
        inference_function=fake,
        max_concurrency=1,
        output_file=str(out),
        workdir_path=str(tmp_path),
    )

    assert len(results) == 5
    assert sorted(_read_ids(out)) == ["q0", "q1", "q2", "q3", "q4"]


def test_prepare_output_file_keeps_completions_when_resuming(tmp_path):
    """``resume=True`` returns the ids already in the file and leaves it alone."""
    out = tmp_path / "out.jsonl"
    _write_jsonl(out, [{"question_id": "q0", "completion": "SELECT 0"}])

    done = prepare_output_file(str(out), resume=True)

    assert done == {"q0"}
    assert _read_ids(out) == ["q0"]


def test_prepare_output_file_truncates_without_resume(tmp_path):
    out = tmp_path / "out.jsonl"
    _write_jsonl(out, [{"question_id": "q0", "completion": "OLD"}])

    done = prepare_output_file(str(out), resume=False)

    assert done == set()
    assert out.read_text(encoding="utf-8") == ""


def test_resume_ignores_truncated_last_line(tmp_path):
    """A half-written final line must not break resume (it is just skipped)."""
    out = tmp_path / "out.jsonl"
    out.write_text(
        '{"question_id": "q0", "completion": "SELECT 0"}\n{"question_id": "q1", "com',
        encoding="utf-8",
    )

    done = prepare_output_file(str(out), resume=True)

    assert done == {"q0"}


def test_resume_on_missing_file_starts_fresh(tmp_path):
    out = tmp_path / "does_not_exist.jsonl"

    done = prepare_output_file(str(out), resume=True)

    assert done == set()
    assert out.exists()


def test_skip_completed_keeps_order_and_unknown_questions():
    questions = [{"question_id": f"q{i}"} for i in range(4)]

    assert skip_completed(questions, set()) == questions
    assert skip_completed(questions, {"q1", "q3"}) == [
        {"question_id": "q0"},
        {"question_id": "q2"},
    ]
    assert skip_completed(questions, {"other"}) == questions


def test_prepare_benchmark_applies_limit(tmp_path):
    _make_fixtures(tmp_path, n_questions=10)

    run = prepare_benchmark(version="2.0", workdir_path=str(tmp_path), limit=3)

    assert [q["question_id"] for q in run.questions] == ["q0", "q1", "q2"]
    assert set(run.tables) == {"t1"}

    run = prepare_benchmark(version="2.0", workdir_path=str(tmp_path), limit=0.5)

    assert len(run.questions) == 5


@pytest.mark.parametrize("limit", [0, -1, 1.5, 0.0])
def test_prepare_benchmark_rejects_bad_limit(tmp_path, limit):
    _make_fixtures(tmp_path)

    with pytest.raises(ValueError):
        prepare_benchmark(version="2.0", workdir_path=str(tmp_path), limit=limit)
