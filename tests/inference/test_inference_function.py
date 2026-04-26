"""Tests for the async inference_function implementation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmsql.inference.inference_function import inference_function


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _make_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    questions = [
        {"question_id": "q1", "question": "What is 1+1?", "table_id": "t1"},
        {"question_id": "q2", "question": "What is 2+2?", "table_id": "t1"},
    ]
    tables = [{"table_id": "t1", "header": ["col"], "types": ["text"], "rows": [["x"]]}]
    qpath = tmp_path / "questions.jsonl"
    tpath = tmp_path / "tables.jsonl"
    outpath = tmp_path / "out.jsonl"
    _write_jsonl(qpath, questions)
    _write_jsonl(tpath, tables)
    return qpath, tpath, outpath


def test_runs_with_custom_async_callable(tmp_path):
    _make_fixtures(tmp_path)

    async def fake_infer(prompt, **kwargs):
        assert isinstance(prompt, str)
        assert kwargs["temperature"] == 0.0
        assert "question" in kwargs
        assert "table" in kwargs
        return "SELECT 1"

    results = inference_function(
        inference_function=fake_infer,
        function_kwargs={"temperature": 0.0},
        output_file=str(tmp_path / "out.jsonl"),
        workdir_path=str(tmp_path),
    )

    assert len(results) == 2
    assert all(r["completion"] == "SELECT 1" for r in results)


def test_limit_works(tmp_path):
    _make_fixtures(tmp_path)

    async def fake_infer(prompt, **kwargs):
        return "SELECT 1"

    results = inference_function(
        inference_function=fake_infer,
        output_file=str(tmp_path / "out.jsonl"),
        workdir_path=str(tmp_path),
        limit=1,
    )

    assert len(results) == 1


def test_rejects_non_async_result(tmp_path):
    _make_fixtures(tmp_path)

    def bad_infer(prompt, **kwargs):
        return "not-awaitable"

    with pytest.raises(TypeError, match="must return an awaitable"):
        inference_function(
            inference_function=bad_infer,  # type: ignore[arg-type]
            output_file=str(tmp_path / "out.jsonl"),
            workdir_path=str(tmp_path),
        )
