"""Tests for the async inference_function implementation."""

from __future__ import annotations

import asyncio
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


def test_rate_limiter_rejects_non_positive_rpm():
    from llmsql.inference.inference_function import _AsyncRateLimiter

    with pytest.raises(ValueError, match="requests_per_minute must be > 0"):
        _AsyncRateLimiter(0)

    with pytest.raises(ValueError):
        _AsyncRateLimiter(-5)


@pytest.mark.asyncio
async def test_rate_limiter_waits(monkeypatch):
    from llmsql.inference.inference_function import _AsyncRateLimiter

    limiter = _AsyncRateLimiter(requests_per_minute=60)  # 1 request/sec

    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    # First call should not sleep
    await limiter.acquire()
    # Second call should trigger wait
    await limiter.acquire()

    assert len(sleep_calls) == 1
    assert sleep_calls[0] > 0


def test_rejects_non_callable_inference_function(tmp_path):
    _make_fixtures(tmp_path)

    with pytest.raises(TypeError, match="must be callable"):
        inference_function(
            inference_function="not-a-function",  # type: ignore
            output_file=str(tmp_path / "out.jsonl"),
            workdir_path=str(tmp_path),
        )


@pytest.mark.parametrize("bad_limit", [0.0, -0.1, 1.5])
def test_limit_float_out_of_range(tmp_path, bad_limit):
    _make_fixtures(tmp_path)

    async def fake_infer(prompt, **kwargs):
        return "SELECT 1"

    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        inference_function(
            inference_function=fake_infer,
            output_file=str(tmp_path / "out.jsonl"),
            workdir_path=str(tmp_path),
            limit=bad_limit,
        )


@pytest.mark.parametrize("bad_limit", [0, -1, "foo", None])
def test_limit_invalid_type_or_value(tmp_path, bad_limit):
    _make_fixtures(tmp_path)

    async def fake_infer(prompt, **kwargs):
        return "SELECT 1"

    if bad_limit is None:
        # None is valid (means no limit), skip
        return

    with pytest.raises(ValueError, match="must be a positive integer"):
        inference_function(
            inference_function=fake_infer,
            output_file=str(tmp_path / "out.jsonl"),
            workdir_path=str(tmp_path),
            limit=bad_limit,  # type: ignore[arg-type]
        )


def test_runs_inside_existing_event_loop(monkeypatch, tmp_path):
    _make_fixtures(tmp_path)

    async def fake_infer(prompt, **kwargs):
        return "SELECT 1"

    applied = {"called": False}

    def fake_apply(loop):
        applied["called"] = True

    monkeypatch.setattr("nest_asyncio.apply", fake_apply)

    # Create a real loop we control
    real_loop = asyncio.new_event_loop()

    class FakeLoop:
        def is_running(self):
            return True

        def run_until_complete(self, coro):
            return real_loop.run_until_complete(coro)

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: FakeLoop())

    try:
        results = inference_function(
            inference_function=fake_infer,
            output_file=str(tmp_path / "out.jsonl"),
            workdir_path=str(tmp_path),
        )
    finally:
        real_loop.close()

    assert applied["called"] is True
    assert len(results) > 0
