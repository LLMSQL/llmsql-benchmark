"""Tests for the async inference_api implementation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import llmsql.inference.inference_api as api_mod
from llmsql.inference.inference_api import (
    _AsyncRateLimiter,
    _HTTPStatusError,
    _retry_delay,
    inference_api,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _make_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return (questions_path, tables_path, out_path) pre-populated with minimal data."""
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


def _fake_http_response(content: str = "SELECT 1") -> MagicMock:
    """Build a mock that looks like a successful aiohttp response."""
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(
        return_value={"choices": [{"message": {"content": content}}]}
    )
    # Support async context-manager usage: `async with session.post(...) as resp`
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx


# ---------------------------------------------------------------------------
# _AsyncRateLimiter unit tests
# ---------------------------------------------------------------------------


class TestAsyncRateLimiter:
    def test_raises_on_non_positive_rpm(self):
        with pytest.raises(ValueError, match="requests_per_minute must be > 0"):
            _AsyncRateLimiter(0)
        with pytest.raises(ValueError, match="requests_per_minute must be > 0"):
            _AsyncRateLimiter(-10)

    def test_no_limit_returns_immediately(self):
        """acquire() with rpm=None should not sleep at all."""
        limiter = _AsyncRateLimiter(None)

        async def _run():
            t0 = time.monotonic()
            await limiter.acquire()
            await limiter.acquire()
            await limiter.acquire()
            return time.monotonic() - t0

        elapsed = asyncio.run(_run())
        assert elapsed < 0.05  # well under any sleep threshold

    def test_slots_are_spaced_correctly(self):
        """With 60 RPM the limiter should space 3 calls ~1 s apart (2 s total)."""
        limiter = _AsyncRateLimiter(60)  # 1 request per second

        timestamps: list[float] = []

        async def _run():
            for _ in range(3):
                await limiter.acquire()
                timestamps.append(time.monotonic())

        asyncio.run(_run())
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        for gap in gaps:
            assert gap == pytest.approx(1.0, abs=0.15), f"gap {gap:.3f}s not ~1 s"

    def test_concurrent_slots_are_serialised(self):
        """
        Multiple concurrent coroutines must each get a distinct slot —
        _next_allowed must advance monotonically even under concurrency.
        """
        limiter = _AsyncRateLimiter(600)  # 0.1 s interval — fast enough for tests
        TASKS = 5
        slots: list[float] = []

        async def _worker():
            await limiter.acquire()
            slots.append(time.monotonic())

        async def _run():
            await asyncio.gather(*[_worker() for _ in range(TASKS)])

        asyncio.run(_run())
        slots.sort()
        gaps = [slots[i + 1] - slots[i] for i in range(len(slots) - 1)]
        for gap in gaps:
            # Each gap should be ≥ interval minus a small scheduling tolerance.
            assert gap >= 0.08, f"slots overlap too closely: gap={gap:.4f}s"


# ---------------------------------------------------------------------------
# _post_chat_completion_async unit test
# ---------------------------------------------------------------------------


class TestPostChatCompletionAsync:
    def test_returns_parsed_response(self):
        mock_ctx = _fake_http_response("SELECT 42")

        async def _run():
            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_ctx)
            return await api_mod._post_chat_completion_async(
                session=mock_session,
                base_url="http://fake/v1",
                endpoint="chat/completions",
                payload={"model": "x", "messages": []},
                timeout=5.0,
            )

        result = asyncio.run(_run())
        assert result["choices"][0]["message"]["content"] == "SELECT 42"

    def test_raises_on_missing_choices(self):
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value={"error": "oops"})
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        async def _run():
            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_ctx)
            return await api_mod._post_chat_completion_async(
                session=mock_session,
                base_url="http://fake/v1",
                endpoint="chat/completions",
                payload={},
                timeout=5.0,
            )

        with pytest.raises(ValueError, match="does not contain `choices`"):
            asyncio.run(_run())

    def test_http_error_status_raises_with_retry_after(self):
        """A 4xx/5xx answer becomes ``_HTTPStatusError`` carrying ``Retry-After``."""
        resp = _FakeHttpResponse(
            status=429, payload=None, headers={"Retry-After": "2"}, text="slow down"
        )
        session = MagicMock()
        session.post = MagicMock(return_value=resp)

        async def _run():
            return await api_mod._post_chat_completion_async(
                session=session,
                base_url="http://fake/v1",
                endpoint="chat/completions",
                payload={},
                timeout=5.0,
            )

        with pytest.raises(_HTTPStatusError) as exc_info:
            asyncio.run(_run())

        assert exc_info.value.status == 429
        assert exc_info.value.retry_after == 2.0


class _FakeHttpResponse:
    """Minimal stand-in for an ``aiohttp`` response with a real status code."""

    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status = status
        self.headers = headers or {}
        self._payload = payload if payload is not None else {}
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_retry_delay_prefers_retry_after(self):
        """A server-supplied ``Retry-After`` wins over the exponential backoff."""
        assert _retry_delay(_HTTPStatusError(429, retry_after=2.5), 0, 1.0, 30.0) == 2.5

    def test_retry_delay_doubles_and_is_capped(self):
        assert _retry_delay(ValueError("boom"), 0, 1.0, 30.0) == 1.0
        assert _retry_delay(ValueError("boom"), 1, 1.0, 30.0) == 2.0
        assert _retry_delay(ValueError("boom"), 4, 1.0, 5.0) == 5.0

    def test_retry_delay_is_never_negative(self):
        assert _retry_delay(_HTTPStatusError(429, retry_after=-1), 0, 1.0, 30.0) == 0.0

    def test_transient_error_is_retried_then_succeeds(self, monkeypatch):
        """429 -> 200: the run must continue with the retried response."""
        calls = {"n": 0}

        async def fake_post(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _HTTPStatusError(429)
            return {"choices": [{"message": {"content": "SELECT 1"}}]}

        monkeypatch.setattr(api_mod, "_post_chat_completion_async", fake_post)

        result = asyncio.run(
            api_mod._post_with_retries(
                session=None,
                base_url="http://fake/v1",
                endpoint="chat/completions",
                payload={},
                timeout=5.0,
                max_retries=3,
                retry_base_delay=0.0,
                retry_max_delay=0.0,
            )
        )

        assert result["choices"][0]["message"]["content"] == "SELECT 1"
        assert calls["n"] == 2

    def test_permanent_error_is_not_retried(self, monkeypatch):
        """A 400 will fail identically every time — don't waste attempts."""
        calls = {"n": 0}

        async def fake_post(**_kwargs):
            calls["n"] += 1
            raise _HTTPStatusError(400)

        monkeypatch.setattr(api_mod, "_post_chat_completion_async", fake_post)

        with pytest.raises(_HTTPStatusError):
            asyncio.run(
                api_mod._post_with_retries(
                    session=None,
                    base_url="http://fake/v1",
                    endpoint="chat/completions",
                    payload={},
                    timeout=5.0,
                    max_retries=3,
                    retry_base_delay=0.0,
                    retry_max_delay=0.0,
                )
            )

        assert calls["n"] == 1

    def test_retries_are_bounded_by_max_retries(self, monkeypatch):
        calls = {"n": 0}

        async def fake_post(**_kwargs):
            calls["n"] += 1
            raise _HTTPStatusError(503)

        monkeypatch.setattr(api_mod, "_post_chat_completion_async", fake_post)

        with pytest.raises(_HTTPStatusError):
            asyncio.run(
                api_mod._post_with_retries(
                    session=None,
                    base_url="http://fake/v1",
                    endpoint="chat/completions",
                    payload={},
                    timeout=5.0,
                    max_retries=2,
                    retry_base_delay=0.0,
                    retry_max_delay=0.0,
                )
            )

        assert calls["n"] == 3  # 1 initial attempt + 2 retries

    def test_connection_errors_are_retried(self, monkeypatch):
        calls = {"n": 0}

        async def fake_post(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise api_mod.aiohttp.ClientConnectionError("connection reset")
            return {"choices": [{"message": {"content": "SELECT 1"}}]}

        monkeypatch.setattr(api_mod, "_post_chat_completion_async", fake_post)

        asyncio.run(
            api_mod._post_with_retries(
                session=None,
                base_url="http://fake/v1",
                endpoint="chat/completions",
                payload={},
                timeout=5.0,
                max_retries=1,
                retry_base_delay=0.0,
                retry_max_delay=0.0,
            )
        )
        assert calls["n"] == 2


# ---------------------------------------------------------------------------
# inference_api integration-style tests (HTTP mocked)
# ---------------------------------------------------------------------------


class TestInferenceApi:
    """Patch aiohttp.ClientSession so no real HTTP calls are made."""

    def _patch_session(self, monkeypatch, content: str = "SELECT 1"):
        """Replace aiohttp.ClientSession with a fully async-compatible mock."""
        mock_ctx = _fake_http_response(content)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_ctx)

        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(
            api_mod.aiohttp, "ClientSession", MagicMock(return_value=session_ctx)
        )

    def test_returns_results_for_all_questions(self, monkeypatch, tmp_path):
        self._patch_session(monkeypatch)
        qpath, tpath, outpath = _make_fixtures(tmp_path)

        results = inference_api(
            model_name="dummy",
            base_url="http://localhost:9999/v1",
            output_file=str(outpath),
            workdir_path=str(tmp_path),
        )

        assert len(results) == 2
        assert all("question_id" in r and "completion" in r for r in results)

    def test_output_file_written_correctly(self, monkeypatch, tmp_path):
        self._patch_session(monkeypatch, content="SELECT 99")
        qpath, tpath, outpath = _make_fixtures(tmp_path)

        inference_api(
            model_name="dummy",
            base_url="http://localhost:9999/v1",
            output_file=str(outpath),
            workdir_path=str(tmp_path),
        )

        lines = outpath.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            row = json.loads(line)
            assert row["completion"] == "SELECT 99"
            assert "question_id" in row

    def test_limit_integer(self, monkeypatch, tmp_path):
        self._patch_session(monkeypatch)
        qpath, tpath, outpath = _make_fixtures(tmp_path)

        results = inference_api(
            model_name="dummy",
            base_url="http://localhost:9999/v1",
            output_file=str(outpath),
            workdir_path=str(tmp_path),
            limit=1,
        )

        assert len(results) == 1

    def test_limit_float(self, monkeypatch, tmp_path):
        self._patch_session(monkeypatch)
        qpath, tpath, outpath = _make_fixtures(tmp_path)

        results = inference_api(
            model_name="dummy",
            base_url="http://localhost:9999/v1",
            output_file=str(outpath),
            workdir_path=str(tmp_path),
            limit=0.5,  # 50% of 2 questions → 1
        )

        assert len(results) == 1

    def test_invalid_limit_raises(self, monkeypatch, tmp_path):
        self._patch_session(monkeypatch)
        qpath, tpath, outpath = _make_fixtures(tmp_path)

        with pytest.raises(ValueError):
            inference_api(
                model_name="dummy",
                base_url="http://localhost:9999/v1",
                output_file=str(outpath),
                workdir_path=str(tmp_path),
                limit=1.5,  # float out of (0, 1]
            )

    def test_negative_limit_raises(self, monkeypatch, tmp_path):
        self._patch_session(monkeypatch)
        qpath, tpath, outpath = _make_fixtures(tmp_path)

        with pytest.raises(ValueError):
            inference_api(
                model_name="dummy",
                base_url="http://localhost:9999/v1",
                output_file=str(outpath),
                workdir_path=str(tmp_path),
                limit=-10,  # float out of (0, 1]
            )

    def test_api_key_set_in_header(self, monkeypatch, tmp_path):
        """Authorization header must be forwarded to the session."""
        captured_headers: dict = {}

        def fake_client_session(headers=None, **_):
            captured_headers.update(headers or {})
            mock_ctx = _fake_http_response()
            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_ctx)
            session_ctx = AsyncMock()
            session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            session_ctx.__aexit__ = AsyncMock(return_value=False)
            return session_ctx

        monkeypatch.setattr(api_mod.aiohttp, "ClientSession", fake_client_session)
        qpath, tpath, outpath = _make_fixtures(tmp_path)

        inference_api(
            model_name="dummy",
            base_url="http://localhost:9999/v1",
            api_key="sk-test-key",
            output_file=str(outpath),
            workdir_path=str(tmp_path),
        )

        assert captured_headers.get("Authorization") == "Bearer sk-test-key"

    def test_no_rate_limit_completes(self, monkeypatch, tmp_path):
        """rpm=None should not raise and should still return all results."""
        self._patch_session(monkeypatch)
        qpath, tpath, outpath = _make_fixtures(tmp_path)

        results = inference_api(
            model_name="dummy",
            base_url="http://localhost:9999/v1",
            output_file=str(outpath),
            workdir_path=str(tmp_path),
            requests_per_minute=None,
        )

        assert len(results) == 2

    def test_notebook_compat_with_running_loop(self, monkeypatch, tmp_path):
        """
        Simulate a Jupyter environment: inference_api is called from inside a
        running event loop.  nest_asyncio should be applied and results returned.
        """
        self._patch_session(monkeypatch)
        qpath, tpath, outpath = _make_fixtures(tmp_path)

        apply_called = {"n": 0}
        real_apply = api_mod.nest_asyncio.apply

        def spy_apply(loop=None):
            apply_called["n"] += 1
            real_apply(loop)

        monkeypatch.setattr(api_mod.nest_asyncio, "apply", spy_apply)

        async def _run_inside_loop():
            return inference_api(
                model_name="dummy",
                base_url="http://localhost:9999/v1",
                output_file=str(outpath),
                workdir_path=str(tmp_path),
            )

        import nest_asyncio

        nest_asyncio.apply()  # allow the outer asyncio.run below to nest
        results = asyncio.run(_run_inside_loop())

        assert apply_called["n"] >= 1, "nest_asyncio.apply() should have been called"
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Failure handling (issue #123)
# ---------------------------------------------------------------------------


class TestInferenceApiFailureHandling:
    """One bad request must not cost the whole run."""

    @staticmethod
    def _patch_post(monkeypatch, handler):
        monkeypatch.setattr(api_mod, "_post_chat_completion_async", handler)

    def test_null_content_becomes_empty_string(self, monkeypatch, tmp_path):
        """
        Regression: ``"content": null`` used to be written out as-is, and
        ``evaluate()`` then failed the ``completion must be str`` assertion
        for the entire predictions file.
        """

        async def handler(**_kwargs):
            return {"choices": [{"message": {"content": None}}]}

        self._patch_post(monkeypatch, handler)
        _, _, outpath = _make_fixtures(tmp_path)

        results = inference_api(
            model_name="dummy",
            base_url="http://localhost:9999/v1",
            output_file=str(outpath),
            workdir_path=str(tmp_path),
        )

        assert [r["completion"] for r in results] == ["", ""]
        for line in outpath.read_text().strip().splitlines():
            assert json.loads(line)["completion"] == ""

    def test_permanent_failure_records_empty_completion(self, monkeypatch, tmp_path):
        """With ``raise_on_error=False`` a dead request yields an empty completion."""

        async def handler(**_kwargs):
            raise _HTTPStatusError(400)

        self._patch_post(monkeypatch, handler)
        _, _, outpath = _make_fixtures(tmp_path)

        results = inference_api(
            model_name="dummy",
            base_url="http://localhost:9999/v1",
            output_file=str(outpath),
            workdir_path=str(tmp_path),
            limit=1,
            max_retries=0,
        )

        assert results == [{"question_id": "q1", "completion": ""}]

    def test_raise_on_error_propagates(self, monkeypatch, tmp_path):
        """With ``raise_on_error=True`` the failure aborts the run, as before."""

        async def handler(**_kwargs):
            raise _HTTPStatusError(400)

        self._patch_post(monkeypatch, handler)
        _, _, outpath = _make_fixtures(tmp_path)

        with pytest.raises(_HTTPStatusError):
            inference_api(
                model_name="dummy",
                base_url="http://localhost:9999/v1",
                output_file=str(outpath),
                workdir_path=str(tmp_path),
                limit=1,
                max_retries=0,
                raise_on_error=True,
            )

    def test_transient_failure_is_retried_and_run_completes(
        self, monkeypatch, tmp_path
    ):
        """A single 500 is retried, so the run still produces real completions."""
        calls = {"n": 0}

        async def handler(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _HTTPStatusError(500)
            return {"choices": [{"message": {"content": "SELECT 1"}}]}

        self._patch_post(monkeypatch, handler)
        _, _, outpath = _make_fixtures(tmp_path)

        results = inference_api(
            model_name="dummy",
            base_url="http://localhost:9999/v1",
            output_file=str(outpath),
            workdir_path=str(tmp_path),
            limit=1,
            max_retries=2,
            retry_base_delay=0.0,
            retry_max_delay=0.0,
        )

        assert results == [{"question_id": "q1", "completion": "SELECT 1"}]
        assert calls["n"] == 2

    def test_empty_choices_are_treated_as_failure(self, monkeypatch, tmp_path):
        """``choices: []`` must not raise IndexError out of the run."""

        async def handler(**_kwargs):
            return {"choices": []}

        self._patch_post(monkeypatch, handler)
        _, _, outpath = _make_fixtures(tmp_path)

        results = inference_api(
            model_name="dummy",
            base_url="http://localhost:9999/v1",
            output_file=str(outpath),
            workdir_path=str(tmp_path),
            limit=1,
            max_retries=0,
        )

        assert results == [{"question_id": "q1", "completion": ""}]

    def test_invalid_max_concurrency_raises(self, monkeypatch, tmp_path):
        self._patch_post(monkeypatch, _fake_post_success())
        _, _, outpath = _make_fixtures(tmp_path)

        with pytest.raises(ValueError, match="max_concurrency"):
            inference_api(
                model_name="dummy",
                base_url="http://localhost:9999/v1",
                output_file=str(outpath),
                workdir_path=str(tmp_path),
                max_concurrency=0,
            )

    def test_invalid_max_retries_raises(self, monkeypatch, tmp_path):
        self._patch_post(monkeypatch, _fake_post_success())
        _, _, outpath = _make_fixtures(tmp_path)

        with pytest.raises(ValueError, match="max_retries"):
            inference_api(
                model_name="dummy",
                base_url="http://localhost:9999/v1",
                output_file=str(outpath),
                workdir_path=str(tmp_path),
                max_retries=-1,
            )

    def test_concurrency_cap_is_respected(self, monkeypatch, tmp_path):
        """``max_concurrency`` must bound how many requests run at once."""
        in_flight = {"now": 0, "peak": 0}

        async def handler(**_kwargs):
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
            await asyncio.sleep(0.01)
            in_flight["now"] -= 1
            return {"choices": [{"message": {"content": "SELECT 1"}}]}

        self._patch_post(monkeypatch, handler)
        _, _, outpath = _make_fixtures(tmp_path)

        inference_api(
            model_name="dummy",
            base_url="http://localhost:9999/v1",
            output_file=str(outpath),
            workdir_path=str(tmp_path),
            max_concurrency=1,
        )

        assert in_flight["peak"] == 1


def _fake_post_success():
    """Return a handler that always answers with a valid completion."""

    async def handler(**_kwargs):
        return {"choices": [{"message": {"content": "SELECT 1"}}]}

    return handler
