"""
LLMSQL OpenAI-Compatible API Inference Function
===============================================

This module provides ``inference_api()`` for text-to-SQL generation against an
OpenAI-compatible Chat Completions API.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from typing import Any, Literal

import aiohttp
from dotenv import load_dotenv
import nest_asyncio
from tqdm.asyncio import tqdm

from llmsql.config.config import (
    DEFAULT_LLMSQL_VERSION,
    DEFAULT_WORKDIR_PATH,
    get_repo_id,
)
from llmsql.loggers.logging_config import log
from llmsql.utils.inference_utils import _maybe_download, _setup_seed
from llmsql.utils.utils import (
    choose_prompt_builder,
    load_jsonl,
    overwrite_jsonl,
    save_jsonl_lines,
)

load_dotenv()


class _AsyncRateLimiter:
    """
    Token-bucket style async rate limiter.

    Releases one token every (60 / requests_per_minute) seconds,
    so requests are spaced from their *start* time — not from when
    the previous one finished.  This allows concurrent in-flight
    requests while still honouring the RPM cap.
    """

    def __init__(self, requests_per_minute: float | None) -> None:
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0 when provided.")
        self._interval: float | None = (
            60.0 / requests_per_minute if requests_per_minute is not None else None
        )
        self._next_allowed: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available, then claim it."""
        if self._interval is None:
            return

        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
            # Claim the next slot *before* releasing the lock so the
            # following coroutine waits for exactly one more interval.
            self._next_allowed = time.monotonic() + self._interval


async def _post_chat_completion_async(
    *,
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    ep = endpoint.lstrip("/")
    url = f"{base}/{ep}"

    async with session.post(
        url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as resp:
        resp.raise_for_status()
        parsed: dict[str, Any] = await resp.json()

    if "choices" not in parsed:
        raise ValueError("API response does not contain `choices`.")
    return parsed


async def _inference_api_async(
    model_name: str,
    *,
    base_url: str,
    endpoint: str,
    headers: dict[str, str],
    timeout: float,
    requests_per_minute: float | None,
    api_kwargs: dict[str, Any],
    questions: list[dict[str, Any]],
    tables: dict[str, Any],
    prompt_builder: Any,
    output_file: str,
) -> list[dict[str, str]]:
    limiter = _AsyncRateLimiter(requests_per_minute)
    all_results: list[dict[str, str]] = []
    # Lock to serialise file writes while allowing concurrent HTTP calls.
    write_lock = asyncio.Lock()

    async with aiohttp.ClientSession(headers=headers) as session:

        async def process_question(q: dict[str, Any]) -> dict[str, str]:
            tbl = tables[q["table_id"]]
            example_row = tbl["rows"][0] if tbl["rows"] else []
            prompt = prompt_builder(
                q["question"], tbl["header"], tbl["types"], example_row
            )

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                **api_kwargs,
            }

            # Acquire a rate-limit slot *before* firing the request so that
            # the HTTP round-trip time doesn't count against the interval.
            await limiter.acquire()

            response = await _post_chat_completion_async(
                session=session,
                base_url=base_url,
                endpoint=endpoint,
                payload=payload,
                timeout=timeout,
            )
            completion = response["choices"][0]["message"]["content"]

            result = {
                "question_id": q.get("question_id", q.get("id", "")),
                "completion": completion,
            }

            async with write_lock:
                save_jsonl_lines(output_file, [result])

            return result

        tasks = [process_question(q) for q in questions]
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Generating",
        ):
            result = await coro
            all_results.append(result)

    return all_results


def inference_api(
    model_name: str,
    *,
    base_url: str,
    endpoint: str = "chat/completions",
    api_key: str | None = None,
    timeout: float = 120.0,
    requests_per_minute: float | None = None,
    api_kwargs: dict[str, Any] | None = None,
    request_headers: dict[str, str] | None = None,
    version: Literal["1.0", "2.0"] = DEFAULT_LLMSQL_VERSION,
    output_file: str = "llm_sql_predictions.jsonl",
    questions_path: str | None = None,
    tables_path: str | None = None,
    workdir_path: str = DEFAULT_WORKDIR_PATH,
    limit: int | float | None = None,
    num_fewshots: int = 5,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Run SQL generation using an OpenAI-compatible Chat Completions API.

    Requests are dispatched concurrently so that HTTP round-trip time does
    not count against the rate-limit interval — achieving a true
    `requests_per_minute` throughput rather than
    ``requests_per_minute / (1 + latency_in_minutes)``.

    Args:
        model_name: The model name of the api.

        base_url: e.g. "https://api.openai.com/v1/"
        endpoint: e.g. "chat/completions"

        # Benchmark:
        version: LLMSQL version
        output_file: Path to write outputs (will be overwritten).
        questions_path: Path to questions.jsonl (auto-downloads if missing).
        tables_path: Path to tables.jsonl (auto-downloads if missing).
        workdir_path: Directory to store downloaded data.
        num_fewshots: Number of few-shot examples (0, 1, or 5).
        batch_size: Number of questions per generation batch.
        seed: Random seed for reproducibility.
        limit: Limit the number of questions to evaluate. If an integer, evaluates
               the first N samples. If a float between 0.0 and 1.0, evaluates the
               first X*100% of samples. If None, evaluates all samples (default).

    Returns:
        List of dicts containing `question_id` and generated `completion`.
    """
    _setup_seed(seed=seed)
    api_kwargs = api_kwargs or {}
    request_headers = request_headers or {}

    workdir = Path(workdir_path)
    workdir.mkdir(parents=True, exist_ok=True)

    repo_id = get_repo_id(version)
    questions_path = _maybe_download(repo_id, "questions.jsonl", questions_path)
    tables_path = _maybe_download(repo_id, "tables.jsonl", tables_path)

    questions = load_jsonl(questions_path)
    tables_list = load_jsonl(tables_path)
    tables = {t["table_id"]: t for t in tables_list}

    if limit is not None:
        if isinstance(limit, float):
            if not (0.0 < limit <= 1.0):
                raise ValueError(
                    f"When a float, `limit` must be between 0.0 and 1.0, got {limit}."
                )
            limit = max(1, int(len(questions) * limit))
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(
                f"`limit` must be a positive integer or a float in (0.0, 1.0], got {limit!r}."
            )
        questions = questions[:limit]

    key = api_key or os.environ.get("OPENAI_API_KEY")
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        **request_headers,
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"

    prompt_builder = choose_prompt_builder(num_fewshots)

    overwrite_jsonl(output_file)

    coro = _inference_api_async(
        model_name,
        base_url=base_url,
        endpoint=endpoint,
        headers=headers,
        timeout=timeout,
        requests_per_minute=requests_per_minute,
        api_kwargs=api_kwargs,
        questions=questions,
        tables=tables,
        prompt_builder=prompt_builder,
        output_file=output_file,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Inside a Jupyter notebook (or any other environment that already
        # owns an event loop) — patch the loop so nested runs are allowed.
        nest_asyncio.apply(loop)
        all_results = loop.run_until_complete(coro)
    else:
        all_results = asyncio.run(coro)

    log.info(f"Generation completed. {len(all_results)} results saved to {output_file}")
    return all_results
