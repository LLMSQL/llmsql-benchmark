"""
LLMSQL Custom Function Inference
================================

This module provides ``inference_function()`` for text-to-SQL generation using
an arbitrary user-provided async inference callable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import inspect
import time
from typing import Any, Literal

from dotenv import load_dotenv
import nest_asyncio
from tqdm.asyncio import tqdm

from llmsql.config.config import DEFAULT_LLMSQL_VERSION, get_repo_id
from llmsql.loggers.logging_config import log
from llmsql.utils.inference_utils import (
    _maybe_download,
    _setup_seed,
    resolve_workdir_path,
)
from llmsql.utils.utils import (
    build_all_requests,
    choose_prompt_builder,
    load_jsonl,
    overwrite_jsonl,
    save_jsonl_lines,
)

load_dotenv()


class _AsyncRateLimiter:
    """Token-bucket style async rate limiter with request-start spacing."""

    def __init__(self, requests_per_minute: float | None) -> None:
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0 when provided.")
        self._interval: float | None = (
            60.0 / requests_per_minute if requests_per_minute is not None else None
        )
        self._next_allowed: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._interval is None:
            return

        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_allowed = time.monotonic() + self._interval


async def _inference_function_async(
    *,
    inference_callable: Callable[..., Awaitable[str]],
    requests_per_minute: float | None,
    function_kwargs: dict[str, Any],
    questions: list[dict[str, Any]],
    tables: dict[str, Any],
    prompt_builder: Any,
    output_file: str,
) -> list[dict[str, str]]:
    limiter = _AsyncRateLimiter(requests_per_minute)
    all_results: list[dict[str, str]] = []
    write_lock = asyncio.Lock()

    prompts = build_all_requests(questions, tables, prompt_builder)

    async def process_question(q: dict[str, Any], prompt: str) -> dict[str, str]:
        await limiter.acquire()

        completion = await inference_callable(
            prompt,
            question=q,
            table=tables[q["table_id"]],
            **function_kwargs,
        )

        result = {
            "question_id": q.get("question_id", q.get("id", "")),
            "completion": str(completion),
        }

        async with write_lock:
            save_jsonl_lines(output_file, [result])

        return result

    tasks = [process_question(q, p) for q, p in zip(questions, prompts, strict=False)]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
        all_results.append(await coro)

    return all_results


def inference_function(
    *,
    inference_function: Callable[..., Awaitable[str]],
    requests_per_minute: float | None = None,
    function_kwargs: dict[str, Any] | None = None,
    version: Literal["1.0", "2.0"] = DEFAULT_LLMSQL_VERSION,
    output_file: str = "llm_sql_predictions.jsonl",
    workdir_path: str | None = None,
    limit: int | float | None = None,
    num_fewshots: int = 5,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Run SQL generation using a user-provided async callable.

    The callable is awaited as:
    ``await inference_function(prompt, question=..., table=..., **function_kwargs)``.

    If your callable needs sampling parameters, pass them through ``function_kwargs``.
    """
    _setup_seed(seed=seed)

    if not callable(inference_function):
        raise TypeError("`inference_function` must be callable.")

    function_kwargs = function_kwargs or {}
    workdir = resolve_workdir_path(workdir_path)

    repo_id = get_repo_id(version)
    questions_path = _maybe_download(repo_id, "questions.jsonl", workdir)
    tables_path = _maybe_download(repo_id, "tables.jsonl", workdir)

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

    prompt_builder = choose_prompt_builder(num_fewshots)

    overwrite_jsonl(output_file)

    async def _validated_callable(*args: Any, **kwargs: Any) -> str:
        out = inference_function(*args, **kwargs)
        if inspect.isawaitable(out):
            return str(await out)
        raise TypeError("`inference_function` must return an awaitable value.")

    coro = _inference_function_async(
        inference_callable=_validated_callable,
        requests_per_minute=requests_per_minute,
        function_kwargs=function_kwargs,
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
        nest_asyncio.apply(loop)
        all_results = loop.run_until_complete(coro)
    else:
        all_results = asyncio.run(coro)

    log.info(f"Generation completed. {len(all_results)} results saved to {output_file}")
    return all_results
