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


class _NotAwaitableError(TypeError):
    """Raised when the user callable does not return an awaitable."""


class _AsyncRateLimiter:
    """Async rate limiter that spaces out request *start* times.

    Each call to :meth:`acquire` waits until at least ``60 / requests_per_minute``
    seconds have passed since the previous request was allowed to start. Requests
    themselves may still run concurrently (bounded separately by
    ``max_concurrency``), so slow responses do not reduce the throughput.

    Args:
        requests_per_minute: Maximum number of request starts per minute, or
            ``None`` to disable rate limiting.

    Raises:
        ValueError: If ``requests_per_minute`` is provided and is not positive.
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
        """Wait until the next request is allowed to start."""
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
    max_concurrency: int | None,
    raise_on_error: bool,
    function_kwargs: dict[str, Any],
    questions: list[dict[str, Any]],
    tables: dict[str, Any],
    prompt_builder: Any,
    output_file: str,
) -> list[dict[str, str]]:
    limiter = _AsyncRateLimiter(requests_per_minute)
    semaphore = (
        asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
    )
    all_results: list[dict[str, str]] = []
    write_lock = asyncio.Lock()
    n_failed = 0

    prompts = build_all_requests(questions, tables, prompt_builder)

    async def call_user_function(q: dict[str, Any], prompt: str) -> str:
        try:
            out = inference_callable(
                prompt,
                question=q,
                table=tables[q["table_id"]],
                **function_kwargs,
            )
            if not inspect.isawaitable(out):
                raise _NotAwaitableError(
                    "`inference_function` must return an awaitable value "
                    f"(define it with `async def`), got {type(out).__name__}."
                )
            return str(await out)
        except _NotAwaitableError:
            # Programming error: never swallowed, regardless of ``raise_on_error``.
            raise
        except Exception as e:
            if raise_on_error:
                raise
            nonlocal n_failed
            n_failed += 1
            qid = q.get("question_id", q.get("id", ""))
            log.error(
                f"`inference_function` failed for question_id={qid!r}: "
                f"{type(e).__name__}: {e}. Recording an empty completion."
            )
            return ""

    async def process_question(q: dict[str, Any], prompt: str) -> dict[str, str]:
        if semaphore is not None:
            async with semaphore:
                await limiter.acquire()
                completion = await call_user_function(q, prompt)
        else:
            await limiter.acquire()
            completion = await call_user_function(q, prompt)

        result = {
            "question_id": q.get("question_id", q.get("id", "")),
            "completion": completion,
        }

        async with write_lock:
            save_jsonl_lines(output_file, [result])

        return result

    tasks = [
        asyncio.ensure_future(process_question(q, p))
        for q, p in zip(questions, prompts, strict=False)
    ]
    try:
        for fut in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Generating"
        ):
            all_results.append(await fut)
    except BaseException:
        # Stop all in-flight / pending requests before propagating the error.
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    if n_failed:
        log.warning(
            f"{n_failed}/{len(tasks)} calls to `inference_function` failed; "
            "their completions were recorded as empty strings."
        )

    return all_results


def inference_function(
    *,
    inference_function: Callable[..., Awaitable[str]],
    requests_per_minute: float | None = None,
    max_concurrency: int | None = 32,
    raise_on_error: bool = False,
    function_kwargs: dict[str, Any] | None = None,
    version: Literal["1.0", "2.0"] = DEFAULT_LLMSQL_VERSION,
    output_file: str = "llm_sql_predictions.jsonl",
    workdir_path: str | None = None,
    limit: int | float | None = None,
    num_fewshots: int = 5,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Run SQL generation using a user-provided async callable.

    LLMSQL downloads the benchmark, builds the prompt for every question (with
    the requested number of few-shot examples) and awaits your callable for each
    of them. This lets you plug in any engine, API client, router or agent while
    keeping the standard LLMSQL prompts and output format, so the resulting file
    can be passed directly to :func:`llmsql.evaluate`.

    The callable is awaited as::

        await inference_function(
            prompt,                      # str, the fully built LLMSQL prompt
            question=question,           # dict, the raw benchmark question row
            table=table,                 # dict, the table the question refers to
            **function_kwargs,
        )

    and must return the model completion (it is converted with ``str()``).
    Calls run concurrently on a single event loop, bounded by
    ``max_concurrency`` and spaced out by ``requests_per_minute``. Results are
    appended to ``output_file`` as soon as each call finishes, so the file
    order follows completion order, not question order.

    Error handling: by default (``raise_on_error=False``) an exception raised by
    the callable is logged together with the ``question_id`` and an empty
    completion is recorded for that question (it will be counted as incorrect
    by the evaluator); a summary of failures is logged at the end. With
    ``raise_on_error=True`` the first exception cancels all remaining calls and
    is re-raised. A callable that does not return an awaitable always raises
    ``TypeError``.

    The function can be called from synchronous code as well as from within a
    running event loop (e.g. Jupyter), in which case ``nest_asyncio`` is applied
    to that loop.

    Example:
        >>> from llmsql import inference_function
        >>> async def my_model(prompt, **kwargs):
        ...     return "SELECT 1"
        >>> results = inference_function(
        ...     inference_function=my_model,
        ...     requests_per_minute=60,
        ...     max_concurrency=8,
        ... )  # doctest: +SKIP

    Args:
        inference_function: Async callable (``async def``) that receives the
            prompt as the first positional argument plus ``question``,
            ``table`` and ``**function_kwargs`` keyword arguments, and returns
            the generated SQL completion.
        requests_per_minute: Maximum number of calls started per minute. If
            ``None`` (default), calls are not rate limited.
        max_concurrency: Maximum number of calls in flight at the same time.
            Defaults to 32. Use ``None`` to disable the cap (not recommended
            for the full benchmark, as all questions would be dispatched at
            once).
        raise_on_error: If ``True``, re-raise the first exception raised by the
            callable and abort the run. If ``False`` (default), log the error
            and record an empty completion for that question.
        function_kwargs: Extra keyword arguments forwarded to every call, e.g.
            sampling parameters such as ``{"temperature": 0.0}``.
        version: LLMSQL benchmark version (``"1.0"`` or ``"2.0"``).
        output_file: Path of the JSONL file to write outputs to (it is
            overwritten).
        workdir_path: Directory to store downloaded benchmark files. If
            omitted, a temporary directory is created automatically.
        limit: Limit the number of questions to evaluate. If an integer,
            evaluates the first N samples. If a float between 0.0 and 1.0,
            evaluates the first X*100% of samples. If None, evaluates all
            samples (default).
        num_fewshots: Number of few-shot examples (0, 1, or 5).
        seed: Random seed for reproducibility.

    Returns:
        List of dicts containing ``question_id`` and generated ``completion``,
        in completion order.

    Raises:
        TypeError: If ``inference_function`` is not callable or does not return
            an awaitable.
        ValueError: If ``requests_per_minute``, ``max_concurrency`` or
            ``limit`` has an invalid value.
        Exception: Any exception raised by ``inference_function`` when
            ``raise_on_error=True``.
    """
    _setup_seed(seed=seed)

    if not callable(inference_function):
        raise TypeError("`inference_function` must be callable.")
    if requests_per_minute is not None and requests_per_minute <= 0:
        raise ValueError("requests_per_minute must be > 0 when provided.")
    if max_concurrency is not None and (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency <= 0
    ):
        raise ValueError(
            f"`max_concurrency` must be a positive integer or None, got {max_concurrency!r}."
        )

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

    coro = _inference_function_async(
        inference_callable=inference_function,
        requests_per_minute=requests_per_minute,
        max_concurrency=max_concurrency,
        raise_on_error=raise_on_error,
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
