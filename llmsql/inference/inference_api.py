"""
LLMSQL OpenAI-Compatible API Inference Function
===============================================

This module provides ``inference_api()`` for text-to-SQL generation against an
OpenAI-compatible Chat Completions API.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Literal

import aiohttp
from dotenv import load_dotenv
import nest_asyncio
from tqdm.asyncio import tqdm

from llmsql.config.config import (
    DEFAULT_LLMSQL_VERSION,
    get_repo_id,
)
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


# Retry policy for transient API failures (see ``inference_api``).
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0
DEFAULT_RETRY_MAX_DELAY = 30.0

# Statuses worth retrying: request timeouts, conflicts, rate limiting.
# Anything >= 500 is retried as well; 4xx client errors are not (they will
# fail identically on every attempt).
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429})

# Same default as ``inference_function`` so both backends behave alike.
DEFAULT_MAX_CONCURRENCY = 32


class _HTTPStatusError(RuntimeError):
    """Raised when the API answers with an HTTP error status.

    ``aiohttp``'s ``raise_for_status()`` does not expose the response headers,
    so the status is checked explicitly and the ``Retry-After`` hint (when the
    server sends one) is carried along for the retry loop to honour.
    """

    def __init__(self, status: int, retry_after: float | None = None,
                 message: str = "") -> None:
        super().__init__(message or f"API returned HTTP {status}")
        self.status = status
        self.retry_after = retry_after


def _retry_after_seconds(headers: Any) -> float | None:
    """Parse the ``Retry-After`` header (seconds form), if present."""
    try:
        value = headers.get("Retry-After")
    except (AttributeError, TypeError):
        return None
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        # HTTP-date form — not parsed; the exponential backoff is used instead.
        return None
    return seconds if seconds >= 0 else None


def _is_retryable_error(exc: BaseException) -> bool:
    """Whether a failed request is worth another attempt."""
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS_CODES or status >= 500
    # Connection resets, DNS hiccups and read timeouts.
    return isinstance(exc, (asyncio.TimeoutError, aiohttp.ClientConnectionError))


def _retry_delay(exc: BaseException, attempt: int, base_delay: float,
                 max_delay: float) -> float:
    """Seconds to wait before retry ``attempt + 1`` (``attempt`` is 0-based)."""
    hint = getattr(exc, "retry_after", None)
    if hint is None:
        hint = base_delay * (2 ** attempt)
    return max(0.0, min(float(hint), max_delay))


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
        status = getattr(resp, "status", None)
        if isinstance(status, int) and status >= 400:
            retry_after = _retry_after_seconds(getattr(resp, "headers", None))
            try:
                detail = (await resp.text())[:200]
            except Exception:  # pragma: no cover - body may be unreadable
                detail = ""
            raise _HTTPStatusError(status, retry_after, detail)
        resp.raise_for_status()
        parsed: dict[str, Any] = await resp.json()

    if "choices" not in parsed:
        raise ValueError("API response does not contain `choices`.")
    return parsed


async def _post_with_retries(
    *,
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> dict[str, Any]:
    """Send one chat completion request, retrying transient failures.

    A single HTTP 429 / 5xx or a timeout used to propagate out of
    ``asyncio.as_completed`` and abort the entire run — with the output file
    already wiped, one transient error cost a whole benchmark. Transient
    failures are therefore retried with exponential backoff (honouring
    ``Retry-After``); permanent ones (400, 401, ...) are raised immediately.
    """
    attempt = 0
    while True:
        try:
            return await _post_chat_completion_async(
                session=session,
                base_url=base_url,
                endpoint=endpoint,
                payload=payload,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised below when final
            if attempt >= max_retries or not _is_retryable_error(exc):
                raise
            delay = _retry_delay(exc, attempt, retry_base_delay, retry_max_delay)
            log.warning(
                f"API request failed ({type(exc).__name__}: {exc}); retrying in "
                f"{delay:.2f}s ({attempt + 1}/{max_retries})."
            )
            attempt += 1
            await asyncio.sleep(delay)


async def _inference_api_async(
    model_name: str,
    *,
    base_url: str,
    endpoint: str,
    headers: dict[str, str],
    timeout: float,
    requests_per_minute: float | None,
    max_concurrency: int | None,
    raise_on_error: bool,
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
    api_kwargs: dict[str, Any],
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
    # Lock to serialise file writes while allowing concurrent HTTP calls.
    write_lock = asyncio.Lock()
    n_failed = 0

    async with aiohttp.ClientSession(headers=headers) as session:

        # Pre-build all prompts using the shared function
        prompts = build_all_requests(questions, tables, prompt_builder)

        async def request_completion(q: dict[str, Any], prompt: str) -> str:
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

            response = await _post_with_retries(
                session=session,
                base_url=base_url,
                endpoint=endpoint,
                payload=payload,
                timeout=timeout,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
                retry_max_delay=retry_max_delay,
            )

            choices = response.get("choices") or []
            if not choices:
                raise ValueError("API response contains no choices.")

            content = choices[0].get("message", {}).get("content")
            # Some APIs answer with "content": null (refusals, reasoning and
            # tool-call responses). Written out as-is it makes `evaluate()`
            # fail the `completion must be str` assertion for the whole file.
            return "" if content is None else content

        async def call_with_policy(q: dict[str, Any], prompt: str) -> str:
            try:
                return await request_completion(q, prompt)
            except Exception as e:  # noqa: BLE001 - policy applied below
                if raise_on_error:
                    raise
                nonlocal n_failed
                n_failed += 1
                qid = q.get("question_id", q.get("id", ""))
                log.error(
                    f"API request failed for question_id={qid!r}: "
                    f"{type(e).__name__}: {e}. Recording an empty completion."
                )
                return ""

        async def process_question(q: dict[str, Any], prompt: str) -> dict[str, str]:
            if semaphore is not None:
                async with semaphore:
                    completion = await call_with_policy(q, prompt)
            else:
                completion = await call_with_policy(q, prompt)

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
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Generating",
            ):
                result = await coro
                all_results.append(result)
        except BaseException:
            # Stop all in-flight / pending requests before propagating.
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    if n_failed:
        log.warning(
            f"{n_failed}/{len(tasks)} API requests failed; their completions "
            "were recorded as empty strings."
        )

    return all_results


def inference_api(
    model_name: str,
    *,
    base_url: str,
    endpoint: str = "chat/completions",
    api_key: str | None = None,
    timeout: float = 120.0,
    requests_per_minute: float | None = None,
    max_concurrency: int | None = DEFAULT_MAX_CONCURRENCY,
    raise_on_error: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    api_kwargs: dict[str, Any] | None = None,
    request_headers: dict[str, str] | None = None,
    version: Literal["1.0", "2.0"] = DEFAULT_LLMSQL_VERSION,
    output_file: str = "llm_sql_predictions.jsonl",
    workdir_path: str | None = None,
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

        max_concurrency: Maximum number of HTTP requests in flight at the same
            time, or ``None`` to leave it unbounded (aiohttp's connector limit
            then applies). Defaults to 32, like ``inference_function``.
        raise_on_error: If ``True``, re-raise the first error and cancel the
            remaining requests. If ``False`` (default), log the error and
            record an empty completion for that question so a single failure
            cannot abort the whole run.
        max_retries: How many times a *transient* failure (HTTP 429/5xx,
            timeout, connection reset) is retried before giving up. ``0``
            disables retries. Permanent errors (e.g. 400, 401) are never
            retried.
        retry_base_delay: First retry delay in seconds; doubled on every
            further attempt, unless the server sends ``Retry-After``.
        retry_max_delay: Upper bound for a single retry delay.

        # Benchmark:
        version: LLMSQL version
        output_file: Path to write outputs (will be overwritten).
        workdir_path: Directory to store downloaded benchmark files. If omitted, a
            temporary directory is created automatically.
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

    if max_concurrency is not None and (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency <= 0
    ):
        raise ValueError(
            f"`max_concurrency` must be a positive integer or None, got {max_concurrency!r}."
        )
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise ValueError(
            f"`max_retries` must be a non-negative integer, got {max_retries!r}."
        )

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
        max_concurrency=max_concurrency,
        raise_on_error=raise_on_error,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
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
