import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from huggingface_hub import hf_hub_download

from llmsql.loggers.logging_config import log
from llmsql.utils.regex_extractor import find_sql

# Time budget for a single SQL statement, in seconds.
#
# The statements handed to `execute_sql` are extracted from model output, so a
# pathological one (a self-join, an unbounded recursive CTE) can run forever and
# hang the whole evaluation. Without a budget there is no way to get control
# back short of killing the process.
DEFAULT_QUERY_TIMEOUT = 5.0

# Granularity of the SQLite progress callback: the deadline is checked once
# every this many VM instructions. Small enough to catch runaway queries
# quickly, large enough not to slow normal ones down.
PROGRESS_HANDLER_OPS = 1000


def _clear_progress_handler(conn: sqlite3.Connection) -> None:
    """Remove any progress handler installed on `conn`."""
    clear = getattr(conn, "set_progress_handler", None)
    if callable(clear):
        clear(None, 0)


def execute_sql(
    conn: sqlite3.Connection,
    sql: str,
    query_timeout: float = DEFAULT_QUERY_TIMEOUT,
) -> list[tuple] | None:
    """
    Execute a SQL query on the given SQLite connection and return its results.

    The results are always sorted to avoid differences caused by row order (order agnostic).
    If the query fails, the function logs the error and returns None.

    Args:
        conn (sqlite3.Connection): An active SQLite database connection.
        sql (str): SQL query string to execute.
        query_timeout (float): Time budget for this one statement, in seconds.
            A statement that runs longer is aborted and reported as a failure
            (``None``), which callers count as a SQL error. ``<= 0`` disables
            the budget, restoring the previous unbounded behaviour.

    Returns:
        Optional[List[Tuple]]:
            - Sorted list of result rows (each row as a tuple) if successful.
            - [(None,)] if the query executed but returned NULL values.
            - None if the SQL execution failed due to an exception.
    """
    use_timeout = query_timeout is not None and query_timeout > 0

    try:
        if use_timeout:
            # SQLite calls this back while the statement runs; returning a
            # non-zero value aborts it with an "interrupted" error, which the
            # except branch below already turns into None.
            deadline = time.monotonic() + query_timeout

            def _abort_when_over_budget() -> int:
                return 1 if time.monotonic() > deadline else 0

            conn.set_progress_handler(_abort_when_over_budget, PROGRESS_HANDLER_OPS)

        cur = conn.cursor()
        cur.execute(sql)
        results = cur.fetchall()
        return sorted(results)
    except Exception:
        return None
    finally:
        # The handler lives on the connection, so it has to be removed again or
        # the next statement on this connection inherits an expired deadline.
        if use_timeout:
            _clear_progress_handler(conn)


def fix_table_name(sql: str, table_id: str) -> str:
    """
    Replace placeholder table name in the SQL query with the actual table ID.

    During evaluation, the LLM is instructed to always generate queries using
    a generic placeholder table name (`FROM Table`, `FROM "Table"`, or `FROM 'Table'`).
    This keeps the model’s task simpler and avoids requiring it to memorize or
    reproduce arbitrary, dataset-specific table IDs.

    This function post-processes the model’s SQL output by replacing the placeholder
    with the true table identifier for the current question.

    Args:
        sql (str): SQL query string produced by the model, using "Table" as placeholder.
        table_id (str): Actual table name/identifier for the current question.

    Returns:
        str: SQL query with the correct table name substituted.
    """
    return (
        sql.replace("FROM 'Table'", f'FROM "{table_id}"')
        .replace('FROM "Table"', f'FROM "{table_id}"')
        .replace("FROM Table", f'FROM "{table_id}"')
        .strip()
    )


def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL string for exact string match comparison.

    Strips surrounding whitespace and trailing semicolons and collapses runs of
    whitespace into a single space. Identifier and literal casing is preserved.

    Args:
        sql (str): SQL query string.

    Returns:
        str: Normalized SQL string.
    """
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").strip())


def normalize_question_id(value: Any) -> int:
    """
    Coerce a ``question_id`` coming from a predictions file into an ``int``.

    Other tools commonly serialise ids as strings (``"123"``), so an int-like
    string is accepted and converted. Anything else fails with a message that
    names the offending value.

    Args:
        value: raw ``question_id`` value from a prediction entry.

    Returns:
        The id as ``int``.

    Raises:
        TypeError: if the value is neither an int nor a string.
        ValueError: if the value is a string that does not parse as an int.
    """
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            raise ValueError(
                f"question_id must be an int or an int-like string, got {value!r}."
            ) from None

    raise TypeError(
        f"question_id must be an int or an int-like string, got {type(value).__name__}."
    )


def resolve_prediction_coverage(
    outputs_list: list[dict[str, Any]],
    questions: dict[int, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Deduplicate predictions and measure how much of the benchmark they cover.

    ``evaluate()`` used to score whatever rows happened to be in the predictions
    file, so a crashed run (or ``--limit``) produced a full-looking accuracy
    with no hint that most questions were never answered. Duplicate ids were
    also counted twice.

    Args:
        outputs_list: predictions as loaded from a JSONL file or passed by the caller.
        questions: benchmark questions keyed by ``question_id``.

    Returns:
        A tuple ``(unique_outputs, coverage)`` where ``coverage`` holds
        ``expected`` (benchmark size), ``answered`` (unique matching ids),
        ``missing`` (benchmark questions with no prediction) and ``duplicates``.

    Raises:
        ValueError: if a prediction references an id that is not in the benchmark.
    """
    expected = len(questions)
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    duplicates = 0

    for item in outputs_list:
        qid = normalize_question_id(item["question_id"])

        if qid not in questions:
            raise ValueError(
                f"question_id {qid} from the predictions file is not part of the "
                f"benchmark ({expected} questions loaded)."
            )

        if qid in seen:
            duplicates += 1
            continue

        seen.add(qid)
        unique.append(item)

    return unique, {
        "expected": expected,
        "answered": len(seen),
        "missing": expected - len(seen),
        "duplicates": duplicates,
    }


def evaluate_sample(
    item: dict[str, int | str],
    questions: dict[int, dict[str, str]],
    conn: sqlite3.Connection,
    query_timeout: float = DEFAULT_QUERY_TIMEOUT,
) -> tuple[int, dict[str, Any] | None, dict[Any, Any]]:
    """
    Evaluate a single model prediction against the gold (ground-truth) SQL query.

    This function:
    - Retrieves the gold SQL query and question metadata for the given `question_id`.
    - Executes the gold SQL and the model's at most 10 predicted SQL queries on the SQLite DB.
    - Compares their results to determine whether the gold and at least one prediction are matched.
    - Tracks special cases such as SQL errors or queries returning NULL results.
    - Returns evaluation metrics and mismatch details (if any).

    Args:
        item (dict): A single model prediction entry. Must contain:
                     - "question_id": ID of the benchmark question.
                     - "completion": The raw SQL string predicted by the model.
        questions (dict): Dictionary mapping `question_id` → question metadata:
                          {"sql": ..., "table_id": ..., "question": ...}.
        conn (sqlite3.Connection): Active SQLite connection used to run queries.
        query_timeout (float): Per-statement time budget in seconds; a statement
            that overruns it is aborted and counted as a SQL error. ``<= 0``
            disables the budget.

    Returns:
        tuple:
            is_match (int): 1 if prediction matches gold SQL results, else 0.
            mismatch_info (dict or None): Details about the mismatch if incorrect,
                                          otherwise None. Includes question, gold SQL,
                                          model output, and query results.
            metrics_update (dict): Partial metrics for this prediction:
                                   {
                                     "pred_none": int,
                                     "gold_none": int,
                                     "sql_error": int
                                   }
    """
    # Extract question metadata.
    # These used to be plain asserts: ``python -O`` strips asserts entirely, so
    # malformed input silently slipped through instead of being reported.
    qid = normalize_question_id(item["question_id"])

    if qid not in questions:
        raise ValueError(
            f"question_id {qid} from the predictions file is not part of the "
            f"benchmark ({len(questions)} questions loaded)."
        )

    q_info = questions[qid]
    table_id, gold_sql, question_text = (
        q_info["table_id"],
        q_info["sql"],
        q_info["question"],
    )

    # Execute the gold (ground-truth) SQL
    gold_results = execute_sql(conn, gold_sql, query_timeout=query_timeout)

    # Initialize counters for this sample
    pred_none = gold_none = sql_error = exact_string_match = 0

    # Track if gold query returned a NULL-equivalent result
    if gold_results == [(None,)]:
        gold_none = 1

    # Flag for whether the prediction was correct
    is_match = 0
    last_pred_res = None  # store last prediction results for mismatch logging

    # Loop over all SQL queries extracted from the model output
    completion = item["completion"]
    if not isinstance(completion, str):
        raise TypeError(
            f"completion must be of type str, got {type(completion).__name__}: "
            f"{completion!r}"
        )
    for pred_sql in find_sql(completion):
        # Replace placeholder table names with the actual one
        pred_sql_fixed = fix_table_name(pred_sql, table_id)

        # Execute predicted SQL
        pred_res = execute_sql(conn, pred_sql_fixed, query_timeout=query_timeout)
        last_pred_res = pred_res

        # Exact string match after whitespace / trailing semicolon normalization.
        # The extractor drops the trailing ";" while gold SQL keeps it.
        if normalize_sql(pred_sql_fixed) == normalize_sql(gold_sql):
            exact_string_match = 1

        # Update metrics
        if pred_res is None:  # execution failed
            sql_error += 1
        elif pred_res == [(None,)]:  # returned NULL-equivalent
            pred_none += 1

        # If both gold and prediction executed successfully and match → success
        if (
            gold_results is not None
            and pred_res is not None
            and gold_results == pred_res
        ):
            is_match = 1

    # If no match was found, prepare mismatch details for debugging/logging
    mismatch_info = None
    if not is_match:
        mismatch_info = {
            "question_id": qid,
            "question": question_text,
            "gold_sql": gold_sql,
            "model_output": completion,
            "gold_results": gold_results,
            "prediction_results": last_pred_res,
        }

    return (
        is_match,
        mismatch_info,
        {
            "pred_none": pred_none,
            "gold_none": gold_none,
            "sql_error": sql_error,
            "exact_string_match": exact_string_match,
        },
    )


def download_benchmark_file(repo_id: str, filename: str, local_dir: Path) -> str:
    """Download a benchmark file from HuggingFace Hub."""
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
    )
    assert isinstance(file_path, str)
    log.info(f"Downloaded {filename} to: {file_path}")
    return file_path


def connect_sqlite(db_path: str) -> sqlite3.Connection:
    """
    Open the benchmark SQLite database read-only.

    The statements that :func:`execute_sql` runs are extracted from model
    output, and the database is cached under ``workdir_path`` and reused by
    every later run. A writable connection therefore means one destructive
    candidate (``DROP TABLE``, ``DELETE FROM``, ...) silently corrupts the
    benchmark for good. Opening with ``mode=ro`` plus ``PRAGMA query_only``
    makes those statements fail instead, which :func:`execute_sql` already
    reports as a SQL error.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        sqlite3.Connection: A read-only connection.

    Raises:
        FileNotFoundError: if `db_path` does not exist.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")

    # as_uri() needs an absolute path and takes care of escaping, so paths with
    # spaces or '#' don't break the URI.
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn
