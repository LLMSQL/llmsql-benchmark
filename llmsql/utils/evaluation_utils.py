import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from huggingface_hub import hf_hub_download

from llmsql.loggers.logging_config import log
from llmsql.utils.matching import extract_sql, results_match
from llmsql.utils.regex_extractor import find_sql

# Wall-clock limit for executing a predicted query in the LLMSQL 2.0 protocol.
SQL_TIMEOUT_SECONDS: float = 10.0


def execute_sql(conn: sqlite3.Connection, sql: str) -> list[tuple] | None:
    """
    Execute a SQL query on the given SQLite connection and return its results.

    The results are always sorted to avoid differences caused by row order (order agnostic).
    If the query fails, the function logs the error and returns None.

    Args:
        conn (sqlite3.Connection): An active SQLite database connection.
        sql (str): SQL query string to execute.

    Returns:
        Optional[List[Tuple]]:
            - Sorted list of result rows (each row as a tuple) if successful.
            - [(None,)] if the query executed but returned NULL values.
            - None if the SQL execution failed due to an exception.
    """
    try:
        cur = conn.cursor()
        cur.execute(sql)
        results = cur.fetchall()
        return sorted(results)
    except Exception:
        return None


def execute_sql_with_timeout(
    conn: sqlite3.Connection, sql: str, timeout: float | None = SQL_TIMEOUT_SECONDS
) -> list[tuple] | None:
    """
    Execute a SQL query and return its rows unchanged (not sorted).

    Used by the LLMSQL 2.0 protocol, where the comparison
    (:func:`llmsql.utils.matching.results_match`) is itself order-insensitive.
    A query running longer than ``timeout`` seconds is interrupted (e.g. an
    accidental cross join on large tables) and counted as failed.

    Args:
        conn: An active SQLite database connection.
        sql: SQL query string to execute.
        timeout: Time limit in seconds, or ``None`` for no limit.

    Returns:
        The result rows, or None if the execution failed or timed out.
    """
    if timeout is not None:
        deadline = time.monotonic() + timeout
        conn.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0, 10000
        )
    try:
        return conn.execute(sql).fetchall()
    except Exception:
        return None
    finally:
        if timeout is not None:
            conn.set_progress_handler(None, 0)


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


def evaluate_sample(
    item: dict[str, int | str],
    questions: dict[int, dict[str, Any]],
    conn: sqlite3.Connection,
) -> tuple[int, dict[str, Any] | None, dict[Any, Any]]:
    """
    Evaluate a single model prediction against the gold (ground-truth) SQL query.

    LLMSQL 2.0 questions (those with an ``answer`` field) are delegated to
    :func:`evaluate_sample_v2` (last fenced ``sql`` code block, lenient comparison with the
    verified answer). For LLMSQL 1.0 questions this function:
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
    # Extract question metadata
    qid = item["question_id"]
    assert isinstance(
        qid, int
    ), "question_id in the outputs file needs to be of type int."
    q_info = questions[qid]

    # LLMSQL 2.0 questions carry their verified reference result
    if "answer" in q_info:
        return evaluate_sample_v2(item, q_info, conn)
    table_id, gold_sql, question_text = (
        q_info["table_id"],
        q_info["sql"],
        q_info["question"],
    )

    # Execute the gold (ground-truth) SQL
    gold_results = execute_sql(conn, gold_sql)

    # Initialize counters for this sample
    pred_none = gold_none = sql_error = exact_string_match = 0

    # Track if gold query returned a NULL-equivalent result
    if gold_results == [(None,)]:
        gold_none = 1

    # Flag for whether the prediction was correct
    is_match = 0
    last_pred_res = None  # store last prediction results for mismatch logging

    # Loop over all SQL queries extracted from the model output
    assert isinstance(
        item["completion"], str
    ), f"Completion filed in outputs file must be of type string: {item['completion']}. Type: {type(item['completion'])}"
    for pred_sql in find_sql(item["completion"]):
        # Replace placeholder table names with the actual one
        pred_sql_fixed = fix_table_name(pred_sql, table_id)

        # Execute predicted SQL
        pred_res = execute_sql(conn, pred_sql_fixed)
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
            "model_output": item["completion"],
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


def evaluate_sample_v2(
    item: dict[str, Any],
    q_info: dict[str, Any],
    conn: sqlite3.Connection,
    timeout: float | None = SQL_TIMEOUT_SECONDS,
) -> tuple[int, dict[str, Any] | None, dict[str, int]]:
    """
    Evaluate one prediction with the LLMSQL 2.0 protocol.

    1. The SQL is extracted with :func:`llmsql.utils.matching.extract_sql`
       (last fenced ``sql`` code block, falling back to the first WITH/SELECT statement).
       The model is shown the real table names, so no table-name fixing is done.
    2. It is executed on the benchmark database (``timeout`` seconds limit).
    3. Its rows are compared with the verified reference ``answer`` using the
       lenient :func:`llmsql.utils.matching.results_match`.

    Counters: ``sql_error`` is 1 if no SQL could be extracted or the query
    failed; ``pred_none`` is 1 if the query returned a single NULL value;
    ``gold_none`` is 1 if the reference answer is a single NULL value;
    ``exact_string_match`` is 1 if the extracted SQL equals the reference SQL
    up to whitespace and the trailing semicolon.

    Args:
        item: Prediction with ``question_id`` and ``completion``.
        q_info: The benchmark question (with ``sql`` and ``answer``).
        conn: Active SQLite connection to the benchmark database.
        timeout: Time limit for the predicted query in seconds.

    Returns:
        Same structure as :func:`evaluate_sample`.
    """
    completion = item["completion"]
    assert isinstance(
        completion, str
    ), f"Completion filed in outputs file must be of type string: {completion}. Type: {type(completion)}"
    gold_sql = q_info["sql"]
    gold_results = q_info["answer"]

    pred_none = sql_error = exact_string_match = 0
    gold_none = int(gold_results == [[None]])

    pred_sql = extract_sql(completion)
    pred_res = None
    if pred_sql is None:
        sql_error = 1
    else:
        pred_res = execute_sql_with_timeout(conn, pred_sql, timeout)
        if pred_res is None:
            sql_error = 1
        elif pred_res == [(None,)]:
            pred_none = 1
        if normalize_sql(pred_sql) == normalize_sql(gold_sql):
            exact_string_match = 1

    is_match = int(results_match(gold_results, pred_res))

    mismatch_info = None
    if not is_match:
        mismatch_info = {
            "question_id": item["question_id"],
            "question": q_info["question"],
            "gold_sql": gold_sql,
            "model_output": completion,
            "gold_results": gold_results,
            "prediction_results": pred_res,
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
    """Create SQLite connection."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    return sqlite3.connect(db_path)
