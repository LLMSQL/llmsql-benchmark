import sqlite3
from typing import Any

from llmsql.loggers.logging_config import log
from llmsql.utils.regex_extractor import find_sql


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
    except Exception as e:
        log.error(f"SQL Execution Error: {e} | Query: {sql}")
        return None


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


def evaluate_sample(
    item: dict[str, int | str],
    questions: dict[int, dict[str, str]],
    conn: sqlite3.Connection,
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
    table_id, gold_sql, question_text = (
        q_info["table_id"],
        q_info["sql"],
        q_info["question"],
    )

    # Execute the gold (ground-truth) SQL
    gold_results = execute_sql(conn, gold_sql)

    # Initialize counters for this sample
    pred_none = gold_none = sql_error = 0

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
        {"pred_none": pred_none, "gold_none": gold_none, "sql_error": sql_error},
    )
