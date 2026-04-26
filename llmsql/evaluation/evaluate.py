"""
LLMSQL Evaluation Module
=========================

Provides the `evaluate()` function to benchmark Text-to-SQL model outputs
on the LLMSQL benchmark.

See the documentation for full usage details.
"""

from datetime import datetime, timezone
import uuid

from rich.progress import track

from llmsql.config.config import (
    DEFAULT_LLMSQL_VERSION,
    get_repo_id,
)
from llmsql.utils.evaluation_utils import (
    connect_sqlite,
    evaluate_sample,
)
from llmsql.utils.inference_utils import _maybe_download, resolve_workdir_path
from llmsql.utils.rich_utils import log_mismatch, print_summary
from llmsql.utils.utils import load_jsonl, load_jsonl_dict_by_key, save_json_report


def evaluate(
    outputs: str | list[dict[int, str | int]],
    *,
    version: str = DEFAULT_LLMSQL_VERSION,
    workdir_path: str | None = None,
    save_report: str | None = None,
    show_mismatches: bool = True,
    max_mismatches: int = 5,
) -> dict:
    """
    Evaluate predicted SQL queries against the LLMSQL benchmark.

    Args:
        version: LLMSQL version
        outputs: Either a JSONL file path or a list of dicts.
        workdir_path: Directory to store downloaded benchmark files. If omitted, a
            temporary directory is created automatically.
        save_report: Optional manual save path. If None → auto-generated.
        show_mismatches: Print mismatches while evaluating.
        max_mismatches: Max mismatches to print.

    Returns:
        dict: Metrics and mismatches.
    """

    # Determine input type
    input_mode = "jsonl_path" if isinstance(outputs, str) else "dict_list"
    workdir = resolve_workdir_path(workdir_path)

    repo_id = get_repo_id(version)

    questions_path = _maybe_download(repo_id, "questions.jsonl", workdir)
    db_path = _maybe_download(repo_id, "sqlite_tables.db", workdir)

    # --- Load benchmark questions ---
    questions = load_jsonl_dict_by_key(questions_path, key="question_id")

    # --- Load predictions (path or list) ---
    if isinstance(outputs, str):
        outputs_list = load_jsonl(outputs)
    elif isinstance(outputs, list):
        outputs_list = outputs
    else:
        raise TypeError(
            "outputs must be file path or list of dicts in format {'question_id': int, 'completion': str}"
        )

    # --- Connect to DB ---
    conn = connect_sqlite(db_path)

    # --- Evaluation loop ---
    metrics = {
        "total": 0,
        "matches": 0,
        "pred_none": 0,
        "gold_none": 0,
        "sql_errors": 0,
    }
    mismatches: list[dict] = []

    for item in track(outputs_list, description="Evaluating"):
        metrics["total"] += 1

        is_match, mismatch_info, m = evaluate_sample(item, questions, conn)

        metrics["matches"] += is_match
        metrics["pred_none"] += m["pred_none"]
        metrics["gold_none"] += m["gold_none"]
        metrics["sql_errors"] += m["sql_error"]

        if mismatch_info:
            mismatches.append(mismatch_info)
            if show_mismatches and len(mismatches) <= max_mismatches:
                log_mismatch(**mismatch_info)

    print_summary(
        metrics["total"],
        metrics["matches"],
        metrics["pred_none"],
        metrics["gold_none"],
        metrics["sql_errors"],
    )

    # --- Build report structure ---
    report = {
        **metrics,
        "accuracy": metrics["matches"] / metrics["total"] if metrics["total"] else 0,
        "mismatches": mismatches,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_mode": input_mode,
    }

    # --- Auto-generate report filename (if not provided) ---
    if save_report is None:
        save_report = f"evaluation_results_{uuid.uuid4()}.json"

    save_json_report(save_report, report)

    conn.close()
    return report
