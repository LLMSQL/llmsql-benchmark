"""
LLMSQL Evaluation Module
=========================

Provides the `evaluate()` function to benchmark Text-to-SQL model outputs
on the LLMSQL benchmark.

See the documentation for full usage details.
"""

from datetime import datetime, timezone
from typing import Any
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
from llmsql.utils.leaderboard_utils import (
    build_leaderboard_record,
    write_leaderboard_yaml,
)
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
    model_name: str | None = None,
    save_leaderboard_yaml: str | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict:
    """
    Evaluate predicted SQL queries against the LLMSQL benchmark.

    LLMSQL 1.0: up to 10 SQL candidates are extracted from each completion and
    the prediction counts as correct if one of them returns exactly the same
    (sorted) rows as the gold query.

    LLMSQL 2.0: the SQL is taken from the last ```sql block of the completion
    (falling back to the first WITH/SELECT statement), executed, and compared
    with the verified reference ``answer`` by the lenient execution match of
    :func:`llmsql.utils.matching.results_match` (insensitive to row order,
    duplicates, number formatting and extra columns). The report additionally
    contains ``category_accuracy`` (``lookup``, ``convention``,
    ``text_quantity``).

    Args:
        version: LLMSQL version
        outputs: Either a JSONL file path or a list of dicts.
        workdir_path: Directory to store downloaded benchmark files. If omitted, a
            temporary directory is created automatically.
        save_report: Optional manual save path. If None → auto-generated.
        show_mismatches: Print mismatches while evaluating.
        max_mismatches: Max mismatches to print.
        model_name: Name of the evaluated model (e.g. ``Qwen/Qwen3-0.6B``).
            Stored in the JSON report and in the leaderboard YAML.
        save_leaderboard_yaml: Optional path to additionally save the
            results in the leaderboard ``run.yaml`` format (see the
            ``leaderboard/`` folder). If None, no YAML is written.
        run_metadata: Optional dict deep-merged into the leaderboard YAML to
            fill fields that cannot be detected automatically, e.g.
            ``{"type": "open-source", "inference": {"backend": "vllm",
            "arguments": {"num_fewshots": 5}}}``.

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
        "exact_string_matches": 0,
        "pred_none": 0,
        "gold_none": 0,
        "sql_errors": 0,
    }
    mismatches: list[dict] = []
    # Per-category counts (LLMSQL 2.0 questions have a ``category`` field)
    per_category: dict[str, dict[str, int]] = {}

    for item in track(outputs_list, description="Evaluating"):
        metrics["total"] += 1

        is_match, mismatch_info, m = evaluate_sample(item, questions, conn)

        category = questions.get(item.get("question_id"), {}).get("category")
        if category is not None:
            cat = per_category.setdefault(category, {"total": 0, "matches": 0})
            cat["total"] += 1
            cat["matches"] += is_match

        metrics["matches"] += is_match
        metrics["pred_none"] += m["pred_none"]
        metrics["gold_none"] += m["gold_none"]
        metrics["sql_errors"] += m["sql_error"]
        metrics["exact_string_matches"] += m["exact_string_match"]

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
        metrics["exact_string_matches"],
        category_accuracy=per_category or None,
    )

    # --- Build report structure ---
    accuracy = metrics["matches"] / metrics["total"] if metrics["total"] else 0.0
    report = {
        "model_name": model_name,
        "version": version,
        **metrics,
        "accuracy": accuracy,
        "exact_string_match_accuracy": (
            metrics["exact_string_matches"] / metrics["total"]
            if metrics["total"]
            else 0.0
        ),
        **(
            {
                "category_accuracy": {
                    name: {**c, "accuracy": c["matches"] / c["total"]}
                    for name, c in sorted(per_category.items())
                }
            }
            if per_category
            else {}
        ),
        "mismatches": mismatches,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_mode": input_mode,
    }

    # --- Auto-generate report filename (if not provided) ---
    if save_report is None:
        save_report = f"evaluation_results_{uuid.uuid4()}.json"

    save_json_report(save_report, report)

    if save_leaderboard_yaml is not None:
        record = build_leaderboard_record(
            accuracy=accuracy,
            total=metrics["total"],
            version=version,
            model_name=model_name,
            answers_path=outputs if isinstance(outputs, str) else None,
            run_metadata=run_metadata,
        )
        write_leaderboard_yaml(save_leaderboard_yaml, record)

    conn.close()
    return report
