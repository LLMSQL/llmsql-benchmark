"""
LLMSQL Benchmark Evaluator

This module provides a high-level class `LLMSQLEvaluator` for
evaluating model-generated SQL queries against the gold benchmark,
logging mismatches and generating detailed evaluation reports.

Example usage:

    from llmsql import LLMSQLEvaluator

    evaluator = LLMSQLEvaluator(workdir_path="llmsql_workdir")
    report = evaluator.evaluate("examples/test_output.jsonl")
    print(report)

Notes:
  - If `questions.jsonl` or `sqlite_tables.db` are not found locally, they
    will be automatically downloaded from the Hugging Face Hub.
  - The `evaluate` method returns a metrics dictionary and optionally saves
    a JSON report with mismatches and statistics.
"""

import json
import os
from pathlib import Path
import sqlite3

from huggingface_hub import hf_hub_download
from rich.progress import track

from llmsql.loggers.logging_config import log
from llmsql.utils.evaluation_utils import evaluate_sample
from llmsql.utils.rich_utils import log_mismatch, print_summary


class LLMSQLEvaluator:
    """
    High-level class for managing LLMSQL benchmark evaluation:
      - Downloading the benchmark SQLite database.
      - Evaluating model predictions against gold queries.
    """

    def __init__(self, workdir_path: str = "llmsql_workdir"):
        """
        Initialize evaluator.
        """
        self.conn: sqlite3.Connection | None = None
        self.workdir_path = Path(workdir_path)
        self.workdir_path.mkdir(parents=True, exist_ok=True)
        self.repo_id = "llmsql-bench/llmsql-benchmark"

    def _download_file(self, filename: str) -> str:
        """
        Download the official SQLite DB from Hugging Face Hub.
        Will be downloaded to the workdir specified in init.

        Returns:
            str: Local path to the downloaded DB file.
        """
        file_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=self.workdir_path,
        )
        assert isinstance(
            file_path, str
        ), f"file path to the {filename} is not string. File path: {file_path}, type: {type(file_path)}"
        log.info(f"File saved at: {file_path}")
        return file_path

    def connect(self, db_path: str) -> None:
        """Establish SQLite connection."""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at {db_path}")
        self.conn = sqlite3.connect(db_path)

    def close(self) -> None:
        """Close SQLite connection if open."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def evaluate(
        self,
        outputs_path: str,
        questions_path: str | None = None,
        db_path: str | None = None,
        save_report: str | None = None,
        show_mismatches: bool = True,
        max_mismatches: int = 5,
    ) -> dict:
        """
        Evaluate predicted SQL queries against benchmark ground truth.

        Args:
            outputs_path (str): Path to JSONL file with model predictions.
            questions_path (str): Path to benchmark questions JSONL.
            save_report (str, optional): If set, saves detailed JSON report.
            show_mismatches (bool, optional): Print mismatches to log.
            max_mismatches (int, optional): Max mismatches to display.

        Returns:
            Dict: Metrics summary.

        Example:
            ```
            from llmsql import LLMSQLEvaluator

            evaluator = LLMSQLEvaluator(workdir_path="llmsql_workdir")
            report = evaluator.evaluate("examples/test_output.jsonl")
            print(report)
            ```
        """
        if (
            questions_path is None
            and not (self.workdir_path / Path("questions.jsonl")).is_file()
        ):
            questions_path = self._download_file("questions.jsonl")
        elif questions_path is None:
            questions_path = f"{self.workdir_path}/questions.jsonl"

        if (
            db_path is None
            and not (self.workdir_path / Path("sqlite_tables.db")).is_file()
        ):
            db_path = self._download_file("sqlite_tables.db")
        elif db_path is None:
            db_path = f"{self.workdir_path}/sqlite_tables.db"

        if self.conn is None:
            self.connect(db_path=db_path)

        with open(questions_path, encoding="utf-8") as f:
            questions = {q["question_id"]: q for q in map(json.loads, f)}

        with open(outputs_path, encoding="utf-8") as f:
            outputs = [json.loads(line) for line in f]

        metrics = {
            "total": 0,
            "matches": 0,
            "pred_none": 0,
            "gold_none": 0,
            "sql_errors": 0,
        }
        mismatches: list[dict] = []

        for item in track(outputs, description="Evaluating"):
            metrics["total"] += 1
            is_match, mismatch_info, m_update = evaluate_sample(
                item,
                questions,
                self.conn,  # type: ignore
            )

            metrics["matches"] += is_match
            metrics["pred_none"] += m_update["pred_none"]
            metrics["gold_none"] += m_update["gold_none"]
            metrics["sql_errors"] += m_update["sql_error"]

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

        report = {
            **metrics,
            "accuracy": metrics["matches"] / metrics["total"]
            if metrics["total"] > 0
            else 0,
            "mismatches": mismatches,
        }

        if save_report:
            with open(save_report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            log.info(f"Saved report to {save_report}")

        return report
