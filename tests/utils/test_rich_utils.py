"""Tests for llmsql.utils.rich_utils module."""

from rich.console import Console

from llmsql.utils import rich_utils


def test_log_mismatch_builds_rich_table(monkeypatch) -> None:
    """Ensure log_mismatch renders expected table structure."""
    recording_console = Console(record=True)
    monkeypatch.setattr(rich_utils, "console", recording_console)

    rich_utils.log_mismatch(
        question_id="42",
        question="What is the answer?",
        model_output="SELECT 1",
        gold_sql="SELECT 42",
        prediction_results=[(1,)],
        gold_results=[(42,)],
    )

    output = recording_console.export_text()
    assert "Mismatch for Question 42" in output
    assert "Question" in output
    assert "What is the answer?" in output
    assert "Model Output SQL" in output
    assert "SELECT 1" in output
    assert "Gold SQL" in output
    assert "SELECT 42" in output


def test_print_summary_includes_metrics(monkeypatch) -> None:
    """Ensure print_summary displays computed metrics."""
    recording_console = Console(record=True)
    monkeypatch.setattr(rich_utils, "console", recording_console)

    rich_utils.print_summary(total=5, matches=3, pred_none=1, gold_none=0, sql_errors=2)

    output = recording_console.export_text()
    assert "Evaluation Summary" in output
    assert "Total Samples" in output and "5" in output
    assert "Correct Results" in output and "3 (60.00%)" in output
    assert "Prediction None" in output and "1/5" in output
    assert "Ground Truth None" in output and "0/5" in output
    assert "SQL Errors" in output and "2" in output
