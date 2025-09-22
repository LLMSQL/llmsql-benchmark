from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


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


def log_mismatch(
    question_id: str,
    question: str,
    model_output: str,
    gold_sql: str,
    prediction_results: Any,
    gold_results: Any,
) -> None:
    """Pretty-print a mismatch with Rich."""
    table = Table(
        title=f"[red]Mismatch for Question {question_id}[/red]", show_lines=True
    )
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", overflow="fold")

    table.add_row("Question", question)
    table.add_row("Model Output SQL", model_output.strip())
    table.add_row("Gold SQL", gold_sql.strip())
    table.add_row("Prediction Results", str(prediction_results))
    table.add_row("Gold Results", str(gold_results))

    console.print(table)


def print_summary(
    total: int,
    matches: int,
    pred_none: int,
    gold_none: int,
    sql_errors: int,
) -> None:
    """Pretty-print summary with Rich."""
    table = Table(title="[green]Evaluation Summary[/green]", show_lines=True)
    table.add_column("Metric", style="bold green")
    table.add_column("Value", style="bold yellow")

    table.add_row("Total Samples", str(total))
    table.add_row("Correct Results", f"{matches} ({matches / total:.2%})")
    table.add_row("Prediction None", f"{pred_none}/{total}")
    table.add_row("Ground Truth None", f"{gold_none}/{total}")
    table.add_row("SQL Errors", str(sql_errors))

    console.print(table)
