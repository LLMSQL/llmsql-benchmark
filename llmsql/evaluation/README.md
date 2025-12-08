# Evaluation: Benchmarking Text-to-SQL Models on LLMSQL

This module provides a **pipeline for evaluating Text-to-SQL model outputs** on the **LLMSQL benchmark**.
It checks your model’s SQL predictions against the gold-standard queries and database, logs mismatches, and generates detailed evaluation reports.

You can now use it directly via the `evaluate()` function.

---

## Quick Start

### Install

```bash
pip install llmsql
```

### Evaluate Model Predictions

```python
from llmsql import evaluate

# Evaluate outputs from a JSONL file
report = evaluate("path_to_your_outputs.jsonl")
print(report)
```

```python
# Or evaluate from a list of prediction dicts
predictions = [
    {"question_id": "1", "predicted_sql": "SELECT name FROM Table WHERE age > 30"},
    {"question_id": "2", "predicted_sql": "SELECT COUNT(*) FROM Table"},
]
report = evaluate(predictions)
print(report)
```

---

## Function Arguments

```python
evaluate(
    outputs,
    *,
    workdir_path: str | None = "llmsql_workdir",
    questions_path: str | None = None,
    db_path: str | None = None,
    save_report: str | None = None,
    show_mismatches: bool = True,
    max_mismatches: int = 5,
)
```

| Argument          | Description                                                                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `outputs`         | **Required**. Either a path to a JSONL file or a list of dicts with predictions.                                                                |
| `workdir_path`    | Directory for automatic download of benchmark files (ignored if both `questions_path` and `db_path` are provided). Default: `"llmsql_workdir"`. |
| `questions_path`  | Optional path to benchmark questions JSONL file.                                                                                                |
| `db_path`         | Optional path to SQLite DB with evaluation tables.                                                                                              |
| `save_report`     | Optional path to save detailed JSON report. Defaults to `evaluation_results_{uuid}.json`.                                                       |
| `show_mismatches` | Print mismatches while evaluating. Default: `True`.                                                                                             |
| `max_mismatches`  | Maximum number of mismatches to print. Default: `5`.                                                                                            |

---

## Input Format

Your model predictions must be in **JSONL format** (one JSON object per line):

```json
{"question_id": "1", "predicted_sql": "SELECT name FROM Table WHERE age > 30"}
{"question_id": "2", "predicted_sql": "SELECT COUNT(*) FROM Table"}
{"question_id": "3", "predicted_sql": "SELECT * FROM Table WHERE active=1"}
```

* `question_id` must match IDs in `questions.jsonl`.
* `predicted_sql` should contain your model’s SQL output (extra text is allowed; SQL is extracted automatically).

---

## Output & Metrics

The evaluation returns a dictionary containing:

* `total` – Total queries evaluated
* `matches` – Queries where predicted SQL results match gold results
* `pred_none` – Queries where the model returned `NULL` or no result
* `gold_none` – Queries where gold reference is `NULL` or no result
* `sql_errors` – Invalid SQL or execution errors
* `accuracy` – Overall exact match accuracy
* `mismatches` – List of mismatched queries with details
* `timestamp` – Evaluation timestamp
* `input_mode` – Whether results were provided as JSONL path or dict list

Example console output:

```
Evaluating ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 100/100
Total: 100 | Matches: 82 | Pred None: 5 | Gold None: 3 | SQL Errors: 2
```

**Report Saving**

* By default, the report is saved as `evaluation_results_{uuid}.json` in the current directory.
* Includes timestamp and input mode (JSONL path or dict list).
* You can override the save path via the `save_report` argument.
