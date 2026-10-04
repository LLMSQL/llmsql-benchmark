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
    workdir_path: str | None = None,
    save_report: str | None = None,
    show_mismatches: bool = True,
    max_mismatches: int = 5,
    model_name: str | None = None,
    save_leaderboard_yaml: str | None = None,
    run_metadata: dict | None = None,
)
```

| Argument          | Description                                                                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `outputs`         | **Required**. Either a path to a JSONL file or a list of dicts with predictions.                                                                |
| `workdir_path`    | Directory used to cache downloaded benchmark files. If omitted, a temporary directory is created automatically. |
| `save_report`     | Optional path to save detailed JSON report. Defaults to `evaluation_results_{uuid}.json`.                                                       |
| `show_mismatches` | Print mismatches while evaluating. Default: `True`.                                                                                             |
| `max_mismatches`  | Maximum number of mismatches to print. Default: `5`.                                                                                            |
| `model_name`      | Name of the evaluated model (e.g. `Qwen/Qwen3-0.6B`). Stored in the JSON report and the leaderboard YAML. Default: `None`.                      |
| `save_leaderboard_yaml` | Optional path to also save the results in the leaderboard `run.yaml` format (see [`leaderboard/`](../../leaderboard)). Default: `None` (not saved). |
| `run_metadata`    | Optional dict deep-merged into the leaderboard YAML to fill fields that cannot be detected automatically (model details, `type`, `inference` backend/arguments, `device`, ...). |

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
* `model_name` – Name of the evaluated model (if provided)
* `version` – LLMSQL benchmark version used for evaluation
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

---

## Leaderboard Format (`run.yaml`)

Pass `save_leaderboard_yaml` to additionally save the results in the same format as the
files in the [`leaderboard/`](../../leaderboard) folder. The evaluation date, `llmsql`
package version, benchmark version, OS name, Python version, execution accuracy, number of
samples and outputs path are filled automatically; everything else is `null` unless
provided via `run_metadata`:

```python
from llmsql import evaluate

report = evaluate(
    "outputs.jsonl",
    model_name="Qwen/Qwen3-0.6B",
    save_leaderboard_yaml="leaderboard/Qwen3-0.6B/5fewshots/run.yaml",
    run_metadata={
        "type": "open-source",
        "model": {"dtype": "bfloat16", "parameter_count": "0.6B"},
        "device": "1xH200",
        "inference": {
            "backend": "vllm",
            "arguments": {"num_fewshots": 5, "temperature": 0.0},
        },
    },
)
```

The same is available from the CLI (`--run-metadata` accepts a YAML/JSON file):

```bash
llmsql evaluate --outputs outputs.jsonl \
    --model-name Qwen/Qwen3-0.6B \
    --save-leaderboard-yaml run.yaml \
    --run-metadata metadata.yaml
```
