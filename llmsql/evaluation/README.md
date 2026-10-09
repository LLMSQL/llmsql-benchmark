# Evaluation: Benchmarking Text-to-SQL Models on LLMSQL

This module provides a **pipeline for evaluating Text-to-SQL model outputs** on the **LLMSQL benchmark**.
It executes your model’s SQL predictions on the benchmark database, compares the results with the reference answers, logs mismatches, and generates detailed evaluation reports.

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

# Evaluate outputs from a JSONL file (LLMSQL 2.0 is the default version)
report = evaluate("path_to_your_outputs.jsonl", version="2.0")
print(report["accuracy"], report["category_accuracy"])
```

```python
# Or evaluate from a list of prediction dicts
predictions = [
    {"question_id": 1, "completion": "```sql\nSELECT 1;\n```"},
    {"question_id": 2, "completion": "```sql\nSELECT 2;\n```"},
]
report = evaluate(predictions, version="2.0")
print(report)
```

---

## How predictions are scored

### LLMSQL 2.0 (`version="2.0"`)

1. **SQL extraction.** The SQL is taken from the **last** ```` ```sql ```` block of the
   completion (also ```` ```sqlite ```` or a bare ```` ``` ```` block), falling back to the
   first `WITH`/`SELECT` statement if there is no block. The prompt shows the real table
   names, so no table-name substitution is done.
2. **Execution.** The query is executed on `sqlite_tables.db` with a 10-second time limit.
3. **Lenient execution match** against the verified `answer` of the question
   ([`llmsql/utils/matching.py`](../utils/matching.py)). A prediction is correct if, after
   normalisation, it returns the same rows, where:
   * row order and duplicate rows are ignored;
   * numbers are compared as floats rounded to 2 decimals (`42 == "42" == 42.0`);
   * thousands separators, currency signs (`$ £ €`), units / magnitude words
     (`million`, `%`, `km`, ...) and parentheses around a number (`(0)`) are ignored;
   * a trailing count in parentheses is ignored (`Al Horford (15)` == `Al Horford`);
   * a two-part answer may be split into two columns (`("KeyArena", "10,891")` ==
     `"KeyArena 10,891"`);
   * extra columns are accepted if one predicted column equals the single answer column.

The report additionally contains `category_accuracy` for the three categories
(`lookup`, `convention`, `text_quantity`).

### LLMSQL 1.0 (`version="1.0"`)

Up to 10 SQL candidates are extracted from the completion, the placeholder table name
`"Table"` is replaced by the real one, and the prediction is correct if one candidate
returns exactly the same (sorted) rows as the gold query.

---

## Function Arguments

```python
evaluate(
    outputs,
    *,
    version: str = "2.0",
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
| `version`         | Benchmark version: `"2.0"` (default) or `"1.0"`.                                                                                                |
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
{"question_id": 1, "completion": "```sql\nSELECT \"Singer(s)\" FROM \"1-29135051-2\" WHERE \"Comedian\" = 'Joe Wilkinson';\n```"}
{"question_id": 2, "completion": "Reasoning... ```sql\nSELECT COUNT(*) FROM \"1-10399701-2\";\n```"}
```

* `question_id` (integer) must match IDs in `questions.jsonl`.
* `completion` should contain your model’s raw output (extra text is allowed; SQL is extracted automatically, see above).
* This is exactly the format written by the `inference_*` functions.

---

## Output & Metrics

The evaluation returns a dictionary containing:

* `total` – Total queries evaluated
* `matches` – Queries where predicted SQL results match gold results
* `pred_none` – Queries where the model returned `NULL` or no result
* `gold_none` – Queries where gold reference is `NULL` or no result
* `sql_errors` – Invalid SQL or execution errors (for 2.0 also completions without any SQL, and queries over the time limit)
* `exact_string_matches` – Predictions whose SQL equals the gold SQL up to whitespace
* `accuracy` – Overall execution accuracy
* `category_accuracy` – (LLMSQL 2.0 only) `total`, `matches` and `accuracy` per category
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
