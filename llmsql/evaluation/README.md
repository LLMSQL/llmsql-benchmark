# Evaluation: Benchmarking Text-to-SQL Models on LLMSQL

This folder contains the evaluation pipeline for the **LLMSQL benchmark**.
It allows you to test your model’s Text-to-SQL outputs against the gold-standard queries and database, and generate evaluation metrics and mismatch reports.

Now, you can use it directly with the package:

## Quick Start

### Install

Make sure you have the package installed:

```bash
pip3 install llmsql
```

### Quick Start

```python
from llmsql import LLMSQLEvaluator

evaluator = LLMSQLEvaluator(workdir_path="llmsql_workdir")
report = evaluator.evaluate(outputs_path="path_to_your_outputs.jsonl")
print(report)
```

### Arguments (evaluate method)

* `outputs_path` (**required**) – Path to your model’s predictions in JSONL format.
* `questions_path` (optional) – Gold benchmark questions + reference SQL queries.
* `db_path` (optional) – SQLite DB with all tables used in evaluation.
* `save_report` (optional) – Path to save a detailed JSON report with results.
* `show_mismatches` (default: True) – Print mismatches to log.
* `max_mismatches` (default: 5) – Maximum number of mismatches to display.

### Expected Input Format

Your model’s predictions (`outputs_path`) must be stored in **JSONL format** (one JSON object per line):

```json
{"question_id": "1", "completion": "SELECT name FROM Table WHERE age > 30"}
{"question_id": "2", "completion": "SELECT COUNT(*) FROM Table"}
{"question_id": "3", "completion": "The model answer can also be raw and unstructured: SELECT smth FROM smt"}
```

* `question_id` must match the IDs in `questions.jsonl`.
* `completion` should contain the model’s SQL prediction (can include extra text; SQL is extracted automatically).

---

## Output & Metrics

The evaluation returns:

* **Total queries** evaluated
* **Exact matches** (predicted SQL results == gold SQL results)
* **Predicted None** (model returned `NULL` or no result)
* **Gold None** (reference result was `NULL` or no result)
* **SQL Errors** (invalid SQL or execution error)

Example console output:

```
Evaluating ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 100/100
Total: 100 | Matches: 82 | Pred None: 5 | Gold None: 3 | SQL Errors: 2
```

If `save_report` is provided, a detailed JSON report is saved, including mismatches.
