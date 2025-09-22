

# Evaluation: Benchmarking Text-to-SQL Models on LLMSQL

This folder contains the evaluation pipeline for the **LLMSQL benchmark**.
It allows you to test your model’s Text-to-SQL outputs against the gold-standard queries and database, and generate evaluation metrics and mismatch reports.

---

## Folder Structure

```
evaluation/
├── download_db.py          # Download the SQLite database used for evaluation
├── evaluate_answers.py     # Main script: evaluate model outputs against gold queries
├── utils/                  # Helper functions and utilities
│   ├── evaluation_utils.py # Core evaluation logic (SQL execution, comparison, logging)
│   ├── logging_config.py   # Standard logging configuration
│   ├── regex_extractor.py  # Extract SQL queries from raw model completions
│   └── utils.py            # Additional helpers
└── README.md               # You are here :)
```

---


## ⚠️ Important: Run Commands from Project Root

All commands shown below should be executed from the **project root folder** (`LLMSQL/`), not from inside `evaluation/`.

Example:

```bash
cd LLMSQL
python3 evaluation/download_db.py --target_dir dataset
```

---

## Install

Make sure you have the dependencies installed:

```bash
pip3 install -r requirements.txt
```

Key libraries:

* `sqlite3` (Python built-in) – for executing SQL queries
* `huggingface_hub` – for downloading the benchmark DB
* `rich` – for progress bars and nice console output

# Basic Usage

🚨 Important: All commands shown below should be executed from the **project root folder** (`LLMSQL/`), not from inside `evaluation/`.

## Step 1: Download the Benchmark Database

Before running evaluation, download the official SQLite DB file from Hugging Face Hub(`llmsql-bench/llmsql-benchmark`). You can do it with the help of script:

```bash
python3 evaluation/download_db.py
```

Arguments:

* `--repo_id` (optional, default: `llmsql-bench/llmsql-benchmark`)
  Hugging Face dataset repository containing the database.
* `--target_dir` (optional, default: `dataset`)
  Directory where the DB file will be saved.

This will download:

```
dataset/sqlite_tables.db
```

---

## Step 2: Evaluate Model Outputs

Run the evaluation with:

```bash
python3 evaluation/evaluate_answers.py \
    --outputs_path path/to/your_model_outputs.jsonl \
```

### Arguments

* `--outputs_path` (**required**)
  Path to your model’s predictions in JSONL format.
* `--questions_path` (default: `dataset/questions.jsonl`)
  Gold benchmark questions + reference SQL queries.
* `--db_file_path` (default: `dataset/sqlite_tables.db`)
  SQLite DB with all tables used in evaluation.
* `--save_report` (optional)
  Path to save a detailed JSON report with results, e.g. `evaluation_report.json`.
* `--no_mismatches` (optional flag)
  Suppress mismatch details in the console output.
* `--max_mismatches` (default: 5)
  Maximum number of mismatches to print to the console.


### Expected Input Format

Your model’s predictions (`--outputs_path`) must be stored in **JSONL format** (one JSON object per line):

```json
{"question_id": "1", "completion": "SELECT name FROM Table WHERE age > 30"}
{"question_id": "2", "completion": "SELECT COUNT(*) FROM Table"}
{"question_id": "3", "completion": "The model answer can also be raw and unstructured: SELECT smth FROM smt"}
...
```

* `question_id` must match the IDs in `questions.jsonl`.
* `completion` should contain the model’s SQL prediction (can include extra text; SQL is extracted automatically).

---

## Output & Metrics

The script reports:

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

If `--save_report` is provided, a detailed JSON report is saved, including mismatches:

```json
{
  "total": 100,
  "matches": 82,
  "pred_none": 5,
  "gold_none": 3,
  "sql_errors": 2,
  "accuracy": 0.82,
  "mismatches": [
    {
      "question_id": "q17",
      "question": "How many users are older than 50?",
      "gold_sql": "SELECT COUNT(*) FROM users WHERE age > 50",
      "model_output": "SELECT COUNT(*) FROM Table WHERE age > 60",
      "gold_results": [[42]],
      "prediction_results": [[25]]
    }
  ]
}
```

---

## 🧠 How It Works (Under the Hood)

* The model is only required to generate queries using the placeholder table name `"Table"`.
* The script automatically replaces `"Table"` with the correct table name (`fix_table_name`).
* Queries are executed on the SQLite DB, and results are compared against the gold query results.
* Errors and mismatches are logged for analysis.

---

## ✅ Summary

With this evaluation suite you can:

1. **Download** the official evaluation database.
2. **Run your model** to generate predictions.
3. **Evaluate** those predictions against gold SQL.
4. **Inspect** mismatches and metrics in a clean report.

This makes it easy to benchmark different Text-to-SQL models on **LLMSQL** in a standardized way.
