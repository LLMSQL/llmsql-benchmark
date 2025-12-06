Evaluation API Reference
========================

The `evaluate()` function allows you to benchmark Text-to-SQL model outputs
against the LLMSQL gold queries and SQLite database. It prints metrics, logs
mismatches, and saves detailed reports automatically.

Features
--------
- Evaluate model predictions from JSONL files or Python dicts.
- Automatically download benchmark questions and SQLite DB if missing.
- Prints mismatch summaries and supports configurable reporting.
- Saves detailed JSON report with metrics, mismatches, timestamp, and input mode.

Usage Examples
--------------

Evaluate from a JSONL file:

.. code-block:: python

    from llmsql.evaluation.evaluate import evaluate

    report = evaluate("path_to_outputs.jsonl")
    print(report)

Evaluate from a list of Python dicts:

.. code-block:: python

    predictions = [
        {"question_id": "1", "predicted_sql": "SELECT name FROM Table WHERE age > 30"},
        {"question_id": "2", "predicted_sql": "SELECT COUNT(*) FROM Table"},
    ]

    report = evaluate(predictions)
    print(report)

Providing your own DB and questions (skip workdir):

.. code-block:: python

    report = evaluate(
        "path_to_outputs.jsonl",
        questions_path="bench/questions.jsonl",
        db_path="bench/sqlite_tables.db",
        workdir_path=None
    )

Function Arguments
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Argument
     - Description
   * - outputs
     - Path to JSONL file or a list of prediction dicts (required).
   * - workdir_path
     - Directory for automatic benchmark downloads. Ignored if both questions_path and db_path are provided. Default: "llmsql_workdir".
   * - questions_path
     - Optional path to benchmark questions JSONL file.
   * - db_path
     - Optional path to SQLite DB with evaluation tables.
   * - save_report
     - Path to save detailed JSON report. Defaults to "evaluation_results_{uuid}.json".
   * - show_mismatches
     - Print mismatches while evaluating. Default True.
   * - max_mismatches
     - Maximum number of mismatches to display. Default 5.

Input Format
------------

The predictions should be in JSONL format:

.. code-block:: json

    {"question_id": "1", "predicted_sql": "SELECT name FROM Table WHERE age > 30"}
    {"question_id": "2", "predicted_sql": "SELECT COUNT(*) FROM Table"}
    {"question_id": "3", "predicted_sql": "SELECT * FROM Table WHERE active=1"}

Output Metrics
--------------

The function returns a dictionary with the following keys:

- total – Total queries evaluated
- matches – Queries where predicted SQL results match gold results
- pred_none – Queries where the model returned NULL or no result
- gold_none – Queries where the reference result was NULL or no result
- sql_errors – Invalid SQL or execution errors
- accuracy – Overall exact match accuracy
- mismatches – List of mismatched queries with details
- timestamp – Evaluation timestamp
- input_mode – How results were provided ("jsonl_path" or "dict_list")

Report Saving
-------------

By default, a report is saved automatically as `evaluation_results_{uuid}.json` in the current directory.
It contains metrics, mismatches, timestamp, and input mode. You can override this path using `save_report`.

---

.. automodule:: llmsql.evaluation.evaluate
   :members:
   :undoc-members:
   :show-inheritance:


---

.. raw:: html

   <div style="text-align:center; margin-top:2rem; color:#666;">
     💬 Made with ❤️ by the LLMSQL Team
   </div>
