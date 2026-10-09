Evaluation API Reference
========================

The `evaluate()` function allows you to benchmark Text-to-SQL model outputs
against the LLMSQL reference answers and SQLite database. It prints metrics, logs
mismatches, and saves detailed reports automatically.

Scoring
-------

**LLMSQL 2.0** (``version="2.0"``, default):

1. The SQL is taken from the **last** fenced ``sql`` code block of the completion (also
   ``sqlite`` or a bare fence), falling back to the first ``WITH``/``SELECT``
   statement. The prompt shows the real table names, so no table-name substitution is done.
2. The query is executed on ``sqlite_tables.db`` with a 10-second time limit.
3. The rows are compared with the verified ``answer`` of the question by a lenient
   execution match (:func:`llmsql.utils.matching.results_match`):

   * row order and duplicate rows are ignored;
   * numbers are compared as floats rounded to 2 decimals;
   * thousands separators, currency signs, units / magnitude words (``million``, ``%``,
     ``km``, ...) and parentheses around a number are ignored;
   * a trailing count in parentheses is ignored (``Al Horford (15)`` == ``Al Horford``);
   * a two-part answer may be split into two columns;
   * extra columns are accepted if one predicted column equals the single answer column.

The report additionally contains ``category_accuracy`` for ``lookup``, ``convention`` and
``text_quantity``.

**LLMSQL 1.0** (``version="1.0"``): up to 10 SQL candidates are extracted from the
completion, the placeholder table name ``"Table"`` is replaced by the real one, and the
prediction is correct if one candidate returns exactly the same (sorted) rows as the gold
query.

Features
--------
- Evaluate model predictions from JSONL files or Python dicts.
- Automatically download benchmark questions and SQLite DB if missing.
- Prints mismatch summaries and supports configurable reporting.
- Saves detailed JSON report with metrics, mismatches, timestamp, and input mode.
- Optionally saves the results in the leaderboard ``run.yaml`` format.

Usage Examples
--------------

Evaluate from a JSONL file:

.. code-block:: python

    from llmsql.evaluation.evaluate import evaluate

    report = evaluate("path_to_outputs.jsonl", version="2.0")
    print(report["accuracy"], report["category_accuracy"])

Evaluate from a list of Python dicts:

.. code-block:: python

    predictions = [
        {"question_id": 1, "completion": "```sql\nSELECT 1;\n```"},
        {"question_id": 2, "completion": "```sql\nSELECT 2;\n```"},
    ]

    report = evaluate(predictions, version="2.0")
    print(report)

Using a persistent cache directory for benchmark downloads:

.. code-block:: python

    report = evaluate(
        "path_to_outputs.jsonl",
        workdir_path="./benchmark-cache",
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
   * - version
     - Benchmark version, ``"2.0"`` (default) or ``"1.0"``.
   * - workdir_path
     - Directory used to cache downloaded benchmark files. If omitted, a temporary directory is created automatically.
   * - save_report
     - Path to save detailed JSON report. Defaults to "evaluation_results_{uuid}.json".
   * - show_mismatches
     - Print mismatches while evaluating. Default True.
   * - max_mismatches
     - Maximum number of mismatches to display. Default 5.
   * - model_name
     - Name of the evaluated model (e.g. ``Qwen/Qwen3-0.6B``). Stored in the JSON report and the leaderboard YAML. Default None.
   * - save_leaderboard_yaml
     - Optional path to also save the results in the leaderboard ``run.yaml`` format. Default None (not saved).
   * - run_metadata
     - Optional dict deep-merged into the leaderboard YAML (model details, ``type``, ``inference`` backend/arguments, ``device``, ...).

Input Format
------------

The predictions should be in JSONL format:

.. code-block:: json

    {"question_id": 1, "completion": "```sql\nSELECT 1;\n```"}
    {"question_id": 2, "completion": "Some reasoning... ```sql\nSELECT 2;\n```"}

``question_id`` is the integer id of the benchmark question and ``completion`` the raw
model output; this is the format written by the ``inference_*`` functions.

Output Metrics
--------------

The function returns a dictionary with the following keys:

- total – Total queries evaluated
- matches – Queries where predicted SQL results match gold results
- pred_none – Queries where the model returned NULL or no result
- gold_none – Queries where the reference result was NULL or no result
- sql_errors – Invalid SQL or execution errors (for 2.0 also completions without SQL and queries over the time limit)
- exact_string_matches – Predictions whose SQL equals the gold SQL up to whitespace
- accuracy – Overall execution accuracy
- category_accuracy – (LLMSQL 2.0 only) total, matches and accuracy per category
- model_name – Name of the evaluated model (if provided)
- version – LLMSQL benchmark version used for evaluation
- mismatches – List of mismatched queries with details
- timestamp – Evaluation timestamp
- input_mode – How results were provided ("jsonl_path" or "dict_list")

Report Saving
-------------

By default, a report is saved automatically as `evaluation_results_{uuid}.json` in the current directory.
It contains metrics, mismatches, timestamp, and input mode. You can override this path using `save_report`.

Leaderboard Format
------------------

Pass ``save_leaderboard_yaml`` to additionally save the results in the same format as the
``run.yaml`` files in the ``leaderboard/`` folder of the repository. The evaluation date,
``llmsql`` package version, benchmark version, OS name, Python version, execution accuracy,
number of samples and outputs path are filled automatically; all other fields are ``null``
unless provided via ``run_metadata``:

.. code-block:: python

    report = evaluate(
        "outputs.jsonl",
        model_name="Qwen/Qwen3-0.6B",
        save_leaderboard_yaml="run.yaml",
        run_metadata={
            "type": "open-source",
            "model": {"dtype": "bfloat16", "parameter_count": "0.6B"},
            "inference": {
                "backend": "vllm",
                "arguments": {"num_fewshots": 5, "temperature": 0.0},
            },
        },
    )

From the CLI (``--run-metadata`` accepts a YAML/JSON file with the same structure):

.. code-block:: bash

    llmsql evaluate --outputs outputs.jsonl \
        --model-name Qwen/Qwen3-0.6B \
        --save-leaderboard-yaml run.yaml \
        --run-metadata metadata.yaml

---

.. automodule:: llmsql.evaluation.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

---

.. automodule:: llmsql.utils.matching
   :members: extract_sql, norm, results_match


---

.. raw:: html

   <div style="text-align:center; margin-top:2rem; color:#666;">
     💬 Made with ❤️ by the LLMSQL Team
   </div>
