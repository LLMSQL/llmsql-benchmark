Usage Overview
==============

LLMSQL package provides two primary components:

1. **Inference** – running LLM models to generate SQL queries.
2. **Evaluation** – computing accuracy and task-level performance.

Benchmark versions
------------------

Every inference function and ``evaluate()`` take a ``version`` argument:

* ``"2.0"`` (default) — **LLMSQL 2.0**, a test-only benchmark of **2,000** questions over
  942 Wikipedia tables, in three categories: ``lookup`` (420, WikiSQL-style lookups),
  ``convention`` (1,158, sports game logs whose cells follow a convention that must be
  read from the data; some questions show distractor tables from the same Wikipedia page)
  and ``text_quantity`` (422, dates stored as text). Every reference answer was verified
  independently (second implementation, hand audit, or both).
* ``"1.0"`` — **LLMSQL 1.0**, the cleaned WikiSQL with train/validation/test splits.

The earlier, unpublished LLMSQL 2.0 draft (LLMSQL 1.0 with corrected aggregation
operators) is available at the Hugging Face tag ``legacy-2.0-aggregation-fix`` of
``llmsql-bench/llmsql-2.0``.

LLMSQL 2.0 protocol
~~~~~~~~~~~~~~~~~~~

1. **Prompt** (zero-shot only): the ``CREATE TABLE`` schema with the real table name, the
   Wikipedia page/section and the first 3 rows of every table in the question's ``tables``
   field (target plus distractors), then the question; the model must answer with a single
   SQLite query in a fenced ``sql`` code block. The prompt is identical to the ``prompt`` field
   of the dataset.
2. **SQL extraction**: the last fenced ``sql`` code block of the completion (fallback: first
   ``WITH``/``SELECT`` statement), executed on the benchmark SQLite database.
3. **Lenient execution match** with the verified answer: insensitive to row order and
   duplicates, tolerant to number formatting (thousands separators, currency signs, units,
   rounding to 2 decimals) and to a trailing count in parentheses, and accepting extra
   columns if one of them equals the single answer column.
4. **Metric**: execution accuracy over the 2,000 questions, also per category.

``num_fewshots`` defaults to 0 for 2.0 (5 for 1.0); a non-zero value for 2.0 raises
``ValueError``. ``max_new_tokens`` defaults to 4096 for 2.0 (256 for 1.0). The reported
results for reasoning models were obtained with up to 16k new tokens.

Results (zero-shot, vLLM, a single run): gpt-oss-120b solves **21.9%** with medium
reasoning effort and **39.0%** with high reasoning effort. The questions were selected
from a larger pool of verified candidates as those that gpt-oss-120b (medium effort)
failed in earlier runs; the numbers come from fresh runs not used for selection.

Quickstart (LLMSQL 2.0)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from llmsql import inference_vllm, evaluate

    inference_vllm(
        "openai/gpt-oss-20b",
        version="2.0",
        output_file="outputs.jsonl",
        max_new_tokens=16384,
    )
    report = evaluate("outputs.jsonl", version="2.0")
    print(report["accuracy"], report["category_accuracy"])

Or with the CLI:

.. code-block:: bash

    llmsql inference vllm --model-name openai/gpt-oss-20b --version 2.0 \
        --output-file outputs.jsonl --max-new-tokens 16384
    llmsql evaluate --outputs outputs.jsonl --version 2.0

Typical workflow
----------------

1. Run inference on dataset examples (Transformers or vLLM)
2. Pass predictions to `evaluate()`
3. Inspect evaluation metrics

Basic Example (LLMSQL 1.0, 5-shot)
----------------------------------

Using transformers backend.

.. code-block:: python

    from llmsql import inference_transformers
    from llmsql import evaluate

    # Run inference (will take some time)
    results = inference_transformers(
        model_or_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        version="1.0",
        output_file="outputs/preds_transformers.jsonl",
        workdir_path="./benchmark-cache",
        num_fewshots=5,
        batch_size=8,
        max_new_tokens=256,
        temperature=0.7,
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16",
        },
        generation_kwargs={
            "do_sample": False,
        },
    )

    # Evaluate the results
    report = evaluate(outputs="outputs/preds_transformers.jsonl", version="1.0")
    print(report)

Using vllm backend.

.. code-block:: python

    from llmsql import inference_vllm
    from llmsql import evaluate

    # Run inference (will take some time)
    results = inference_vllm(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        version="1.0",
        output_file="outputs/preds_vllm.jsonl",
        workdir_path="./benchmark-cache",
        num_fewshots=5,
        batch_size=8,
        max_new_tokens=256,
        do_sample=False,
        llm_kwargs={
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
        },
    )

    # Evaluate the results
    report = evaluate(outputs="outputs/preds_vllm.jsonl", version="1.0")
    print(report)


Using OpenAI-compatible API (LLMSQL 2.0, zero-shot).

.. code-block:: python

    from llmsql import inference_api, evaluate
    from dotenv import load_dotenv
    import os
    load_dotenv()

    # Run inference (will take some time)
    results = inference_api(
        model_name="gpt-5-mini",
        base_url="https://api.openai.com/v1/",
        api_key=os.environ["OPENAI_API_KEY"],
        api_kwargs={
            "response_format": {
                    "type": "text"
                },
                "verbosity": "medium",
                "reasoning_effort": "medium",
                "store": False
        },
        requests_per_minute=100,
        output_file="test_output_api.jsonl",
        limit=50,
        seed=42,
        version="2.0"
    )

    # Evaluate the results
    report = evaluate(outputs="test_output_api.jsonl", version="2.0")
    print(report)

---

.. raw:: html

   <div style="text-align:center; margin-top:2rem; color:#666;">
     💬 Made with ❤️ by the LLMSQL Team
   </div>
