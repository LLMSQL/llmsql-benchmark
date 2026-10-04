Usage Overview
==============

LLMSQL package provides two primary components:

1. **Inference** – running LLM models to generate SQL queries.
2. **Evaluation** – computing accuracy and task-level performance.

Typical workflow
----------------

1. Run inference on dataset examples (Transformers or vLLM)
2. Pass predictions to `Evaluator`
3. Inspect evaluation metrics

Basic Example
-------------

Using transformers backend.

.. code-block:: python

    from llmsql import inference_transformers
    from llmsql import LLMSQLEvaluator

    # Run inference (will take some time)
    results = inference_transformers(
        model_or_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        output_file="outputs/preds_transformers.jsonl",
        questions_path="data/questions.jsonl",
        tables_path="data/tables.jsonl",
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
    evaluator = LLMSQLEvaluator()
    report = evaluator.evaluate(outputs_path="outputs/preds_transformers.jsonl")
    print(report)

Using vllm backend.

.. code-block:: python

    from llmsql import inference_vllm
    from llmsql import LLMSQLEvaluator

    # Run inference (will take some time)
    results = inference_vllm(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        output_file="outputs/preds_vllm.jsonl",
        questions_path="data/questions.jsonl",
        tables_path="data/tables.jsonl",
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
    evaluator = LLMSQLEvaluator()
    report = evaluator.evaluate(outputs_path="outputs/preds_transformers.jsonl")
    print(report)


---

.. raw:: html

   <div style="text-align:center; margin-top:2rem; color:#666;">
     💬 Made with ❤️ by the LLMSQL Team
   </div>
