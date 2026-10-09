LLMSQL package Documentation
============================

.. raw:: html

   <a href="../index.html" class="sidebar-button">
     ← Back to main page
   </a>


Welcome to the LLMSQL documentation!
This guide covers everything you need to use the project, from running inference
to evaluating Text-to-SQL models.

The default benchmark is **LLMSQL 2.0**: 2,000 hard, verified test questions over
Wikipedia tables, evaluated zero-shot with a lenient execution match (see
:doc:`usage`). LLMSQL 1.0 remains available with ``version="1.0"``.

---

Getting Started
===============

Installation
------------

Install LLMSQL:

.. code-block:: bash

    pip install llmsql

Example: Running your first evaluation (with transformers backend)
--------------------------------------------------------------------

.. code-block:: python

    from llmsql import evaluate, inference_transformers

    results = inference_transformers(
        model_or_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        version="2.0",  # zero-shot
        output_file="outputs/preds_transformers.jsonl",
        batch_size=8,
        max_new_tokens=4096,
        model_kwargs={
            "torch_dtype": "bfloat16",
        },
    )
    report = evaluate("outputs/preds_transformers.jsonl", version="2.0")
    print(report["accuracy"])


Full Documentation
------------------

.. toctree::
   :maxdepth: 1
   :caption: Contents

   usage
   inference
   evaluation


---

.. raw:: html

   <div style="text-align:center; margin-top:2rem; color:#666;">
     💬 Made with ❤️ by the LLMSQL Team
   </div>
