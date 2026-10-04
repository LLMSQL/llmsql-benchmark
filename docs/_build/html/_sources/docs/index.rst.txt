LLMSQL package Documentation
============================

.. raw:: html

   <a href="../index.html" class="sidebar-button">
     ← Back to main page
   </a>


Welcome to the LLMSQL documentation!
This guide covers everything you need to use the project, from running inference
to evaluating Text-to-SQL models.

---

Getting Started
===============

Installation
------------

Install LLMSQL from source:

.. code-block:: bash

    pip install llmsql

Example: Running your first evaluation (with transformers backend)
--------------------------------------------------------------------

.. code-block:: python

    from llmsql import inference_transformers

    results = inference_transformers(
        model_or_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        output_file="outputs/preds_transformers.jsonl",
        questions_path="data/questions.jsonl",
        tables_path="data/tables.jsonl",
        num_fewshots=5,
        batch_size=8,
        max_new_tokens=256,
        temperature=0.7,
        model_args={
            "torch_dtype": "bfloat16",
        },
        generate_kwargs={
            "do_sample": False,
        },
    )
    print(results)


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
