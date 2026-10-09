Inference API Reference
=======================

All inference functions take ``version`` (``"2.0"`` by default, or ``"1.0"``).
LLMSQL 2.0 is zero-shot: ``num_fewshots`` defaults to 0 for 2.0 (5 for 1.0) and a
non-zero value for 2.0 raises ``ValueError``. ``max_new_tokens`` defaults to 4096 for
2.0 (256 for 1.0); the reported LLMSQL 2.0 results of reasoning models were obtained
with up to 16k new tokens.

Prompts
-------

For LLMSQL 2.0 questions the prompt is built by
:func:`llmsql.prompts.prompts.build_prompt_v2`: the schema, Wikipedia page/section and
first 3 rows of every table in the question's ``tables`` field (target plus distractor
tables), then the question. It is identical to the dataset's ``prompt`` field.

.. autofunction:: llmsql.prompts.prompts.build_prompt_v2

.. autofunction:: llmsql.prompts.prompts.render_table_schema_v2

---

.. automodule:: llmsql.inference.inference_transformers
   :members:
   :undoc-members:

---

.. automodule:: llmsql.inference.inference_vllm
   :members:
   :undoc-members:


---

.. automodule:: llmsql.inference.inference_api
   :members:
   :undoc-members:

---


.. automodule:: llmsql.inference.inference_function
   :members:
   :undoc-members:

---


.. raw:: html

   <div style="text-align:center; margin-top:2rem; color:#666;">
     💬 Made with ❤️ by the LLMSQL Team
   </div>
