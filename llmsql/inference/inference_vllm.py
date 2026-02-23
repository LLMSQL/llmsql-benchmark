"""
LLMSQL vLLM Inference Function
==============================

This module provides a single function `inference_vllm()` that performs
text-to-SQL generation using large language models via the vLLM backend.

Example
-------

.. code-block:: python

    from llmsql.inference import inference_vllm

    results = inference_vllm(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        version="2.0",
        output_file="outputs/predictions.jsonl",
        questions_path="data/questions.jsonl",
        tables_path="data/tables.jsonl",
        num_fewshots=5,
        batch_size=8,
        max_new_tokens=256,
        temperature=0.7,
        tensor_parallel_size=1,
        lora_path="path/to/lora"
    )

Notes
~~~~~

This function uses the vLLM backend. Outputs may differ from the Transformers
backend due to differences in implementation, batching, and numerical precision.

"""

from __future__ import annotations

import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from llmsql.config.config import (
    DEFAULT_LLMSQL_VERSION,
    DEFAULT_WORKDIR_PATH,
    get_repo_id,
)
from llmsql.loggers.logging_config import log
from llmsql.utils.inference_utils import _maybe_download, _setup_seed
from llmsql.utils.utils import (
    choose_prompt_builder,
    load_jsonl,
    overwrite_jsonl,
    save_jsonl_lines,
)

load_dotenv()


def inference_vllm(
    model_name: str,
    *,
    # === Model Loading Parameters ===
    trust_remote_code: bool = True,
    tensor_parallel_size: int = 1,
    hf_token: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    use_chat_template: bool = True,
    # === LoRA Parameters ===
    lora_config: dict[str, Any] | None = None,  # new optional dict
    # === Generation Parameters ===
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    do_sample: bool = True,
    sampling_kwargs: dict[str, Any] | None = None,
    # === Benchmark Parameters ===
    version: Literal["1.0", "2.0"] = DEFAULT_LLMSQL_VERSION,
    output_file: str = "llm_sql_predictions.jsonl",
    questions_path: str | None = None,
    tables_path: str | None = None,
    workdir_path: str = DEFAULT_WORKDIR_PATH,
    limit: int | float | None = None,
    num_fewshots: int = 5,
    batch_size: int = 8,
    seed: int = 42,
) -> list[dict[str, str]]:
    """
    Run SQL generation using vLLM.

    Args:
        model_name: Hugging Face model name or path.

        # Model Loading:
        trust_remote_code: Whether to trust remote code (default: True).
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1).
        hf_token: Hugging Face authentication token.
        llm_kwargs: Additional arguments for vllm.LLM().
                   Note: 'model', 'tokenizer', 'tensor_parallel_size',
                   'trust_remote_code' are handled separately and will
                   override values here.

        lora_config: Optional dict with LoRA parameters:
            - lora_path: Path to the pretrained LoRA adapter (required if enable_lora)
            - lora_name: Logical name for the LoRA adapter
            - lora_scale: Scaling factor for LoRA weights
            - max_lora_rank: Maximum LoRA rank supported by vLLM
             LoRA usage rules:
                - If `lora_config` is provided, `enable_lora` must be True in `llm_kwargs`.
                - If `enable_lora` is True, a valid `lora_config` must be provided.
                - Otherwise, an exception is raised to prevent inconsistent configuration.

        # Generation:
        max_new_tokens: Maximum tokens to generate per sequence.
        temperature: Sampling temperature (0.0 = greedy).
        do_sample: Whether to use sampling vs greedy decoding.
        sampling_kwargs: Additional arguments for vllm.SamplingParams().
                        Note: 'temperature', 'max_tokens' are handled
                        separately and will override values here.

        # Benchmark:
        version: LLMSQL version
        output_file: Path to write outputs (will be overwritten).
        questions_path: Path to questions.jsonl (auto-downloads if missing).
        tables_path: Path to tables.jsonl (auto-downloads if missing).
        workdir_path: Directory to store downloaded data.
        num_fewshots: Number of few-shot examples (0, 1, or 5).
        batch_size: Number of questions per generation batch.
        seed: Random seed for reproducibility.
        limit: Limit the number of questions to evaluate. If an integer, evaluates
               the first N samples. If a float between 0.0 and 1.0, evaluates the
               first X*100% of samples. If None, evaluates all samples (default).



    Returns:
        List of dicts containing `question_id` and generated `completion`.
    """
    # --- setup ---
    llm_kwargs = llm_kwargs or {}
    sampling_kwargs = sampling_kwargs or {}
    _setup_seed(seed=seed)

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    workdir = Path(workdir_path)
    workdir.mkdir(parents=True, exist_ok=True)

    # --- load input data ---
    log.info("Preparing questions and tables...")

    repo_id = get_repo_id(version)

    questions_path = _maybe_download(repo_id, "questions.jsonl", questions_path)
    tables_path = _maybe_download(repo_id, "tables.jsonl", tables_path)

    questions = load_jsonl(questions_path)
    tables_list = load_jsonl(tables_path)
    tables = {t["table_id"]: t for t in tables_list}

    # --- Apply limit ---
    if limit is not None:
        if isinstance(limit, float):
            if not (0.0 < limit <= 1.0):
                raise ValueError(
                    f"When a float, `limit` must be between 0.0 and 1.0, got {limit}."
                )
            limit = max(1, int(len(questions) * limit))
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(
                f"`limit` must be a positive integer or a float in (0.0, 1.0], got {limit!r}."
            )
        log.info(
            f"Limiting evaluation to first {limit} questions out of {len(questions)}"
        )
        questions = questions[:limit]

    # --- Validate LoRA usage ---
    enable_lora = llm_kwargs.get("enable_lora", False)
    if lora_config is not None and not enable_lora:
        raise ValueError(
            "LoRA config provided but `enable_lora` is not True in llm_kwargs."
        )
    if enable_lora and lora_config is None:
        raise ValueError("`enable_lora` is True but no `lora_config` was provided.")
    if lora_config is not None and not enable_lora:
        raise ValueError(
            "`lora_config` provided but `enable_lora` is not True in llm_kwargs."
        )

    # --- init model ---
    llm_init_args = {
        "model": model_name,
        "tokenizer": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
        **llm_kwargs,  # user overrides
    }

    log.info(f"Loading vLLM model '{model_name}' (tp={tensor_parallel_size})...")
    llm = LLM(**llm_init_args)

    # --- LoRA request ---
    lora_request = None
    if enable_lora and lora_config is not None:
        log.info(f"Loading LoRA adapter from {lora_config['lora_path']}")
        lora_request = LoRARequest(
            lora_name=lora_config["lora_name"],
            lora_path=lora_config["lora_path"],
            scaling=lora_config["lora_scale"],
        )

    tokenizer = llm.get_tokenizer()
    if use_chat_template:
        use_chat_template = getattr(tokenizer, "chat_template", None)  # type: ignore

    # --- prepare output file ---
    overwrite_jsonl(output_file)
    log.info(f"Output will be written to {output_file}")

    # --- prompt builder and sampling params ---
    prompt_builder = choose_prompt_builder(num_fewshots)

    effective_temperature = 0.0 if not do_sample else temperature

    sampling_params_args = {
        "temperature": effective_temperature,
        "max_tokens": max_new_tokens,
        **sampling_kwargs,
    }

    sampling_params = SamplingParams(**sampling_params_args)

    # --- main inference loop ---
    all_results: list[dict[str, str]] = []
    total = len(questions)

    for batch_start in tqdm(range(0, total, batch_size), desc="Generating"):
        batch = questions[batch_start : batch_start + batch_size]

        prompts = []
        for q in batch:
            tbl = tables[q["table_id"]]
            example_row = tbl["rows"][0] if tbl["rows"] else []

            raw_text = prompt_builder(
                q["question"], tbl["header"], tbl["types"], example_row
            )

            if use_chat_template:
                messages = [{"role": "user", "content": raw_text}]

                final_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                final_prompt = raw_text

            prompts.append(final_prompt)

        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
        )

        batch_results: list[dict[str, str]] = []
        for q, out in zip(batch, outputs, strict=False):
            text = out.outputs[0].text
            batch_results.append(
                {
                    "question_id": q.get("question_id", q.get("id", "")),
                    "completion": text,
                }
            )

        save_jsonl_lines(output_file, batch_results)
        all_results.extend(batch_results)
        log.info(
            f"Saved batch {batch_start // batch_size + 1}: {len(all_results)}/{total}"
        )

    log.info(f"Generation completed. {len(all_results)} results saved to {output_file}")
    return all_results
