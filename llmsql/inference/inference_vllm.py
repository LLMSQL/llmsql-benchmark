"""
LLMSQL vLLM Inference Function

This module provides a single function `inference_vllm()` that performs
text-to-SQL generation using large language models via the vLLM backend.

Example:

    from llmsql.inference import inference_vllm

    results = inference_vllm(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        output_file="outputs/predictions.jsonl",
        questions_path="data/questions.jsonl",
        tables_path="data/tables.jsonl",
        num_fewshots=5,
        batch_size=8,
        max_new_tokens=256,
        temperature=0.7,
        tensor_parallel_size=1,
    )
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm
from vllm import LLM, SamplingParams

from llmsql.config.config import DEFAULT_WORKDIR_PATH
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
    output_file: str,
    questions_path: str | None = None,
    tables_path: str | None = None,
    hf_token: str | None = None,
    tensor_parallel_size: int = 1,
    seed: int = 42,
    workdir_path: str = DEFAULT_WORKDIR_PATH,
    num_fewshots: int = 5,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    do_sample: bool = True,
    llm_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Run SQL generation using vLLM.

    Args:
        model_name: Hugging Face model name or path.
        output_file: Path to write outputs (will be overwritten).
        questions_path: Path to questions.jsonl (optional, auto-download if missing).
        tables_path: Path to tables.jsonl (optional, auto-download if missing).
        hf_token: Hugging Face auth token.
        tensor_parallel_size: Degree of tensor parallelism (for multi-GPU).
        seed: Random seed.
        workdir_path: Directory to store any downloaded data.
        num_fewshots: Number of examples per prompt (0, 1, or 5).
        batch_size: Number of questions per generation batch.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        do_sample: Whether to sample or use greedy decoding.
        **llm_kwargs: Extra kwargs forwarded to vllm.LLM().

    Returns:
        List of dicts containing `question_id` and generated `completion`.
    """
    # --- setup ---
    _setup_seed(seed=seed)

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    workdir = Path(workdir_path)
    workdir.mkdir(parents=True, exist_ok=True)

    # --- load input data ---
    log.info("Preparing questions and tables...")
    questions_path = _maybe_download("questions.jsonl", questions_path)
    tables_path = _maybe_download("tables.jsonl", tables_path)
    questions = load_jsonl(questions_path)
    tables_list = load_jsonl(tables_path)
    tables = {t["table_id"]: t for t in tables_list}

    # --- init model ---
    llm_kwargs = llm_kwargs or {}
    if "tensor_parallel_size" in llm_kwargs:
        tensor_parallel_size = llm_kwargs.pop("tensor_parallel_size")

    log.info(f"Loading vLLM model '{model_name}' (tp={tensor_parallel_size})...")

    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        **llm_kwargs,
    )

    # --- prepare output file ---
    overwrite_jsonl(output_file)
    log.info(f"Output will be written to {output_file}")

    # --- prompt builder and sampling params ---
    prompt_builder = choose_prompt_builder(num_fewshots)
    temperature = 0.0 if not do_sample else temperature
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    # --- main inference loop ---
    all_results: list[dict[str, str]] = []
    total = len(questions)

    for batch_start in tqdm(range(0, total, batch_size), desc="Generating"):
        batch = questions[batch_start : batch_start + batch_size]

        prompts = []
        for q in batch:
            tbl = tables[q["table_id"]]
            example_row = tbl["rows"][0] if tbl["rows"] else []
            prompts.append(
                prompt_builder(q["question"], tbl["header"], tbl["types"], example_row)
            )

        outputs = llm.generate(prompts, sampling_params)

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
