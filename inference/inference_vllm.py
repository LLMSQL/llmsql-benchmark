"""
Generate SQL predictions for a dataset of questions using vLLM.

Usage (example from project root):
    python3 inference/inference_vllm.py \
        --questions_file dataset/questions.jsonl \
        --tables_file dataset/tables.jsonl \
        --output_file outputs/my_model_preds.jsonl \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --shots 5 \
        --batch_size 16 \
        --max_new_tokens 256 \
        --do_sample false

The script will process ALL questions (starts fresh) and write JSONL with lines like:
    {"question_id": "<id>", "completion": "<model generated SQL>"}
"""

import argparse
import os
import random
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from utils.logging_config import log
from utils.utils import (
    choose_prompt_builder,
    load_jsonl,
    overwrite_jsonl,
    save_jsonl_lines,
)
from vllm import LLM, SamplingParams

load_dotenv()

# Helper types
Question = Dict[str, object]
Table = Dict[str, object]


def main(
    questions_file: str,
    tables_file: str,
    output_file: str,
    model_name: str,
    shots: int = 5,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    do_sample: bool = True,
    seed: int = 42,
    hf_token: str | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.95,
) -> None:
    """
    Generate SQL queries for all questions in `questions_file` using `model_name` with vLLM.

    Args:
        questions_file: path to JSONL with questions (each item contains "question", "table_id", "sql", and "question_id").
        tables_file: path to JSONL with tables metadata (with keys like id, header, types, rows).
        output_file: path to JSONL to write outputs; overwritten at start.
        model_name: HF model name / path used for vLLM LLM.
        shots: 0, 1 or 5 — selects prompt builder (build_prompt_0/1/5shot).
        batch_size: number of requests processed per vLLM generate call.
        max_new_tokens: max tokens to generate per sample.
        temperature: sampling temperature.
        do_sample: whether to use sampling (temperature > 0) or greedy decoding.
        seed: random seed for reproducibility.
        hf_token: optional HF token to pass to vLLM (defaults to env HF_TOKEN).
        tensor_parallel_size: number of GPUs to use for tensor parallelism.
        gpu_memory_utilization: fraction of GPU memory to use (0.0 to 1.0).
    """
    # seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    # load data
    log.info("Loading questions and tables...")
    questions = load_jsonl(questions_file)
    tables_list = load_jsonl(tables_file)
    tables = {t["table_id"]: t for t in tables_list}

    log.info(f"Questions loaded: {len(questions)}")
    log.info(f"Tables loaded: {len(tables)}")

    # prepare output file (start fresh)
    overwrite_jsonl(output_file)
    log.info(f"Output will be written to: {output_file} (overwritten)")

    # select prompt builder
    prompt_builder = choose_prompt_builder(shots)
    log.info(f"Using {shots}-shot prompt builder: {prompt_builder.__name__}")

    # Initialize vLLM
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    log.info(f"Initializing vLLM with model: {model_name}")

    # vLLM initialization arguments
    vllm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "seed": seed,
        "trust_remote_code": True,
    }

    try:
        llm = LLM(**vllm_kwargs)
        log.info("vLLM model loaded successfully")
    except Exception as e:
        log.error(f"Failed to load model with vLLM: {e}")
        log.info("Trying with reduced memory utilization...")
        vllm_kwargs["gpu_memory_utilization"] = 0.8
        llm = LLM(**vllm_kwargs)
        log.info("vLLM model loaded successfully with reduced memory")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, **({} if not hf_token else {"token": hf_token})
    )

    # Set up sampling parameters
    # If do_sample is False, set temperature to 0 for greedy decoding
    actual_temperature = temperature if do_sample else 0.0
    sampling_params = SamplingParams(
        temperature=actual_temperature,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    log.info(
        f"Sampling params: temperature={actual_temperature}, max_tokens={max_new_tokens}"
    )

    total = len(questions)
    saved = 0

    for batch_start in tqdm(range(0, total, batch_size), desc="Generating"):
        batch = questions[batch_start : batch_start + batch_size]

        prompts = []
        batch_questions = []

        for q in batch:
            tbl = tables[q["table_id"]]
            example_row = tbl["rows"][0]
            prompt_content = prompt_builder(
                q["question"], tbl["header"], tbl["types"], example_row
            )

            # Format as chat message and apply chat template
            try:
                # Try to use chat template if available
                if (
                    hasattr(tokenizer, "apply_chat_template")
                    and tokenizer.chat_template
                ):
                    formatted_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_content}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                else:
                    # Fallback to plain prompt
                    formatted_prompt = prompt_content
            except Exception as e:
                log.warning(f"Failed to apply chat template: {e}, using plain prompt")
                formatted_prompt = prompt_content

            prompts.append(formatted_prompt)
            batch_questions.append(q)

        # Generate completions
        try:
            outputs = llm.generate(prompts, sampling_params)
        except Exception as e:
            log.error(
                f"Generation failed for batch {batch_start // batch_size + 1}: {e}"
            )

        # Process outputs
        batch_results: List[Dict[str, str]] = []
        for q, output in zip(batch_questions, outputs):
            completion = output.outputs[0].text

            batch_results.append(
                {
                    "question_id": q["question_id"],
                    "completion": completion,
                }
            )

        # Save batch results (append)
        save_jsonl_lines(output_file, batch_results)
        saved += len(batch_results)
        log.info(
            f"Saved batch {batch_start // batch_size + 1} — total saved: {saved}/{total}"
        )

    log.info(f"Generation completed. Total saved: {saved}. Output file: {output_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate SQL predictions using vLLM.")
    p.add_argument(
        "--questions_file",
        type=str,
        default="dataset/questions.jsonl",
        help="Path to questions JSONL (question objects).",
    )
    p.add_argument(
        "--tables_file",
        type=str,
        default="dataset/tables.jsonl",
        help="Path to tables JSONL (table metadata).",
    )
    p.add_argument(
        "--output_file",
        required=True,
        type=str,
        help="Path to write outputs JSONL (will overwrite).",
    )
    p.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="Hugging Face model id / path.",
    )
    p.add_argument(
        "--shots",
        type=int,
        choices=[0, 1, 5],
        default=5,
        help="Select prompt: 0, 1 or 5 shots.",
    )
    p.add_argument("--batch_size", type=int, default=8, help="Generation batch size.")
    p.add_argument(
        "--max_new_tokens", type=int, default=256, help="Maximum tokens to generate."
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature to use for generation.",
    )
    p.add_argument(
        "--do_sample",
        type=bool,
        default=True,
        help="Whether to sample or use greedy decoding.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (optional).",
    )
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (0.0 to 1.0).",
    )
    args = p.parse_args()

    main(
        questions_file=args.questions_file,
        tables_file=args.tables_file,
        output_file=args.output_file,
        model_name=args.model_name,
        shots=args.shots,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        seed=args.seed,
        hf_token=args.hf_token,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
