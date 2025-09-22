"""
Generate SQL predictions for a dataset of questions using a Hugging Face causal LM.

Usage (example from project root):
    python3 inference/inference.py \
        --questions_file dataset/questions.jsonl \
        --tables_file dataset/tables.jsonl \
        --output_file outputs/my_model_preds.jsonl \
        --model_name dataset/gpt_oss_output.jsonl \
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
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from utils.logging_config import log
from utils.utils import (
    choose_prompt_builder,
    load_jsonl,
    overwrite_jsonl,
    save_jsonl_lines,
)

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
    temperature: float=1.0,
    do_sample: bool=True,
    seed: int = 42,
    hf_token: str | None = None,
) -> None:
    """
    Generate SQL queries for all questions in `questions_file` using `model_name`.

    Args:
        questions_file: path to JSONL with questions (each item contains "question", "table_id", "sql", and "question_id").
        tables_file: path to JSONL with tables metadata (with keys like id, header, types, rows).
        output_file: path to JSONL to write outputs; overwritten at start.
        model_name: HF model name / path used for AutoTokenizer & AutoModelForCausalLM.
        shots: 0, 1 or 5 — selects prompt builder (build_prompt_0/1/5shot).
        batch_size: number of requests processed per model.generate call.
        max_new_tokens: max tokens to generate per sample.
        seed: random seed for reproducibility.
        hf_token: optional HF token to pass to from_pretrained (defaults to env HF_TOKEN).
    """
    # seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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

    # load model & tokenizer
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading tokenizer and model on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, **({} if not hf_token else {"token": hf_token}), padding_side='left'
    )
    # Ensure pad token exists for batch padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        **({} if not hf_token else {"token": hf_token}),
    )
    model.eval()

    # batch generation loop
    total = len(questions)
    saved = 0
    for batch_start in tqdm(range(0, total, batch_size), desc="Generating"):
        batch = questions[batch_start : batch_start + batch_size]

        # build prompts for each example in the batch
        # The builder should accept (question, header, types, example_row) and return a string or chat-format.
        prompts = []
        for q in batch:
            tbl = tables[q["table_id"]]
            # example_row: we reuse the first row as example context if present
            example_row = tbl["rows"][0]
            p = prompt_builder(q["question"], tbl["header"], tbl["types"], example_row)
            prompts.append(p)

        # Tokenize
        # Using chat template api
        tokenized = tokenizer.apply_chat_template(
            [[{"role": "user", "content": p}] for p in prompts],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)

        # We'll compute input lengths to strip input from tokens after generation
        input_lengths = (tokenized["attention_mask"] != 0).sum(dim=1).tolist()
        gen_kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )

        decoded = [
            tokenizer.decode(out[input_len:], skip_special_tokens=False)
            for out, input_len in zip(outputs, input_lengths)
        ]



        # here we keep full decoding as "completion".
        batch_results: List[Dict[str, str]] = []
        for q, completion in zip(batch, decoded):
            batch_results.append(
                {
                    "question_id": q.get("question_id", q.get("id", "")),
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
    p = argparse.ArgumentParser(
        description="Generate SQL predictions using a HF causal LM."
    )
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
        "--temperature", type=float, default=1.0, help="Temperature to use for generation."
    )
    p.add_argument(
        "--do_sample", type=bool, default=True, help="Whether to sample or use greedy decoding."
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (optional).",
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
    )
