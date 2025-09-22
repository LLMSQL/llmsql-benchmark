"""
Fine-tune a causal LM on Text-to-SQL benchmark.

This script uses Hugging Face TRL's `SFTTrainer` to train a model
where the input is a prompt (built from question + table context)
and the output is the ground-truth SQL query.

Usage (example from project root, tested for 2xH100(80GB)):
    python3 finetune/finetune.py \
        --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
        --output_dir outputs/finetuned-llama \
        --num_train_epochs 2 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 
"""

import argparse
import os
from typing import Dict

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from utils.logging_config import log
from utils.utils import choose_prompt_builder, load_jsonl


def build_dataset(file_path: str, tables: Dict, prompt_builder) -> Dataset:
    """Convert JSONL file to HF dataset samples."""
    questions = load_jsonl(file_path)
    samples = []
    for q in questions:
        tbl = tables[q["table_id"]]
        example_row = tbl["rows"][0]
        prompt = prompt_builder(q["question"], tbl["header"], tbl["types"], example_row)
        samples.append({"prompt": prompt, "completion": q["sql"]})
    return Dataset.from_list(samples)


def main(
    train_file: str,
    val_file: str,
    tables_file: str,
    model_name_or_path: str,
    output_dir: str,
    shots: int = 5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    learning_rate: float = 5e-5,
    save_steps: int = 500,
    logging_steps: int = 100,
    seed: int = 42,
    hf_token: str | None = None,
):
    # load tables
    tables_list = load_jsonl(tables_file)
    tables = {t["table_id"]: t for t in tables_list}

    # select prompt builder
    prompt_builder = choose_prompt_builder(shots)
    log.info(f"Using {shots}-shot prompt builder: {prompt_builder.__name__}")

    # build datasets
    train_dataset = build_dataset(train_file, tables, prompt_builder)
    val_dataset = build_dataset(val_file, tables, prompt_builder)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto" if device == "cuda" else None,
        token=hf_token,
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        save_steps=save_steps,
        save_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=0.1,
        save_total_limit=2,
        bf16=True,
        logging_dir=os.path.join(output_dir, "logs"),
        seed=seed,
        completion_only_loss=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    log.info("Starting training...")
    trainer.train()
    log.info("Training complete. Saving final model...")
    trainer.save_model(output_dir)
    log.info(f"Model saved at {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fine-tune a causal LM on Text-to-SQL benchmark."
    )
    p.add_argument("--train_file", type=str, default="dataset/train_questions.jsonl")
    p.add_argument("--val_file", type=str, default="dataset/val_questions.jsonl")
    p.add_argument("--tables_file", type=str, default="dataset/tables.jsonl")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--shots", type=int, choices=[0, 1, 5], default=5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--save_steps", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    main(
        train_file=args.train_file,
        val_file=args.val_file,
        tables_file=args.tables_file,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        shots=args.shots,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        hf_token=args.hf_token,
    )
