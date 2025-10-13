"""
LLMSQL Fine-Tuning Script

This module provides CLI and programmatic utilities for fine-tuning
causal language models on the LLMSQL benchmark dataset. It supports
loading training/validation data, prompt construction, dataset building,
and training using Hugging Face TRL's `SFTTrainer`.

Example usage (CLI):

    llmsql finetune --config_file examples/example_finetune_args.yaml

Notes:
  - Configuration can be passed via command-line arguments or a YAML
    config file (`--config_file`). Example configuration, can be found in `examples/example_finetune_args.yaml`
  - If training/validation/tables files are not found locally, they are
    automatically downloaded from the Hugging Face Hub.
  - Training progress, metrics, and reports are logged via the
    configured logger and optionally integrated with Weights & Biases.
"""

import argparse
from collections.abc import Callable
import os
from pathlib import Path
import shutil
from typing import Any

from datasets import Dataset
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
import yaml

from llmsql.loggers.logging_config import log
from llmsql.utils.utils import choose_prompt_builder, load_jsonl


def parse_args_and_config() -> argparse.Namespace:
    """Parse CLI args and optionally merge with YAML config."""
    p = argparse.ArgumentParser(
        description="Fine-tune a causal LM on Text-to-SQL benchmark."
    )
    # Basic CLI args
    p.add_argument("--config_file", type=str, help="Path to YAML config file.")
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--val_file", type=str, default=None)
    p.add_argument("--tables_file", type=str, default=None)
    p.add_argument("--model_name_or_path", type=str)
    p.add_argument("--output_dir", type=str)
    p.add_argument("--shots", type=int, choices=[0, 1, 5], default=None)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=None)
    p.add_argument("--per_device_eval_batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    p.add_argument("--no_eval", type=bool, default=None)
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)

    args = vars(p.parse_args())

    # Load YAML config
    config = {}
    if args.get("config_file"):
        with open(args["config_file"]) as f:
            config = yaml.safe_load(f)

    def flatten(d: Any, parent_key: str = "", sep: str = "_") -> dict[str, Any]:
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    flat_config = flatten(config)

    for k, v in flat_config.items():
        if args.get(k) is None:
            args[k] = v

    return argparse.Namespace(**args)


def build_dataset(file_path: str, tables: dict, prompt_builder: Callable) -> Dataset:
    """Convert JSONL file to HF dataset samples."""
    questions = load_jsonl(file_path)
    samples = []
    for q in questions:
        tbl = tables[q["table_id"]]
        example_row = tbl["rows"][0]
        prompt = prompt_builder(q["question"], tbl["header"], tbl["types"], example_row)
        samples.append({"prompt": prompt, "completion": q["sql"]})
    return Dataset.from_list(samples)


def _download_file(filename: str, repo_id: str, workdir_path: str) -> str:
    """
    Download the file from Hugging Face Hub, using the global cache.
    If workdir_path is given, copy the cached file there.
    """
    # Step 1: always download into global HF cache
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )
    local_path = os.path.join(workdir_path, filename)
    shutil.copy(cached_path, local_path)
    return local_path


def main(
    model_name_or_path: str,
    output_dir: str,
    train_file: str | None = None,
    val_file: str | None = None,
    tables_file: str | None = None,
    shots: int = 5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    learning_rate: float = 5e-5,
    save_steps: int = 500,
    logging_steps: int = 100,
    seed: int = 42,
    hf_token: str | None = None,
    max_length: int = 32768,
    no_eval: bool = False,
    eval_steps: int = 100,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_key: str | None = None,
    wandb_offline: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set WandB env if specified
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = wandb_run_name
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    if (
        train_file is None
        and not (Path(output_dir) / Path("train_questions.jsonl")).is_file()
    ):
        train_file = _download_file(
            "train_questions.jsonl", "llmsql-bench/llmsql-benchmark", output_dir
        )
    elif train_file is None:
        train_file = f"{output_dir}/train_questions.jsonl"

    if (
        val_file is None
        and not (Path(output_dir) / Path("val_questions.jsonl")).is_file()
    ):
        val_file = _download_file(
            "val_questions.jsonl", "llmsql-bench/llmsql-benchmark", output_dir
        )
    elif val_file is None:
        val_file = f"{output_dir}/val_questions.jsonl"

    if tables_file is None and not (Path(output_dir) / Path("tables.jsonl")).is_file():
        tables_file = _download_file(
            "tables.jsonl", "llmsql-bench/llmsql-benchmark", output_dir
        )
    elif tables_file is None:
        tables_file = f"{output_dir}/tables.jsonl"

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

    hf_token = hf_token or os.environ.get("HF_TOKEN")

    params = {
        "model_name_or_path": model_name_or_path,
        "output_dir": output_dir,
        "train_file": train_file,
        "val_file": val_file,
        "tables_file": tables_file,
        "shots": shots,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "learning_rate": learning_rate,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "seed": seed,
        "hf_token": hf_token,
        "max_length": max_length,
        "no_eval": no_eval,
        "eval_steps": eval_steps,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_offline": wandb_offline,
    }

    log.info("===== Finetuning parameters =====")
    for k, v in params.items():
        log.info(f"{k}: {v}")
    log.info("================================")

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
        eval_strategy="steps" if not no_eval else "no",
        eval_steps=eval_steps,
        save_total_limit=2,
        bf16=True,
        logging_dir=os.path.join(output_dir, "logs"),
        seed=seed,
        completion_only_loss=True,
        report_to="wandb" if wandb_project else "none",
        max_length=max_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None if no_eval else val_dataset,
        args=training_args,
    )
    log.info("Starting training...")
    trainer.train()
    log.info("Training complete. Saving final model...")
    trainer.save_model(f"{output_dir}/final_model")
    log.info(f"Model saved at {output_dir}/final_model")


def run_cli() -> None:
    args = parse_args_and_config()
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
        max_length=args.max_length,
        eval_steps=args.eval_steps,
    )


if __name__ == "__main__":
    args = parse_args_and_config()
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
        max_length=args.max_length,
        eval_steps=args.eval_steps,
    )
