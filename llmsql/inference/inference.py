"""
LLMSQL vLLM Inference Module

This module defines the `LLMSQLVLLMInference` class, which provides
utilities for running text-to-SQL generation using large language models
via the [vLLM](https://github.com/vllm-project/vllm) inference backend.

Example usage:

    from llmsql import LLMSQLVLLMInference

    inference = LLMSQLVLLMInference(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        tensor_parallel_size=1,
    )

    results = inference.generate(
        output_file="outputs/predictions.jsonl",
        questions_path="data/questions.jsonl",
        tables_path="data/tables.jsonl",
        shots=5,
        batch_size=8,
        max_new_tokens=256,
        temperature=0.7,
    )

Notes:
  - If `questions.jsonl` or `tables.jsonl` are not provided, they will be
    automatically downloaded from the Hugging Face Hub dataset
    `llmsql-bench/llmsql-benchmark`.
  - The `generate` method overwrites the given output file at the start.
  - Logging provides progress updates, including per-batch saves and totals.
"""

import os
from pathlib import Path
import random
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from llmsql.loggers.logging_config import log
from llmsql.utils.utils import (
    choose_prompt_builder,
    load_jsonl,
    overwrite_jsonl,
    save_jsonl_lines,
)

load_dotenv()

Question = dict[str, object]
Table = dict[str, object]


class LLMSQLVLLMInference:
    """
    Run SQL generation using Hugging Face causal models with vLLM backend.
    """

    def __init__(
        self,
        model_name: str,
        hf_token: str | None = None,
        tensor_parallel_size: int = 1,
        seed: int = 42,
        workdir_path: str = "llmsql_workdir",
        **llm_kwargs: Any,
    ):
        """
        Initialize vLLM model for SQL inference.

        Args:
            model_name: Hugging Face model name / path.
            hf_token: Hugging Face auth token.
            tensor_parallel_size: Degree of tensor parallelism (for multi-GPU).
            seed: Random seed.
            **llm_kwargs: Extra kwargs forwarded to vllm.LLM().
        """
        # set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model_name = model_name
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        self.workdir_path = Path(workdir_path)
        self.workdir_path.mkdir(parents=True, exist_ok=True)
        self.repo_id = "llmsql-bench/llmsql-benchmark"

        if "device" not in llm_kwargs:
            llm_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = llm_kwargs["device"]

        log.info(
            f"Loading vLLM model {model_name} with tensor_parallel_size={tensor_parallel_size}..."
        )
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            tokenizer=model_name,
            **llm_kwargs,
        )

    def _download_file(self, filename: str) -> str:
        """
        Download the official SQLite DB from Hugging Face Hub.
        Will be downloaded to the workdir specified in init.

        Returns:
            str: Local path to the downloaded DB file.
        """
        file_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=self.workdir_path,
        )
        log.info(f"File saved at: {file_path}")
        assert isinstance(
            file_path, str
        ), f"file path to the {filename} is not string. File path: {file_path}, type: {type(file_path)}"
        return file_path

    def generate(
        self,
        output_file: str,
        questions_path: str | None = None,
        tables_path: str | None = None,
        shots: int = 5,
        batch_size: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
        **sampling_kwargs: Any,
    ) -> list[dict[str, str]]:
        """
        Generate SQL queries for all benchmark questions.

        Args:
            questions_path (str): Path to JSONL with questions.
            tables_file (str): Path to JSONL with tables.
            output_file (str): Path to write outputs (overwritten at start).
            shots (int): 0, 1, or 5 — prompt builder choice.
            batch_size (int): Number of questions per batch.
            max_new_tokens (int): Max tokens per generation.
            temperature (float): Sampling temperature.
            do_sample (bool): Whether to sample (True) or use greedy decoding.

        Returns:
            List[Dict[str, str]]: All results as dicts.

        Example:
            ```
            from llmsql import LLMSQLVLLMInference

            inference = LLMSQLVLLMInference(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                tensor_parallel_size=1,
            )

            results = inference.generate(
                output_file="outputs/predictions.jsonl",
                questions_path="data/questions.jsonl",
                tables_path="data/tables.jsonl",
                shots=5,
                batch_size=8,
                max_new_tokens=256,
                temperature=0.7,
            )
            ```
        """
        log.info("Loading questions and tables...")
        if (
            questions_path is None
            and not (self.workdir_path / Path("questions.jsonl")).is_file()
        ):
            questions_path = self._download_file("questions.jsonl")
        elif questions_path is None:
            questions_path = f"{self.workdir_path}/questions.jsonl"

        if (
            tables_path is None
            and not (self.workdir_path / Path("tables.jsonl")).is_file()
        ):
            tables_path = self._download_file("tables.jsonl")
        elif tables_path is None:
            tables_path = f"{self.workdir_path}/tables.jsonl"

        questions = load_jsonl(questions_path)
        tables_list = load_jsonl(tables_path)
        tables = {t["table_id"]: t for t in tables_list}

        overwrite_jsonl(output_file)
        log.info(f"Output will be written to: {output_file} (overwritten)")

        prompt_builder = choose_prompt_builder(shots)
        log.info(f"Using {shots}-shot prompt builder: {prompt_builder.__name__}")

        all_results: list[dict[str, str]] = []
        total = len(questions)

        temperature = 0.0 if not do_sample else temperature

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            **sampling_kwargs,
        )

        for batch_start in tqdm(range(0, total, batch_size), desc="Generating"):
            batch = questions[batch_start : batch_start + batch_size]

            prompts = []
            for q in batch:
                tbl = tables[q["table_id"]]
                example_row = tbl["rows"][0] if tbl["rows"] else []
                p = prompt_builder(
                    q["question"], tbl["header"], tbl["types"], example_row
                )
                prompts.append(p)

            # Generate with vLLM
            outputs = self.llm.generate(prompts, sampling_params)

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
                f"Saved batch {batch_start // batch_size + 1} — total saved: {len(all_results)}/{total}"
            )

        log.info(
            f"Generation completed. Total saved: {len(all_results)}. Output file: {output_file}"
        )
        return all_results
