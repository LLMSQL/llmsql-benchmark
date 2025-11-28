"""
LLMSQL Transformers Inference Function
======================================

This module provides a single function `inference_transformers()` that performs
text-to-SQL generation using large language models via the Transformers backend.

Example
-------

.. code-block:: python

    from llmsql.inference import inference_transformers

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

Notes
~~~~~

This function uses the HuggingFace Transformers backend and may produce
slightly different outputs than the vLLM backend even with the same inputs
due to differences in implementation and numerical precision.

"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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

Question = dict[str, Any]
Table = dict[str, Any]


@torch.inference_mode()  # type: ignore
def inference_transformers(
    model_or_model_name_or_path: str | AutoModelForCausalLM,
    tokenizer_or_name: str | Any | None = None,
    *,
    chat_template: str | None = None,
    model_args: dict[str, Any] | None = None,
    hf_token: str | None = None,
    output_file: str = "outputs/predictions.jsonl",
    questions_path: str | None = None,
    tables_path: str | None = None,
    workdir_path: str = DEFAULT_WORKDIR_PATH,
    num_fewshots: int = 5,
    trust_remote_code: bool = True,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: bool = False,
    top_p: float = 1.0,
    top_k: int = 50,
    seed: int = 42,
    dtype: torch.dtype = torch.float16,
    device_map: str | dict[str, int] | None = "auto",
    generate_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Inference a causal model (Transformers) on the LLMSQL benchmark.

    Args:
        model_or_model_name_or_path: Model object or HF model name/path.
        tokenizer_or_name: Tokenizer object or HF tokenizer name/path.
        chat_template: Optional chat template to apply before tokenization.
        model_args: Optional kwargs passed to `from_pretrained` if needed.
        hf_token: Hugging Face token (optional).
        output_file: Output JSONL file for completions.
        questions_path: Path to benchmark questions JSONL.
        tables_path: Path to benchmark tables JSONL.
        workdir_path: Work directory (default: "llmsql_workdir").
        num_fewshots: 0, 1, or 5 — prompt builder choice.
        batch_size: Batch size for inference.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        do_sample: Whether to sample or use greedy decoding.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        seed: Random seed.
        dtype: Torch dtype (default: float16).
        device_map: Device map ("auto" for multi-GPU).
        **generate_kwargs: Extra arguments for `model.generate`.

    Returns:
        List[dict[str, str]]: Generated SQL results.
    """
    # --- Setup ---
    _setup_seed(seed=seed)

    workdir = Path(workdir_path)
    workdir.mkdir(parents=True, exist_ok=True)

    if generate_kwargs is None:
        generate_kwargs = {}

    model_args = model_args or {}
    if "torch_dtype" in model_args:
        dtype = model_args.pop("torch_dtype")
    if "trust_remote_code" in model_args:
        trust_remote_code = model_args.pop("trust_remote_code")

    # --- Load model ---
    if isinstance(model_or_model_name_or_path, str):
        model_args = model_args or {}
        log.info(f"Loading model from: {model_or_model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_or_model_name_or_path,
            torch_dtype=dtype,
            device_map=device_map,
            token=hf_token,
            trust_remote_code=trust_remote_code,
            **model_args,
        )
    else:
        model = model_or_model_name_or_path
        log.info(f"Using provided model object: {type(model)}")

    # --- Load tokenizer ---
    if tokenizer_or_name is None:
        if isinstance(model_or_model_name_or_path, str):
            tokenizer = AutoTokenizer.from_pretrained(
                model_or_model_name_or_path,
                token=hf_token,
                trust_remote_code=True,
            )
        else:
            raise ValueError("Tokenizer must be provided if model is passed directly.")
    elif isinstance(tokenizer_or_name, str):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_or_name,
            token=hf_token,
            trust_remote_code=True,
        )
    else:
        tokenizer = tokenizer_or_name

    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # --- Load necessary files ---
    questions_path = _maybe_download("questions.jsonl", questions_path)
    tables_path = _maybe_download("tables.jsonl", tables_path)

    questions = load_jsonl(questions_path)
    tables_list = load_jsonl(tables_path)
    tables = {t["table_id"]: t for t in tables_list}

    # --- Chat template setup ---
    use_chat_template = chat_template or getattr(tokenizer, "chat_template", None)
    if use_chat_template:
        log.info("Using chat template for prompt formatting.")

    # --- Output setup ---
    overwrite_jsonl(output_file)
    log.info(f"Writing results to {output_file}")

    prompt_builder = choose_prompt_builder(num_fewshots)
    log.info(f"Using {num_fewshots}-shot prompt builder: {prompt_builder.__name__}")

    results: list[dict[str, str]] = []
    total = len(questions)

    # --- Inference loop ---
    for start in tqdm(range(0, total, batch_size), desc="Generating"):
        batch = questions[start : start + batch_size]
        prompts = []

        for q in batch:
            tbl = tables[q["table_id"]]
            example_row = tbl["rows"][0] if tbl["rows"] else []
            text = prompt_builder(
                q["question"], tbl["header"], tbl["types"], example_row
            )

            # Apply chat template if available
            if use_chat_template:
                messages = [{"role": "user", "content": text}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=use_chat_template,
                )
            prompts.append(text)

        inputs = tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            **generate_kwargs,
        )

        input_lengths = [len(ids) for ids in inputs["input_ids"]]

        # Slice off the prompt part
        generated_texts = []
        for output, input_len in zip(outputs, input_lengths, strict=False):
            generated_part = output[input_len:]  # tokens generated after the prompt
            text = tokenizer.decode(generated_part, skip_special_tokens=True).strip()
            generated_texts.append(text)

        batch_results = []
        for q, text in zip(batch, generated_texts, strict=False):
            batch_results.append(
                {
                    "question_id": q.get("question_id", q.get("id", "")),
                    "completion": text.strip(),
                }
            )

        save_jsonl_lines(output_file, batch_results)
        results.extend(batch_results)
        log.info(f"Saved batch {start // batch_size + 1}: {len(results)}/{total}")

    log.info(f"Generation complete — total: {len(results)}")
    return results
