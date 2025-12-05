# LLMSQL Inference

LLMSQL provides two inference backends for **Text-to-SQL generation** with large language models:

* 🧠 **Transformers** — runs inference using the standard Hugging Face `transformers` pipeline.
* ⚡ **vLLM** — runs inference using the high-performance [vLLM](https://github.com/vllm-project/vllm) backend.

Both backends load benchmark questions and table schemas, build prompts (with few-shot examples), and generate SQL queries in parallel batches.

---

## Installation

Install the base package:

```bash
pip install llmsql
```

To enable the vLLM backend:

```bash
pip install llmsql[vllm]
```

---

## Quick Start

### ✅ Option 1 — Using the **Transformers** backend

```python
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
```

---

### ⚡ Option 2 — Using the **vLLM** backend

```python
from llmsql import inference_vllm

results = inference_vllm(
    model_name="EleutherAI/pythia-14m",
    output_file="test_output.jsonl",
    batch_size=5000,
    do_sample=False,
)
```

---

## Command-Line Interface (CLI)

You can also run inference directly from the command line:

```bash
llmsql inference --method vllm \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --output-file outputs/preds.jsonl \
    --batch-size 8 \
    --num_fewshots 5 \
    --temperature 0.0
```

Or use the Transformers backend:

```bash
llmsql inference --method transformers \
    --model-or-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
    --output-file outputs/preds.jsonl \
    --batch-size 8 \
    --temperature 0.0 \
```

👉 Run `llmsql inference --help` for more detailed examples and parameter options.

---

## API Reference

### `inference_transformers(...)`

Runs inference using the Hugging Face `transformers` backend.

**Parameters:**

#### Model Loading

| Argument                        | Type                  | Default       | Description                                                    |
| ------------------------------- | --------------------- | ------------- | -------------------------------------------------------------- |
| `model_or_model_name_or_path`   | `str \| AutoModelForCausalLM` | *required* | Model object, HuggingFace model name, or local path. |
| `tokenizer_or_name`             | `str \| Any \| None`  | `None`        | Tokenizer object, name, or None (infers from model).           |
| `trust_remote_code`             | `bool`                | `True`        | Whether to trust remote code when loading models.              |
| `dtype`                         | `torch.dtype`         | `torch.float16` | Model precision (e.g., `torch.float16`, `torch.bfloat16`).   |
| `device_map`                    | `str \| dict \| None` | `"auto"`      | Device placement strategy for multi-GPU.                       |
| `hf_token`                      | `str \| None`         | `None`        | Hugging Face authentication token.                             |
| `model_kwargs`                  | `dict \| None`        | `None`        | Additional kwargs for `AutoModelForCausalLM.from_pretrained()`. |
| `tokenizer_kwargs`              | `dict \| None`        | `None`        | Additional kwargs for `AutoTokenizer.from_pretrained()`.       |

#### Prompt & Chat

| Argument                        | Type           | Default | Description                                      |
| ------------------------------- | -------------- | ------- | ------------------------------------------------ |
| `chat_template`                 | `str \| None`  | `None`  | Optional chat template string to apply.          |

#### Generation

| Argument                        | Type           | Default | Description                                      |
| ------------------------------- | -------------- | ------- | ------------------------------------------------ |
| `max_new_tokens`                | `int`          | `256`   | Maximum tokens to generate per sequence.         |
| `temperature`                   | `float`        | `0.0`   | Sampling temperature (0.0 = greedy).             |
| `do_sample`                     | `bool`         | `False` | Whether to use sampling vs greedy decoding.      |
| `top_p`                         | `float`        | `1.0`   | Nucleus sampling parameter.                      |
| `top_k`                         | `int`          | `50`    | Top-k sampling parameter.                        |
| `generation_kwargs`             | `dict \| None` | `None`  | Additional kwargs for `model.generate()`.        |

#### Benchmark

| Argument                        | Type    | Default                   | Description                                      |
| ------------------------------- | ------- | ------------------------- | ------------------------------------------------ |
| `output_file`                   | `str`   | `"outputs/predictions.jsonl"` | Path to write predictions as JSONL.          |
| `questions_path`                | `str \| None` | `None`          | Path to questions.jsonl (auto-downloads if missing). |
| `tables_path`                   | `str \| None` | `None`          | Path to tables.jsonl (auto-downloads if missing).    |
| `workdir_path`                  | `str`   | `"llmsql_workdir"`        | Working directory for downloaded files.          |
| `num_fewshots`                  | `int`   | `5`                       | Number of few-shot examples (0, 1, or 5).        |
| `batch_size`                    | `int`   | `8`                       | Batch size for inference.                        |
| `seed`                          | `int`   | `42`                      | Random seed for reproducibility.                 |

**Note:** Explicit parameters (e.g., `dtype`, `trust_remote_code`) override any values specified in `model_kwargs` or `tokenizer_kwargs`.

---

### `inference_vllm(...)`

Runs inference using the [vLLM](https://github.com/vllm-project/vllm) backend for high-speed batched decoding.

**Parameters:**

#### Model Loading

| Argument                        | Type           | Default | Description                                      |
| ------------------------------- | -------------- | ------- | ------------------------------------------------ |
| `model_name`                    | `str`          | *required* | Hugging Face model name or local path.        |
| `trust_remote_code`             | `bool`         | `True`  | Whether to trust remote code when loading.       |
| `tensor_parallel_size`          | `int`          | `1`     | Number of GPUs for tensor parallelism.           |
| `hf_token`                      | `str \| None`  | `None`  | Hugging Face authentication token.               |
| `llm_kwargs`                    | `dict \| None` | `None`  | Additional kwargs for `vllm.LLM()`.              |
| `llm_kwargs`                    | `bool` | `True`  | Whether to use chat template of the tokenizer              |

#### Generation

| Argument                        | Type           | Default | Description                                      |
| ------------------------------- | -------------- | ------- | ------------------------------------------------ |
| `max_new_tokens`                | `int`          | `256`   | Maximum tokens to generate per sequence.         |
| `temperature`                   | `float`        | `1.0`   | Sampling temperature (0.0 = greedy).             |
| `do_sample`                     | `bool`         | `True`  | Whether to use sampling vs greedy decoding.      |
| `sampling_kwargs`               | `dict \| None` | `None`  | Additional kwargs for `vllm.SamplingParams()`.   |

#### Benchmark

| Argument                        | Type           | Default                       | Description                                      |
| ------------------------------- | -------------- | ----------------------------- | ------------------------------------------------ |
| `output_file`                   | `str`          | `"outputs/predictions.jsonl"` | Path to write predictions as JSONL.              |
| `questions_path`                | `str \| None`  | `None`                        | Path to questions.jsonl (auto-downloads if missing). |
| `tables_path`                   | `str \| None`  | `None`                        | Path to tables.jsonl (auto-downloads if missing).    |
| `workdir_path`                  | `str`          | `"llmsql_workdir"`            | Working directory for downloaded files.          |
| `num_fewshots`                  | `int`          | `5`                           | Number of few-shot examples (0, 1, or 5).        |
| `batch_size`                    | `int`          | `8`                           | Number of prompts per batch.                     |
| `seed`                          | `int`          | `42`                          | Random seed for reproducibility.                 |

**Note:** Explicit parameters (e.g., `tensor_parallel_size`, `trust_remote_code`) override any values specified in `llm_kwargs` or `sampling_kwargs`.

---

## Output Format

Both inference methods return a list of dictionaries and write results to `output_file` in JSONL format:

```json
{"question_id": "1", "completion": "SELECT name FROM students WHERE age > 18;"}
{"question_id": "2", "completion": "SELECT COUNT(*) FROM courses;"}
{"question_id": "3", "completion": "SELECT name FROM teachers WHERE department = 'Physics';"}
```

---

## Choosing Between Backends

| Backend          | Pros                             | Ideal For                            |
| ---------------- | -------------------------------- | ------------------------------------ |
| **Transformers** | Easy setup, CPU/GPU compatible   | Small models, simple runs            |
| **vLLM**         | Much faster, optimized GPU usage | Large models |
