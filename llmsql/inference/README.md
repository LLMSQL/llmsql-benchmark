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
    shots=5,
    batch_size=8,
    max_new_tokens=256,
    temperature=0.7,
    model_args={
        "attn_implementation": "flash_attention_2",
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
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    output_file="outputs/preds_vllm.jsonl",
    questions_path="data/questions.jsonl",
    tables_path="data/tables.jsonl",
    shots=5,
    batch_size=8,
    max_new_tokens=256,
    do_sample=False,
    llm_kwargs={
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 4096,
    },
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
    --shots 5 \
    --temperature 0.0
```

Or use the Transformers backend:

```bash
llmsql inference --method transformers \
    --model-or-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
    --output-file outputs/preds.jsonl \
    --batch-size 8 \
    --temperature 0.9 \
    --generate-kwargs '{"do_sample": false, "top_p": 0.95}'
```

👉 Run `llmsql inference --help` for more detailed examples and parameter options.

---

## API Reference

### `inference_transformers(...)`

Runs inference using the Hugging Face `transformers` backend.

**Parameters:**

| Argument                        | Type    | Description                                                    |
| ------------------------------- | ------- | -------------------------------------------------------------- |
| `model_or_model_name_or_path`   | `str`   | Model name or local path (any causal LM).                      |
| `output_file`                   | `str`   | Path to write predictions as JSONL.                            |
| `questions_path`, `tables_path` | `str`   | Benchmark files (auto-downloaded if missing).                  |
| `shots`                         | `int`   | Number of few-shot examples (0, 1, 5).                         |
| `batch_size`                    | `int`   | Batch size for inference.                                      |
| `max_new_tokens`                | `int`   | Maximum length of generated SQL queries.                       |
| `temperature`                   | `float` | Sampling temperature.                                          |
| `do_sample`                     | `bool`  | Whether to use sampling.                                       |
| `model_args`                    | `dict`  | Extra kwargs passed to `AutoModelForCausalLM.from_pretrained`. |
| `generate_kwargs`               | `dict`  | Extra kwargs passed to `model.generate()`.                     |

---

### `inference_vllm(...)`

Runs inference using the [vLLM](https://github.com/vllm-project/vllm) backend for high-speed batched decoding.

**Parameters:**

| Argument                        | Type    | Description                                      |
| ------------------------------- | ------- | ------------------------------------------------ |
| `model_name`                    | `str`   | Hugging Face model name or path.                 |
| `output_file`                   | `str`   | Path to write predictions as JSONL.              |
| `questions_path`, `tables_path` | `str`   | Benchmark files (auto-downloaded if missing).    |
| `shots`                         | `int`   | Number of few-shot examples (0, 1, 5).           |
| `batch_size`                    | `int`   | Number of prompts per batch.                     |
| `max_new_tokens`                | `int`   | Maximum tokens per generation.                   |
| `temperature`                   | `float` | Sampling temperature.                            |
| `do_sample`                     | `bool`  | Whether to sample or use greedy decoding.        |
| `llm_kwargs`                    | `dict`  | Extra kwargs forwarded to `vllm.LLM`.            |
| `sampling_kwargs`               | `dict`  | Extra kwargs forwarded to `vllm.SamplingParams`. |

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
