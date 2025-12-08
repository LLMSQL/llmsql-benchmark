![Downloads](https://img.shields.io/pypi/dm/llmsql)
[![codecov](https://codecov.io/gh/LLMSQL/llmsql-benchmark/branch/main/graph/badge.svg)](https://codecov.io/gh/LLMSQL/llmsql-benchmark)
![PyPI Version](https://img.shields.io/pypi/v/llmsql)
![CI](https://github.com/LLMSQL/llmsql-benchmark/actions/workflows/tests.yml/badge.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/llmsql)
![License](https://img.shields.io/pypi/l/llmsql)

# LLMSQL

Patched and improved version of the original large crowd-sourced dataset for developing natural language interfaces for relational databases, [WikiSQL](https://github.com/salesforce/WikiSQL).


Our datasets are available for different scenarios on our [HuggingFace page](https://huggingface.co/llmsql-bench).
---

## Overview

### Install

```bash
pip3 install llmsql
```

This repository provides the **LLMSQL Benchmark** — a modernized, cleaned, and extended version of WikiSQL, designed for evaluating large language models (LLMs) on **Text-to-SQL** tasks.

### Note
The package doesn't have the dataset, it is stored on our [HuggingFace page](https://huggingface.co/llmsql-bench).

### This package contains
- Support for modern LLMs.
- Tools for **inference** and **evaluation**.
- Support for Hugging Face models out-of-the-box.
- Structured for reproducibility and benchmarking.

---

## Usage Recommendations

Modern LLMs are already strong at **producing SQL queries without finetuning**.
We therefore recommend that most users:

1. **Run inference** directly on the full benchmark:
    model_or_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    output_file="path_to_your_outputs.jsonl",
   - Use [`llmsql.inference_transformers`](./llmsql/inference/inference_transformers.py) (the function for transformers inference) for generation of SQL predictions with your model. If you want to do vllm based inference, use [`llmsql.inference_vllm`](./llmsql/inference/inference_vllm.py). Works both with HF model id, e.g. `Qwen/Qwen2.5-1.5B-Instruct` and model instance passed directly, e.g. `inference_transformers(model_or_model_name_or_path=model, ...)`
   - Evaluate results against the benchmark with the [`llmsql.LLMSQLEvaluator`](./llmsql/evaluation/evaluator.py) evaluator class.

2. **Optional finetuning**:
   - For research or domain adaptation, we provide finetuning version for HF models. Use [Finetune Ready](https://huggingface.co/datasets/llmsql-bench/llmsql-benchmark-finetune-ready) dataset from HuggingFace.

> [!Tip]
> You can find additional manuals in the README files of each folder([Inferece Readme](./llmsql/inference/README.md), [Evaluation Readme](./llmsql/evaluation/README.md))

> [!Tip]
> vllm based inference require vllm optional dependency group installed: `pip install llmsql[vllm]`
---

## Repository Structure

```

llmsql/
├── evaluation/          # Scripts for downloading DB + evaluating predictions
└── inference/           # Generate SQL queries with your LLM
```



## Quickstart


### Install

Make sure you have the package installed (we used python3.11):

```bash
pip3 install llmsql
```

### 1. Run Inference

#### Transformers inference

```python
from llmsql import inference_transformers

# Run generation directly with transformers
results = inference_transformers(
    model_or_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    output_file="path_to_your_outputs.jsonl",
    num_fewshots=5,
    batch_size=8,
    max_new_tokens=256,
    do_sample=False,
    model_args={
        "torch_dtype": "bfloat16",
    }
)
```

#### Vllm inference (Recommended)

To speed up your inference we recommend using vllm inference. You can do it with optional llmsql[vllm] dependency group
```bash
pip install llmsql[vllm]
```

After that run
```python
from llmsql import inference_vllm
results = inference_vllm(
    "Qwen/Qwen2.5-1.5B-Instruct",
    "test_results.jsonl",
    do_sample=False,
    batch_size=20000
)
```
for fast inference.

### 2. Evaluate Results

```python
from llmsql import evaluate

report =evaluate(outputs="path_to_your_outputs.jsonl")
print(report)
```

Or with ther results from the infernece:

```python
from llmsql import evaluate

# results = inference_transformers(...) or infernce_vllm(...)

report =evaluate(outputs=results)
print(report)
```


## Command Line Interface (CLI)

LLMSQL also exposes all inference and evaluation capabilities through a convenient command-line interface.
After installing the package:

```bash
pip install llmsql
```

you can run:

```bash
llmsql --help
```

to view the top-level commands:

```
LLMSQL: LLM-powered SQL generation toolkit

Top-level commands:
  inference    Run SQL generation using a chosen LLM backend
  evaluate     Evaluate predictions against the LLMSQL benchmark
```

### CLI Inference

You can run inference with **Transformers** or **vLLM** directly from the CLI.

### Transformers backend

```bash
llmsql inference --method transformers \
    --model-or-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
    --output-file outputs/preds_transformers.jsonl \
    --batch-size 8 \
    --num-fewshots 5
```

With custom model init arguments:

```bash
llmsql inference --method transformers \
    --model-or-model-name-or-path meta-llama/Llama-3-8b-instruct \
    --output-file outputs/llama_preds.jsonl \
    --model-kwargs '{"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"}'
```

Override generation parameters:

```bash
llmsql inference --method transformers \
    --model-or-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
    --output-file outputs/temp_0.9.jsonl \
    --temperature 0.9 \
    --generation-kwargs '{"do_sample": true, "top_p": 0.9, "top_k": 40}'
```


### vLLM backend (recommended for speed)

Install the vLLM optional dependencies:

```bash
pip install llmsql[vllm]
```

Then run:

```bash
llmsql inference --method vllm \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --output-file outputs/preds_vllm.jsonl \
    --batch-size 20000 \
    --num-fewshots 5 \
    --do-sample false
```

With `llm-kwargs`:

```bash
llmsql inference --method vllm \
    --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --output-file outputs/mixtral_preds.jsonl \
    --llm-kwargs '{"max_model_len": 4096, "gpu_memory_utilization": 0.9}'
```


### CLI Evaluation

You can evaluate model predictions (either from a file or inline JSON):

```bash
llmsql evaluate \
    --outputs outputs/preds_vllm.jsonl
```

Evaluate inline JSON:

```bash
llmsql evaluate \
    --outputs '[{"id": 1, "sql": "SELECT ..."}]'
```

Specify custom paths:

```bash
llmsql evaluate \
    --outputs outputs/preds.jsonl \
    --questions-path data/questions.jsonl \
    --db-path data/database.sqlite \
    --show-mismatches true \
    --max-mismatches 10
```

Specify where to save the evaluation report:

```bash
llmsql evaluate \
    --outputs outputs/preds.jsonl \
    --save-report results/eval_report.json
```


## Suggested Workflow

* **Primary**: Run inference on all questions with vllm or transformers → Evaluate with `evaluate()`.
* **Secondary (optional)**: Fine-tune on `train/val` → Test on `test_questions.jsonl`. You can find the datasets here [HF Finetune Ready](https://huggingface.co/datasets/llmsql-bench/llmsql-benchmark-finetune-ready).


## Contributing

Check out our [open issues](https://github.com/LLMSQL/llmsql-benchmark/issues) and feel free to submit pull requests!

We also encourage you to submit new issues!

To get started with development, first fork the repository and install the dev dependencies.

For more information on the contributing: check [CONTRIBUTING.md](./CONTRIBUTING.md) and our [documentation page](https://llmsql.github.io/llmsql-benchmark/).



## License & Citation

Please cite LLMSQL if you use it in your work:
```text
@inproceedings{llmsql_bench,
  title={LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL},
  author={Pihulski, Dzmitry and  Charchut, Karol and Novogrodskaia, Viktoria and Koco{'n}, Jan},
  booktitle={2025 IEEE International Conference on Data Mining Workshops (ICDMW)},
  year={2025},
  organization={IEEE}
}
```
