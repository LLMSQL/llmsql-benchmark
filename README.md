<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/logo_black.svg">
  <img alt="LLMSQL Logo" src="./assets/logo_white.svg">
</picture>



![Downloads](https://img.shields.io/pypi/dm/llmsql)
[![codecov](https://codecov.io/gh/LLMSQL/llmsql-benchmark/branch/main/graph/badge.svg)](https://codecov.io/gh/LLMSQL/llmsql-benchmark)
![PyPI Version](https://img.shields.io/pypi/v/llmsql)
![CI](https://github.com/LLMSQL/llmsql-benchmark/actions/workflows/tests.yml/badge.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/llmsql)
![License](https://img.shields.io/pypi/l/llmsql)

# LLMSQL

Patched and improved version of the original large crowd-sourced dataset for developing natural language interfaces for relational databases, [WikiSQL](https://github.com/salesforce/WikiSQL).


Our datasets are available for different scenarios on our [HuggingFace page](https://huggingface.co/llmsql-bench).

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



## Latest News 📣

* [2025/12] Evaluation class converted to function see [new `evaluate(...)` function](./llmsql/evaluation/evaluate.py#evaluate)

* New page version added to [`https://llmsql.github.io/llmsql-benchmark/`](https://llmsql.github.io/llmsql-benchmark/)

* Vllm inference method now supports chat templates, see [`inference_vllm(...)`](./llmsql/inference/inference_vllm.py#inference_vllm).
* Transformers inference now supports custom chat tempalates with `chat_template` argument, see [`inference_transformers(...)`](./llmsql/inference/inference_transformers.py#inference_transformers)

* More stable and deterministic inference with  [`inference_vllm(...)`](./llmsql/inference/inference_vllm.py#inference_vllm) function added by setting [some envars](./llmsql/inference/inference_vllm.py)

* `padding_side` argument added to [`inference_transformers(...)`](./llmsql/inference/inference_transformers.py#inference_transformers) function with default `left` option.


## Usage Recommendations

Modern LLMs are already strong at **producing SQL queries without finetuning**.
We therefore recommend that most users:

1. **Run inference** directly on the full benchmark:
    model_or_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    output_file="path_to_your_outputs.jsonl",
   - Use [`llmsql.inference_transformers`](./llmsql/inference/inference_transformers.py) (the function for transformers inference) for generation of SQL predictions with your model. If you want to do vllm based inference, use [`llmsql.inference_vllm`](./llmsql/inference/inference_vllm.py). Works both with HF model id, e.g. `Qwen/Qwen2.5-1.5B-Instruct` and model instance passed directly, e.g. `inference_transformers(model_or_model_name_or_path=model, ...)`
   - Evaluate results against the benchmark with the [`llmsql.evaluate`](./llmsql/evaluation/evaluator.py) function.

2. **Optional finetuning**:
   - For research or domain adaptation, we provide finetuning version for HF models. Use [Finetune Ready](https://huggingface.co/datasets/llmsql-bench/llmsql-benchmark-finetune-ready) dataset from HuggingFace.

> [!Tip]
> You can find additional manuals in the README files of each folder([Inferece Readme](./llmsql/inference/README.md), [Evaluation Readme](./llmsql/evaluation/README.md))

> [!Tip]
> vllm based inference require vllm optional dependency group installed: `pip install llmsql[vllm]`


## Repository Structure

```

llmsql/
├── evaluation/          # Scripts for downloading DB + evaluating predictions
└── inference/           # Generate SQL queries with your LLM
```



## Quickstart

For the full tutorial, check out the Colab notebook: [Open in Colab](https://colab.research.google.com/drive/1i0A7t_iSnDTikGqzG5Gq3ETswFsK5RQw#scrollTo=jcUR9wxvRBBb)

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
    model_kwargs={
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


## Prompt Template

The prompt defines explicit constraints on the generated output. 
The model is instructed to output only a valid SQL `SELECT` query, to use a fixed table name (`"Table"`) **(which will be replaced with the actual table name during evaluation)**, to quote all table and column names, and to restrict generation to the specified SQL functions, condition operators, and keywords. 
The full prompt specification is provided in the prompt template.

Below is an example of the **5-shot prompt template** used during inference.

```
Your task: Given a question and a table schema, output ONLY a valid SQL SELECT query.
⚠️ STRICT RULES:
 - Output ONLY SQL (no explanations, no markdown, no ``` fences)
 - Use table name "Table"
 - Allowed functions: ['MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
 - Allowed condition operators: ['=', '>', '<', '!=']
 - Allowed SQL keywords: ['SELECT', 'WHERE', 'AND']
 - Always use "" with all column names and table name, even one word: "Price", "General column", "Something #"

### EXAMPLE 1:
Question: What is the price of the Samsung Galaxy S23?
Columns: ['Brand', 'Model', 'Price', 'Storage', 'Color']
Types: ['text', 'text', 'real', 'text', 'text']
Sample row: ['Apple', 'iPhone 14', 899.99, '128GB', 'White']
SQL: SELECT "Price" FROM "Table" WHERE "Brand" = "Samsung" AND "Model" = "Galaxy S23";

### EXAMPLE 2:
Question: How many books did Maya Chen publish?
Columns: ['Author', 'Books Published', 'Genre', 'Country', 'Years Active']
Types: ['text', 'real', 'text', 'text', 'text']
Sample row: ['John Smith', 3, 'Non-fiction', 'Canada', '2005–2015']
SQL: SELECT "Books Published" FROM "Table" WHERE "Author" = "Maya Chen";

### EXAMPLE 3:
Question: What is the total population of cities in California?
Columns: ['City', 'State', 'Population', 'Area', 'Founded']
Types: ['text', 'text', 'real', 'real', 'text']
Sample row: ['Houston', 'Texas', 2304580, 1651.1, '1837']
SQL: SELECT SUM("Population") FROM "Table" WHERE "State" = "California";

### EXAMPLE 4:
Question: How many restaurants serve Italian cuisine?
Columns: ['Restaurant', 'Cuisine', 'Rating', 'City', 'Price Range']
Types: ['text', 'text', 'real', 'text', 'text']
Sample row: ['Golden Dragon', 'Chinese', 4.2, 'Boston', '$$']
SQL: SELECT COUNT(*) FROM "Table" WHERE "Cuisine" = "Italian";

### EXAMPLE 5:
Question: What is the average salary for Software Engineers?
Columns: ['Job Title', 'Salary', 'Experience', 'Location', 'Company Size']
Types: ['text', 'real', 'text', 'text', 'text']
Sample row: ['Data Analyst', 70000, 'Junior', 'Chicago', '200–500']
SQL: SELECT AVG("Salary") FROM "Table" WHERE "Job Title" = "Software Engineer";

### NOW ANSWER:
Question: {question}
Columns: {headers}
Types: {types}
Sample row: {sample_row}
SQL:"""
```

Implementations of 0-shot, 1-shot, and 5-shot prompt templates are available here:
👉 [link-to-file](./llmsql/prompts/prompts.py)



## Suggested Workflow

* **Primary**: Run inference on all questions with vllm or transformers → Evaluate with `evaluate()`.
* **Secondary (optional)**: Fine-tune on `train/val` → Test on `test_questions.jsonl`. You can find the datasets here [HF Finetune Ready](https://huggingface.co/datasets/llmsql-bench/llmsql-benchmark-finetune-ready).


## Contributing

Check out our [open issues](https://github.com/LLMSQL/llmsql-benchmark/issues), fork this repo and feel free to submit pull requests!

We also encourage you to submit new issues!

To get started with development, first fork the repository and install basic dependencies with dev dependencies.

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
