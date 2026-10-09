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

Text-to-SQL benchmarks for evaluating large language models, built on the tables of [WikiSQL](https://github.com/salesforce/WikiSQL):

* **LLMSQL 2.0** (default, `version="2.0"`) — a small, hard and verified **test-only** benchmark of **2,000** questions over 942 real Wikipedia tables. See [LLMSQL 2.0](#llmsql-20).
* **LLMSQL 1.0** (`version="1.0"`) — the patched and cleaned version of WikiSQL with train/validation/test splits.

Our datasets are available on our [HuggingFace page](https://huggingface.co/llmsql-bench).

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

* [2026/10] **LLMSQL 2.0 is released**: 2,000 hard, verified test questions in three categories, a zero-shot prompt with sample rows and distractor tables, and a lenient execution match. gpt-oss-120b solves 21.9% (medium reasoning effort) and 39.0% (high). See [LLMSQL 2.0](#llmsql-20) and the [dataset](https://huggingface.co/datasets/llmsql-bench/llmsql-2.0).

* [2026/10] Bring your own model: run the benchmark with any custom async callable via [`inference_function()`](./llmsql/inference/inference_function.py#inference_function), with built-in rate limiting and concurrency control.

* [2026/03] Fully functional CLI commands for inference and evaluation. See this [guide](./llmsql/_cli/README.md).

* [2026/03] Added support for API inference, for now only for OpenAI-compatable APIs, see [`inference_api()` function](./llmsql/inference/inference_api.py#inference_api)

* [2026/03] The page now contains first version of [leaderboard](https://llmsql.github.io/llmsql-benchmark/#:~:text=%F0%9F%93%8A%20Leaderboard%20%E2%80%94%20Execution%20Accuracy%20%28EX)!

* [2026/02] A first LLMSQL 2.0 draft (LLMSQL 1.0 with corrected aggregation operators) was published. It has been replaced by the new LLMSQL 2.0 and is still available at the Hugging Face tag [`legacy-2.0-aggregation-fix`](https://huggingface.co/datasets/llmsql-bench/llmsql-2.0/tree/legacy-2.0-aggregation-fix).



## LLMSQL 2.0

LLMSQL 2.0 ([`llmsql-bench/llmsql-2.0`](https://huggingface.co/datasets/llmsql-bench/llmsql-2.0)) is a **test-only** benchmark of **2,000** natural-language questions over **942** real Wikipedia tables. Every question comes with a reference SQLite query and its **verified answer**: each answer was re-derived independently (by a second implementation, a hand audit, or both), and questions whose answer depends on an interpretation the table does not settle were removed.

| category | questions | what it tests |
|---|---|---|
| `lookup` | 420 | WikiSQL-style lookups (cleaned LLMSQL questions): choosing the right filter column and reproducing the exact cell value from terse questions |
| `convention` | 1,158 | Sports game logs whose cells follow a convention that must be read from the data, e.g. winner-first scores (`L 37–0` = the team lost 0–37), tennis set scores, mixed time formats. Some questions show additional tables from the same Wikipedia page (distractors). |
| `text_quantity` | 422 | Dates stored as text (`September 9, 1985`, `9September1985`, …): spans in days, earliest/latest, counting rows after a date. Plain string comparison gives the wrong answer. |

**Protocol** (implemented end to end by this package):

1. **Prompt.** Zero-shot (there are no train or validation splits). The model sees the `CREATE TABLE` schema with the real table name, the Wikipedia page/section and the **first 3 rows** of every table in the question's `tables` field — the target table plus, for some `convention` questions, distractor tables from the same page — and is asked to return a single SQLite query in a ```` ```sql ```` block. The exact prompt is also stored in the `prompt` field of every question.
2. **SQL extraction.** The SQL is taken from the **last** ```` ```sql ```` block of the completion (falling back to the first `WITH`/`SELECT` statement) and executed on `sqlite_tables.db`.
3. **Lenient execution match.** The result is compared to the verified `answer`. The comparison is insensitive to row order and duplicate rows, tolerant to number formatting (thousands separators, currency signs, units, rounding to 2 decimals) and to a trailing count in parentheses (`Al Horford (15)`), and accepts extra columns if one of them equals the single answer column. See [`llmsql/utils/matching.py`](./llmsql/utils/matching.py).
4. **Metric.** Execution accuracy over all 2,000 questions, also reported per category.

**Results** (zero-shot, vLLM, a single run):

| model | accuracy |
|---|---|
| gpt-oss-120b (reasoning effort: high) | 39.0% |
| gpt-oss-120b (reasoning effort: medium) | 21.9% |

> [!Note]
> The questions were selected from a larger pool of verified candidates as those that gpt-oss-120b (medium reasoning effort) failed in earlier runs. The numbers above come from fresh runs that were not used for selection.

> [!Tip]
> LLMSQL 2.0 is zero-shot only: `num_fewshots` defaults to `0` for `version="2.0"` (and to `5` for `version="1.0"`); requesting few-shot examples for 2.0 raises an error. The prompt asks for a ```` ```sql ```` block and many models reason before answering, so `max_new_tokens` defaults to 4096 for 2.0 (256 for 1.0). The results above were obtained with up to 16k new tokens — give reasoning models a large budget.

**Versions of the `llmsql-bench/llmsql-2.0` dataset:** `main` is the 2,000-question benchmark described here. The earlier, unpublished LLMSQL 2.0 draft (LLMSQL 1.0 with corrected aggregation operators, 80k questions with train/val/test splits) is available at the tag [`legacy-2.0-aggregation-fix`](https://huggingface.co/datasets/llmsql-bench/llmsql-2.0/tree/legacy-2.0-aggregation-fix). LLMSQL 1.0 is [`llmsql-bench/llmsql-benchmark`](https://huggingface.co/datasets/llmsql-bench/llmsql-benchmark) (`version="1.0"`).


## Usage Recommendations

Modern LLMs are already strong at producing SQL queries without finetuning.
We therefore recommend that most users:

1. **Run inference** directly on the full benchmark:
   - Use one of the supported inference frameworks:
      * [`llmsql.inference_transformers`](./llmsql/inference/inference_transformers.py) - the function for transformers inference, for generation of SQL predictions with your model. Works both with HF model id, e.g. `Qwen/Qwen2.5-1.5B-Instruct` and model instance passed directly, e.g. `inference_transformers(model_or_model_name_or_path=model, ...)`.
      * [`llmsql.inference_vllm`](./llmsql/inference/inference_vllm.py) - if you want to do vllm based inference.
      * [`llmsql.inference_api`](./llmsql/inference/inference_api.py#inference_api) - for OpenAI-compatible API inference.
      * [`llmsql.inference_function`](./llmsql/inference/inference_function.py#inference_function) - for passing your own async function (any engine, API client or agent) for inference, with `requests_per_minute` and `max_concurrency` control.
   - Evaluate results against the benchmark with the [`llmsql.evaluate`](./llmsql/evaluation/evaluate.py) function.

2. **Optional finetuning** (LLMSQL 1.0):
   - For research or domain adaptation, we provide finetuning version for HF models. Use [Finetune Ready](https://huggingface.co/collections/llmsql-bench/fine-tune-ready-versions-of-the-llmsql-benchmark) datasets from HuggingFace. LLMSQL 2.0 is a test-only benchmark.

> [!Tip]
> You can find additional manuals in the README files of each folder([Inferece Readme](./llmsql/inference/README.md), [Evaluation Readme](./llmsql/evaluation/README.md))

> [!Tip]
> vllm based inference require vllm optional dependency group installed: `pip install llmsql[vllm]`


## Repository Structure

```

llmsql/
├── evaluation/          # Scripts for evaluation
└── inference/           # Generate SQL queries with your LLM
```



## Quickstart

For the full tutorial, check out the Colab notebook: [Open in Colab](https://colab.research.google.com/drive/1i0A7t_iSnDTikGqzG5Gq3ETswFsK5RQw#scrollTo=jcUR9wxvRBBb)

### Install

Make sure you have the package installed (we used python3.11):

```bash
pip3 install llmsql
```

### LLMSQL 2.0 (default)

```python
from llmsql import inference_vllm, evaluate

# zero-shot; give reasoning models a large generation budget
results = inference_vllm(
    "openai/gpt-oss-20b",
    version="2.0",
    output_file="outputs.jsonl",
    max_new_tokens=16384,
)
report = evaluate("outputs.jsonl", version="2.0")
print(report["accuracy"], report["category_accuracy"])
```

The same with the CLI:

```bash
llmsql inference vllm \
    --model-name openai/gpt-oss-20b \
    --version 2.0 \
    --output-file outputs.jsonl \
    --max-new-tokens 16384

llmsql evaluate --outputs outputs.jsonl --version 2.0
```

`inference_transformers`, `inference_api` and `inference_function` take the same `version` argument.

### LLMSQL 1.0

#### 1. Run Inference

##### Transformers inference

```python
from llmsql import inference_transformers

# Run generation directly with transformers
results = inference_transformers(
    model_or_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    version="1.0",
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

##### Vllm inference (Recommended)

To speed up your inference we recommend using vllm inference. You can do it with optional llmsql[vllm] dependency group
```bash
pip install llmsql[vllm]
```

After that run
```python
from llmsql import inference_vllm
results = inference_vllm(
    "Qwen/Qwen2.5-1.5B-Instruct",
    version="1.0",
    output_file="test_results.jsonl",
    do_sample=False,
    batch_size=20000
)
```
for fast inference.

#### 2. Evaluate Results

```python
from llmsql import evaluate

report = evaluate(outputs="path_to_your_outputs.jsonl", version="1.0")
print(report)
```

Or with the results from the inference:

```python
from llmsql import evaluate

# results = inference_transformers(...) or inference_vllm(...)

report = evaluate(outputs=results, version="1.0")
print(report)
```


For more examples check the [examples folder](./examples/)

## Prompt Templates

### LLMSQL 2.0

Zero-shot: schema with the real table names, the Wikipedia page/section and the first 3 rows of every table shown (target plus distractors), then the question. Example:

````
You are an expert SQLite query writer. Given the database schema with a few sample rows and a question,
write a single SQLite query that answers the question. Use the exact table names. Return only the SQL in a ```sql block.

CREATE TABLE "1-29135051-2" ("Episode" REAL, "Broadcast date" TEXT, "Guest(s)" TEXT, "Singer(s)" TEXT, "Comedian" TEXT, "Ratings" TEXT);
-- Wikipedia: The Rob Brydon Show / Series 2
-- first 3 of 6 rows:
[1.0, "22July2011", "Matt Lucas", "The Script", "Nina Conti", "2.08m"]
[2.0, "29July2011", "Bill Bailey", "Beverley Knight", "Celia Pacquola", "1.45m"]
[3.0, "5August2011", "Bruce Forsyth", "Sophie Ellis-Bextor", "Elis James", "Under 1.41m"]

Question: Name the singer for Joe Wilkinson
````

Implemented by `build_prompt_v2` in [`llmsql/prompts/prompts.py`](./llmsql/prompts/prompts.py); it reproduces the `prompt` field of the dataset byte for byte.

### LLMSQL 1.0

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

Implementations of the LLMSQL 1.0 0-shot, 1-shot, and 5-shot prompt templates are available here:
👉 [link-to-file](./llmsql/prompts/prompts.py)


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
