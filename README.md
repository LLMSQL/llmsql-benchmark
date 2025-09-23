# LLMSQL

Patched and improved version of the original large crowd-sourced dataset for developing natural language interfaces for relational databases, [WikiSQL](https://github.com/salesforce/WikiSQL).

---

## Overview

This repository provides the **LLMSQL Benchmark** — a modernized, cleaned, and extended version of WikiSQL, designed for evaluating and fine-tuning large language models (LLMs) on **Text-to-SQL** tasks.

### ✨ Highlights
- Updated schema and improved SQL annotations.
- Support for modern LLMs.
- Tools for **evaluation**, **inference**, and **finetuning**.
- Support for Hugging Face models out-of-the-box.
- Structured for reproducibility and benchmarking.

---

## 🚨 Version Notice

This is the **first release** of the LLMSQL Benchmark.  
Expect refinements, new features, and additional tools in future updates.

---

## Usage Recommendations

Modern LLMs are already strong at **producing SQL queries without finetuning**.  
We therefore recommend that most users:

1. **Run inference** directly on the full benchmark:
   - Use `dataset/questions.jsonl` (the main evaluation set).
   - Generate SQL predictions with your LLM.
   - Evaluate results against the benchmark.

2. **Optional finetuning**:
   - For research or domain adaptation, we provide `train_questions.jsonl`, `val_questions.jsonl`, and `test_questions.jsonl`.
   - Use the `finetune/` scripts if you want to adapt a base model.

---

## Repository Structure

```

WikiSQLv2/
├── dataset/             # JSONL files (questions, tables, splits)
├── evaluation/          # Scripts for downloading DB + evaluating predictions
├── inference/           # Generate SQL queries with your LLM
├── finetune/            # Fine-tuning with TRL's SFTTrainer
├── outputs/             # Example location for your model outputs
└── utils/               # Shared helpers (prompt builders, logging, etc.)

````

---

## Quickstart

## Install

Make sure you have the repo cloned (we used python3.11):

```bash
git clone https://github.com/LLMSQL/llmsql-benchmark.git
cd LLMSQL
pip3 install -r requirements.txt
```

### 1. Download the Benchmark Database
```bash
python3 evaluation/download_db.py
````

This will fetch `sqlite_tables.db` into `dataset/`.

### 2. Run Inference

```bash
python3 inference/inference.py \
    --questions_file dataset/questions.jsonl \
    --tables_file dataset/tables.jsonl \
    --output_file outputs/my_model_preds.jsonl \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --shots 5 \
    --batch_size 16
```

### 3. Evaluate Results

```bash
python3 evaluation/evaluate_answers.py \
    --pred_file outputs/my_model_preds.jsonl
```

---

## Finetuning (Optional)

If you want to adapt a base model on LLMSQL:

```bash
python3 finetune/finetune.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --output_dir outputs/finetuned-llama \
    --train_file dataset/train_questions.jsonl \
    --val_file dataset/val_questions.jsonl \
    --tables_file dataset/tables.jsonl \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1
```

This will train a model on the train/val splits and save it under `outputs/`.

---

## Suggested Workflow

* **Primary**: Run inference on `dataset/questions.jsonl` → Evaluate with `evaluation/`.
* **Secondary (optional)**: Fine-tune on `train/val` → Test on `test_questions.jsonl`.

---

## License & Citation

This project builds on the original [WikiSQL](https://github.com/salesforce/WikiSQL) dataset.
Please cite LLMSQL if you use it in your work:
```
@inproceedings{llmsql_bench,
  title={LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQLels},
  author={Pihulski, Dzmitry and  Charchut, Karol and Novogrodskaia, Viktoria and Koco{'n}, Jan},
  booktitle={2025 IEEE International Conference on Data Mining Workshops (ICDMW)},
  pages={...},
  year={2025},
  organization={IEEE}
}
```


