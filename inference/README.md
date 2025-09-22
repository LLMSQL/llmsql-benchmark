# Inference: Generate SQL Predictions

This directory contains `inference.py`, a script to generate **SQL predictions** for a dataset of natural language questions using a Hugging Face causal LM.

The script takes:
- A dataset of questions
- Table metadata
- A model (pretrained or fine-tuned)

It then produces a JSONL file with SQL completions.

---

## 🚨 Important
Run all commands from the **project root folder** (`LLMSQL/`), not from inside `inference/`.

---

## Step 1: Input Data

You need two inputs:

1. **Questions JSONL** (e.g. `dataset/questions.jsonl`):
```json
{
  "question_id": "1",
  "question": "How many players are older than 30?",
  "sql": "SELECT COUNT(*) FROM players WHERE age > 30",
  "table_id": "players_1"
}
````

2. **Tables JSONL** (e.g. `dataset/tables.jsonl`):

```json
{
  "table_id": "players_1",
  "header": ["name", "age", "team"],
  "types": ["text", "number", "text"],
  "rows": [["Alice", "31", "Red Sox"], ["Bob", "29", "Yankees"]]
}
```

---

## Step 2: Run Inference

Example command:

```bash
python3 inference/inference.py \
    --questions_file dataset/val_questions.jsonl \
    --tables_file dataset/tables.jsonl \
    --output_file outputs/my_model_preds.jsonl \
    --model_name outputs/finetuned-llama \
    --shots 5 \
    --batch_size 16 \
    --max_new_tokens 256 \
    --do_sample false
```

This will:

* Load the model from `outputs/finetuned-llama` (or any HF model ID).
* Generate SQL queries for each question.
* Save results into `outputs/my_model_preds.jsonl`.

Output format:

```json
{"question_id": "1", "completion": "SELECT COUNT(*) FROM players WHERE age > 30"}
```

---

## Arguments

| Argument           | Default                   | Description                                 |
| ------------------ | ------------------------- | ------------------------------------------- |
| `--questions_file` | `dataset/questions.jsonl` | Input JSONL with natural language questions |
| `--tables_file`    | `dataset/tables.jsonl`    | Table metadata JSONL                        |
| `--output_file`    | *required*                | Output JSONL file (will overwrite)          |
| `--model_name`     | *required*                | Hugging Face model name or local path       |
| `--shots`          | `5`                       | Prompt style (0, 1, or 5-shot)(you can find them in `finetune/utils/prompts.py`)              |
| `--batch_size`     | `8`                       | Number of samples per batch                 |
| `--max_new_tokens` | `256`                     | Maximum tokens per completion               |
| `--temperature`    | `1.0`                     | Sampling temperature                        |
| `--do_sample`      | `true`                    | Whether to sample (otherwise greedy)        |
| `--seed`           | `42`                      | Random seed for reproducibility             |
| `--hf_token`       | `$HF_TOKEN`               | Hugging Face token (optional)               |

Check full list:

```bash
python3 inference/inference.py --help
```

---

## Step 3: Next Steps

Evaluate your predictions with:

   ```bash
   python3 evaluation/evaluate_answers.py --pred_file outputs/my_model_preds.jsonl
   ```

