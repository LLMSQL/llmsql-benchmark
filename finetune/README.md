
# Fine-tuning on LLMSQL Benchmark

This directory contains a training script (`finetune.py`) to fine-tune a causal LM (e.g., LLaMA, Mistral, GPT-OSS) on the **LLMSQL Text-to-SQL benchmark**.  
The model learns to map natural language questions + table context into SQL queries.

---

## 🚨 Important
Run all commands from the **project root folder** (`LLMSQL/`), not from inside `finetune/`.

---

## Step 1: Prepare the Data

Make sure you have the benchmark dataset in `dataset/`:
- `train_questions.jsonl`
- `val_questions.jsonl`
- `tables.jsonl`

Each `questions.jsonl` file must contain:
```json
{
  "question_id": "1",
  "question": "How many players are older than 30?",
  "sql": "SELECT COUNT(*) FROM players WHERE age > 30",
  "table_id": "players_1"
}
````

Each `tables.jsonl` file must contain table metadata with:

* `table_id`
* `header` (column names)
* `types` (column types)
* `rows` (sample rows)

---

## Step 2: Run Fine-tuning

Example command (tested with **2× H100 80GB**):

```bash
python3 finetune/finetune.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --output_dir outputs/finetuned-llama \
    --train_file dataset/train_questions.jsonl \
    --val_file dataset/val_questions.jsonl \
    --tables_file dataset/tables.jsonl \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --shots 5
```

This will:

* Load the base model from Hugging Face Hub (requires `HF_TOKEN` in your `.env` or environment).
* Build prompts from the question + table metadata.
* Train the model using **TRL’s `SFTTrainer`** with completion-only loss.
* Save the fine-tuned model to `outputs/finetuned-llama`.

---

## Arguments

| Argument                        | Default                         | Description                                                        |
| ------------------------------- | ------------------------------- | ------------------------------------------------------------------ |
| `--train_file`                  | `dataset/train_questions.jsonl` | Training set in JSONL format                                       |
| `--val_file`                    | `dataset/val_questions.jsonl`   | Validation set in JSONL format                                     |
| `--tables_file`                 | `dataset/tables.jsonl`          | Table metadata                                                     |
| `--model_name_or_path`          | *required*                      | Base model name or path (e.g., `meta-llama/Llama-3.2-1B-Instruct`) |
| `--output_dir`                  | *required*                      | Where to save the fine-tuned model                                 |
| `--shots`                       | `5`                             | Prompt type (0, 1, or 5-shot)(you can find them in `finetune/utils/prompts.py`)                                     |
| `--num_train_epochs`            | `3`                             | Number of epochs                                                   |
| `--per_device_train_batch_size` | `4`                             | Training batch size per device                                     |
| `--per_device_eval_batch_size`  | `4`                             | Evaluation batch size per device                                   |
| `--learning_rate`               | `5e-5`                          | Learning rate                                                      |
| `--save_steps`                  | `500`                           | Save checkpoint every N steps                                      |
| `--logging_steps`               | `100`                           | Log metrics every N steps                                          |
| `--seed`                        | `42`                            | Random seed                                                        |
| `--hf_token`                    | `$HF_TOKEN`                     | Hugging Face token (if not set, will try env var)                  |

For a full list, run:

```bash
python3 finetune/finetune.py --help
```

---

## Step 3: Next Steps

After fine-tuning, feel free to:

1. Use the model in **inference** (`inference/inference.py`) by setting `--model_name` to your fine-tuned model’s path (`outputs/finetuned-llama`).
2. Evaluate results using **evaluation** (`evaluation/evaluate_answers.py`).
