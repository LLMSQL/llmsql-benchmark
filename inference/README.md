# Inference: Generate SQL Predictions

This directory contains scripts to generate **SQL predictions** for a dataset of natural language questions using language models. We provide two inference engines:

- **`inference.py`**: Uses Hugging Face Transformers (AutoModel)
- **`inference_vllm.py`**: Uses vLLM for faster, more stable inference ⚡

Both scripts take:
- A dataset of questions
- Table metadata
- A model (pretrained or fine-tuned)

And produce a JSONL file with SQL completions.

---

## 🚨 Important
Run all commands from the **project root folder** (`LLMSQL/`), not from inside `inference/`.

---

## Choosing Your Inference Engine

### Standard Inference (`inference/inference.py`)
- Uses Hugging Face Transformers
- Good for small models or limited resources
- May have memory issues with large models

### vLLM Inference (`inference/inference_vllm.py`) - **Recommended**
- 2-10x faster inference speed
- Better memory management and stability
- Multi-GPU support with tensor parallelism
- Supports most popular model architectures

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

### Option A: vLLM Inference (Recommended)

```bash
python3 inference/inference_vllm.py \
    --questions_file dataset/val_questions.jsonl \
    --tables_file dataset/tables.jsonl \
    --output_file outputs/my_model_preds.jsonl \
    --model_name outputs/finetuned-llama \
    --shots 5 \
    --batch_size 32 \
    --max_new_tokens 256 \
    --do_sample false \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9
```

### Option B: Standard Transformers Inference

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

Both will:
* Load the model from `outputs/finetuned-llama` (or any HF model ID).
* Generate SQL queries for each question.
* Save results into `outputs/my_model_preds.jsonl`.

Output format:
```json
{"question_id": "1", "completion": "SELECT COUNT(*) FROM players WHERE age > 30"}
```

---

## Arguments

### Common Arguments (Both Scripts)

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

### vLLM-Specific Arguments

| Argument                   | Default | Description                                    |
| -------------------------- | ------- | ---------------------------------------------- |
| `--tensor_parallel_size`   | `1`     | Number of GPUs for tensor parallelism         |
| `--gpu_memory_utilization` | `0.9`   | Fraction of GPU memory to use (0.0 to 1.0)    |

### Performance Tips

**For vLLM:**
- Use larger `--batch_size` (32-128) for better throughput
- Set `--tensor_parallel_size` to number of available GPUs
- Adjust `--gpu_memory_utilization` if you get OOM errors (try 0.7-0.8)

**For Standard Inference:**
- Keep `--batch_size` smaller (8-16) to avoid OOM
- Monitor GPU memory usage

Check full argument list:
```bash
python3 inference/inference_vllm.py --help
python3 inference/inference.py --help
```

---

## Installation

### For Standard Inference
```bash
pip install transformers torch
```

### For vLLM Inference
```bash
pip install vllm
# vLLM automatically includes transformers and torch
```

---

## Step 3: Next Steps

Evaluate your predictions with:

```bash
python3 evaluation/evaluate_answers.py --pred_file outputs/my_model_preds.jsonl
```