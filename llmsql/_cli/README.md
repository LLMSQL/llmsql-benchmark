# LLMSQL CLI Manual

The `llmsql` CLI provides two main workflows:

- `inference`: generate SQL predictions with a selected backend.
- `evaluate`: score predictions on the LLMSQL benchmark.

## Command Structure

```bash
llmsql <command> [options]
```

Available commands:

- `llmsql inference transformers ...`
- `llmsql inference vllm ...`
- `llmsql inference api ...`
- `llmsql evaluate ...`

## Inference Commands

### 1) Transformers backend

```bash
llmsql inference transformers \
  --model-or-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --output-file outputs/preds_transformers.jsonl
```

This command calls [`inference_transformers()`](../inference/inference_transformers.py).

### 2) vLLM backend

```bash
llmsql inference vllm \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --output-file outputs/preds_vllm.jsonl
```

This command calls [`inference_vllm()`](../inference/inference_vllm.py).

### 3) OpenAI-compatible API backend

```bash
llmsql inference api \
  --model-name gpt-5-mini \
  --base-url https://api.openai.com/v1 \
  --output-file outputs/preds_api.jsonl
```

This command calls [`inference_api()`](../inference/inference_api.py).

## Evaluation Command

```bash
llmsql evaluate --outputs outputs/preds_transformers.jsonl
```

This command calls [`evaluate()`](../evaluation/evaluate.py).

## Help

Use built-in help to see all options:

```bash
llmsql --help
llmsql inference --help
llmsql inference transformers --help
llmsql inference vllm --help
llmsql inference api --help
llmsql evaluate --help
```
