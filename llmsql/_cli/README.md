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

## Benchmark version, few-shot examples and generation length

All `inference` subcommands accept `--version` (`2.0`, the default, or `1.0`):

- `--num-fewshots` defaults to 5 for `--version 1.0` and 0 for `--version 2.0`.
  LLMSQL 2.0 is zero-shot only; a non-zero value is rejected.
- `--max-new-tokens` (transformers / vllm) defaults to 256 for 1.0 and 4096 for 2.0.
  Reasoning models need more: the reported LLMSQL 2.0 results were obtained with up to
  16k new tokens.

```bash
# LLMSQL 2.0 (zero-shot)
llmsql inference vllm --model-name openai/gpt-oss-20b --version 2.0 \
  --output-file outputs.jsonl --max-new-tokens 16384
llmsql evaluate --outputs outputs.jsonl --version 2.0

# LLMSQL 1.0 (5-shot)
llmsql inference vllm --model-name Qwen/Qwen2.5-1.5B-Instruct --version 1.0 \
  --num-fewshots 5 --output-file outputs_v1.jsonl
llmsql evaluate --outputs outputs_v1.jsonl --version 1.0
```

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

### Passing model constructor arguments (`--model-args`)

The `transformers` and `vllm` subcommands accept `--model-args` (alias `--model_args`),
a comma-separated list of `key=value` pairs in the style of
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
llmsql inference transformers \
  --model-or-model-name-or-path EleutherAI/pythia-160m \
  --model-args dtype=float32,revision=main,low_cpu_mem_usage=true

llmsql inference vllm \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --model-args gpu_memory_utilization=0.8,max_model_len=4096,enforce_eager=true
```

- **transformers**: the arguments go to `AutoModelForCausalLM.from_pretrained()`
  (the `model_kwargs` of `inference_transformers()`). A `dtype` (or `torch_dtype`)
  given here overrides `--dtype`; dtype names such as `float32`, `bfloat16` or `auto` are accepted.
- **vllm**: the arguments go to `vllm.LLM()` (the `llm_kwargs` of `inference_vllm()`).

Parsing rules:

- Each pair is split on the first `=`, so values may contain `=`; whitespace around keys/values
  and empty items (e.g. a trailing comma) are ignored.
- Values are converted to `int`, `float`, `bool` (`true`/`false`) or `None` (`none`/`null`)
  where possible; everything else stays a string. Quote a value to force a string,
  e.g. `revision='"123"'`.
- Values cannot contain commas. For such values or nested structures use the JSON flags
  `--model-kwargs` (transformers) / `--llm-kwargs` (vllm).
- Malformed items (no `=`, empty key), duplicate keys, and `pretrained=` are rejected —
  the model is always given by `--model-or-model-name-or-path` / `--model-name`.
- `--model-args` can be combined with `--model-kwargs` / `--llm-kwargs`; the two are merged and
  **the JSON flag wins** on conflicting keys.

## Evaluation Command

```bash
llmsql evaluate --outputs outputs/preds_transformers.jsonl --version 2.0
```

For LLMSQL 2.0 the report also contains the accuracy per question category.

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
