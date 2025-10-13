# Fine-Tuning: Training Text-to-SQL Models on LLMSQL

This module provides the fine-tuning pipeline for **LLMSQL**, allowing you to adapt a causal language model to the Text-to-SQL benchmark.
It supports dataset preparation, prompt construction, and training with Hugging Face [TRL](https://github.com/huggingface/trl)’s `SFTTrainer`.

---

## Installation

Make sure package is installed:

```bash
pip3 install llmsql
```

## Usage

### CLI (Recommended)

Run fine-tuning with a config file:

```bash
llmsql finetune --config_file examples/example_finetune_args.yaml
```

Example YAML (`examples/example_finetune_args.yaml`):

```yaml
# -------------------------
# Model
# -------------------------
model_name_or_path: "Qwen/Qwen2.5-1.5B-Instruct"  # Path or HF repo ID

# -------------------------
# Dataset / Files
# -------------------------
output_dir: "outputs/finetuned-llama"

# -------------------------
# Training hyperparameters
# -------------------------
shots: 5
num_train_epochs: 3
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
learning_rate: !!float 5e-5
save_steps: 10000
logging_steps: 1
seed: 42

seq_len: 32768
max_steps: 300
no_eval: false
eval_steps: 100
max_length: 2048

# -------------------------
# Misc / Logging
# -------------------------
wandb:
  project: ""
  run_name: ""
  key: ""
  offline: false
```

---

## Arguments

| Argument                        | Description                             | Default       |
| ------------------------------- | --------------------------------------- | ------------- |
| `--config_file`                 | Path to YAML config file                | `None`        |
| `--train_file`                  | Training dataset (JSONL)                | auto-download |
| `--val_file`                    | Validation dataset (JSONL)              | auto-download |
| `--tables_file`                 | Table schema file                       | auto-download |
| `--model_name_or_path`          | HF model ID or path                     | **required**  |
| `--output_dir`                  | Save directory for checkpoints          | **required**  |
| `--shots`                       | Few-shot prompt style (0/1/5)           | `5`           |
| `--num_train_epochs`            | Number of epochs                        | `3`           |
| `--per_device_train_batch_size` | Train batch size per device             | `4`           |
| `--per_device_eval_batch_size`  | Eval batch size per device              | `4`           |
| `--learning_rate`               | Learning rate                           | `5e-5`        |
| `--save_steps`                  | Save checkpoint every N steps           | `500`         |
| `--logging_steps`               | Log metrics every N steps               | `100`         |
| `--eval_steps`                  | Run eval every N steps                  | `100`         |
| `--max_steps`                   | Limit training steps (overrides epochs) | `None`        |
| `--max_length`                  | Max sequence length                     | `2048`        |
| `--seed`                        | Random seed                             | `42`          |
| `--no_eval`                     | Disable evaluation                      | `False`       |
| `--wandb_project`               | W\&B project name                       | `None`        |
| `--wandb_run_name`              | W\&B run name                           | `None`        |
| `--wandb_key`                   | W\&B API key                            | `None`        |
| `--wandb_offline`               | Run W\&B offline mode                   | `False`       |

---

## Logs & Checkpoints

* Logs are written to `<output_dir>/logs/`
* Models are saved to `<output_dir>` at each checkpoint
* Final model is saved at the end of training
* If Weights & Biases is configured, training metrics are also tracked there
