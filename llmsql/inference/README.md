# LLMSQL vLLM Inference

This module provides the `LLMSQLVLLMInference` class for running **Text-to-SQL generation** using large language models with the [vLLM](https://github.com/vllm-project/vllm) backend.
It loads benchmark questions and table schemas, builds prompts with configurable few-shot settings, and generates SQL queries efficiently in batches.

---


## Example Usage

### Installation

Ensure you have the required dependencies installed:

```bash
pip3 install llmsql
```

### Quick Start

```python
from llmsql import LLMSQLVLLMInference

# Initialize inference engine
inference = LLMSQLVLLMInference(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",  # or any Hugging Face causal LM
    tensor_parallel_size=1,
)

# Run generation
results = inference.generate(
    output_file="path_to_your_outputs.jsonl",
    questions_path="data/questions.jsonl",
    tables_path="data/tables.jsonl",
    shots=5,
    batch_size=8,
    max_new_tokens=256,
    temperature=0.7,
)
```

---

## Arguments

### Initialization (`LLMSQLVLLMInference`)

* `model_name` (str): Hugging Face model name or path.
* `hf_token` (str, optional): Hugging Face Hub authentication token.
* `tensor_parallel_size` (int): Tensor parallelism for multi-GPU inference.
* `seed` (int): Random seed for reproducibility.
* `workdir_path` (str): Local work directory for benchmark files.
* `**llm_kwargs`: Extra keyword arguments forwarded to `vllm.LLM`.

### Generation (`generate`)

* `output_file` (str): Path to write JSONL outputs (file is overwritten at start).
* `questions_path` (str, optional): Path to `questions.jsonl`. Auto-downloaded if missing.
* `tables_path` (str, optional): Path to `tables.jsonl`. Auto-downloaded if missing.
* `shots` (int): Number of examples for prompt builder (0, 1, or 5).
* `batch_size` (int): Number of questions processed per batch.
* `max_new_tokens` (int): Maximum tokens per generation.
* `temperature` (float): Sampling temperature (ignored if `do_sample=False`).
* `do_sample` (bool): Whether to sample or use greedy decoding.
* `**sampling_kwargs`: Extra keyword arguments for `SamplingParams`.

---

## Output Format

The `generate` method returns a list of dictionaries and also writes them incrementally to `output_file` in JSONL format:

```json
{"question_id": "1", "completion": "SELECT name FROM students WHERE age > 18;"}
{"question_id": "2", "completion": "SELECT COUNT(*) FROM courses;"}
{"question_id": "3", "completion": "or any unstructured output from the model: SELECT COUNT(*) FROM courses; even with special tokens <eos>"}
```
