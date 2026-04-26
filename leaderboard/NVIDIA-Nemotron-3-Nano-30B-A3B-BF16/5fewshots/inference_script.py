import os

from dotenv import load_dotenv

from llmsql import evaluate, inference_vllm

load_dotenv()

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

results = inference_vllm(
    model_name=MODEL_NAME,
    output_file=f"{MODEL_NAME}_outputs.jsonl",
    batch_size=20000,
    tensor_parallel_size=4,
    do_sample=True,
    hf_token=os.environ["HF_TOKEN"],
    max_new_tokens=1024,
    temperature=1.0,
    sampling_kwargs={"top_p": 1.0},
    num_fewshots=5,
    seed=42,
    llm_kwargs={"dtype": "bfloat16"},
)

evaluate(results)
