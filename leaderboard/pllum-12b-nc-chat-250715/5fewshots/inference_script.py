import os

from dotenv import load_dotenv

from llmsql import evaluate, inference_vllm

load_dotenv()

MODEL_NAME = "CYFRAGOVPL/pllum-12b-nc-chat-250715"

results = inference_vllm(
    model_name=MODEL_NAME,
    output_file=f"{MODEL_NAME}_outputs.jsonl",
    batch_size=20000,
    tensor_parallel_size=4,
    do_sample=False,
    hf_token=os.environ["HF_TOKEN"],
    max_new_tokens=256,
    temperature=0.0,
    num_fewshots=5,
    seed=42,
    llm_kwargs={"dtype": "bfloat16"},
)

evaluate(results)
