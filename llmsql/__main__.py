import argparse
import inspect
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmsql", description="LLMSQL CLI")
    subparsers = parser.add_subparsers(dest="command")

    # ================================================================
    # Inference command
    # ================================================================
    inference_examples = r"""
Examples:

  # 1️⃣ Run inference with Transformers backend
  llmsql inference --method transformers \
      --model-or-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
      --output-file outputs/preds_transformers.jsonl \
      --batch-size 8 \
      --num-fewshots 5

  # 2️⃣ Run inference with vLLM backend
  llmsql inference --method vllm \
      --model-name Qwen/Qwen2.5-1.5B-Instruct \
      --output-file outputs/preds_vllm.jsonl \
      --batch-size 8 \
      --num-fewshots 5

  # 3️⃣ Pass model-specific kwargs (for Transformers)
  llmsql inference --method transformers \
      --model-or-model-name-or-path meta-llama/Llama-3-8b-instruct \
      --output-file outputs/llama_preds.jsonl \
      --model-kwargs '{"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"}'

  # 4️⃣ Pass LLM init kwargs (for vLLM)
  llmsql inference --method vllm \
      --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 \
      --output-file outputs/mixtral_preds.jsonl \
      --llm-kwargs '{"max_model_len": 4096, "gpu_memory_utilization": 0.9}'

  # 5️⃣ Override generation parameters dynamically
  llmsql inference --method transformers \
      --model-or-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
      --output-file outputs/temp_0.9.jsonl \
      --temperature 0.9 \
      --generation-kwargs '{"do_sample": true, "top_p": 0.9, "top_k": 40}'
"""

    inf_parser = subparsers.add_parser(
        "inference",
        help="Run inference using either Transformers or vLLM backend.",
        description="Run SQL generation using a chosen inference method "
        "(either 'transformers' or 'vllm').",
        epilog=inference_examples,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    inf_parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["transformers", "vllm"],
        help="Inference backend to use ('transformers' or 'vllm').",
    )

    # ================================================================
    # Parse CLI
    # ================================================================
    args, extra = parser.parse_known_args()

    # ------------------------------------------------
    # Inference
    # ------------------------------------------------
    if args.command == "inference":
        if args.method == "vllm":
            from llmsql import inference_vllm as inference_fn
        elif args.method == "transformers":
            from llmsql import inference_transformers as inference_fn  # type: ignore
        else:
            raise ValueError(f"Unknown inference method: {args.method}")

        # Dynamically create parser from the function signature
        fn_parser = argparse.ArgumentParser(
            prog=f"llmsql inference --method {args.method}",
            description=f"Run inference using {args.method} backend",
        )

        sig = inspect.signature(inference_fn)
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                fn_parser.add_argument(
                    "--llm-kwargs",
                    default="{}",
                    help="Additional LLM kwargs as a JSON string, e.g. '{\"top_p\": 0.9}'",
                )
                fn_parser.add_argument(
                    "--generate-kwargs",
                    default="{}",
                    help="",
                )
                continue
            arg_name = f"--{name.replace('_', '-')}"
            default = param.default
            if default is inspect.Parameter.empty:
                fn_parser.add_argument(arg_name, required=True)
            else:
                if isinstance(default, bool):
                    fn_parser.add_argument(
                        arg_name,
                        action="store_true" if not default else "store_false",
                        help=f"(default: {default})",
                    )
                elif default is None:
                    fn_parser.add_argument(arg_name, type=str, default=None)
                else:
                    fn_parser.add_argument(
                        arg_name, type=type(default), default=default
                    )

        fn_args = fn_parser.parse_args(extra)
        fn_kwargs = vars(fn_args)

        if "llm_kwargs" in fn_kwargs and isinstance(fn_kwargs["llm_kwargs"], str):
            try:
                fn_kwargs["llm_kwargs"] = json.loads(fn_kwargs["llm_kwargs"])
            except json.JSONDecodeError:
                print("⚠️  Could not parse --llm-kwargs JSON, passing as string.")

        if fn_kwargs.get("model_kwargs") is not None:
            try:
                fn_kwargs["model_kwargs"] = json.loads(fn_kwargs["model_kwargs"])
            except json.JSONDecodeError:
                raise

        if fn_kwargs.get("generation_kwargs") is not None:
            try:
                fn_kwargs["generation_kwargs"] = json.loads(
                    fn_kwargs["generation_kwargs"]
                )
            except json.JSONDecodeError:
                raise

        print(f"🔹 Running {args.method} inference with arguments:")
        for k, v in fn_kwargs.items():
            print(f"  {k}: {v}")

        results = inference_fn(**fn_kwargs)
        print(f"✅ Inference complete. Generated {len(results)} results.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
