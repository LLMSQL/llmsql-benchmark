import argparse
import json
import textwrap
from typing import Any
import platform

from llmsql import inference_transformers
from llmsql import VLLM_SUPPORTED_PLATFORMS
from llmsql.loggers.logging_config import log

CURRENT_OS = platform.system()
IS_OS_SUPPORTED = CURRENT_OS in VLLM_SUPPORTED_PLATFORMS


def _json_arg(value: str | None) -> dict[str, Any] | Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as dec:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {value}") from dec


class InferenceCommand:
    """CLI registration + dispatch for `llmsql inference`."""

    @staticmethod
    def register(subparsers: Any) -> None:
        """Register method"""

        inference_examples = textwrap.dedent("""
    Examples:

      # 1️⃣ Run inference with Transformers backend
      llmsql inference --method transformers \\
          --model-or-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \\
          --output-file outputs/preds_transformers.jsonl \\
          --batch-size 8 \\
          --num-fewshots 5

      # 2️⃣ Run inference with vLLM backend
      llmsql inference --method vllm \\
          --model-name Qwen/Qwen2.5-1.5B-Instruct \\
          --output-file outputs/preds_vllm.jsonl \\
          --batch-size 8 \\
          --num-fewshots 5

      # 3️⃣ Pass model init kwargs (Transformers)
      llmsql inference --method transformers \\
          --model-or-model-name-or-path meta-llama/Llama-3-8b-instruct \\
          --output-file outputs/llama_preds.jsonl \\
          --model-kwargs '{"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"}'

      # 4️⃣ Pass LLM init kwargs (vLLM)
      llmsql inference --method vllm \\
          --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 \\
          --output-file outputs/mixtral_preds.jsonl \\
          --llm-kwargs '{"max_model_len": 4096, "gpu_memory_utilization": 0.9}'

      # 5️⃣ Override generation parameters dynamically (Transformers)
      llmsql inference --method transformers \\
          --model-or-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \\
          --output-file outputs/temp_0.9.jsonl \\
          --temperature 0.9 \\
          --generation-kwargs '{"do_sample": true, "top_p": 0.9, "top_k": 40}'

See `llmsql inference --method vllm` or `llmsql inference --method transformers` for more information about the arguments.
""")
        parser = subparsers.add_parser(
            "inference",
            help="Run inference using Transformers or vLLM backend.",
            description="Run SQL generation using a chosen inference backend.",
            epilog=inference_examples,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        backend_subparsers = parser.add_subparsers(
            title="Backend",
            description="Select backend for inference",
            dest="backend",
            required=True,
        )

        InferenceCommand._register_transformers_parser(backend_subparsers)
        if IS_OS_SUPPORTED:
            InferenceCommand._register_vllm_parser(backend_subparsers)

        parser.set_defaults(func=InferenceCommand.dispatch)

    # ------------------------------------------------------------------
    @staticmethod
    def dispatch(args: Any) -> None:
        """Dispatch to the correct backend"""
        kwargs = vars(args).copy()  # copy to avoid modifying original
        # Remove keys not used by the backend functions
        kwargs.pop("func", None)      # added by set_defaults
        kwargs.pop("command", None)   # top-level command
        kwargs.pop("backend", None)   # subcommand itself

        if args.backend == "vllm":
            if not IS_OS_SUPPORTED:
                raise RuntimeError(
                    f"vLLM backend is not supported on {CURRENT_OS}. "
                    f"Supported platforms: {', '.join(VLLM_SUPPORTED_PLATFORMS)}. "
                    "Please choose the 'transformers' backend."
                )
            from llmsql import inference_vllm
            results = inference_vllm(**kwargs)
        else:
            results = inference_transformers(**kwargs)

        log.info(f"Generated {len(results)} results.")

    # ------------------------------------------------------------------
    @staticmethod
    def _register_vllm_parser(subparsers: Any) -> None:
        """vLLM backend arguments"""
        parser = subparsers.add_parser(
            "vllm",
            help="Inference using vLLM backend",
            description="Run SQL generation using the vLLM backend",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # === Model Loading Parameters ===
        parser.add_argument(
            "--model-name", required=True, help="Hugging Face model name or path."
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            default=True,
            help="Whether to trust remote code when loading the model (default: True).",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            type=int,
            default=1,
            help="Number of GPUs to use for tensor parallelism (default: 1).",
        )
        parser.add_argument(
            "--hf-token",
            type=str,
            default=None,
            help="Hugging Face authentication token (optional).",
        )
        parser.add_argument(
            "--llm-kwargs",
            type=_json_arg,
            default=None,
            help=(
                "Additional keyword arguments for vllm.LLM(). "
                "Note: 'model', 'tokenizer', 'tensor_parallel_size', 'trust_remote_code' "
                "are handled separately and will override values here."
            ),
        )
        parser.add_argument(
            "--use-chat-template",
            action="store_true",
            default=True,
            help="Whether to use a chat template for prompting (default: True).",
        )

        # === Generation Parameters ===
        parser.add_argument(
            "--max-new-tokens",
            type=int,
            default=256,
            help="Maximum number of tokens to generate per sequence (default: 256).",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="Sampling temperature (0.0 = greedy, default: 1.0).",
        )
        parser.add_argument(
            "--do-sample",
            action="store_true",
            default=True,
            help="Whether to use sampling instead of greedy decoding (default: True).",
        )
        parser.add_argument(
            "--sampling-kwargs",
            type=_json_arg,
            default=None,
            help=(
                "Additional arguments for vllm.SamplingParams(). "
                "Note: 'temperature', 'max_tokens' are handled separately and will override values here."
            ),
        )

        # === Benchmark / Input-Output Parameters ===
        parser.add_argument(
            "--output-file",
            default="llm_sql_predictions.jsonl",
            help="Path to write outputs (will be overwritten if exists).",
        )
        parser.add_argument(
            "--questions-path",
            type=str,
            default=None,
            help="Path to questions.jsonl (auto-downloads if missing).",
        )
        parser.add_argument(
            "--tables-path",
            type=str,
            default=None,
            help="Path to tables.jsonl (auto-downloads if missing).",
        )
        parser.add_argument(
            "--workdir-path",
            default="./llmsql_workdir",
            help="Directory to store downloaded data and temporary files.",
        )
        parser.add_argument(
            "--num-fewshots",
            type=int,
            default=5,
            help="Number of few-shot examples to use (0, 1, or 5; default: 5).",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Number of questions per generation batch (default: 8).",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: 42).",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _register_transformers_parser(subparsers: Any) -> None:
        """vLLM backend arguments"""
        parser = subparsers.add_parser(
            "transformers",
            help="Inference using vLLM backend",
            description="Run SQL generation using the vLLM backend",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # === Model Loading Parameters ===
        parser.add_argument(
            "--model-or-model-name-or-path",
            required=True,
            help="Model object or Hugging Face model name/path.",
        )
        parser.add_argument(
            "--tokenizer-or-name",
            default=None,
            help="Tokenizer object or Hugging Face tokenizer name/path.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            default=True,
            help="Whether to trust remote code when loading the model (default: True).",
        )
        parser.add_argument(
            "--dtype",
            default="float16",
            help="Torch dtype for model (default: float16).",
        )
        parser.add_argument(
            "--device-map",
            default="auto",
            help="Device placement strategy, e.g., 'auto', 'cuda:0', or dict mapping (default: 'auto').",
        )
        parser.add_argument(
            "--hf-token",
            type=str,
            default=None,
            help="Hugging Face authentication token (optional).",
        )
        parser.add_argument(
            "--model-kwargs",
            type=_json_arg,
            default=None,
            help=(
                "Additional arguments for AutoModelForCausalLM.from_pretrained(). "
                "Note: 'dtype', 'device_map', 'trust_remote_code', 'token' are handled separately and will override values here."
            ),
        )

        # === Tokenizer Loading Parameters ===
        parser.add_argument(
            "--tokenizer-kwargs",
            type=_json_arg,
            default=None,
            help=(
                "Additional arguments for AutoTokenizer.from_pretrained(). 'padding_side' defaults to 'left'. "
                "Note: 'trust_remote_code', 'token' are handled separately and will override values here."
            ),
        )

        # === Prompt & Chat Parameters ===
        parser.add_argument(
            "--chat-template",
            type=str,
            default=None,
            help="Optional chat template to apply before tokenization.",
        )

        # === Generation Parameters ===
        parser.add_argument(
            "--max-new-tokens",
            type=int,
            default=256,
            help="Maximum number of tokens to generate per sequence (default: 256).",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.0,
            help="Sampling temperature (0.0 = greedy, default: 0.0).",
        )
        parser.add_argument(
            "--do-sample",
            action="store_true",
            default=False,
            help="Whether to use sampling instead of greedy decoding (default: False).",
        )
        parser.add_argument(
            "--top-p",
            type=float,
            default=1.0,
            help="Nucleus sampling parameter (default: 1.0).",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=50,
            help="Top-k sampling parameter (default: 50).",
        )
        parser.add_argument(
            "--generation-kwargs",
            type=_json_arg,
            default=None,
            help=(
                "Additional arguments for model.generate(). "
                "Note: 'max_new_tokens', 'temperature', 'do_sample', 'top_p', 'top_k' are handled separately and will override values here."
            ),
        )

        # === Benchmark / Input-Output Parameters ===
        parser.add_argument(
            "--output-file",
            default="llm_sql_predictions.jsonl",
            help="Output JSONL file path for completions.",
        )
        parser.add_argument(
            "--questions-path",
            type=str,
            default=None,
            help="Path to benchmark questions JSONL (optional, auto-downloads if missing).",
        )
        parser.add_argument(
            "--tables-path",
            type=str,
            default=None,
            help="Path to benchmark tables JSONL (optional, auto-downloads if missing).",
        )
        parser.add_argument(
            "--workdir-path",
            default="./llmsql_workdir",
            help="Working directory path for downloads and temporary files.",
        )
        parser.add_argument(
            "--num-fewshots",
            type=int,
            default=5,
            help="Number of few-shot examples to use (0, 1, or 5; default: 5).",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Batch size for inference (default: 8).",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: 42).",
        )
