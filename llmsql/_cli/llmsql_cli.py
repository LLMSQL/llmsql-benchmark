import argparse
import textwrap

from llmsql._cli.inference import Inference


class ParserCLI:
    """Main CLI parser that manages all subcommands."""

    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser(
            prog="llmsql",
            description="LLMSQL CLI",
            epilog=textwrap.dedent("""
            Examples:

            # 1️⃣ Transformers backend
            llmsql inference transformers \
                --model-or-model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
                --output-file outputs/preds_transformers.jsonl \
                --batch-size 8 \
                --num-fewshots 5

            # 2️⃣ vLLM backend
            llmsql inference vllm \
                --model-name Qwen/Qwen2.5-1.5B-Instruct \
                --output-file outputs/preds_vllm.jsonl \
                --batch-size 8 \
                --num-fewshots 5

            # 3️⃣ Transformers with model kwargs
            llmsql inference transformers \
                --model-or-model-name-or-path meta-llama/Llama-3-8b-instruct \
                --model-kwargs '{"attn_implementation": "flash_attention_2"}'

            # 4️⃣ vLLM with LLM kwargs
            llmsql inference vllm \
                --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 \
                --llm-kwargs '{"max_model_len": 4096}'

            Visit https://github.com/LLMSQL/llmsql-benchmark for more
            """),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        self._parser.set_defaults(func=lambda args: self._parser.print_help())

        self._subparsers = self._parser.add_subparsers(
            dest="command",
            metavar="COMMAND",
            required=True,
        )

        Inference(self._subparsers)

    def parse_args(self) -> argparse.Namespace:
        """Parse CLI arguments."""
        return self._parser.parse_args()

    def execute(self, args: argparse.Namespace) -> None:
        """Execute selected command."""
        args.func(args)
