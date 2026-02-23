import argparse
import json
from typing import Any

from llmsql._cli.subparsers import SubCommand


def parse_limit(value: str) -> float | int:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("limit must be int or float") from err


class Inference(SubCommand):
    """Command for running language model evaluation."""

    def __init__(
        self, subparsers: argparse._SubParsersAction, *args: Any, **kwargs: Any
    ) -> None:
        self._parser = subparsers.add_parser(
            "inference",
            help="Run inference",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        inference_subparsers = self._parser.add_subparsers(dest="method", required=True)

        self._parser_transformers = inference_subparsers.add_parser(
            "transformers",
            help="Use HuggingFace Transformers backend",
        )

        self._parser_vllm = inference_subparsers.add_parser(
            "vllm",
            help="Use vLLM backend",
        )

        self._add_args()

        self._parser_transformers.set_defaults(func=self._execute_transformers)
        self._parser_vllm.set_defaults(func=self._execute_vllm)

    def _add_args(self) -> None:
        # =========================
        # COMMON BENCHMARK ARGS
        # =========================
        def add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--version", default="2.0", choices=["1.0", "2.0"])
            parser.add_argument("--output-file", default="llm_sql_predictions.jsonl")
            parser.add_argument("--questions-path")
            parser.add_argument("--tables-path")
            parser.add_argument("--workdir-path", default="./workdir")
            parser.add_argument("--num-fewshots", type=int, default=5)
            parser.add_argument("--batch-size", type=int, default=8)
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--limit", type=parse_limit)

        # =========================
        # COMMON GENERATION ARGS
        # =========================
        def add_common_generation_args(
            parser: argparse.ArgumentParser, default_temp: float, default_sample: int
        ) -> None:
            parser.add_argument("--max-new-tokens", type=int, default=256)
            parser.add_argument("--temperature", type=float, default=default_temp)
            parser.add_argument(
                "--do-sample", action="store_true", default=default_sample
            )

        # =========================
        # TRANSFORMERS
        # =========================
        self._parser_transformers.add_argument(
            "--model-or-model-name-or-path",
            required=True,
            help="HF model name or local path",
        )

        self._parser_transformers.add_argument("--tokenizer-or-name")

        self._parser_transformers.add_argument(
            "--trust-remote-code",
            action="store_true",
            default=True,
        )
        self._parser_transformers.add_argument("--dtype", default="float16")
        self._parser_transformers.add_argument("--device-map", default="auto")
        self._parser_transformers.add_argument("--hf-token")
        self._parser_transformers.add_argument(
            "--model-kwargs",
            type=json.loads,
            help="JSON string for AutoModel kwargs",
        )
        self._parser_transformers.add_argument(
            "--tokenizer-kwargs",
            type=json.loads,
            help="JSON string for tokenizer kwargs",
        )
        self._parser_transformers.add_argument("--chat-template")

        # Generation
        add_common_generation_args(self._parser_transformers, 0.0, False)
        self._parser_transformers.add_argument("--top-p", type=float, default=1.0)
        self._parser_transformers.add_argument("--top-k", type=int, default=50)
        self._parser_transformers.add_argument(
            "--generation-kwargs",
            type=json.loads,
            help="JSON string for generate() kwargs",
        )

        add_common_benchmark_args(self._parser_transformers)

        self._parser_transformers.set_defaults(func=self._execute_transformers)

        # =========================
        # vLLM
        # =========================
        self._parser_vllm.add_argument(
            "--model-name",
            required=True,
            help="HF model name or path",
        )

        self._parser_vllm.add_argument(
            "--trust-remote-code",
            action="store_true",
            default=True,
        )
        self._parser_vllm.add_argument(
            "--tensor-parallel-size",
            type=int,
            default=1,
        )
        self._parser_vllm.add_argument("--hf-token")
        self._parser_vllm.add_argument(
            "--llm-kwargs",
            type=json.loads,
            help="JSON string for vllm.LLM kwargs",
        )
        self._parser_vllm.add_argument(
            "--use-chat-template",
            action="store_true",
            default=True,
        )

        add_common_generation_args(self._parser_vllm, 1.0, True)
        self._parser_vllm.add_argument(
            "--sampling-kwargs",
            type=json.loads,
            help="JSON string for SamplingParams kwargs",
        )

        add_common_benchmark_args(self._parser_vllm)

        self._parser_vllm.set_defaults(func=self._execute_vllm)

    @staticmethod
    def _execute_transformers(args: argparse.Namespace) -> None:
        from llmsql import inference_transformers

        inference_transformers(
            model_or_model_name_or_path=args.model_or_model_name_or_path,
            tokenizer_or_name=args.tokenizer_or_name,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            device_map=args.device_map,
            hf_token=args.hf_token,
            model_kwargs=args.model_kwargs,
            tokenizer_kwargs=args.tokenizer_kwargs,
            chat_template=args.chat_template,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            generation_kwargs=args.generation_kwargs,
            version=args.version,
            output_file=args.output_file,
            questions_path=args.questions_path,
            tables_path=args.tables_path,
            workdir_path=args.workdir_path,
            num_fewshots=args.num_fewshots,
            batch_size=args.batch_size,
            limit=args.limit,
            seed=args.seed,
        )

    @staticmethod
    def _execute_vllm(args: argparse.Namespace) -> None:
        from llmsql import inference_vllm

        inference_vllm(
            model_name=args.model_name,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            hf_token=args.hf_token,
            llm_kwargs=args.llm_kwargs,
            use_chat_template=args.use_chat_template,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            sampling_kwargs=args.sampling_kwargs,
            version=args.version,
            output_file=args.output_file,
            questions_path=args.questions_path,
            tables_path=args.tables_path,
            workdir_path=args.workdir_path,
            limit=args.limit,
            num_fewshots=args.num_fewshots,
            batch_size=args.batch_size,
            seed=args.seed,
        )
