import argparse
import json
from typing import Any

from llmsql._cli.subparsers import SubCommand
from llmsql.config.config import get_available_versions


def parse_limit(value: str) -> float | int:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("limit must be int or float") from err


_RESERVED_MODEL_ARGS = frozenset({"pretrained"})


def _coerce_model_arg_value(value: str) -> Any:
    """Convert a raw ``--model-args`` value to a Python object.

    Coercion rules (applied in order):

    * a value wrapped in matching single or double quotes is returned as a
      string with the quotes stripped and no further coercion
      (e.g. ``revision="123"`` stays the string ``"123"``);
    * ``true`` / ``false`` (case-insensitive) -> ``bool``;
    * ``none`` / ``null`` (case-insensitive) -> ``None``;
    * integers -> ``int``;
    * numeric literals containing a digit (``0.9``, ``1e-5``) -> ``float``;
    * anything else is kept as ``str``.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None

    try:
        return int(value)
    except ValueError:
        pass

    # Require a digit so that strings like "inf" or "nan" stay strings.
    if any(ch.isdigit() for ch in value):
        try:
            return float(value)
        except ValueError:
            pass

    return value


def parse_model_args(value: str) -> dict[str, Any]:
    """Parse an lm-evaluation-harness style ``"k1=v1,k2=v2"`` string into a dict.

    * Tokens are separated by commas; surrounding whitespace is ignored and
      empty tokens (e.g. from a trailing comma) are skipped.
    * Each token is split on the **first** ``=``, so values may contain ``=``.
    * Values are type-coerced with :func:`_coerce_model_arg_value`.
    * Values cannot contain commas; use the JSON flags
      (``--model-kwargs`` / ``--llm-kwargs``) for such values or nested objects.

    Raises:
        argparse.ArgumentTypeError: on tokens without ``=``, empty keys or
            duplicated keys.
    """
    result: dict[str, Any] = {}
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "=" not in token:
            raise argparse.ArgumentTypeError(
                f"invalid model argument {token!r}: expected 'key=value'"
            )
        key, raw_value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(
                f"invalid model argument {token!r}: empty key"
            )
        if key in result:
            raise argparse.ArgumentTypeError(f"duplicate model argument {key!r}")
        result[key] = _coerce_model_arg_value(raw_value.strip())
    return result


def _parse_backend_model_args(value: str) -> dict[str, Any]:
    """``parse_model_args`` plus rejection of keys the CLI handles elsewhere."""
    parsed = parse_model_args(value)
    reserved = sorted(_RESERVED_MODEL_ARGS.intersection(parsed))
    if reserved:
        raise argparse.ArgumentTypeError(
            f"{', '.join(reserved)} is not supported in --model-args; "
            "pass the model via the backend's model flag "
            "(--model-or-model-name-or-path / --model-name)"
        )
    return parsed


def merge_model_args(
    model_args: dict[str, Any] | None, json_kwargs: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Merge ``--model-args`` with the JSON kwargs flag.

    On key conflicts the JSON flag (``--model-kwargs`` / ``--llm-kwargs``) wins,
    since it is the more explicit, typed form. Returns ``None`` if both are empty.
    """
    if not model_args and not json_kwargs:
        return json_kwargs
    return {**(model_args or {}), **(json_kwargs or {})}


def _add_model_args_flag(parser: argparse.ArgumentParser, target: str) -> None:
    parser.add_argument(
        "--model-args",
        "--model_args",
        dest="model_args",
        type=_parse_backend_model_args,
        default=None,
        metavar="KEY=VALUE[,KEY=VALUE...]",
        help=(
            "Comma-separated keyword arguments for the model constructor, "
            "e.g. 'dtype=bfloat16,revision=main'. Values are coerced to "
            "int/float/bool/None where possible (quote a value to keep it a string). "
            f"Merged into {target}; on conflicting keys the JSON flag wins."
        ),
    )


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

        self._parser_api = inference_subparsers.add_parser(
            "api",
            help="Use OpenAI-compatible API backend",
        )

        self._add_args()

        self._parser_transformers.set_defaults(func=self._execute_transformers)
        self._parser_vllm.set_defaults(func=self._execute_vllm)
        self._parser_api.set_defaults(func=self._execute_api)

    def _add_args(self) -> None:
        # =========================
        # COMMON BENCHMARK ARGS
        # =========================
        def add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--version", default="2.0", choices=get_available_versions())
            parser.add_argument("--output-file", default="llm_sql_predictions.jsonl")
            parser.add_argument(
                "--workdir-path",
                default=None,
                help="Directory for benchmark downloads. If omitted, a temporary directory is used.",
            )
            parser.add_argument("--num-fewshots", type=int, default=5)
            parser.add_argument("--batch-size", type=int, default=8)
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--limit", type=parse_limit)

        # =========================
        # COMMON GENERATION ARGS
        # =========================
        def add_common_generation_args(
            parser: argparse.ArgumentParser, default_temp: float, default_sample: bool
        ) -> None:
            parser.add_argument("--max-new-tokens", type=int, default=256)
            parser.add_argument("--temperature", type=float, default=default_temp)
            parser.add_argument(
                "--do-sample",
                action=argparse.BooleanOptionalAction,
                default=default_sample,
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
            action=argparse.BooleanOptionalAction,
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
        _add_model_args_flag(self._parser_transformers, "--model-kwargs")
        self._parser_transformers.add_argument(
            "--tokenizer-kwargs",
            type=json.loads,
            help="JSON string for tokenizer kwargs",
        )
        self._parser_transformers.add_argument("--chat-template")

        add_common_generation_args(self._parser_transformers, 0.0, False)
        self._parser_transformers.add_argument("--top-p", type=float, default=1.0)
        self._parser_transformers.add_argument("--top-k", type=int, default=50)
        self._parser_transformers.add_argument(
            "--generation-kwargs",
            type=json.loads,
            help="JSON string for generate() kwargs",
        )

        add_common_benchmark_args(self._parser_transformers)

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
            action=argparse.BooleanOptionalAction,
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
        _add_model_args_flag(self._parser_vllm, "--llm-kwargs")
        self._parser_vllm.add_argument(
            "--use-chat-template",
            action=argparse.BooleanOptionalAction,
            default=True,
        )

        add_common_generation_args(self._parser_vllm, 1.0, True)
        self._parser_vllm.add_argument(
            "--sampling-kwargs",
            type=json.loads,
            help="JSON string for SamplingParams kwargs",
        )

        add_common_benchmark_args(self._parser_vllm)

        # =========================
        # OpenAI-compatible API
        # =========================
        self._parser_api.add_argument(
            "--model-name",
            required=True,
            help="Target model name expected by the API",
        )
        self._parser_api.add_argument(
            "--base-url",
            required=True,
            help="API base URL, e.g. https://api.openai.com/v1",
        )
        self._parser_api.add_argument(
            "--endpoint",
            default="chat/completions",
            help="Completion endpoint path relative to --base-url",
        )
        self._parser_api.add_argument("--api-key")
        self._parser_api.add_argument("--timeout", type=float, default=120.0)
        self._parser_api.add_argument(
            "--requests-per-minute",
            type=float,
            help="Rate limit for API requests",
        )
        self._parser_api.add_argument(
            "--api-kwargs",
            type=json.loads,
            help="JSON string merged into API request payload",
        )
        self._parser_api.add_argument(
            "--request-headers",
            type=json.loads,
            help="JSON string merged into HTTP request headers",
        )
        add_common_benchmark_args(self._parser_api)

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
            model_kwargs=merge_model_args(args.model_args, args.model_kwargs),
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
            llm_kwargs=merge_model_args(args.model_args, args.llm_kwargs),
            use_chat_template=args.use_chat_template,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            sampling_kwargs=args.sampling_kwargs,
            version=args.version,
            output_file=args.output_file,
            workdir_path=args.workdir_path,
            limit=args.limit,
            num_fewshots=args.num_fewshots,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    @staticmethod
    def _execute_api(args: argparse.Namespace) -> None:
        from llmsql import inference_api

        inference_api(
            model_name=args.model_name,
            base_url=args.base_url,
            endpoint=args.endpoint,
            api_key=args.api_key,
            timeout=args.timeout,
            requests_per_minute=args.requests_per_minute,
            api_kwargs=args.api_kwargs,
            request_headers=args.request_headers,
            version=args.version,
            output_file=args.output_file,
            workdir_path=args.workdir_path,
            limit=args.limit,
            num_fewshots=args.num_fewshots,
            seed=args.seed,
        )
