"""Tests for the ``--model-args`` CLI flag (issue #34)."""

import argparse
import json
import sys
from unittest.mock import MagicMock

import pytest

from llmsql._cli.inference import merge_model_args, parse_model_args
from llmsql._cli.llmsql_cli import ParserCLI

# ---------------------------------------------------------------------------
# parse_model_args
# ---------------------------------------------------------------------------


def test_parse_model_args_type_coercion():
    parsed = parse_model_args(
        "a=1,b=true,c=float32,d=None,e=0.5,f=False,g=null,h=1e-5,i=-3"
    )
    assert parsed == {
        "a": 1,
        "b": True,
        "c": "float32",
        "d": None,
        "e": 0.5,
        "f": False,
        "g": None,
        "h": 1e-5,
        "i": -3,
    }
    assert type(parsed["a"]) is int
    assert type(parsed["b"]) is bool


def test_parse_model_args_lm_eval_example():
    assert parse_model_args("pretrained=EleutherAI/pythia-160m,dtype=float32") == {
        "pretrained": "EleutherAI/pythia-160m",
        "dtype": "float32",
    }


@pytest.mark.parametrize("value", ["", "   ", ",", " , ,"])
def test_parse_model_args_empty(value):
    assert parse_model_args(value) == {}


def test_parse_model_args_whitespace_and_trailing_comma():
    assert parse_model_args(" a = 1 , b= x ,") == {"a": 1, "b": "x"}


def test_parse_model_args_value_with_equals_sign():
    assert parse_model_args("url=http://h/?x=1&y=2,k=a=b") == {
        "url": "http://h/?x=1&y=2",
        "k": "a=b",
    }


def test_parse_model_args_empty_value_is_empty_string():
    assert parse_model_args("revision=") == {"revision": ""}


def test_parse_model_args_quoted_values_stay_strings():
    assert parse_model_args("revision=\"123\",x='true'") == {
        "revision": "123",
        "x": "true",
    }


@pytest.mark.parametrize("value", ["inf", "nan", "Infinity"])
def test_parse_model_args_non_numeric_words_stay_strings(value):
    assert parse_model_args(f"x={value}") == {"x": value}


@pytest.mark.parametrize(
    "value",
    ["novalue", "a=1,broken", "=1", " =1", "a=1,a=2"],
)
def test_parse_model_args_malformed(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_model_args(value)


# ---------------------------------------------------------------------------
# merge_model_args
# ---------------------------------------------------------------------------


def test_merge_model_args():
    assert merge_model_args(None, None) is None
    assert merge_model_args({}, None) is None
    assert merge_model_args({"a": 1}, None) == {"a": 1}
    assert merge_model_args(None, {"b": 2}) == {"b": 2}
    # JSON kwargs win on conflicting keys
    assert merge_model_args({"a": 1, "b": 1}, {"b": 2}) == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["llmsql", *argv])
    cli = ParserCLI()
    args = cli.parse_args()
    cli.execute(args)


@pytest.mark.parametrize("flag", ["--model-args", "--model_args"])
def test_transformers_model_args_reach_model_kwargs(monkeypatch, flag):
    mock_inference = MagicMock(return_value=[])
    monkeypatch.setattr("llmsql.inference_transformers", mock_inference)

    _run_cli(
        monkeypatch,
        [
            "inference",
            "transformers",
            "--model-or-model-name-or-path",
            "EleutherAI/pythia-160m",
            flag,
            "dtype=float32,low_cpu_mem_usage=true,revision=main",
        ],
    )

    mock_inference.assert_called_once()
    kwargs = mock_inference.call_args.kwargs
    assert kwargs["model_or_model_name_or_path"] == "EleutherAI/pythia-160m"
    assert kwargs["model_kwargs"] == {
        "dtype": "float32",
        "low_cpu_mem_usage": True,
        "revision": "main",
    }


def test_transformers_model_args_merged_with_json_kwargs(monkeypatch):
    mock_inference = MagicMock(return_value=[])
    monkeypatch.setattr("llmsql.inference_transformers", mock_inference)

    _run_cli(
        monkeypatch,
        [
            "inference",
            "transformers",
            "--model-or-model-name-or-path",
            "m",
            "--model-args",
            "revision=main,attn_implementation=eager",
            "--model-kwargs",
            json.dumps({"attn_implementation": "sdpa"}),
        ],
    )

    assert mock_inference.call_args.kwargs["model_kwargs"] == {
        "revision": "main",
        "attn_implementation": "sdpa",
    }


def test_transformers_without_model_args_keeps_none(monkeypatch):
    mock_inference = MagicMock(return_value=[])
    monkeypatch.setattr("llmsql.inference_transformers", mock_inference)

    _run_cli(
        monkeypatch,
        ["inference", "transformers", "--model-or-model-name-or-path", "m"],
    )

    assert mock_inference.call_args.kwargs["model_kwargs"] is None


@pytest.mark.parametrize("flag", ["--model-args", "--model_args"])
def test_vllm_model_args_reach_llm_kwargs(monkeypatch, flag):
    mock_inference = MagicMock(return_value=[])
    # Set the name in the package namespace directly: `monkeypatch.setattr` would
    # first read the old value, which triggers the lazy vLLM import.
    import llmsql

    monkeypatch.setitem(llmsql.__dict__, "inference_vllm", mock_inference)

    _run_cli(
        monkeypatch,
        [
            "inference",
            "vllm",
            "--model-name",
            "Qwen/Qwen2.5-1.5B-Instruct",
            flag,
            "gpu_memory_utilization=0.8,max_model_len=4096,enforce_eager=True",
            "--llm-kwargs",
            json.dumps({"max_model_len": 2048}),
        ],
    )

    mock_inference.assert_called_once()
    assert mock_inference.call_args.kwargs["llm_kwargs"] == {
        "gpu_memory_utilization": 0.8,
        "max_model_len": 2048,
        "enforce_eager": True,
    }


def test_vllm_lora_path_produces_lora_config(monkeypatch):
    mock_inference = MagicMock(return_value=[])
    import llmsql

    monkeypatch.setitem(llmsql.__dict__, "inference_vllm", mock_inference)

    _run_cli(
        monkeypatch,
        [
            "inference",
            "vllm",
            "--model-name",
            "m",
            "--lora-path",
            "/path/to/adapter",
        ],
    )

    mock_inference.assert_called_once()
    kwargs = mock_inference.call_args.kwargs
    assert kwargs["lora_config"] == {
        "lora_path": "/path/to/adapter",
        "lora_name": "default",
        "lora_scale": 1.0,
    }
    # `--lora-path` implies `enable_lora=True`
    assert kwargs["llm_kwargs"]["enable_lora"] is True


def test_vllm_lora_name_and_scale_are_forwarded(monkeypatch):
    mock_inference = MagicMock(return_value=[])
    import llmsql

    monkeypatch.setitem(llmsql.__dict__, "inference_vllm", mock_inference)

    _run_cli(
        monkeypatch,
        [
            "inference",
            "vllm",
            "--model-name",
            "m",
            "--lora-path",
            "/path/to/adapter",
            "--lora-name",
            "my-adapter",
            "--lora-scale",
            "0.5",
        ],
    )

    kwargs = mock_inference.call_args.kwargs
    assert kwargs["lora_config"] == {
        "lora_path": "/path/to/adapter",
        "lora_name": "my-adapter",
        "lora_scale": 0.5,
    }


def test_vllm_lora_config_json_implies_enable_lora(monkeypatch):
    mock_inference = MagicMock(return_value=[])
    import llmsql

    monkeypatch.setitem(llmsql.__dict__, "inference_vllm", mock_inference)

    lora_config = {"lora_path": "/path/to/adapter", "lora_name": "n"}
    _run_cli(
        monkeypatch,
        [
            "inference",
            "vllm",
            "--model-name",
            "m",
            "--lora-config",
            json.dumps(lora_config),
        ],
    )

    kwargs = mock_inference.call_args.kwargs
    assert kwargs["lora_config"] == lora_config
    assert kwargs["llm_kwargs"]["enable_lora"] is True


def test_vllm_without_lora_keeps_none(monkeypatch):
    mock_inference = MagicMock(return_value=[])
    import llmsql

    monkeypatch.setitem(llmsql.__dict__, "inference_vllm", mock_inference)

    _run_cli(monkeypatch, ["inference", "vllm", "--model-name", "m"])

    kwargs = mock_inference.call_args.kwargs
    assert kwargs["lora_config"] is None
    assert kwargs["llm_kwargs"] is None


@pytest.mark.parametrize(
    "argv",
    [
        ["transformers", "--model-or-model-name-or-path", "m", "--model-args", "x"],
        ["vllm", "--model-name", "m", "--model-args", "a=1,a=2"],
        # `pretrained` is rejected: the model is given by the backend's model flag
        [
            "transformers",
            "--model-or-model-name-or-path",
            "m",
            "--model-args",
            "pretrained=m",
        ],
    ],
)
def test_invalid_model_args_exit(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["llmsql", "inference", *argv])
    cli = ParserCLI()
    with pytest.raises(SystemExit) as exc:
        cli.parse_args()
    assert exc.value.code == 2
    assert "--model-args" in capsys.readouterr().err


def test_api_backend_has_no_model_args(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llmsql",
            "inference",
            "api",
            "--model-name",
            "m",
            "--base-url",
            "http://localhost",
            "--model-args",
            "a=1",
        ],
    )
    with pytest.raises(SystemExit):
        ParserCLI().parse_args()
