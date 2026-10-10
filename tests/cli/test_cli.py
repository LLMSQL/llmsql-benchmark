import json
import sys
from unittest.mock import MagicMock

import pytest

import llmsql
from llmsql._cli.llmsql_cli import ParserCLI


@pytest.mark.asyncio
async def test_transformers_backend_called(monkeypatch):
    """
    Ensure transformers backend is correctly invoked.
    """
    # Mock backend function
    mock_inference = MagicMock(return_value=[])

    monkeypatch.setattr(
        "llmsql.inference_transformers",
        mock_inference,
    )

    test_args = [
        "llmsql",
        "inference",
        "transformers",
        "--model-or-model-name-or-path",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "--temperature",
        "0.9",
        "--generation-kwargs",
        json.dumps({"top_p": 0.9}),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    cli = ParserCLI()
    args = cli.parse_args()

    cli.execute(args)

    # Assert backend was called
    mock_inference.assert_called_once()

    call_kwargs = mock_inference.call_args.kwargs
    assert call_kwargs["model_or_model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert call_kwargs["temperature"] == 0.9
    assert call_kwargs["dtype"] == "auto"
    assert call_kwargs["trust_remote_code"] is False
    assert call_kwargs["generation_kwargs"]["top_p"] == 0.9


@pytest.mark.asyncio
async def test_vllm_backend_called(monkeypatch):
    """
    Ensure vLLM backend is correctly invoked.
    """
    pytest.importorskip("vllm")
    mock_inference = MagicMock(return_value=[])

    monkeypatch.setitem(llmsql.__dict__, "inference_vllm", mock_inference)

    test_args = [
        "llmsql",
        "inference",
        "vllm",
        "--model-name",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "--tensor-parallel-size",
        "2",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    cli = ParserCLI()
    args = cli.parse_args()

    cli.execute(args)

    mock_inference.assert_called_once()

    call_kwargs = mock_inference.call_args.kwargs
    assert call_kwargs["model_name"] == "mistralai/Mixtral-8x7B-Instruct-v0.1"
    assert call_kwargs["tensor_parallel_size"] == 2
    assert call_kwargs["trust_remote_code"] is False


def test_boolean_options():
    parser = ParserCLI()._parser
    cases = [
        (
            ["inference", "transformers", "--model-or-model-name-or-path", "m"],
            "trust_remote_code",
            False,
            "--trust-remote-code",
        ),
        (
            ["inference", "transformers", "--model-or-model-name-or-path", "m"],
            "do_sample",
            False,
            "--do-sample",
        ),
        (
            ["inference", "vllm", "--model-name", "m"],
            "trust_remote_code",
            False,
            "--trust-remote-code",
        ),
        (
            ["inference", "vllm", "--model-name", "m"],
            "use_chat_template",
            True,
            "--use-chat-template",
        ),
        (
            ["inference", "vllm", "--model-name", "m"],
            "do_sample",
            True,
            "--do-sample",
        ),
        (
            ["evaluate", "--outputs", "out.jsonl"],
            "show_mismatches",
            True,
            "--show-mismatches",
        ),
    ]
    for args, attribute, default, flag in cases:
        assert getattr(parser.parse_args(args), attribute) is default
        assert getattr(parser.parse_args([*args, flag]), attribute) is True
        assert getattr(parser.parse_args([*args, f"--no-{flag[2:]}"]), attribute) is False


@pytest.mark.asyncio
async def test_api_backend_called(monkeypatch):
    """
    Ensure API backend is correctly invoked.
    """
    mock_inference = MagicMock(return_value=[])

    monkeypatch.setattr(
        "llmsql.inference_api",
        mock_inference,
    )

    test_args = [
        "llmsql",
        "inference",
        "api",
        "--model-name",
        "gpt-4o-mini",
        "--base-url",
        "https://api.openai.com/v1",
        "--requests-per-minute",
        "30",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    cli = ParserCLI()
    args = cli.parse_args()

    cli.execute(args)

    mock_inference.assert_called_once()

    call_kwargs = mock_inference.call_args.kwargs
    assert call_kwargs["model_name"] == "gpt-4o-mini"
    assert call_kwargs["base_url"] == "https://api.openai.com/v1"
    assert call_kwargs["requests_per_minute"] == 30.0


@pytest.mark.asyncio
async def test_missing_backend_errors(monkeypatch):
    """
    Ensure missing backend fails.
    """
    test_args = ["llmsql", "inference"]

    monkeypatch.setattr(sys, "argv", test_args)

    cli = ParserCLI()

    with pytest.raises(SystemExit):
        cli.parse_args()


@pytest.mark.asyncio
async def test_invalid_json_kwargs(monkeypatch):
    """
    Invalid JSON should raise argparse error.
    """
    test_args = [
        "llmsql",
        "inference",
        "transformers",
        "--model-or-model-name-or-path",
        "test-model",
        "--generation-kwargs",
        "{invalid_json}",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    cli = ParserCLI()

    with pytest.raises(SystemExit):
        cli.parse_args()


@pytest.mark.asyncio
async def test_help_shows_without_crashing(monkeypatch, capsys):
    """
    Running with no args should print help.
    """
    test_args = ["llmsql"]

    monkeypatch.setattr(sys, "argv", test_args)

    cli = ParserCLI()

    with pytest.raises(SystemExit):
        cli.parse_args()

    captured = capsys.readouterr()
    assert "usage:" in captured.err.lower() or "usage:" in captured.out.lower()



@pytest.mark.asyncio
async def test_evaluate_command_called(monkeypatch):
    """
    Ensure the evaluate command is correctly invoked with arguments.
    """
    mock_evaluate = MagicMock(return_value={})

    monkeypatch.setattr(
        "llmsql._cli.evaluate.evaluate",
        mock_evaluate,
    )
    
    test_args = [
        "llmsql",
        "evaluate",
        "--outputs",
        "dummy_file.jsonl",
        "--show-mismatches",
        "--max-mismatches",
        "10"
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    
    cli = ParserCLI()
    args = cli.parse_args()
    cli.execute(args)
    
    mock_evaluate.assert_called_once() 

    call_kwargs = mock_evaluate.call_args.kwargs
    assert call_kwargs["outputs"] == "dummy_file.jsonl"
    assert call_kwargs["show_mismatches"] is True 
    assert call_kwargs["max_mismatches"] == 10