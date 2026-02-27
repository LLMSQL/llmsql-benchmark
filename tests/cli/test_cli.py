import json
import sys
from unittest.mock import AsyncMock

import pytest

from llmsql._cli.llmsql_cli import ParserCLI


@pytest.mark.asyncio
async def test_transformers_backend_called(monkeypatch):
    """
    Ensure transformers backend is correctly invoked.
    """
    # Mock backend function
    mock_inference = AsyncMock(return_value=[])

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
    assert call_kwargs["generation_kwargs"]["top_p"] == 0.9


@pytest.mark.asyncio
async def test_vllm_backend_called(monkeypatch):
    """
    Ensure vLLM backend is correctly invoked.
    """
    mock_inference = AsyncMock(return_value=[])

    monkeypatch.setattr(
        "llmsql.inference_vllm",
        mock_inference,
    )

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
