from unittest.mock import patch

import pytest

from llmsql._cli.llmsql_cli import LLMSQLCLI


# ------------------------------
# Helper to run CLI with args
# ------------------------------
def run_cli(cli, arglist):
    args = cli._parser.parse_args(arglist)
    return cli.execute(args)


# ------------------------------
# Tests for InferenceCommand
# ------------------------------
@patch("llmsql._cli.inference.inference_transformers")
@patch("llmsql._cli.inference.inference_vllm")
def test_inference_dispatch(mock_vllm, mock_transformers):
    mock_vllm.return_value = [{"sql": "SELECT 1"}]
    mock_transformers.return_value = [{"sql": "SELECT 2"}]

    cli = LLMSQLCLI()

    # vLLM backend
    run_cli(
        cli,
        [
            "inference",
            "vllm",
            "--model-name",
            "test-model",
        ],
    )
    mock_vllm.assert_called_once()
    mock_transformers.assert_not_called()
    mock_vllm.reset_mock()

    # Transformers backend
    run_cli(
        cli,
        [
            "inference",
            "transformers",
            "--model-or-model-name-or-path",
            "test-model",
        ],
    )
    mock_transformers.assert_called_once()
    mock_vllm.assert_not_called()


# ------------------------------
# Tests for EvaluationCommand
# ------------------------------
@patch("llmsql._cli.evaluation.evaluate")
def test_evaluation_dispatch(mock_evaluate):
    mock_evaluate.return_value = {"accuracy": 1.0}

    cli = LLMSQLCLI()

    # JSONL file path
    run_cli(cli, ["evaluate", "--outputs", "fake_path.jsonl"])
    mock_evaluate.assert_called_once()
    args_passed = mock_evaluate.call_args[1]
    assert args_passed["outputs"] == "fake_path.jsonl"

    mock_evaluate.reset_mock()

    # Inline JSON
    inline_json = '[{"id":1,"sql":"SELECT 1"}]'
    run_cli(cli, ["evaluate", "--outputs", inline_json])
    mock_evaluate.assert_called_once()
    args_passed = mock_evaluate.call_args[1]
    assert args_passed["outputs"] == [{"id": 1, "sql": "SELECT 1"}]


# ------------------------------
# Test CLI help outputs (optional)
# ------------------------------
def test_cli_help(capsys):
    cli = LLMSQLCLI()
    with pytest.raises(SystemExit):
        cli._parser.parse_args(["--help"])
    captured = capsys.readouterr()
    assert "LLMSQL: LLM-powered SQL generation toolkit" in captured.out
    assert "inference" in captured.out
    assert "evaluate" in captured.out
