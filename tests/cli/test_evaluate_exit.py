"""Regression tests for evaluate CLI failure handling."""

import sys

import pytest

from llmsql._cli.llmsql_cli import ParserCLI


def test_evaluate_missing_outputs_exits_nonzero(tmp_path, monkeypatch, caplog):
    """A missing outputs file must fail the process instead of exiting 0."""
    missing = tmp_path / "does-not-exist.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["llmsql", "evaluate", "--outputs", str(missing)],
    )

    import llmsql.evaluation.evaluate as evaluation

    monkeypatch.setattr(evaluation, "resolve_workdir_path", lambda path: tmp_path)
    monkeypatch.setattr(
        evaluation, "_maybe_download", lambda *args, **kwargs: tmp_path / "unused"
    )
    monkeypatch.setattr(evaluation, "load_jsonl_dict_by_key", lambda *args, **kwargs: {})

    cli = ParserCLI()
    args = cli.parse_args()

    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit) as exc:
            cli.execute(args)

    assert exc.value.code == 1
    assert any(
        "Error during evaluation" in record.message and "does-not-exist.jsonl" in record.message
        for record in caplog.records
    )
