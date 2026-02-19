import pytest
from llmsql import evaluate
from llmsql.config.config import get_available_versions

VALID_LLMSQL_VERSIONS = [None] + get_available_versions()
INVALID_LLMSQL_VERSION = "1.1"


@pytest.mark.parametrize("version_arg", VALID_LLMSQL_VERSIONS)
def test_evaluate_runs_with_valid_versions(monkeypatch, tmp_path, version_arg):
    outputs_path = tmp_path / "outputs.jsonl"
    outputs_path.write_text('{"question_id":1,"completion":"SELECT 1"}')

    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text('{"question_id":1,"table_id":1,"question":"x","sql":"SELECT 1"}')

    monkeypatch.setattr("llmsql.utils.evaluation_utils.evaluate_sample", lambda *a, **k: (1, None, {}))
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)

    kwargs = {
        "outputs": str(outputs_path),
        "questions_path": str(questions_path),
        "db_path": tmp_path / "dummy.db",
        "show_mismatches": False,
    }

    if version_arg is not None:
        kwargs["version"] = version_arg

    # Should NOT raise
    evaluate(**kwargs)


def test_evaluate_raises_with_invalid_version(monkeypatch, tmp_path):
    outputs_path = tmp_path / "outputs.jsonl"
    outputs_path.write_text('{"question_id":1,"completion":"SELECT 1"}')

    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text('{"question_id":1,"table_id":1,"question":"x","sql":"SELECT 1"}')

    monkeypatch.setattr("llmsql.utils.evaluation_utils.evaluate_sample", lambda *a, **k: (1, None, {}))
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)

    kwargs = {
        "outputs": str(outputs_path),
        "questions_path": str(questions_path),
        "db_path": tmp_path / "dummy.db",
        "show_mismatches": False,
        "version": INVALID_LLMSQL_VERSION,
    }

    with pytest.raises(Exception):
        evaluate(**kwargs)