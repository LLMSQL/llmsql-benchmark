import json
import shutil

import pytest

from llmsql import evaluate
from llmsql.config.config import get_available_versions

VALID_LLMSQL_VERSIONS = [None] + get_available_versions()
INVALID_LLMSQL_VERSION = "1.1"


@pytest.mark.asyncio
@pytest.mark.parametrize("version_arg", VALID_LLMSQL_VERSIONS)
async def test_evaluate_runs_with_valid_versions(
    monkeypatch, temp_dir, dummy_db_file, version_arg
):
    # Fake questions.jsonl
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": 1,
                "table_id": 1,
                "question": "Sample",
                "sql": "SELECT 1",
            }
        )
    )

    shutil.copy(dummy_db_file, temp_dir / "sqlite_tables.db")

    # Fake outputs.jsonl
    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(json.dumps({"question_id": 1, "completion": "SELECT 1"}))

    # Monkeypatch exactly like reference file
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.evaluate_sample",
        lambda *a, **k: (
            1,
            None,
            {"pred_none": 0, "gold_none": 0, "sql_error": 0},
        ),
    )
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)

    kwargs = {
        "outputs": str(outputs_path),
        "workdir_path": str(temp_dir),
        "show_mismatches": False,
    }

    if version_arg is not None:
        kwargs["version"] = version_arg

    evaluate(**kwargs)


@pytest.mark.asyncio
async def test_evaluate_raises_with_invalid_version(
    monkeypatch, temp_dir, dummy_db_file
):
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": 1,
                "table_id": 1,
                "question": "Sample",
                "sql": "SELECT 1",
            }
        )
    )
    shutil.copy(dummy_db_file, temp_dir / "sqlite_tables.db")

    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(json.dumps({"question_id": 1, "completion": "SELECT 1"}))

    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.evaluate_sample",
        lambda *a, **k: (
            1,
            None,
            {"pred_none": 0, "gold_none": 0, "sql_error": 0},
        ),
    )
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)

    with pytest.raises(ValueError):
        evaluate(
            outputs=str(outputs_path),
            workdir_path=str(temp_dir),
            show_mismatches=False,
            version=INVALID_LLMSQL_VERSION,
        )
