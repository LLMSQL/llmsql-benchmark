import json

import pytest

from llmsql import evaluate


@pytest.mark.asyncio
async def test_evaluate_with_mock(monkeypatch, temp_dir, dummy_db_file):
    # Fake questions.jsonl
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": 1,
                "table_id": 1,
                "question": "Sample quesiton",
                "sql": "SELECT 1",
            }
        )
    )

    # Fake outputs.jsonl
    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(json.dumps({"question_id": 1, "completion": "SELECT 1"}))

    # Monkeypatch dependencies
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.evaluate_sample",
        lambda *a, **k: (1, None, {"pred_none": 0, "gold_none": 0, "sql_error": 0}),
    )
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)

    report = evaluate(
        outputs=str(outputs_path),
        questions_path=str(questions_path),
        db_path=dummy_db_file,
        show_mismatches=False,
    )

    assert report["total"] == 1
    assert report["matches"] == 1
    assert report["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_evaluate_saves_report(monkeypatch, temp_dir, dummy_db_file):
    """Test that save_report parameter creates a JSON report file."""

    # Setup test files
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {"question_id": 1, "table_id": 1, "question": "Test", "sql": "SELECT 1"}
        )
    )

    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(json.dumps({"question_id": 1, "completion": "SELECT 1"}))

    report_path = temp_dir / "report.json"

    # Mock dependencies
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.evaluate_sample",
        lambda *a, **k: (1, None, {"pred_none": 0, "gold_none": 0, "sql_error": 0}),
    )
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)

    evaluate(
        outputs=str(outputs_path),
        questions_path=str(questions_path),
        db_path=dummy_db_file,
        save_report=str(report_path),
        show_mismatches=False,
    )

    # Verify report file was created
    assert report_path.exists()
    with open(report_path, encoding="utf-8") as f:
        saved_report = json.load(f)
    assert saved_report["total"] == 1
    assert saved_report["accuracy"] == 1.0
