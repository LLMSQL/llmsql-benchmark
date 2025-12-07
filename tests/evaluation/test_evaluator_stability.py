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


@pytest.mark.asyncio
async def test_evaluate_with_jsonl_file(monkeypatch, temp_dir, dummy_db_file):
    # Create fake questions.jsonl
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {"question_id": 1, "table_id": 1, "question": "Sample", "sql": "SELECT 1"}
        )
    )

    # Create fake outputs.jsonl
    outputs_path = temp_dir / "outputs.jsonl"
    outputs_path.write_text(json.dumps({"question_id": 1, "completion": "SELECT 1"}))

    # Mock dependencies
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.evaluate_sample",
        lambda *a, **k: (1, None, {"pred_none": 0, "gold_none": 0, "sql_error": 0}),
    )
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.connect_sqlite", lambda db: dummy_db_file
    )
    monkeypatch.setattr(
        "llmsql.utils.utils.save_json_report", lambda path, report: None
    )

    report = evaluate(
        outputs=str(outputs_path),
        questions_path=str(questions_path),
        db_path=dummy_db_file,
        show_mismatches=False,
    )

    assert report["total"] == 1
    assert report["matches"] == 1
    assert report["accuracy"] == 1.0
    assert report["input_mode"] == "jsonl_path"


@pytest.mark.asyncio
async def test_evaluate_with_dict_list(monkeypatch, temp_dir, dummy_db_file):
    # Prepare fake questions dict
    questions_path = temp_dir / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {"question_id": 1, "table_id": 1, "question": "Sample", "sql": "SELECT 1"}
        )
    )

    # Output as a list of dicts
    outputs_list = [{"question_id": 1, "completion": "SELECT 1"}]

    # Mock dependencies
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.evaluate_sample",
        lambda *a, **k: (1, None, {"pred_none": 0, "gold_none": 0, "sql_error": 0}),
    )
    monkeypatch.setattr("llmsql.utils.rich_utils.log_mismatch", lambda **k: None)
    monkeypatch.setattr("llmsql.utils.rich_utils.print_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        "llmsql.utils.evaluation_utils.connect_sqlite", lambda db: dummy_db_file
    )
    monkeypatch.setattr(
        "llmsql.utils.utils.load_jsonl_dict_by_key",
        lambda path, key: {1: {"question_id": 1}},
    )
    monkeypatch.setattr(
        "llmsql.utils.utils.save_json_report", lambda path, report: None
    )

    report = evaluate(
        outputs=outputs_list,
        questions_path=str(questions_path),
        db_path=dummy_db_file,
        show_mismatches=False,
    )

    assert report["total"] == 1
    assert report["matches"] == 1
    assert report["accuracy"] == 1.0
    assert report["input_mode"] == "dict_list"


def test_evaluate_with_list_outputs(mock_utils, mocker):
    outputs = [{"question_id": 1, "completion": "SELECT 1"}]

    report = evaluate(outputs, workdir_path=str(mock_utils))

    assert report["total"] == 1
    assert report["matches"] == 1
    assert report["accuracy"] == 1.0
    assert report["input_mode"] == "dict_list"


def test_evaluate_with_jsonl_path(mock_utils, mocker):
    # fake jsonl file
    jsonl_path = mock_utils / "preds.jsonl"
    jsonl_path.write_text("dummy", encoding="utf-8")

    report = evaluate(str(jsonl_path), workdir_path=str(mock_utils))

    assert report["total"] == 1
    assert report["input_mode"] == "jsonl_path"


def test_missing_workdir_and_no_questions_path_raises():
    with pytest.raises(ValueError):
        evaluate(
            outputs=[{"question_id": 1, "completion": "x"}],
            workdir_path=None,
            questions_path=None,
        )


def test_missing_workdir_and_no_db_path_raises():
    with pytest.raises(ValueError):
        evaluate(
            outputs=[{"question_id": 1, "completion": "x"}],
            workdir_path=None,
            db_path=None,
        )


def test_download_occurs_if_files_missing(mock_utils, mocker):
    dl = mocker.patch("llmsql.evaluation.evaluate.download_benchmark_file")

    evaluate(
        [{"question_id": 1, "completion": "SELECT 1"}],
        workdir_path=str(mock_utils),
        questions_path=None,
        db_path=None,
    )

    assert dl.call_count == 2  # questions + sqlite


def test_saves_report_with_auto_filename(mock_utils, mocker):
    save = mocker.patch("llmsql.evaluation.evaluate.save_json_report")

    report = evaluate(
        [{"question_id": 1, "completion": "SELECT 1"}],
        workdir_path=str(mock_utils),
        save_report=None,
    )

    # automatic UUID-based filename
    args, kwargs = save.call_args
    auto_filename = args[0]
    assert auto_filename.startswith("evaluation_results_")
    assert auto_filename.endswith(".json")

    assert report["total"] == 1


def test_mismatch_handling(mock_utils, mocker):
    """Test branch where a mismatch is returned."""
    mocker.patch(
        "llmsql.evaluation.evaluate.evaluate_sample",
        return_value=(
            0,
            {"info": "bad"},
            {"pred_none": 0, "gold_none": 0, "sql_error": 0},
        ),
    )

    log_mis = mocker.patch("llmsql.evaluation.evaluate.log_mismatch")

    report = evaluate(
        [{"question_id": 1, "completion": "SELECT X"}],
        workdir_path=str(mock_utils),
        max_mismatches=3,
    )

    assert report["matches"] == 0
    assert len(report["mismatches"]) == 1
    log_mis.assert_called_once()
