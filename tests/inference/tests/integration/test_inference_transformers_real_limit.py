import json
from pathlib import Path
import tempfile

import pytest

from llmsql.inference.inference_transformers import inference_transformers


def _write_jsonl(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


@pytest.mark.integration
def test_real_inference_limit():
    """
    Real integration test:
    - real transformers backend
    - real HF model (Pythia-14m)
    - real inference pipeline
    - verifies limit argument
    - verifies output format
    """

    LIMIT = 50
    MODEL_NAME = "EleutherAI/pythia-14m"
    TOKENIER_NAME = "EleutherAI/pythia-14m"

    questions = [
        {
            "question_id": f"q{i}",
            "table_id": "t1",
            "question": f"What is row {i}?",
        }
        for i in range(100)
    ]

    tables = [
        {
            "table_id": "t1",
            "header": ["id", "name"],
            "types": ["int", "text"],
            "rows": [[1, "Alice"], [2, "Bob"]],
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        questions_file = tmpdir_path / "questions.jsonl"
        tables_file = tmpdir_path / "tables.jsonl"
        output_file = tmpdir_path / "outputs.jsonl"

        _write_jsonl(questions, questions_file)
        _write_jsonl(tables, tables_file)

        try:
            results = inference_transformers(
                model_or_model_name_or_path=MODEL_NAME,
                tokenizer_or_name=TOKENIER_NAME,
                output_file=str(output_file),
                workdir_path=str(tmpdir_path),
                limit=LIMIT,
                batch_size=4,
                max_new_tokens=16,
                temperature=0.0,
                do_sample=False,
            )
        except (OSError, RuntimeError) as exc:
            pytest.skip(f"Unable to load/run model in CI: {exc}")

        # Verify limit handling
        assert len(results) == 50

        expected_ids = [f"q{i}" for i in range(50)]
        actual_ids = [r["question_id"] for r in results]

        assert actual_ids == expected_ids

        # Verify output file was written
        assert output_file.exists()

        output_lines = output_file.read_text(encoding="utf-8").splitlines()

        assert len(output_lines) == 50

        # Verify output schema
        for line in output_lines:
            row = json.loads(line)

            assert "question_id" in row
            assert "completion" in row

            assert isinstance(row["question_id"], str)
            assert isinstance(row["completion"], str)