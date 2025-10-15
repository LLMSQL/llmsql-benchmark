import json
from pathlib import Path
import tempfile

import pytest

from llmsql.inference.inference_transformers import inference_transformers

# --- Minimal fake benchmark data for testing ---
questions = [
    {"question_id": "q1", "table_id": "t1", "question": "Select name from students;"},
    {
        "question_id": "q2",
        "table_id": "t1",
        "question": "Count students older than 20;",
    },
]

tables = [
    {
        "table_id": "t1",
        "header": ["id", "name", "age"],
        "types": ["int", "str", "int"],
        "rows": [[1, "Alice", 21], [2, "Bob", 19]],
    }
]


# Save minimal JSONL files for testing
def _write_jsonl(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


@pytest.mark.asyncio
async def test_inference_stability():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        questions_file = tmpdir_path / "questions.jsonl"
        tables_file = tmpdir_path / "tables.jsonl"
        output_file = tmpdir_path / "outputs.jsonl"

        _write_jsonl(questions, questions_file)
        _write_jsonl(tables, tables_file)

        results = inference_transformers(
            model_or_model_name_or_path="sshleifer/tiny-gpt2",  # tiny HF model for fast tests
            tokenizer_or_name="sshleifer/tiny-gpt2",
            output_file=str(output_file),
            questions_path=str(questions_file),
            tables_path=str(tables_file),
            batch_size=1,
            max_new_tokens=8,
            temperature=0.0,
            do_sample=False,
        )

        # Basic assertions
        assert isinstance(results, list)
        assert all("question_id" in r and "completion" in r for r in results)
        assert output_file.exists()
