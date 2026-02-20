import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

import llmsql.inference.inference_vllm as mod
from llmsql.config.config import get_available_versions

questions = [
    {"question_id": "q1", "table_id": "t1", "question": "Select name from students;"},
    {"question_id": "q2", "table_id": "t1", "question": "Count students older than 20;"},
]
tables = [
    {
        "table_id": "t1",
        "header": ["id", "name", "age"],
        "types": ["int", "str", "int"],
        "rows": [[1, "Alice", 21], [2, "Bob", 19]],
    }
]

VALID_LLMSQL_VERSIONS = [None] + get_available_versions()
INVALID_LLMSQL_VERSION = "1.1"


@pytest.mark.asyncio
@pytest.mark.parametrize("version_arg", VALID_LLMSQL_VERSIONS)
async def test_inference_vllm_valid_versions(monkeypatch, tmp_path, version_arg):
    """Test inference_vllm with valid version flags using local JSONL files."""
    q_file = tmp_path / "questions.jsonl"
    t_file = tmp_path / "tables.jsonl"
    out_file = tmp_path / "out.jsonl"

    q_file.write_text("\n".join(json.dumps(q) for q in questions))
    t_file.write_text("\n".join(json.dumps(t) for t in tables))

    monkeypatch.setattr(mod, "load_jsonl",
                        lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()])
    monkeypatch.setattr(mod, "overwrite_jsonl", lambda path: None)
    monkeypatch.setattr(mod, "save_jsonl_lines", lambda path, lines: None)
    monkeypatch.setattr(mod, "choose_prompt_builder", lambda shots: lambda *a: "PROMPT")

    fake_llm = MagicMock()
    fake_llm.generate.return_value = [MagicMock(outputs=[MagicMock(text="SELECT 1")])]
    monkeypatch.setattr(mod, "LLM", lambda *a, **kw: fake_llm)

    kwargs = {
        "model_name": "dummy-model",
        "output_file": str(out_file),
        "questions_path": str(q_file),
        "tables_path": str(t_file),
        "num_fewshots": 1,
        "batch_size": 1,
        "max_new_tokens": 8,
        "temperature": 0.0,
    }
    if version_arg is not None:
        kwargs["version"] = version_arg

    results = mod.inference_vllm(**kwargs)

    assert isinstance(results, list)
    assert all("question_id" in r and "completion" in r for r in results)
    assert out_file.exists()


@pytest.mark.asyncio
async def test_inference_vllm_invalid_version(monkeypatch, tmp_path):
    """Test inference_vllm raises exception with invalid version flag."""
    q_file = tmp_path / "questions.jsonl"
    t_file = tmp_path / "tables.jsonl"
    out_file = tmp_path / "out.jsonl"

    q_file.write_text("\n".join(json.dumps(q) for q in questions))
    t_file.write_text("\n".join(json.dumps(t) for t in tables))

    monkeypatch.setattr(mod, "load_jsonl",
                        lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()])
    monkeypatch.setattr(mod, "overwrite_jsonl", lambda path: None)
    monkeypatch.setattr(mod, "save_jsonl_lines", lambda path, lines: None)
    monkeypatch.setattr(mod, "choose_prompt_builder", lambda shots: lambda *a: "PROMPT")

    fake_llm = MagicMock()
    fake_llm.generate.return_value = [MagicMock(outputs=[MagicMock(text="SELECT 1")])]
    monkeypatch.setattr(mod, "LLM", lambda *a, **kw: fake_llm)

    kwargs = {
        "model_name": "dummy-model",
        "output_file": str(out_file),
        "questions_path": str(q_file),
        "tables_path": str(t_file),
        "num_fewshots": 1,
        "batch_size": 1,
        "max_new_tokens": 8,
        "temperature": 0.0,
        "version": INVALID_LLMSQL_VERSION,  # invalid version
    }

    with pytest.raises(Exception):
        mod.inference_vllm(**kwargs)