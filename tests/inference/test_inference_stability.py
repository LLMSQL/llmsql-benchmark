import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import llmsql.inference.inference_vllm as mod  # patch in the correct module


@pytest.mark.asyncio
async def test_inference_vllm_with_local_files(monkeypatch, tmp_path):
    """Test inference_vllm with local JSONL files."""
    # --- Prepare fake JSONL files ---
    questions = [
        {"question_id": "q1", "question": "What is 1+1?", "table_id": "t1"},
        {"question_id": "q2", "question": "What is 2+2?", "table_id": "t1"},
    ]
    tables = [
        {"table_id": "t1", "header": ["col"], "types": ["text"], "rows": [["foo"]]}
    ]

    qpath = tmp_path / "questions.jsonl"
    tpath = tmp_path / "tables.jsonl"
    out_file = tmp_path / "out.jsonl"

    qpath.write_text("\n".join(json.dumps(q) for q in questions))
    tpath.write_text("\n".join(json.dumps(t) for t in tables))

    # --- Patch utility functions ---
    monkeypatch.setattr(
        mod,
        "load_jsonl",
        lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()],
    )
    monkeypatch.setattr(mod, "overwrite_jsonl", lambda path: Path(path).write_text(""))
    monkeypatch.setattr(
        mod,
        "save_jsonl_lines",
        lambda path, lines: Path(path).write_text(
            Path(path).read_text()
            + "\n".join(json.dumps(line) for line in lines)
            + "\n"
        ),
    )
    monkeypatch.setattr(
        mod, "choose_prompt_builder", lambda shots: lambda q, h, t, r: f"PROMPT: {q}"
    )

    # --- Patch LLM.generate ---
    fake_llm = MagicMock()
    fake_llm.generate.return_value = [
        MagicMock(outputs=[MagicMock(text="SELECT 2")]),
        MagicMock(outputs=[MagicMock(text="SELECT 4")]),
    ]
    monkeypatch.setattr(mod, "LLM", lambda *a, **kw: fake_llm)

    # --- Run inference ---
    results = mod.inference_vllm(
        model_name="dummy-model",
        output_file=str(out_file),
        questions_path=str(qpath),
        tables_path=str(tpath),
        num_fewshots=1,
        batch_size=1,
        max_new_tokens=5,
        temperature=0.7,
    )

    # --- Assertions ---
    assert len(results) == 2
    assert all("question_id" in r and "completion" in r for r in results)
    assert out_file.exists()
    written = out_file.read_text().strip().splitlines()
    assert len(written) == 2


@pytest.mark.asyncio
async def test_inference_vllm_download_if_missing(monkeypatch, tmp_path):
    """Test inference_vllm downloads JSONL files if missing."""
    out_file = tmp_path / "out.jsonl"

    called = {"q": 0, "t": 0}

    def fake_download(filename, path, **_):
        called["q" if "questions" in filename else "t"] += 1
        path = tmp_path / filename
        # Write minimal JSONL for subsequent load_jsonl
        if "questions" in filename:
            path.write_text(
                json.dumps({"question_id": "q1", "question": "x?", "table_id": "t1"})
            )
        else:
            path.write_text(
                json.dumps(
                    {
                        "table_id": "t1",
                        "header": ["id"],
                        "types": ["int"],
                        "rows": [[1]],
                    }
                )
            )
        return str(path)

    # Patch module functions
    monkeypatch.setattr(mod, "_maybe_download", fake_download)
    monkeypatch.setattr(
        mod,
        "load_jsonl",
        lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()],
    )
    monkeypatch.setattr(mod, "overwrite_jsonl", lambda path: None)
    monkeypatch.setattr(mod, "save_jsonl_lines", lambda path, lines: None)
    monkeypatch.setattr(mod, "choose_prompt_builder", lambda shots: lambda *a: "PROMPT")

    # Patch LLM.generate
    fake_llm = MagicMock()
    fake_llm.generate.return_value = [MagicMock(outputs=[MagicMock(text="SELECT 1")])]
    monkeypatch.setattr(mod, "LLM", lambda *a, **kw: fake_llm)

    # --- Run inference ---
    results = mod.inference_vllm(
        model_name="dummy-model",
        output_file=str(out_file),
        questions_path=None,
        tables_path=None,
        workdir_path=str(tmp_path),
    )

    # --- Assertions ---
    assert results
    assert called["q"] == 1
    assert called["t"] == 1
