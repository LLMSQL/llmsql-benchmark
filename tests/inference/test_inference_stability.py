import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmsql.inference import inference


@pytest.mark.asyncio
async def test_download_file(monkeypatch, tmp_path):
    """Ensure _download_file calls hf_hub_download correctly."""
    # Patch LLM so __init__ does not try to load a real model
    with patch("llmsql.inference.inference.LLM") as mock_llm:
        mock_llm.return_value = object()  # dummy instance
        inf = inference.LLMSQLVLLMInference("dummy-model")

    called = {}

    def fake_download(**kwargs):
        called.update(kwargs)
        return str(tmp_path / "questions.jsonl")

    monkeypatch.setattr(inference, "hf_hub_download", fake_download)

    path = inf._download_file("questions.jsonl")

    assert "repo_id" in called
    assert called["filename"] == "questions.jsonl"
    assert path.endswith("questions.jsonl")


@pytest.mark.asyncio
async def test_generate_with_local_files(monkeypatch, tmp_path):
    """Generate should read JSONL, call LLM, and write outputs."""
    # Create fake questions/tables
    qpath = tmp_path / "questions.jsonl"
    tpath = tmp_path / "tables.jsonl"

    questions = [
        {"question_id": "q1", "question": "What is 1+1?", "table_id": "t1"},
        {"question_id": "q2", "question": "What is 2+2?", "table_id": "t1"},
    ]
    tables = [
        {"table_id": "t1", "header": ["col"], "types": ["text"], "rows": [["foo"]]}
    ]

    qpath.write_text("\n".join(json.dumps(q) for q in questions))
    tpath.write_text("\n".join(json.dumps(t) for t in tables))

    out_file = tmp_path / "out.jsonl"

    # Patch utils
    monkeypatch.setattr(
        inference,
        "load_jsonl",
        lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()],
    )
    monkeypatch.setattr(
        inference, "overwrite_jsonl", lambda path: Path(path).write_text("")
    )
    monkeypatch.setattr(
        inference,
        "save_jsonl_lines",
        lambda path, lines: Path(path).write_text(
            Path(path).read_text()
            + "\n".join(json.dumps(line) for line in lines)
            + "\n"
        ),
    )
    monkeypatch.setattr(
        inference,
        "choose_prompt_builder",
        lambda shots: lambda q, h, t, r: f"PROMPT: {q}",
    )

    # Patch LLM with a fake generate
    fake_llm = MagicMock()
    fake_llm.generate.return_value = [
        MagicMock(outputs=[MagicMock(text="SELECT 2")]),
        MagicMock(outputs=[MagicMock(text="SELECT 4")]),
    ]

    with patch("llmsql.inference.inference.LLM", return_value=fake_llm):
        inf = inference.LLMSQLVLLMInference("dummy-model")

    results = inf.generate(
        output_file=str(out_file),
        questions_path=str(qpath),
        tables_path=str(tpath),
        shots=1,
        batch_size=1,
        max_new_tokens=5,
        temperature=0.7,
    )

    assert len(results) == 2
    assert all("question_id" in r and "completion" in r for r in results)
    assert out_file.exists()
    written = out_file.read_text().strip().splitlines()
    assert len(written) == 2


@pytest.mark.asyncio
async def test_generate_downloads_if_missing(
    monkeypatch, mock_llm, fake_jsonl_files, tmp_path
):
    """If paths not provided, should use _download_file."""
    qpath, tpath = fake_jsonl_files
    (tmp_path / "questions.jsonl").unlink()  # remove to force download
    (tmp_path / "tables.jsonl").unlink()

    inf = inference.LLMSQLVLLMInference("dummy-model", workdir_path=str(tmp_path))

    monkeypatch.setattr(
        inference,
        "load_jsonl",
        lambda path: [{"question_id": "q1", "question": "x?", "table_id": "t1"}]
        if "questions" in path
        else [{"table_id": "t1", "header": ["id"], "types": ["int"], "rows": [[1]]}],
    )
    monkeypatch.setattr(inference, "overwrite_jsonl", lambda path: None)
    monkeypatch.setattr(inference, "save_jsonl_lines", lambda path, lines: None)
    monkeypatch.setattr(
        inference, "choose_prompt_builder", lambda shots: lambda *a: "PROMPT"
    )

    called = {"q": 0, "t": 0}

    def fake_download(filename, **_):
        called["q" if "questions" in filename else "t"] += 1
        return str(tmp_path / filename)

    monkeypatch.setattr(inference, "hf_hub_download", fake_download)

    results = inf.generate(output_file=str(tmp_path / "out.jsonl"))
    assert results
    assert called["q"] == 1
    assert called["t"] == 1
