"""Mocked tests for the args ``inference_vllm`` forwards to ``vllm.LLM`` (issue #110).

These don't need a GPU: ``vllm.LLM`` and the benchmark data loaders are mocked.
They're skipped entirely when the optional ``vllm`` extra isn't installed.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm")

import llmsql.inference.inference_vllm as mod

questions = [
    {"question_id": "q1", "table_id": "t1", "question": "Select name from students;"},
]
tables = [
    {
        "table_id": "t1",
        "header": ["id", "name"],
        "types": ["int", "str"],
        "rows": [[1, "Alice"]],
    },
]


def _captured_llm_kwargs(monkeypatch, tmp_path, **kwargs):
    q_file = tmp_path / "questions.jsonl"
    t_file = tmp_path / "tables.jsonl"
    out_file = tmp_path / "out.jsonl"
    q_file.write_text("\n".join(json.dumps(q) for q in questions))
    t_file.write_text("\n".join(json.dumps(t) for t in tables))

    monkeypatch.setattr(
        mod,
        "load_jsonl",
        lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()],
    )
    monkeypatch.setattr(mod, "overwrite_jsonl", lambda path: Path(path).touch())
    monkeypatch.setattr(mod, "save_jsonl_lines", lambda path, lines: Path(path).touch())
    monkeypatch.setattr(mod, "choose_prompt_builder", lambda shots: lambda *a: "PROMPT")

    captured = {}
    fake_llm = MagicMock()
    fake_llm.generate.return_value = [MagicMock(outputs=[MagicMock(text="SELECT 1")])]

    def fake_LLM(**llm_kwargs):
        captured.update(llm_kwargs)
        return fake_llm

    monkeypatch.setattr(mod, "LLM", fake_LLM)

    call_kwargs = {
        "model_name": "dummy-model",
        "output_file": str(out_file),
        "workdir_path": str(tmp_path),
        "num_fewshots": 1,
        "batch_size": 1,
        "max_new_tokens": 8,
        "temperature": 0.0,
    }
    call_kwargs.update(kwargs)

    mod.inference_vllm(**call_kwargs)
    return captured


def test_hf_token_reaches_llm_as_hf_token(monkeypatch, tmp_path):
    captured = _captured_llm_kwargs(monkeypatch, tmp_path, hf_token="secret-token")
    assert captured["hf_token"] == "secret-token"
    assert "token" not in captured


def test_hf_token_defaults_to_none_when_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    captured = _captured_llm_kwargs(monkeypatch, tmp_path)
    assert "hf_token" not in captured


def test_llm_kwargs_override_explicit_params(monkeypatch, tmp_path):
    captured = _captured_llm_kwargs(
        monkeypatch,
        tmp_path,
        trust_remote_code=True,
        llm_kwargs={"trust_remote_code": False, "max_model_len": 2048},
    )
    assert captured["trust_remote_code"] is False
    assert captured["max_model_len"] == 2048
