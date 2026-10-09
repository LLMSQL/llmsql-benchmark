"""Inference on LLMSQL 2.0: zero-shot prompts and few-shot validation."""

import json
from unittest.mock import MagicMock

import pytest

from llmsql.inference.inference_api import inference_api
from llmsql.inference.inference_function import inference_function
from llmsql.inference.inference_transformers import inference_transformers


def test_inference_function_v2_uses_dataset_prompts(
    llmsql2_questions, llmsql2_workdir, tmp_path
):
    seen = {}

    async def fake(prompt, *, question, table, **kwargs):
        seen[question["question_id"]] = prompt
        assert table["table_id"] == question["table_id"]
        return "```sql\nSELECT 1\n```"

    out = tmp_path / "out.jsonl"
    results = inference_function(
        inference_function=fake,
        version="2.0",
        output_file=str(out),
        workdir_path=str(llmsql2_workdir),
    )
    assert len(results) == len(llmsql2_questions)
    assert seen == {q["question_id"]: q["prompt"] for q in llmsql2_questions}
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["question_id"] for r in written} == set(seen)


def test_inference_function_explicit_zero_shot_ok(llmsql2_workdir, tmp_path):
    async def fake(prompt, **kwargs):
        return "SELECT 1"

    results = inference_function(
        inference_function=fake,
        version="2.0",
        num_fewshots=0,
        limit=2,
        output_file=str(tmp_path / "out.jsonl"),
        workdir_path=str(llmsql2_workdir),
    )
    assert len(results) == 2


@pytest.mark.parametrize("shots", [1, 5])
def test_inference_function_rejects_fewshot_v2(shots, llmsql2_workdir, tmp_path):
    async def fake(prompt, **kwargs):
        return "SELECT 1"

    with pytest.raises(ValueError, match="zero-shot"):
        inference_function(
            inference_function=fake,
            version="2.0",
            num_fewshots=shots,
            output_file=str(tmp_path / "out.jsonl"),
            workdir_path=str(llmsql2_workdir),
        )


def test_inference_api_rejects_fewshot_v2(llmsql2_workdir, tmp_path):
    with pytest.raises(ValueError, match="zero-shot"):
        inference_api(
            "m",
            base_url="http://localhost:1",
            version="2.0",
            num_fewshots=5,
            output_file=str(tmp_path / "out.jsonl"),
            workdir_path=str(llmsql2_workdir),
        )


def test_inference_transformers_rejects_fewshot_v2_before_loading(tmp_path):
    # Fails before any model download / load is attempted.
    with pytest.raises(ValueError, match="zero-shot"):
        inference_transformers(
            "this/model-does-not-exist",
            version="2.0",
            num_fewshots=5,
            output_file=str(tmp_path / "out.jsonl"),
            workdir_path=str(tmp_path),
        )


def test_inference_function_v1_default_is_5shot(tmp_path):
    questions = [{"question_id": 1, "question": "Who?", "table_id": "t1"}]
    tables = [{"table_id": "t1", "header": ["a"], "types": ["text"], "rows": [["x"]]}]
    (tmp_path / "questions.jsonl").write_text(json.dumps(questions[0]) + "\n")
    (tmp_path / "tables.jsonl").write_text(json.dumps(tables[0]) + "\n")
    prompts = []

    async def fake(prompt, **kwargs):
        prompts.append(prompt)
        return "SELECT 1"

    inference_function(
        inference_function=fake,
        version="1.0",
        output_file=str(tmp_path / "out.jsonl"),
        workdir_path=str(tmp_path),
    )
    assert "### EXAMPLE 5:" in prompts[0]


def test_inference_vllm_v2_prompts_and_defaults(
    monkeypatch, llmsql2_questions, llmsql2_workdir, tmp_path
):
    pytest.importorskip("vllm")
    import llmsql.inference.inference_vllm as vllm_mod

    captured = {}

    def fake_generate(prompts, sampling_params, lora_request=None):
        captured.setdefault("prompts", []).extend(prompts)
        return [MagicMock(outputs=[MagicMock(text="SELECT 1")]) for _ in prompts]

    fake_llm = MagicMock()
    fake_llm.generate.side_effect = fake_generate
    monkeypatch.setattr(vllm_mod, "LLM", lambda *a, **kw: fake_llm)
    monkeypatch.setattr(
        vllm_mod,
        "SamplingParams",
        lambda **kw: captured.setdefault("sampling", kw),
    )

    vllm_mod.inference_vllm(
        "dummy",
        version="2.0",
        use_chat_template=False,
        output_file=str(tmp_path / "out.jsonl"),
        workdir_path=str(llmsql2_workdir),
    )
    assert captured["prompts"] == [q["prompt"] for q in llmsql2_questions]
    assert captured["sampling"]["max_tokens"] == 4096

    with pytest.raises(ValueError, match="zero-shot"):
        vllm_mod.inference_vllm(
            "dummy",
            version="2.0",
            num_fewshots=5,
            output_file=str(tmp_path / "out.jsonl"),
            workdir_path=str(llmsql2_workdir),
        )
