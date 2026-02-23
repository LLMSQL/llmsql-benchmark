import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import llmsql.inference.inference_transformers as transformers_mod
import llmsql.inference.inference_vllm as vllm_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUESTIONS = [
    {"question_id": f"q{i}", "question": f"Question {i}?", "table_id": "t1"}
    for i in range(1, 6)  # 5 questions total
]
TABLES = [{"table_id": "t1", "header": ["col"], "types": ["text"], "rows": [["foo"]]}]


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records))


def _patch_common_vllm(monkeypatch, tmp_path):
    """Patch all vLLM module-level dependencies."""
    monkeypatch.setattr(
        vllm_mod,
        "load_jsonl",
        lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()],
    )
    monkeypatch.setattr(
        vllm_mod, "overwrite_jsonl", lambda path: Path(path).write_text("")
    )
    monkeypatch.setattr(
        vllm_mod,
        "save_jsonl_lines",
        lambda path, lines: Path(path)
        .open("a")
        .write("\n".join(json.dumps(line) for line in lines) + "\n"),
    )
    monkeypatch.setattr(
        vllm_mod,
        "choose_prompt_builder",
        lambda shots: lambda q, h, t, r: f"PROMPT: {q}",
    )

    fake_llm = MagicMock()
    fake_llm.generate.side_effect = lambda prompts, _params: [
        MagicMock(outputs=[MagicMock(text=f"SELECT {i}")]) for i in range(len(prompts))
    ]
    monkeypatch.setattr(vllm_mod, "LLM", lambda *a, **kw: fake_llm)


def _patch_common_transformers(monkeypatch, tmp_path):
    """Patch all Transformers module-level dependencies."""
    monkeypatch.setattr(
        transformers_mod,
        "load_jsonl",
        lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()],
    )
    monkeypatch.setattr(
        transformers_mod, "overwrite_jsonl", lambda path: Path(path).write_text("")
    )
    monkeypatch.setattr(
        transformers_mod,
        "save_jsonl_lines",
        lambda path, lines: Path(path)
        .open("a")
        .write("\n".join(json.dumps(line) for line in lines) + "\n"),
    )
    monkeypatch.setattr(
        transformers_mod,
        "choose_prompt_builder",
        lambda shots: lambda q, h, t, r: f"PROMPT: {q}",
    )

    fake_tokenizer = MagicMock()
    fake_tokenizer.pad_token = "<pad>"
    fake_tokenizer.pad_token_id = 0
    fake_tokenizer.chat_template = None
    fake_tokenizer.return_value = {"input_ids": MagicMock()}

    fake_model = MagicMock()
    # generate returns tensors of shape (batch, input_len + new_tokens)
    fake_model.device = "cpu"
    fake_model.generate.side_effect = lambda **kw: [
        [0] * (len(ids) + 5) for ids in kw["input_ids"]
    ]

    monkeypatch.setattr(
        transformers_mod,
        "AutoModelForCausalLM",
        MagicMock(from_pretrained=MagicMock(return_value=fake_model)),
    )
    monkeypatch.setattr(
        transformers_mod,
        "AutoTokenizer",
        MagicMock(from_pretrained=MagicMock(return_value=fake_tokenizer)),
    )


# ---------------------------------------------------------------------------
# vLLM limit tests
# ---------------------------------------------------------------------------


class TestInferenceVllmLimit:
    @pytest.mark.asyncio
    async def test_limit_integer_restricts_results(self, monkeypatch, tmp_path):
        """Integer limit returns only the first N results."""
        qpath, tpath = tmp_path / "questions.jsonl", tmp_path / "tables.jsonl"
        _write_jsonl(qpath, QUESTIONS)
        _write_jsonl(tpath, TABLES)
        _patch_common_vllm(monkeypatch, tmp_path)

        results = vllm_mod.inference_vllm(
            model_name="dummy",
            output_file=str(tmp_path / "out.jsonl"),
            questions_path=str(qpath),
            tables_path=str(tpath),
            limit=3,
        )

        assert len(results) == 3
        assert [r["question_id"] for r in results] == ["q1", "q2", "q3"]

    @pytest.mark.asyncio
    async def test_limit_float_restricts_results(self, monkeypatch, tmp_path):
        """Float limit of 0.4 on 5 questions returns first 2 (floor, min 1)."""
        qpath, tpath = tmp_path / "questions.jsonl", tmp_path / "tables.jsonl"
        _write_jsonl(qpath, QUESTIONS)
        _write_jsonl(tpath, TABLES)
        _patch_common_vllm(monkeypatch, tmp_path)

        results = vllm_mod.inference_vllm(
            model_name="dummy",
            output_file=str(tmp_path / "out.jsonl"),
            questions_path=str(qpath),
            tables_path=str(tpath),
            limit=0.4,
        )

        assert len(results) == 2
        assert results[0]["question_id"] == "q1"

    @pytest.mark.asyncio
    async def test_limit_none_uses_all_samples(self, monkeypatch, tmp_path):
        """No limit evaluates all questions."""
        qpath, tpath = tmp_path / "questions.jsonl", tmp_path / "tables.jsonl"
        _write_jsonl(qpath, QUESTIONS)
        _write_jsonl(tpath, TABLES)
        _patch_common_vllm(monkeypatch, tmp_path)

        results = vllm_mod.inference_vllm(
            model_name="dummy",
            output_file=str(tmp_path / "out.jsonl"),
            questions_path=str(qpath),
            tables_path=str(tpath),
            limit=None,
        )

        assert len(results) == len(QUESTIONS)

    @pytest.mark.asyncio
    async def test_limit_float_1_uses_all_samples(self, monkeypatch, tmp_path):
        """Float limit of 1.0 evaluates all questions."""
        qpath, tpath = tmp_path / "questions.jsonl", tmp_path / "tables.jsonl"
        _write_jsonl(qpath, QUESTIONS)
        _write_jsonl(tpath, TABLES)
        _patch_common_vllm(monkeypatch, tmp_path)

        results = vllm_mod.inference_vllm(
            model_name="dummy",
            output_file=str(tmp_path / "out.jsonl"),
            questions_path=str(qpath),
            tables_path=str(tpath),
            limit=1.0,
        )

        assert len(results) == len(QUESTIONS)

    @pytest.mark.asyncio
    async def test_limit_invalid_float_raises(self, monkeypatch, tmp_path):
        """Float outside (0.0, 1.0] raises ValueError."""
        qpath, tpath = tmp_path / "questions.jsonl", tmp_path / "tables.jsonl"
        _write_jsonl(qpath, QUESTIONS)
        _write_jsonl(tpath, TABLES)
        _patch_common_vllm(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="0.0 and 1.0"):
            vllm_mod.inference_vllm(
                model_name="dummy",
                output_file=str(tmp_path / "out.jsonl"),
                questions_path=str(qpath),
                tables_path=str(tpath),
                limit=1.5,
            )

    @pytest.mark.asyncio
    async def test_limit_invalid_int_raises(self, monkeypatch, tmp_path):
        """Non-positive integer raises ValueError."""
        qpath, tpath = tmp_path / "questions.jsonl", tmp_path / "tables.jsonl"
        _write_jsonl(qpath, QUESTIONS)
        _write_jsonl(tpath, TABLES)
        _patch_common_vllm(monkeypatch, tmp_path)

        with pytest.raises(ValueError):
            vllm_mod.inference_vllm(
                model_name="dummy",
                output_file=str(tmp_path / "out.jsonl"),
                questions_path=str(qpath),
                tables_path=str(tpath),
                limit=0,
            )

    @pytest.mark.asyncio
    async def test_limit_larger_than_dataset_uses_all(self, monkeypatch, tmp_path):
        """Integer limit larger than dataset size returns all samples."""
        qpath, tpath = tmp_path / "questions.jsonl", tmp_path / "tables.jsonl"
        _write_jsonl(qpath, QUESTIONS)
        _write_jsonl(tpath, TABLES)
        _patch_common_vllm(monkeypatch, tmp_path)

        results = vllm_mod.inference_vllm(
            model_name="dummy",
            output_file=str(tmp_path / "out.jsonl"),
            questions_path=str(qpath),
            tables_path=str(tpath),
            limit=9999,
        )

        assert len(results) == len(QUESTIONS)


# ---------------------------------------------------------------------------
# Transformers limit tests  (same cases, different backend)
# ---------------------------------------------------------------------------


class TestInferenceTransformersLimit:
    def test_limit_invalid_float_raises(self, monkeypatch, tmp_path):
        qpath, tpath = tmp_path / "questions.jsonl", tmp_path / "tables.jsonl"
        _write_jsonl(qpath, QUESTIONS)
        _write_jsonl(tpath, TABLES)
        _patch_common_transformers(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="0.0 and 1.0"):
            transformers_mod.inference_transformers(
                model_or_model_name_or_path="dummy",
                output_file=str(tmp_path / "out.jsonl"),
                questions_path=str(qpath),
                tables_path=str(tpath),
                limit=2.0,
            )

    def test_limit_invalid_int_raises(self, monkeypatch, tmp_path):
        qpath, tpath = tmp_path / "questions.jsonl", tmp_path / "tables.jsonl"
        _write_jsonl(qpath, QUESTIONS)
        _write_jsonl(tpath, TABLES)
        _patch_common_transformers(monkeypatch, tmp_path)

        with pytest.raises(ValueError):
            transformers_mod.inference_transformers(
                model_or_model_name_or_path="dummy",
                output_file=str(tmp_path / "out.jsonl"),
                questions_path=str(qpath),
                tables_path=str(tpath),
                limit=-1,
            )
