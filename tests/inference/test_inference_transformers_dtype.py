"""dtype handling in ``inference_transformers`` (strings from the CLI / model_kwargs)."""

import pytest
import torch

import llmsql.inference.inference_transformers as it
from llmsql.inference.inference_transformers import _resolve_dtype


class _StopLoading(Exception):
    pass


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.bfloat16, torch.bfloat16),
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("torch.float16", torch.float16),
        (" float16 ", torch.float16),
        ("half", torch.float16),
        ("auto", "auto"),
    ],
)
def test_resolve_dtype(value, expected):
    assert _resolve_dtype(value) == expected


@pytest.mark.parametrize("value", ["cuda", "nope", "Tensor", 16, None])
def test_resolve_dtype_invalid(value):
    with pytest.raises(ValueError):
        _resolve_dtype(value)


def _captured_load_kwargs(monkeypatch, **kwargs):
    captured = {}

    def fake_from_pretrained(name, **load_kwargs):
        captured["name"] = name
        captured.update(load_kwargs)
        raise _StopLoading

    monkeypatch.setattr(
        it.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )
    with pytest.raises(_StopLoading):
        it.inference_transformers("some/model", **kwargs)
    return captured


def test_string_dtype_argument_is_converted(monkeypatch):
    captured = _captured_load_kwargs(monkeypatch, dtype="bfloat16")
    assert captured["torch_dtype"] is torch.bfloat16
    assert "dtype" not in captured


def test_default_dtype_is_auto(monkeypatch):
    captured = _captured_load_kwargs(monkeypatch)
    assert captured["torch_dtype"] == "auto"
    assert captured["trust_remote_code"] is False


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_remote_code_setting_is_passed_to_model_and_tokenizer(
    monkeypatch, tmp_path, trust_remote_code
):
    captured = {"model": {}, "tokenizer": {}}

    class FakeModel:
        def eval(self):
            pass

    class FakeTokenizer:
        pad_token = "pad"
        eos_token = "eos"
        pad_token_id = 0

    def fake_model_from_pretrained(name, **kwargs):
        captured["model"].update(kwargs)
        return FakeModel()

    monkeypatch.setattr(
        it.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained
    )

    def fake_tokenizer_from_pretrained(name, **kwargs):
        captured["tokenizer"].update(kwargs)
        return FakeTokenizer()

    monkeypatch.setattr(
        it.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained
    )
    monkeypatch.setattr(it, "_setup_seed", lambda **kwargs: None)
    monkeypatch.setattr(it, "_maybe_download", lambda *args: "unused")
    monkeypatch.setattr(it, "load_jsonl", lambda path: [])

    it.inference_transformers(
        "some/model",
        trust_remote_code=trust_remote_code,
        output_file=str(tmp_path / "predictions.jsonl"),
        workdir_path=str(tmp_path),
    )

    assert captured["model"]["trust_remote_code"] is trust_remote_code
    assert captured["tokenizer"]["trust_remote_code"] is trust_remote_code


@pytest.mark.parametrize("key", ["dtype", "torch_dtype"])
def test_model_kwargs_dtype_overrides_argument(monkeypatch, key):
    model_kwargs = {key: "float32", "revision": "main"}
    captured = _captured_load_kwargs(
        monkeypatch, dtype="float16", model_kwargs=model_kwargs
    )
    assert captured["torch_dtype"] is torch.float32
    assert "dtype" not in captured
    assert captured["revision"] == "main"
    # the caller's dict is not mutated
    assert model_kwargs == {key: "float32", "revision": "main"}


def test_conflicting_dtype_keys_raise(monkeypatch):
    monkeypatch.setattr(
        it.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: pytest.fail("should not load"),
    )
    with pytest.raises(ValueError, match="Conflicting"):
        it.inference_transformers(
            "some/model",
            model_kwargs={"dtype": "float32", "torch_dtype": "float16"},
        )
