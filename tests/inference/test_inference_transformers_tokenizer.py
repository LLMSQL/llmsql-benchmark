"""Tokenizer loading in ``inference_transformers`` (issue #110).

Covers the ``trust_remote_code`` argument reaching ``AutoTokenizer.from_pretrained``
and the documented precedence where ``tokenizer_kwargs`` override explicit params.
"""

from unittest.mock import MagicMock

import pytest

import llmsql.inference.inference_transformers as it


class _StopLoading(Exception):
    pass


def _captured_tokenizer_kwargs(monkeypatch, **kwargs):
    captured = {}

    def fake_tok_from_pretrained(name, **load_kwargs):
        captured["name"] = name
        captured.update(load_kwargs)
        raise _StopLoading

    monkeypatch.setattr(
        it.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(it.AutoTokenizer, "from_pretrained", fake_tok_from_pretrained)
    with pytest.raises(_StopLoading):
        it.inference_transformers("some/model", **kwargs)
    return captured


def test_tokenizer_respects_trust_remote_code_false(monkeypatch):
    captured = _captured_tokenizer_kwargs(monkeypatch, trust_remote_code=False)
    assert captured["name"] == "some/model"
    assert captured["trust_remote_code"] is False


def test_tokenizer_respects_trust_remote_code_true(monkeypatch):
    captured = _captured_tokenizer_kwargs(monkeypatch, trust_remote_code=True)
    assert captured["trust_remote_code"] is True


def test_tokenizer_kwargs_override_explicit_trust_remote_code(monkeypatch):
    captured = _captured_tokenizer_kwargs(
        monkeypatch,
        trust_remote_code=True,
        tokenizer_kwargs={"trust_remote_code": False},
    )
    assert captured["trust_remote_code"] is False


def test_tokenizer_kwargs_override_padding_side_default(monkeypatch):
    captured = _captured_tokenizer_kwargs(
        monkeypatch, tokenizer_kwargs={"padding_side": "right"}
    )
    assert captured["padding_side"] == "right"


def test_tokenizer_padding_side_defaults_to_left(monkeypatch):
    captured = _captured_tokenizer_kwargs(monkeypatch)
    assert captured["padding_side"] == "left"
