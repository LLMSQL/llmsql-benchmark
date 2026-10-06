"""Fixtures for the backend-specific inference tests.

The vLLM backend lives behind the optional `vllm` extra (see the `vllm` group in
pyproject.toml), so importing `llmsql.inference.inference_vllm` at module level
would make the whole test session fail to collect for anyone who installed only
the `dev` group. Import it lazily inside the fixtures that actually need it.
"""

import pytest


@pytest.fixture
def mock_llm(monkeypatch):
    """Mock vLLM LLM to avoid GPU/model loading."""
    pytest.importorskip("vllm")
    import llmsql.inference.inference_vllm as inference_vllm

    class DummyOutput:
        def __init__(self, text="SELECT 1"):
            self.outputs = [type("Obj", (), {"text": text})()]

    class DummyLLM:
        def generate(self, prompts, sampling_params):
            return [DummyOutput(f"-- SQL for: {p}") for p in prompts]

    monkeypatch.setattr(inference_vllm, "LLM", lambda **_: DummyLLM())
    return DummyLLM()
