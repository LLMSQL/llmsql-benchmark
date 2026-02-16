from pathlib import Path
import random
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from llmsql.config.config import DEFAULT_WORKDIR_PATH, get_repo_id, DEFAULT_LLMSQL_VERSION
from llmsql.utils import inference_utils as mod


@pytest.mark.asyncio
async def test_download_file(monkeypatch, tmp_path):
    """_download_file calls hf_hub_download and returns path."""
    expected_path = str(tmp_path / "questions.jsonl")

    def fake_hf_hub_download(repo_id, filename, repo_type, local_dir):
        assert repo_id == get_repo_id(DEFAULT_LLMSQL_VERSION)
        assert repo_type == "dataset"
        assert local_dir == DEFAULT_WORKDIR_PATH
        assert filename == "questions.jsonl"
        return expected_path

    monkeypatch.setattr(mod, "hf_hub_download", fake_hf_hub_download)
    path = mod._download_file(get_repo_id(DEFAULT_LLMSQL_VERSION), "questions.jsonl")
    assert path == expected_path


@pytest.mark.asyncio
async def test_setup_seed(monkeypatch):
    """_setup_seed sets random, numpy, and torch seeds."""
    monkeypatch.setattr(torch, "cuda", MagicMock(is_available=lambda: False))
    # Just check no exception occurs
    mod._setup_seed(42)
    # Optionally check reproducibility for random and numpy
    mod._setup_seed(123)
    r1 = random.randint(0, 100)
    mod._setup_seed(123)
    r2 = random.randint(0, 100)
    assert r1 == r2
    a1 = np.random.randint(0, 100)
    mod._setup_seed(123)
    a2 = np.random.randint(0, 100)
    assert a1 == a2


@pytest.mark.asyncio
async def test_maybe_download_existing_file(tmp_path, monkeypatch):
    """_maybe_download returns existing path without calling hf_hub_download."""
    existing = tmp_path / "questions.jsonl"
    existing.write_text("dummy")
    monkeypatch.setattr(mod, "hf_hub_download", lambda *a, **kw: "FAIL")
    # Should return local path directly
    path = mod._maybe_download("questions.jsonl", local_path=str(existing))
    assert path == str(existing)

    # Should also return target_path if file exists in DEFAULT_WORKDIR_PATH
    monkeypatch.setattr(mod, "hf_hub_download", lambda *a, **kw: "FAIL")
    monkeypatch.setattr(mod, "DEFAULT_WORKDIR_PATH", str(tmp_path))
    path2 = mod._maybe_download("questions.jsonl", local_path=None)
    assert Path(path2).exists() or path2.endswith("questions.jsonl")


@pytest.mark.asyncio
async def test_maybe_download_calls_hf_hub(monkeypatch, tmp_path):
    """_maybe_download downloads file if missing."""
    monkeypatch.setattr(mod, "DEFAULT_WORKDIR_PATH", str(tmp_path))
    filename = "questions.jsonl"
    called = {}

    def fake_hf_hub_download(**kwargs):
        called.update(kwargs)
        # create dummy file to simulate download
        path = tmp_path / filename
        path.write_text("dummy")
        return str(path)

    monkeypatch.setattr(mod, "hf_hub_download", fake_hf_hub_download)

    path = mod._maybe_download(filename, local_path=None)
    assert Path(path).exists()
    assert called["repo_id"] == get_repo_id(DEFAULT_LLMSQL_VERSION)
    assert called["filename"] == filename
