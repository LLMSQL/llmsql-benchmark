from pathlib import Path
import random
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from llmsql.config.config import DEFAULT_LLMSQL_VERSION, get_repo_id
from llmsql.utils import inference_utils as mod


@pytest.mark.asyncio
async def test_download_file(monkeypatch, tmp_path):
    """_download_file calls hf_hub_download and returns path."""
    expected_path = str(tmp_path / "questions.jsonl")

    def fake_hf_hub_download(repo_id, filename, repo_type, local_dir):
        assert repo_id == get_repo_id(DEFAULT_LLMSQL_VERSION)
        assert repo_type == "dataset"
        assert Path(local_dir).is_dir()
        assert filename == "questions.jsonl"
        return expected_path

    monkeypatch.setattr(mod, "hf_hub_download", fake_hf_hub_download)
    path = mod._download_file(
        get_repo_id(DEFAULT_LLMSQL_VERSION), "questions.jsonl", tmp_path
    )
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
async def test_maybe_download_calls_hf_hub(monkeypatch, tmp_path):
    """_maybe_download downloads file if missing."""
    filename = "questions.jsonl"
    called = {}

    def fake_hf_hub_download(**kwargs):
        called.update(kwargs)
        # create dummy file to simulate download
        path = tmp_path / filename
        path.write_text("dummy")
        return str(path)

    monkeypatch.setattr(mod, "hf_hub_download", fake_hf_hub_download)

    path = mod._maybe_download(
        get_repo_id(DEFAULT_LLMSQL_VERSION), filename, workdir_path=tmp_path
    )
    assert Path(path).exists()
    assert called["repo_id"] == get_repo_id(DEFAULT_LLMSQL_VERSION)
    assert called["filename"] == filename


def test_resolve_workdir_path_creates_temp(monkeypatch):
    fake_dir = "/tmp/llmsql-test-dir"

    monkeypatch.setattr(mod.tempfile, "mkdtemp", lambda prefix: fake_dir)

    path = mod.resolve_workdir_path(None)

    assert isinstance(path, Path)
    assert str(path) == fake_dir


def test_resolve_workdir_logs(monkeypatch):
    fake_dir = "/tmp/llmsql-test-dir"
    mock_log = MagicMock()

    monkeypatch.setattr(mod.tempfile, "mkdtemp", lambda prefix: fake_dir)
    monkeypatch.setattr(mod, "log", mock_log)

    mod.resolve_workdir_path(None)

    mock_log.info.assert_called_once()


def test_resolve_workdir_path_raises_if_file(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("data")

    with pytest.raises(ValueError, match="must point to a directory"):
        mod.resolve_workdir_path(file_path)


def test_maybe_download_raises_if_not_file(tmp_path):
    filename = "questions.jsonl"
    target_path = tmp_path / filename

    # create directory instead of file
    target_path.mkdir()

    with pytest.raises(ValueError, match="Expected downloaded benchmark file"):
        mod._maybe_download(
            repo_id="dummy/repo",
            filename=filename,
            workdir_path=tmp_path,
        )


def test_maybe_download_uses_cached_file(tmp_path):
    filename = "questions.jsonl"
    file_path = tmp_path / filename
    file_path.write_text("cached")

    path = mod._maybe_download(
        repo_id="dummy/repo",
        filename=filename,
        workdir_path=tmp_path,
    )

    assert path == str(file_path)
