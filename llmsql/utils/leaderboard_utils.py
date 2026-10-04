"""
Helpers for producing leaderboard-format run descriptions (``run.yaml``).

The structure mirrors the files stored in the ``leaderboard/`` folder of the
repository, so a YAML produced by :func:`llmsql.evaluate` can be dropped into
``leaderboard/<model>/<setting>/run.yaml`` and consumed by
``leaderboard/generate_leaderboard.py``. Fields that cannot be detected
automatically are written as ``null`` and can be supplied via ``run_metadata``.
"""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
import platform
from typing import Any

import yaml

from llmsql import __version__
from llmsql.loggers.logging_config import log


def _detect_os_name() -> str:
    """Return a human-readable OS name (e.g. ``Ubuntu 24.04.3 LTS``)."""
    try:
        pretty = platform.freedesktop_os_release().get("PRETTY_NAME")
        if pretty:
            return pretty
    except (OSError, AttributeError):
        pass
    return platform.platform()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_leaderboard_record(
    *,
    accuracy: float,
    total: int,
    version: str,
    model_name: str | None = None,
    answers_path: str | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a leaderboard-format run description.

    Args:
        accuracy: Execution accuracy obtained by the evaluation.
        total: Number of evaluated samples.
        version: LLMSQL benchmark version used for evaluation.
        model_name: Name of the evaluated model (e.g. ``Qwen/Qwen3-0.6B``).
        answers_path: Location of the evaluated model outputs, if known.
        run_metadata: Extra fields deep-merged on top of the auto-filled record
            (e.g. ``{"type": "open-source", "inference": {"backend": "vllm"}}``).

    Returns:
        dict: Record in the same layout as ``leaderboard/*/*/run.yaml``.
    """
    record: dict[str, Any] = {
        "date": date.today(),
        "model": {
            "name": model_name,
            "revision": None,
            "commit_hash": None,
            "parameter_count": None,
            "dtype": None,
            "thinking": None,
        },
        "type": None,
        "llmsql": {
            "version": __version__,
            "commit_hash": None,
        },
        "version": version,
        "os_name": _detect_os_name(),
        "python_version": platform.python_version(),
        "pip_freeze": None,
        "device": None,
        "inference": {
            "backend": None,
            "arguments": {},
        },
        "results": {
            "execution_accuracy": round(accuracy, 4),
            "num_samples": total,
            "answers_path": answers_path,
        },
    }

    if run_metadata:
        record = _deep_merge(record, run_metadata)

    return record


def write_leaderboard_yaml(path: str | Path, record: dict[str, Any]) -> None:
    """Save a leaderboard record as YAML, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, sort_keys=False, allow_unicode=True)
    log.info(f"Saved leaderboard run description to {path}")
