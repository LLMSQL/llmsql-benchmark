"""Tests for version-dependent defaults (few-shot count, generation budget)."""

import pytest

from llmsql.config.config import resolve_max_new_tokens, resolve_num_fewshots


@pytest.mark.parametrize(
    "version,requested,expected",
    [
        ("1.0", None, 5),
        ("1.0", 0, 0),
        ("1.0", 1, 1),
        ("1.0", 5, 5),
        ("2.0", None, 0),
        ("2.0", 0, 0),
    ],
)
def test_resolve_num_fewshots(version, requested, expected):
    assert resolve_num_fewshots(version, requested) == expected


@pytest.mark.parametrize("shots", [1, 5])
def test_resolve_num_fewshots_rejects_fewshot_for_v2(shots):
    with pytest.raises(ValueError, match="zero-shot"):
        resolve_num_fewshots("2.0", shots)


def test_resolve_num_fewshots_invalid_version():
    with pytest.raises(ValueError, match="version should be one of"):
        resolve_num_fewshots("1.1", None)


@pytest.mark.parametrize(
    "version,requested,expected",
    [("1.0", None, 256), ("2.0", None, 4096), ("1.0", 8, 8), ("2.0", 16384, 16384)],
)
def test_resolve_max_new_tokens(version, requested, expected):
    assert resolve_max_new_tokens(version, requested) == expected
