from typing import Literal

REPO_IDs: dict[str, str] = {
    "1.0": "llmsql-bench/llmsql-benchmark",
    "2.0": "llmsql-bench/llmsql-2.0",
}

DEFAULT_LLMSQL_VERSION: Literal["1.0", "2.0"] = "2.0"

# Number of few-shot examples used when ``num_fewshots`` is not given.
# LLMSQL 2.0 is a zero-shot benchmark (it has no train split to draw examples from).
DEFAULT_NUM_FEWSHOTS: dict[str, int] = {
    "1.0": 5,
    "2.0": 0,
}

# Generation budget used when ``max_new_tokens`` is not given. LLMSQL 1.0
# expects a bare SQL query; the LLMSQL 2.0 prompt asks for a ```sql block and
# models often reason before answering. Reasoning models need more: the
# reported LLMSQL 2.0 results were obtained with up to 16k new tokens.
DEFAULT_MAX_NEW_TOKENS: dict[str, int] = {
    "1.0": 256,
    "2.0": 4096,
}


def get_repo_id(version: str = DEFAULT_LLMSQL_VERSION) -> str:
    try:
        return REPO_IDs[version]
    except KeyError as err:
        raise ValueError(
            f"version should be one of: {list(REPO_IDs.keys())}, not {version}"
        ) from err


def get_available_versions() -> list[str]:
    return list(REPO_IDs.keys())


def resolve_num_fewshots(version: str, num_fewshots: int | None) -> int:
    """Resolve the number of few-shot examples for a benchmark version.

    Args:
        version: LLMSQL version (``"1.0"`` or ``"2.0"``).
        num_fewshots: Requested number of examples, or ``None`` for the
            version default (5 for 1.0, 0 for 2.0).

    Returns:
        The number of few-shot examples to use.

    Raises:
        ValueError: If ``version`` is unknown, or if a non-zero number of
            examples is requested for LLMSQL 2.0, which is zero-shot only.
    """
    get_repo_id(version)  # validates the version
    if num_fewshots is None:
        return DEFAULT_NUM_FEWSHOTS[version]
    if version == "2.0" and num_fewshots != 0:
        raise ValueError(
            f"LLMSQL 2.0 is a zero-shot benchmark, but num_fewshots={num_fewshots} "
            "was requested. Use num_fewshots=0 (or leave it unset), or use "
            'version="1.0" for few-shot evaluation.'
        )
    return num_fewshots


def resolve_max_new_tokens(version: str, max_new_tokens: int | None) -> int:
    """Resolve the generation budget for a benchmark version.

    Args:
        version: LLMSQL version (``"1.0"`` or ``"2.0"``).
        max_new_tokens: Requested budget, or ``None`` for the version default
            (256 for 1.0, 4096 for 2.0).

    Returns:
        The maximum number of new tokens to generate.
    """
    get_repo_id(version)  # validates the version
    if max_new_tokens is None:
        return DEFAULT_MAX_NEW_TOKENS[version]
    return max_new_tokens
