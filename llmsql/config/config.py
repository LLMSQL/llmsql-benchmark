from typing import Literal

REPO_IDs: dict[str, str] = {
    "1.0": "llmsql-bench/llmsql-benchmark",
    "2.0": "llmsql-bench/llmsql-2.0",
}

DEFAULT_LLMSQL_VERSION: Literal["1.0", "2.0"] = "2.0"
DEFAULT_WORKDIR_PATH = "llmsql_workdir"


def get_repo_id(version: str = DEFAULT_LLMSQL_VERSION) -> str:
    try:
        return REPO_IDs[version]
    except KeyError as err:
        raise ValueError(
            f"version should be one of: {list(REPO_IDs.keys())}, not {version}"
        ) from err


def get_available_versions() -> list[str]:
    return list(REPO_IDs.keys())
