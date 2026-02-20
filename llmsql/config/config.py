REPO_IDs: dict = {
    "1.0": "llmsql-bench/llmsql-benchmark",
    "2.0": "llmsql-bench/llmsql-2.0"
}
DEFAULT_LLMSQL_VERSION = "2.0"
DEFAULT_WORKDIR_PATH = "llmsql_workdir"

def get_repo_id(version: str = DEFAULT_LLMSQL_VERSION) -> str:
    return REPO_IDs[version]

def get_available_versions() -> list[str]:
    return list(REPO_IDs.keys())