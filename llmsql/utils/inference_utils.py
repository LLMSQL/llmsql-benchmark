from pathlib import Path
import random
import tempfile

from huggingface_hub import hf_hub_download
import numpy as np
import torch

from llmsql.loggers.logging_config import log


def resolve_workdir_path(workdir_path: str | Path | None) -> Path:
    if workdir_path is None:
        resolved = Path(tempfile.mkdtemp(prefix="llmsql-"))
        log.info(f"Created temporary workdir: {resolved}")
        return resolved

    resolved = Path(workdir_path)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(
            f"workdir_path must point to a directory, got file: {resolved}"
        )

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


# --- Load benchmark data ---
def _download_file(
    repo_id: str, filename: str, workdir_path: str | Path | None = None
) -> str:
    local_dir = resolve_workdir_path(workdir_path)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
    )
    assert isinstance(path, str)
    return path


def _setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_download(
    repo_id: str, filename: str, workdir_path: str | Path | None
) -> str:
    target_dir = resolve_workdir_path(workdir_path)
    target_path = target_dir / filename

    if target_path.exists():
        if not target_path.is_file():
            raise ValueError(
                f"Expected downloaded benchmark file path to be a file: {target_path}"
            )
        log.info(f"Using cached benchmark file: {target_path}")
        return str(target_path)

    log.info(f"Downloading {filename} from Hugging Face Hub...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(target_dir),
    )
    log.info(f"Downloaded {filename} to: {local_path}")

    return local_path  # type: ignore
