from pathlib import Path
import random

from huggingface_hub import hf_hub_download
import numpy as np
import torch

from llmsql.config.config import DEFAULT_WORKDIR_PATH
from llmsql.loggers.logging_config import log


# --- Load benchmark data ---
def _download_file(repo_id: str, filename: str) -> str:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=DEFAULT_WORKDIR_PATH,
    )
    assert isinstance(path, str)
    return path


def _setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_download(repo_id: str, filename: str, local_path: str | None) -> str:
    if local_path is not None:
        return local_path

    target_path = Path(DEFAULT_WORKDIR_PATH) / filename

    if target_path.exists():
        log.info(f"Removing existing path: {target_path}")
        if target_path.is_file() or target_path.is_symlink():
            target_path.unlink()

    log.info(f"Downloading {filename} from Hugging Face Hub...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=DEFAULT_WORKDIR_PATH,
    )
    log.info(f"Downloaded {filename} to: {local_path}")

    return local_path
