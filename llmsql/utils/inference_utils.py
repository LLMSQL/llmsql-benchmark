from pathlib import Path
import random

from huggingface_hub import hf_hub_download
import numpy as np
import torch

from llmsql.config.config import DEFAULT_WORKDIR_PATH, REPO_ID
from llmsql.loggers.logging_config import log


# --- Load benchmark data ---
def _download_file(filename: str) -> str:
    path = hf_hub_download(
        repo_id=REPO_ID,
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


def _maybe_download(filename: str, local_path: str | None) -> str:
    if local_path is not None:
        return local_path
    target_path = Path(DEFAULT_WORKDIR_PATH) / filename
    if not target_path.exists():
        log.info(f"Downloading {filename} from Hugging Face Hub...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=DEFAULT_WORKDIR_PATH,
        )
        log.info(f"Saved {filename} to {local_path}")
        return local_path
    return str(target_path)
