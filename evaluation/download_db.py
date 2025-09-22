"""
Download the official SQLite database required for LLMSQL evaluation.

Usage (example from project root):
    python3 evaluation/download_db.py \
        --repo_id llmsql-bench/llmsql-benchmark \
        --target_dir dataset

This script fetches the benchmark database (`sqlite_tables.db`) from the Hugging Face Hub
and saves it locally. By default, it downloads from the repo:
    llmsql-bench/llmsql-benchmark
into the `dataset/` directory.

After running, the file will be available at:
    dataset/sqlite_tables.db

Notes:
  - The database is required for running evaluation (`evaluate_answers.py`).
  - Run this script from the **project root** (LLMSQL), not inside `evaluation/`.
"""

import argparse
import os

from huggingface_hub import hf_hub_download
from utils.logging_config import log


def main(target_dir: str, repo_id: str):
    """
    Download the SQLite database file for evaluation from the Hugging Face Hub.

    This function:
      - Ensures the target directory exists.
      - Downloads the file `sqlite_tables.db` from the specified dataset repository.
      - Saves it locally under `target_dir`.

    Args:
        target_dir (str): Local directory where the database file should be stored.
        repo_id (str): Hugging Face Hub dataset repo ID containing the DB file.

    Returns:
        None. Logs the local path of the downloaded DB file.
    """
    # Make sure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Download `sqlite_tables.db` from the dataset repo
    db_path = hf_hub_download(
        repo_id=repo_id,
        filename="sqlite_tables.db",
        repo_type="dataset",
        local_dir=target_dir,
    )

    # Confirm to the user where the DB is stored
    log.info(f"Database saved at: {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the SQLite .db file required for evaluation."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="llmsql-bench/llmsql-benchmark",
        help="Hugging Face dataset repo ID containing the .db file "
        "(default: llmsql-bench/llmsql-benchmark).",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="dataset",
        help="Local directory to save the downloaded .db file (default: dataset).",
    )
    args = parser.parse_args()

    main(args.target_dir, args.repo_id)
