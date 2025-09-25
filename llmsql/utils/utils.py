from collections.abc import Callable, Iterable
import json
from pathlib import Path

from llmsql.prompts.prompts import (
    build_prompt_0shot,
    build_prompt_1shot,
    build_prompt_5shot,
)


# ---------- Utility functions ----------
def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl_lines(path: str, items: Iterable[dict]) -> None:
    """Append lines (dicts) to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def overwrite_jsonl(path: str) -> None:
    """Ensure output file is empty (start fresh)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass


def choose_prompt_builder(
    shots: int,
) -> Callable[[str, list[str], list[str], list[str | float | int]], str]:
    """
    Return a prompt-building function according to shots.
    The returned callable signature matches how prompts are used below:
        (question, header, types, example_row) -> prompt_str
    """
    if shots == 0:
        return build_prompt_0shot
    if shots == 1:
        return build_prompt_1shot
    if shots == 5:
        return build_prompt_5shot
    raise ValueError("shots must be one of {0, 1, 5}")
