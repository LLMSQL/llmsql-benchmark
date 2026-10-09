from collections.abc import Callable, Iterable
import json
from pathlib import Path

from llmsql.loggers.logging_config import log
from llmsql.prompts.prompts import (
    build_prompt_0shot,
    build_prompt_1shot,
    build_prompt_5shot,
    build_prompt_v2,
)


# ---------- Utility functions ----------
def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl_dict_by_key(path: str, key: str) -> dict:
    """Load JSONL file into dict indexed by `key`."""
    with open(path, encoding="utf-8") as f:
        return {item[key]: item for item in map(json.loads, f)}


def save_jsonl_lines(path: str, items: Iterable[dict]) -> None:
    """Append lines (dicts) to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json_report(path: str, data: dict) -> None:
    """Save JSON report."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"Saved evaluation report to {path}")


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


def build_question_prompt(
    q: dict,
    tables: dict,
    prompt_builder: Callable[[str, list[str], list[str], list[str | float | int]], str],
) -> str:
    """
    Build the raw prompt text (without chat template) for one question.

    LLMSQL 2.0 questions (those with a ``tables`` field) use the zero-shot
    :func:`llmsql.prompts.prompts.build_prompt_v2` prompt with every table in
    ``q["tables"]`` (target plus distractors, in order); ``prompt_builder`` is
    ignored for them. LLMSQL 1.0 questions use ``prompt_builder`` with the
    header, types and first row of ``q["table_id"]``.

    Args:
        q: Question dict.
        tables: Dict mapping table_id to table metadata.
        prompt_builder: Prompt builder for LLMSQL 1.0 questions.

    Returns:
        The prompt text.
    """
    if "tables" in q:
        return build_prompt_v2(q["question"], [tables[t] for t in q["tables"]])

    tbl = tables[q["table_id"]]
    example_row = tbl["rows"][0] if tbl["rows"] else []
    return prompt_builder(q["question"], tbl["header"], tbl["types"], example_row)


def build_all_requests(
    questions: list[dict],
    tables: dict,
    prompt_builder: Callable[[str, list[str], list[str], list[str | float | int]], str],
    tokenizer=None,
    use_chat_template: bool = True,
) -> list[str]:
    """
    Build all prompts from questions and tables.

    Args:
        questions: List of question dicts with 'question' and 'table_id' keys.
            LLMSQL 2.0 questions also have a 'tables' key and get the zero-shot
            2.0 prompt (see :func:`build_question_prompt`).
        tables: Dict mapping table_id to table metadata (with 'header', 'types', 'rows').
        prompt_builder: Function to build raw prompt text for LLMSQL 1.0 questions.
        tokenizer: Optional tokenizer with apply_chat_template method.
        use_chat_template: Whether to apply chat template (if tokenizer provided).

    Returns:
        List of final prompts (with chat template applied if requested).
    """
    prompts = []
    for q in questions:
        raw_text = build_question_prompt(q, tables, prompt_builder)

        if tokenizer and use_chat_template:
            messages = [{"role": "user", "content": raw_text}]
            final_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            final_prompt = raw_text

        prompts.append(final_prompt)

    return prompts
