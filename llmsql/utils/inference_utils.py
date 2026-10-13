from dataclasses import dataclass
import json
from pathlib import Path
import random
import tempfile
from typing import Any

from huggingface_hub import hf_hub_download
import numpy as np
import torch

from llmsql.config.config import get_repo_id
from llmsql.loggers.logging_config import log
from llmsql.utils.utils import load_jsonl, overwrite_jsonl


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


@dataclass(frozen=True)
class BenchmarkRun:
    """
    一份已经取好、limit 也应用过的基准数据

    Attributes:
        questions: 题目列表
        tables: ``table_id -> table`` 映射
    """

    questions: list[dict[str, Any]]
    tables: dict[str, Any]


def prepare_benchmark(
    *,
    version: str,
    workdir_path: str | Path | None = None,
    limit: int | float | None = None,
) -> BenchmarkRun:
    """
    取基准数据并应用 limit，四个推理后端共用

    这段逻辑原先在 ``inference_transformers`` / ``inference_vllm`` /
    ``inference_api`` / ``inference_function`` 里各抄了一份，任何修复都得改
    四遍。

    Args:
        version: 基准版本
        workdir_path: 下载目录，``None`` 时用临时目录
        limit: 题目数量上限（正整数）或比例（0 到 1 之间的浮点）

    Returns:
        :class:`BenchmarkRun`

    Raises:
        ValueError: ``limit`` 不是正整数也不是 (0, 1] 之间的浮点时抛出
    """
    workdir = resolve_workdir_path(workdir_path)

    repo_id = get_repo_id(version)
    questions_path = _maybe_download(repo_id, "questions.jsonl", workdir)
    tables_path = _maybe_download(repo_id, "tables.jsonl", workdir)

    questions: list[dict[str, Any]] = load_jsonl(questions_path)
    tables_list = load_jsonl(tables_path)
    tables: dict[str, Any] = {t["table_id"]: t for t in tables_list}

    if limit is not None:
        if isinstance(limit, float):
            if not (0.0 < limit <= 1.0):
                raise ValueError(
                    f"When a float, `limit` must be between 0.0 and 1.0, got {limit}."
                )
            limit = max(1, int(len(questions) * limit))
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(
                f"`limit` must be a positive integer or a float in (0.0, 1.0], got {limit!r}."
            )
        log.info(
            f"Limiting evaluation to first {limit} questions out of {len(questions)}"
        )
        questions = questions[:limit]

    return BenchmarkRun(questions=questions, tables=tables)


def _question_id(question: dict[str, Any]) -> str:
    """题目的 ID：基准里写作 ``question_id``，个别写法用 ``id``"""
    return str(question.get("question_id", question.get("id", "")))


def _completed_question_ids(output_file: str) -> set[str]:
    """
    读输出文件里已经写完的 question_id

    上次运行是被杀掉的话，最后一行可能只落了一半，``json.loads`` 会抛错。
    这种残缺行直接跳过（对应题目重跑一次），不能让续跑流程整个崩掉。
    """
    path = Path(output_file)
    if not path.exists():
        return set()

    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            qid = row.get("question_id", row.get("id"))
            if qid is not None:
                done.add(str(qid))

    return done


def prepare_output_file(output_file: str, resume: bool = False) -> set[str]:
    """
    准备输出文件，返回已经完成的 question_id 集合

    ``resume=False``（默认）时清空文件，与旧行为一致；``resume=True`` 且文件里
    已经有结果时保留原文件，让调用方用 :func:`skip_completed` 跳过这批题目。

    Args:
        output_file: 输出文件路径
        resume: 保留输出文件里已有的结果
    """
    if resume:
        done = _completed_question_ids(output_file)
        if done:
            log.info(
                f"Resuming {output_file}: {len(done)} question(s) already answered."
            )
            return done

    overwrite_jsonl(output_file)
    return set()


def skip_completed(
    questions: list[dict[str, Any]], done: set[str]
) -> list[dict[str, Any]]:
    """去掉 ``done`` 里已经出现过的题目"""
    if not done:
        return questions
    return [q for q in questions if _question_id(q) not in done]
