"""Build the small LLMSQL 2.0 test fixture from a local copy of the release files.

The fixture (questions.jsonl, tables.jsonl, sqlite_tables.db in this folder) is
committed to the repository, so the tests need neither network access nor the
full dataset. Re-run this script only to regenerate it:

    python tests/fixtures/llmsql2/build_fixture.py /path/to/llmsql-2.0

where the directory contains the Hugging Face files of llmsql-bench/llmsql-2.0
(questions.jsonl, tables.jsonl, sqlite_tables.db).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

# A mix of all categories: lookup, text_quantity, single-table convention and
# convention questions with a distractor table (614: the target is not first).
QUESTION_IDS = [
    # lookup
    2, 12, 24, 37, 51, 144,
    # text_quantity
    1, 42, 43, 63, 97, 158,
    # convention, one table
    3, 9, 18, 40, 69,
    # convention, target + distractor table
    187, 614, 1099, 1345,
]  # fmt: skip

HERE = Path(__file__).resolve().parent


def build(release_dir: Path, out_dir: Path = HERE) -> None:
    with open(release_dir / "questions.jsonl", encoding="utf-8") as f:
        questions = {q["question_id"]: q for q in map(json.loads, f)}
    selected = [questions[i] for i in QUESTION_IDS]

    needed: list[str] = []
    for q in selected:
        for t in q["tables"]:
            if t not in needed:
                needed.append(t)

    with open(release_dir / "tables.jsonl", encoding="utf-8") as f:
        tables = {t["table_id"]: t for t in map(json.loads, f)}

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in selected:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(out_dir / "tables.jsonl", "w", encoding="utf-8") as f:
        for t in needed:
            f.write(json.dumps(tables[t], ensure_ascii=False) + "\n")

    db_path = out_dir / "sqlite_tables.db"
    db_path.unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{release_dir / 'sqlite_tables.db'}?mode=ro", uri=True)
    dst = sqlite3.connect(db_path)
    for tid in needed:
        (ddl,) = src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tid,)
        ).fetchone()
        dst.execute(ddl)
        rows = src.execute(f'SELECT * FROM "{tid}"').fetchall()
        if rows:
            ph = ",".join("?" * len(rows[0]))
            dst.executemany(f'INSERT INTO "{tid}" VALUES ({ph})', rows)
    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    src.close()
    print(f"{len(selected)} questions, {len(needed)} tables -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    build(parser.parse_args().release_dir)
