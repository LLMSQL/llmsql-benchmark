import json
from pathlib import Path
import shutil

import yaml

BASE_DIR = Path(__file__).parent
DOCS_DIR = Path(__file__).parent.parent / "docs/_static"
BUILD_DIR = DOCS_DIR / "_build/html"


def run_to_row(data: dict) -> dict | None:
    """Convert a parsed run.yaml into a leaderboard row.

    Returns None when the run has no execution accuracy (cannot be ranked).
    Missing optional fields (e.g. produced by ``evaluate(save_leaderboard_yaml=...)``
    without full ``run_metadata``) are tolerated.
    """
    model = data.get("model") or {}
    inference = data.get("inference") or {}
    arguments = inference.get("arguments") or {}
    results = data.get("results") or {}

    accuracy = results.get("execution_accuracy")
    if accuracy is None:
        return None

    return {
        "model": model.get("name") or "",
        "type": data.get("type") or "",
        "fewshots": arguments.get("num_fewshots"),
        "backend": inference.get("backend") or "",
        "accuracy": accuracy,
        "date": str(data.get("date", "")),
    }


def collect_rows(base_dir: Path = BASE_DIR) -> list[dict]:
    rows = []
    for path in base_dir.rglob("run.yaml"):
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        row = run_to_row(data)
        if row is None:
            print(f"⚠️  Skipping {path}: no results.execution_accuracy")
            continue
        rows.append(row)

    rows.sort(key=lambda x: x["accuracy"], reverse=True)
    return rows


def main() -> None:
    rows = collect_rows()

    json_file = DOCS_DIR / "leaderboard.json"
    with open(json_file, "w") as f:
        json.dump(rows, f, indent=2)

    if BUILD_DIR.exists():
        shutil.copy(json_file, BUILD_DIR / "leaderboard.json")

    print(f"✅ leaderboard.json in {json_file}")


if __name__ == "__main__":
    main()
