import yaml
import json
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).parent
DOCS_DIR = Path(__file__).parent.parent / "docs/_static"
BUILD_DIR = DOCS_DIR / "_build/html"

rows = []

for path in BASE_DIR.rglob("run.yaml"):
    with open(path) as f:
        data = yaml.safe_load(f)

    rows.append({
        "model": data["model"]["name"],
        "type": data.get("type", ""),
        "fewshots": data["inference"]["arguments"]["num_fewshots"],
        "backend": data["inference"]["backend"],
        "accuracy": data["results"]["execution_accuracy"],
        "date": str(data["date"]),
    })

rows.sort(key=lambda x: x["accuracy"], reverse=True)

json_file = DOCS_DIR / "leaderboard.json"
with open(json_file, "w") as f:
    json.dump(rows, f, indent=2)

if BUILD_DIR.exists():
    shutil.copy(json_file, BUILD_DIR / "leaderboard.json")

print(f"✅ leaderboard.json in {json_file}")