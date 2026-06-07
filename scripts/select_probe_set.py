import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus" / "abstracts"
PROBE_SET_PATH = PROJECT_ROOT / "experiments" / "study_002" / "probe_set.json"
PROBE_COUNT = 10


def select_probe_set() -> None:
    abstract_files = sorted(CORPUS_DIR.glob("*.json"))
    if not abstract_files:
        print(f"ERROR: No abstract files found in {CORPUS_DIR}", file=sys.stderr)
        sys.exit(1)

    corpus_size = len(abstract_files)
    abstract_ids = [f.stem for f in abstract_files]
    selected_ids = random.sample(abstract_ids, PROBE_COUNT)
    selected_ids.sort()

    probe_set = {
        "study_id": "study_002",
        "selection_method": "random",
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "corpus_size": corpus_size,
        "probe_count": PROBE_COUNT,
        "abstract_ids": selected_ids,
    }

    PROBE_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROBE_SET_PATH.write_text(
        json.dumps(probe_set, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Probe set written to {PROBE_SET_PATH}")
    print(f"Selected {PROBE_COUNT} abstracts from corpus of {corpus_size}:")
    for aid in selected_ids:
        print(f"  {aid}")


def commit_probe_set() -> None:
    result = subprocess.run(
        ["git", "add", str(PROBE_SET_PATH.relative_to(PROJECT_ROOT))],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: git add failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "[study_002] commit probe set — 10 fixed control abstracts",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: git commit failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Probe set committed.")


if __name__ == "__main__":
    select_probe_set()
    commit_probe_set()
