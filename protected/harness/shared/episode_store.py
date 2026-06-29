import json
from pathlib import Path

from protected.harness.shared.edit_protocol import Episode

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _episodes_path(study_id: str) -> Path:
    return _PROJECT_ROOT / "experiments" / study_id / "episodes.jsonl"


def append(study_id: str, iteration_n: int, episode: Episode) -> None:
    path = _episodes_path(study_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "iteration_n": iteration_n,
        "observation": episode.observation,
        "hypothesis": episode.hypothesis,
        "action": episode.action,
        "expectation": episode.expectation,
        "edits_applied": getattr(episode, "edits_applied", False),
        "field_failures": getattr(episode, "field_failures", []),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_all(study_id: str) -> list[dict]:
    path = _episodes_path(study_id)
    if not path.exists():
        return []
    episodes = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    episodes.sort(key=lambda e: e.get("iteration_n", 0))
    return episodes


def count(study_id: str) -> int:
    return len(load_all(study_id))
