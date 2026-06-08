import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def log_anomaly(study_id: str, iteration_n: int, anomaly_type: str, details: dict | None = None) -> None:
    path = _PROJECT_ROOT / "experiments" / study_id / "anomalies.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "iteration_n": iteration_n,
        "anomaly_type": anomaly_type,
    }
    if details:
        record["details"] = details
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
