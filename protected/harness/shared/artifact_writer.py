import json
from datetime import datetime, timezone
from pathlib import Path

from protected.harness.corpus_runner import CorpusRunResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _study_dir(study_id: str) -> Path:
    return _PROJECT_ROOT / "experiments" / study_id


def _iterations_dir(study_id: str) -> Path:
    return _study_dir(study_id) / "iterations"


def _timestamp_safe() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_iteration_artifacts(
    iteration_n: int, study_id: str, corpus_result: CorpusRunResult
) -> None:
    iters = _iterations_dir(study_id)
    iters.mkdir(parents=True, exist_ok=True)

    ts = _timestamp_safe()
    output_records = []
    abstract_texts = corpus_result.abstract_texts or {}
    for r in corpus_result.results:
        output_records.append({
            "abstract_id": r.abstract_id,
            "abstract_text": abstract_texts.get(r.abstract_id, ""),
            "predicted_claims": [c.claim_text for c in r.claims],
        })
    for f in corpus_result.failures:
        output_records.append({
            "abstract_id": f.abstract_id,
            "error": f.error,
            "predicted_claims": [],
        })

    out_path = iters / f"iteration_{iteration_n:02d}_{ts}.json"
    out_path.write_text(json.dumps(output_records, indent=2, ensure_ascii=False), encoding="utf-8")


def snapshot_playground(iteration_n: int, study_id: str) -> None:
    snap_dir = _iterations_dir(study_id) / f"iteration_{iteration_n:02d}_playground"
    if snap_dir.exists():
        import shutil
        shutil.rmtree(snap_dir)

    src_playground = _PROJECT_ROOT / "playground"
    dst_playground = snap_dir / "playground"
    if src_playground.exists():
        import shutil
        try:
            shutil.copytree(src_playground, dst_playground)
        except PermissionError:
            pass

    src_prompts = _PROJECT_ROOT / "prompts"
    dst_prompts = snap_dir / "prompts"
    if src_prompts.exists():
        import shutil
        try:
            shutil.copytree(src_prompts, dst_prompts)
        except PermissionError:
            pass


def append_metrics(iteration_n: int, study_id: str, metrics: dict) -> None:
    path = _study_dir(study_id) / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"iteration_n": iteration_n, **metrics}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_rationale(iteration_n: int, study_id: str, rationale: str) -> None:
    path = _study_dir(study_id) / "agent-rationale.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "iteration_n": iteration_n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rationale": rationale,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
