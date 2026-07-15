import json
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _study_dir(study_id: str) -> Path:
    return _PROJECT_ROOT / "experiments" / study_id


def _perf_path(study_id: str) -> Path:
    return _study_dir(study_id) / "model_performance.jsonl"


def _metrics_path(study_id: str) -> Path:
    return _study_dir(study_id) / "metrics.jsonl"


def _lock_path(study_id: str) -> Path:
    return _study_dir(study_id) / ".perf_start_ts"


def _read_metrics_lines(study_id: str) -> list[str]:
    mp = _metrics_path(study_id)
    if not mp.exists():
        return []
    lines = mp.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return [l for l in lines if l.strip()]


def _parse_metrics(study_id: str) -> list[dict]:
    records = []
    for line in _read_metrics_lines(study_id):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _safe_get(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _get_run_start(study_id: str) -> float:
    lp = _lock_path(study_id)
    if lp.exists():
        try:
            return float(lp.read_text().strip())
        except ValueError:
            pass
    ts = time.time()
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(str(ts))
    return ts


def append_after_iteration(study_id: str, iteration_n: int) -> dict:
    records = _parse_metrics(study_id)
    if not records:
        return {}

    run_start = _get_run_start(study_id)
    now = time.time()

    cum_prompt = 0.0
    cum_completion = 0.0
    cum_total = 0.0
    cum_agent_prompt = 0.0
    cum_agent_completion = 0.0
    cum_repair_prompt = 0.0
    cum_repair_completion = 0.0
    cum_scan_duration = 0.0
    total_abstracts = 0
    total_repairs = 0
    total_anomalies = 0
    total_successful = 0
    iter_count = 0
    peak_f1 = 0.0
    cum_tp = 0
    cum_fp = 0
    cum_fn = 0

    for rec in records:
        n = rec.get("iteration_n", -1)
        if n < 0:
            continue
        iter_count += 1
        cp = _safe_get(rec, "corpus_total_prompt_tokens")
        cc = _safe_get(rec, "corpus_total_completion_tokens")
        cum_prompt += cp
        cum_completion += cc
        cum_scan_duration += _safe_get(rec, "scan_duration_seconds")
        cum_total += cp + cc
        total_repairs += _safe_get(rec, "repair_attempts")
        if rec.get("anomaly"):
            total_anomalies += 1
        else:
            total_successful += 1
        cum_agent_prompt += _safe_get(rec, "agent_prompt_tokens")
        cum_agent_completion += _safe_get(rec, "agent_completion_tokens")
        cum_total += _safe_get(rec, "agent_total_tokens")
        cum_repair_prompt += _safe_get(rec, "repair_prompt_tokens")
        cum_repair_completion += _safe_get(rec, "repair_completion_tokens")
        cum_total += cum_repair_prompt + cum_repair_completion
        if rec.get("scanned"):
            total_abstracts += 200
        f1 = _safe_get(rec, "macro_f1")
        if f1 > peak_f1:
            peak_f1 = f1
        cum_tp += _safe_get(rec, "micro_tp")
        cum_fp += _safe_get(rec, "micro_fp")
        cum_fn += _safe_get(rec, "micro_fn")

    current_f1 = 0.0
    for rec in reversed(records):
        if rec.get("iteration_n") == iteration_n:
            current_f1 = _safe_get(rec, "macro_f1")
            break

    wall_seconds = now - run_start
    avg_tps = cum_total / wall_seconds if wall_seconds > 0 else 0.0
    avg_corpus_tps = (cum_prompt + cum_completion) / cum_scan_duration if cum_scan_duration > 0 else 0.0
    avg_abstracts_per_second = total_abstracts / wall_seconds if wall_seconds > 0 else 0.0

    overall_prec = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
    overall_rec = cum_tp / (cum_tp + cum_fn) if (cum_tp + cum_fn) > 0 else 0.0

    perf = {
        "iteration_n": iteration_n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iterations_completed": iter_count,
        "iterations_successful": total_successful,
        "iterations_anomalous": total_anomalies,
        "cumulative_total_tokens": int(cum_total),
        "cumulative_prompt_tokens": int(cum_prompt),
        "cumulative_completion_tokens": int(cum_completion),
        "cumulative_agent_prompt_tokens": int(cum_agent_prompt),
        "cumulative_agent_completion_tokens": int(cum_agent_completion),
        "cumulative_repair_prompt_tokens": int(cum_repair_prompt),
        "cumulative_repair_completion_tokens": int(cum_repair_completion),
        "cumulative_scan_duration_seconds": round(cum_scan_duration, 2),
        "wall_clock_seconds": round(wall_seconds, 2),
        "avg_tokens_per_second": round(avg_tps, 2),
        "avg_corpus_tokens_per_second": round(avg_corpus_tps, 2),
        "avg_abstracts_per_second": round(avg_abstracts_per_second, 2),
        "total_abstracts_processed": total_abstracts,
        "total_repair_attempts": int(total_repairs),
        "iteration_macro_f1": round(current_f1, 4),
        "peak_macro_f1": round(peak_f1, 4),
        "cumulative_micro_tp": int(cum_tp),
        "cumulative_micro_fp": int(cum_fp),
        "cumulative_micro_fn": int(cum_fn),
        "cumulative_micro_precision": round(overall_prec, 4),
        "cumulative_micro_recall": round(overall_rec, 4),
    }

    mp = _perf_path(study_id)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "a", encoding="utf-8") as f:
        f.write(json.dumps(perf) + "\n")

    return perf


def summarize(study_id: str) -> None:
    mp = _perf_path(study_id)
    if not mp.exists():
        print("No model_performance data found.")
        return

    lines = mp.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if not lines:
        print("No model_performance data found.")
        return

    print("\n=== Model Performance Summary ===")
    print(f"{'Iter':>4} {'Wall(s)':>8} {'Tokens':>12} {'TPS':>8} "
          f"{'Corpus TPS':>11} {'Abst/s':>8} "
          f"{'Scan(s)':>9} {'Abst':>7} "
          f"{'F1':>7} {'Peak F1':>8} {'OK':>4} {'Err':>4}")
    print("-" * 100)

    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        print(
            f"{rec.get('iteration_n', '?'):>4} "
            f"{rec.get('wall_clock_seconds', 0):>8.1f} "
            f"{rec.get('cumulative_total_tokens', 0):>12,} "
            f"{rec.get('avg_tokens_per_second', 0):>8.1f} "
            f"{rec.get('avg_corpus_tokens_per_second', 0):>11.1f} "
            f"{rec.get('avg_abstracts_per_second', 0):>8.2f} "
            f"{rec.get('cumulative_scan_duration_seconds', 0):>9.1f} "
            f"{rec.get('total_abstracts_processed', 0):>7,} "
            f"{rec.get('iteration_macro_f1', 0):>7.4f} "
            f"{rec.get('peak_macro_f1', 0):>8.4f} "
            f"{rec.get('iterations_successful', 0):>4} "
            f"{rec.get('iterations_anomalous', 0):>4}"
        )

    print()
