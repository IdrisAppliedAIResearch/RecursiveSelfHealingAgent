import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from protected.harness.shared.corpus_runner import run_corpus
from protected.scorer import score_corpus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STUDY_ID = "study_002"


def _load_ground_truth() -> dict[str, list[str]]:
    gt_path = PROJECT_ROOT / "corpus" / "ground_truth.jsonl"
    gt = {}
    for line in gt_path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            try:
                entry = json.loads(line)
                gt[entry["abstract_id"]] = entry["claims"]
            except json.JSONDecodeError:
                pass
    return gt


async def _run_validation() -> dict:
    abstracts_dir = PROJECT_ROOT / "corpus" / "abstracts"
    abstract_files = sorted(abstracts_dir.glob("*.json"))

    print(f"Running full corpus validation for {STUDY_ID} ({len(abstract_files)} abstracts)...")
    t0 = time.time()
    corpus_result = await run_corpus(STUDY_ID, abstract_files)
    duration = time.time() - t0

    ground_truth = _load_ground_truth()
    score_result = score_corpus(corpus_result.results, ground_truth)

    validation = {
        "study_id": STUDY_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_abstracts": len(abstract_files),
        "successful_extractions": len(corpus_result.results),
        "failed_extractions": len(corpus_result.failures),
        "duration_seconds": round(duration, 2),
        "macro_precision": score_result["macro_precision"],
        "macro_recall": score_result["macro_recall"],
        "macro_f1": score_result["macro_f1"],
        "micro_tp": score_result["micro_tp"],
        "micro_fp": score_result["micro_fp"],
        "micro_fn": score_result["micro_fn"],
        "avg_claims_per_abstract": score_result["avg_claims_per_abstract"],
        "corpus_total_prompt_tokens": corpus_result.corpus_token_usage.total_prompt_tokens,
        "corpus_total_completion_tokens": corpus_result.corpus_token_usage.total_completion_tokens,
    }

    return validation


def main() -> None:
    validation = asyncio.run(_run_validation())

    out_path = PROJECT_ROOT / "experiments" / STUDY_ID / "validation_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    print(f"\nValidation complete. Results written to {out_path}")
    print(f"  Abstracts: {validation['successful_extractions']}/{validation['total_abstracts']}")
    print(f"  Duration: {validation['duration_seconds']:.0f}s")
    print(f"  Macro P/R/F1: {validation['macro_precision']:.4f} / {validation['macro_recall']:.4f} / {validation['macro_f1']:.4f}")
    print(f"  Avg claims/abstract: {validation['avg_claims_per_abstract']:.2f}")


if __name__ == "__main__":
    main()
