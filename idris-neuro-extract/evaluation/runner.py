import json
import os
from datetime import datetime, timezone
from pathlib import Path

from extractor.extractor import Extractor
from evaluation.scorer import score


def run(threshold: int = 80) -> dict:
    base = Path(__file__).parent.parent
    abstracts_dir = base / "corpus" / "abstracts"
    gt_path = base / "corpus" / "ground_truth.jsonl"
    results_dir = base / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt = {}
    for line in gt_path.read_text().strip().splitlines():
        entry = json.loads(line)
        gt[entry["abstract_id"]] = entry["claims"]

    ext = Extractor()
    per_abstract = []
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    precisions = []
    recalls = []

    for af in sorted(abstracts_dir.glob("*.json")):
        abstract_id = af.stem
        abstract_data = json.loads(af.read_text())
        abstract_text = abstract_data.get("abstract", abstract_data.get("text", ""))
        gt_claims = gt.get(abstract_id, [])

        try:
            result = ext.extract(abstract_id, abstract_text)
            pred_claims = [c.claim_text for c in result.claims]
        except Exception as e:
            print(f"Error on {abstract_id}: {e}")
            pred_claims = []

        s = score(pred_claims, gt_claims, threshold)
        per_abstract.append({
            "abstract_id": abstract_id,
            "predicted_claims": pred_claims,
            "ground_truth_claims": gt_claims,
            "precision": s["precision"],
            "recall": s["recall"],
            "f1": s["f1"],
            "tp": s["tp"],
            "fp": s["fp"],
            "fn": s["fn"],
        })
        micro_tp += s["tp"]
        micro_fp += s["fp"]
        micro_fn += s["fn"]
        precisions.append(s["precision"])
        recalls.append(s["recall"])

    n = len(per_abstract)
    macro_p = sum(precisions) / n if n else 0.0
    macro_r = sum(recalls) / n if n else 0.0
    macro_f1 = (2 * macro_p * macro_r / (macro_p + macro_r)
                if (macro_p + macro_r) > 0 else 0.0)

    model = os.environ.get("LLAMA_CPP_MODEL_ID", "qwen3-27b-mtp-6bit")
    timestamp = datetime.now(timezone.utc).isoformat()

    report = {
        "run_timestamp": timestamp,
        "model": model,
        "match_threshold": threshold,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_tp": micro_tp,
        "micro_fp": micro_fp,
        "micro_fn": micro_fn,
        "per_abstract": per_abstract,
    }

    out_path = results_dir / f"run_{timestamp.replace(':', '-')}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Results written to {out_path}")
    print(f"Macro P={macro_p:.4f} R={macro_r:.4f} F1={macro_f1:.4f}")
    print(f"Micro TP={micro_tp} FP={micro_fp} FN={micro_fn}")
    return report
