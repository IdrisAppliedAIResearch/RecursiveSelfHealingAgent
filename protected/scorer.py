from rapidfuzz.fuzz import token_sort_ratio


def score(
    predicted: list[str],
    ground_truth: list[str],
    threshold: int = 80,
) -> dict:
    matched = set()
    tp = 0
    for pred in predicted:
        best_ratio = 0
        best_idx = None
        for i, gt in enumerate(ground_truth):
            if i in matched:
                continue
            ratio = token_sort_ratio(pred, gt)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
        if best_ratio >= threshold and best_idx is not None:
            tp += 1
            matched.add(best_idx)
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def score_corpus(
    extraction_results: list,
    ground_truth: dict[str, list[str]],
    threshold: int = 80,
) -> dict:
    per_abstract = []
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    precisions = []
    recalls = []
    total_claims = 0

    for result in extraction_results:
        abstract_id = result.abstract_id
        gt_claims = ground_truth.get(abstract_id, [])
        pred_claims = [c.claim_text for c in result.claims]
        total_claims += len(pred_claims)

        s = score(pred_claims, gt_claims, threshold)
        precisions.append(s["precision"])
        recalls.append(s["recall"])
        micro_tp += s["tp"]
        micro_fp += s["fp"]
        micro_fn += s["fn"]

    n = len(extraction_results)
    avg_claims = total_claims / n if n else 0.0
    macro_p = sum(precisions) / n if n else 0.0
    macro_r = sum(recalls) / n if n else 0.0
    macro_f1 = (2 * macro_p * macro_r / (macro_p + macro_r)
                if (macro_p + macro_r) > 0 else 0.0)

    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_tp": micro_tp,
        "micro_fp": micro_fp,
        "micro_fn": micro_fn,
        "avg_claims_per_abstract": avg_claims,
    }
