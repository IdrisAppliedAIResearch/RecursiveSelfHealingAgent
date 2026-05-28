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
