from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RoutingScore:
    abstract_id: str
    score: float
    score_start: float
    score_end: float
    intra_generation_delta: float
    results_attention_fraction: float
    methods_attention_fraction: float
    background_attention_fraction: float
    n_results_tokens: int
    n_methods_tokens: int
    n_background_tokens: int
    n_layers_used: int


def _compute_attention_fractions(
    abstract_attn: torch.Tensor,
    sentence_map: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute attention fractions from a single attention tensor.
    Returns dict with routing_score, fractions, and token counts.
    """
    total_attn = abstract_attn.sum().item()

    if total_attn == 0:
        return {
            "routing_score": 0.0,
            "results_fraction": 0.0,
            "methods_fraction": 0.0,
            "background_fraction": 0.0,
            "n_results_tokens": 0,
            "n_methods_tokens": 0,
            "n_background_tokens": 0,
            "total_abstract_tokens": len(abstract_attn),
        }

    results_attn = 0.0
    methods_attn = 0.0
    background_attn = 0.0
    n_results = n_methods = n_background = 0

    for sent in sentence_map:
        label = sent["label"]
        positions = [p for p in sent["token_positions"] if p < len(abstract_attn)]
        if not positions:
            continue

        sent_attn = abstract_attn[positions].sum().item()

        if label == "RESULTS":
            results_attn += sent_attn
            n_results += len(positions)
        elif label == "METHODS":
            methods_attn += sent_attn
            n_methods += len(positions)
        else:
            background_attn += sent_attn
            n_background += len(positions)

    return {
        "routing_score": results_attn / total_attn,
        "results_fraction": results_attn / total_attn,
        "methods_fraction": methods_attn / total_attn,
        "background_fraction": background_attn / total_attn,
        "n_results_tokens": n_results,
        "n_methods_tokens": n_methods,
        "n_background_tokens": n_background,
        "total_abstract_tokens": len(abstract_attn),
    }


def compute_routing_score(
    start_attn: torch.Tensor,
    end_attn: torch.Tensor,
    sentence_map: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute routing scores from start and end attention tensors.
    Returns dict with all three scores plus fractions from end attention.
    """
    start_dict = _compute_attention_fractions(start_attn, sentence_map)
    end_dict = _compute_attention_fractions(end_attn, sentence_map)

    score_start = start_dict["routing_score"]
    score_end = end_dict["routing_score"]
    intra_delta = score_end - score_start

    return {
        "routing_score": score_end,
        "score_start": score_start,
        "score_end": score_end,
        "intra_generation_delta": intra_delta,
        "results_fraction": end_dict["results_fraction"],
        "methods_fraction": end_dict["methods_fraction"],
        "background_fraction": end_dict["background_fraction"],
        "n_results_tokens": end_dict["n_results_tokens"],
        "n_methods_tokens": end_dict["n_methods_tokens"],
        "n_background_tokens": end_dict["n_background_tokens"],
        "total_abstract_tokens": end_dict["total_abstract_tokens"],
    }
