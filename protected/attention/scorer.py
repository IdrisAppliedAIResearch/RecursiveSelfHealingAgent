from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RoutingScore:
    abstract_id: str
    score: float
    results_attention_fraction: float
    methods_attention_fraction: float
    background_attention_fraction: float
    n_results_tokens: int
    n_methods_tokens: int
    n_background_tokens: int
    n_layers_used: int


def compute_routing_score(
    abstract_attn: torch.Tensor,
    sentence_map: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Returns routing score dict:
    {
        "routing_score": float,
        "results_fraction": float,
        "methods_fraction": float,
        "background_fraction": float,
        "n_results_tokens": int,
        "n_methods_tokens": int,
        "n_background_tokens": int,
        "total_abstract_tokens": int,
    }
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
