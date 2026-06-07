from dataclasses import dataclass

from protected.attention.analyzer import AttentionResult
from protected.attention.segmenter import Sentence


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
    attention_result: AttentionResult,
    segments: list[Sentence],
    tokenizer,
) -> RoutingScore:
    abstract_text = attention_result.abstract_text
    encoding = tokenizer(abstract_text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    abstract_tokens = set()
    for idx, (t_start, t_end) in enumerate(offsets):
        if t_start >= 0 and t_start < len(abstract_text):
            abstract_tokens.add(idx)

    if not abstract_tokens:
        return RoutingScore(
            abstract_id=attention_result.abstract_id,
            score=None,  # type: ignore
            results_attention_fraction=0.0,
            methods_attention_fraction=0.0,
            background_attention_fraction=0.0,
            n_results_tokens=0,
            n_methods_tokens=0,
            n_background_tokens=0,
            n_layers_used=0,
        )

    results_tokens = set()
    methods_tokens = set()
    background_tokens = set()

    for sent in segments:
        for idx in range(sent.token_start, sent.token_end):
            if idx in abstract_tokens:
                if sent.label == "RESULTS":
                    results_tokens.add(idx)
                elif sent.label == "METHODS":
                    methods_tokens.add(idx)
                else:
                    background_tokens.add(idx)

    layers_used = 0
    results_fractions = []
    methods_fractions = []
    background_fractions = []

    for layer_idx, weights in attention_result.attention_weights.items():
        if weights is None:
            continue
        layers_used += 1

        attn_avg = weights.mean(dim=0)
        total_attn = 0.0
        results_attn = 0.0
        methods_attn = 0.0
        background_attn = 0.0

        for tok_idx in abstract_tokens:
            if tok_idx < attn_avg.shape[0]:
                w = attn_avg[tok_idx].item()
                total_attn += w
                if tok_idx in results_tokens:
                    results_attn += w
                elif tok_idx in methods_tokens:
                    methods_attn += w
                elif tok_idx in background_tokens:
                    background_attn += w

        if total_attn > 0:
            results_fractions.append(results_attn / total_attn)
            methods_fractions.append(methods_attn / total_attn)
            background_fractions.append(background_attn / total_attn)
        else:
            results_fractions.append(0.0)
            methods_fractions.append(0.0)
            background_fractions.append(0.0)

    if not results_fractions:
        results_frac = 0.0
        methods_frac = 0.0
        background_frac = 0.0
    else:
        results_frac = sum(results_fractions) / len(results_fractions)
        methods_frac = sum(methods_fractions) / len(methods_fractions)
        background_frac = sum(background_fractions) / len(background_fractions)

    score = results_frac

    return RoutingScore(
        abstract_id=attention_result.abstract_id,
        score=score,
        results_attention_fraction=results_frac,
        methods_attention_fraction=methods_frac,
        background_attention_fraction=background_frac,
        n_results_tokens=len(results_tokens),
        n_methods_tokens=len(methods_tokens),
        n_background_tokens=len(background_tokens),
        n_layers_used=layers_used,
    )
