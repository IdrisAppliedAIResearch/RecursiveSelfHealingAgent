from dataclasses import dataclass

import torch

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
    system_prompt: str,
) -> RoutingScore:
    abstract_text = attention_result.abstract_text

    # Tokenize full chat to find where abstract tokens start
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": abstract_text},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    full_enc = tokenizer(chat_text, return_offsets_mapping=True)
    full_offsets = full_enc["offset_mapping"]

    # Find character position of abstract text in chat template output
    abstract_char_start = chat_text.find(abstract_text)

    # Find first token whose character range starts at or after abstract text
    abstract_token_offset = len(full_offsets)
    for i, (s, e) in enumerate(full_offsets):
        if s >= abstract_char_start:
            abstract_token_offset = i
            break

    # Tokenize just abstract for offset mapping
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

        attn_avg = weights.mean(dim=0)  # [from_pos, to_pos]
        to_seq_len = attn_avg.shape[1]

        # Map abstract-relative indices to full-input indices
        results_full = sorted(
            i + abstract_token_offset for i in results_tokens
            if i + abstract_token_offset < to_seq_len
        )
        methods_full = sorted(
            i + abstract_token_offset for i in methods_tokens
            if i + abstract_token_offset < to_seq_len
        )
        background_full = sorted(
            i + abstract_token_offset for i in background_tokens
            if i + abstract_token_offset < to_seq_len
        )

        # Column sums: total attention received by each category
        results_attn = (
            attn_avg[:, results_full].sum().item() if results_full else 0.0
        )
        methods_attn = (
            attn_avg[:, methods_full].sum().item() if methods_full else 0.0
        )
        background_attn = (
            attn_avg[:, background_full].sum().item() if background_full else 0.0
        )
        total = results_attn + methods_attn + background_attn

        if total > 0:
            results_fractions.append(results_attn / total)
            methods_fractions.append(methods_attn / total)
            background_fractions.append(background_attn / total)
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
