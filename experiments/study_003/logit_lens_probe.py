"""Logit Lens Workspace Probe — Study 003 feasibility check.

Implements dev-history/s3-001-logit-lens-workspace-probe.md.

Question: when Qwen 3.6 27B processes a neuroscience abstract and is about to
extract claims, do result-relevant concepts appear in the logit-lens readout at
middle-band layers? If yes, "workspace grounding" is a candidate intrinsic cost
signal to compare against attention routing fidelity (Study 002) in a later
study. This is a feasibility probe, not a pre-registered study: run once,
inspect output, decide.

Usage:
    python experiments/study_003/logit_lens_probe.py [MODEL_PATH]

MODEL_PATH defaults to $TRANSFORMERS_MODEL_PATH, else the local HF cache entry
for Qwen3.6-27B. A HuggingFace cache dir (models--...) is resolved to its
snapshot automatically.
"""
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Concept word lists — what we hope to see surface in the workspace readout.
RESULT_CONCEPT_WORDS = {
    "activation", "activated", "increase", "increased", "decrease", "decreased",
    "correlation", "correlated", "significant", "greater", "reduced", "region",
    "cortex", "gyrus", "hippocampus", "amygdala", "prefrontal", "response",
    "bilateral", "showed", "found", "revealed", "difference", "effect",
}

METHOD_CONCEPT_WORDS = {
    "scanner", "fmri", "mri", "participants", "subjects", "voxel", "protocol",
    "acquired", "recruited", "session", "tesla", "sequence", "analysis",
    "measured", "recorded", "task", "paradigm", "design",
}


def resolve_hf_cache_path(path: str) -> str:
    """Resolve a HuggingFace cache ref (models--org--name) to its snapshot dir."""
    p = Path(path)
    if p.joinpath("config.json").exists():
        return str(p)
    refs_main = p / "refs" / "main"
    if refs_main.exists():
        commit = refs_main.read_text(encoding="utf-8").strip()
        snapshot = p / "snapshots" / commit
        if snapshot.joinpath("config.json").exists():
            return str(snapshot)
    return str(p)


def default_model_path() -> str:
    env = os.environ.get("TRANSFORMERS_MODEL_PATH", "").strip()
    if env:
        return env
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.6-27B"
    return str(cache)


def load_for_logit_lens(model_path: str):
    """Reuse the attention analyzer's 4-bit load. We need hidden states, not attn."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    resolved = resolve_hf_cache_path(model_path)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        quantization_config=quant,
        attn_implementation="eager",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, trust_remote_code=True, use_fast=False, local_files_only=True
    )
    return model, tokenizer


def _final_norm(model):
    """Locate the final RMSNorm applied before lm_head, robust to nesting."""
    for attr_chain in (("model", "norm"), ("model", "model", "norm"), ("norm",)):
        obj = model
        ok = True
        for a in attr_chain:
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok and callable(obj):
            return obj
    raise AttributeError(
        "Could not locate the final norm before lm_head. Inspect print(model)."
    )


def get_hidden_states(model, tokenizer, system_prompt: str, abstract_text: str):
    """Single forward pass; capture hidden states at all layers."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": abstract_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)
    with torch.no_grad():
        out = model(
            **inputs, output_hidden_states=True, use_cache=False, return_dict=True
        )
    return out.hidden_states, inputs["input_ids"]


def logit_lens_at_position(model, tokenizer, hidden_states, layer_idx,
                           position=-1, top_k=30):
    """Apply final norm + unembedding to hidden state at (layer_idx, position)."""
    norm = _final_norm(model)
    h = hidden_states[layer_idx][0, position, :]
    lm_head = model.get_output_embeddings()
    # Match lm_head weight dtype/device (norm output may differ under 4-bit).
    w = lm_head.weight
    with torch.no_grad():
        normed = norm(h).to(device=w.device, dtype=w.dtype)
        logits = lm_head(normed)
    top_vals, top_ids = torch.topk(logits.float(), top_k)
    tokens = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
    return list(zip(tokens, top_vals.tolist()))


DEPTH_FRACTIONS = [0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def sweep_layers(model, tokenizer, system_prompt, abstract_text, out=sys.stdout):
    """Run the logit lens across depths and print/collect the top tokens."""
    hidden_states, _ = get_hidden_states(
        model, tokenizer, system_prompt, abstract_text
    )
    num_layers = len(hidden_states) - 1  # subtract embedding layer
    rows = []
    print(f"Model has {num_layers} layers", file=out)
    print(f"Abstract (first 200 chars): {abstract_text[:200]}...\n", file=out)
    for frac in DEPTH_FRACTIONS:
        layer_idx = int(num_layers * frac)
        top_tokens = logit_lens_at_position(
            model, tokenizer, hidden_states, layer_idx, position=-1, top_k=30
        )
        token_strs = [t.strip() for t, _ in top_tokens]
        low = [t.lower() for t in token_strs]
        r_hits = sorted({t for t in low if t in RESULT_CONCEPT_WORDS})
        m_hits = sorted({t for t in low if t in METHOD_CONCEPT_WORDS})
        print(f"--- Layer {layer_idx} ({int(frac*100)}% depth) "
              f"| result-concepts={len(r_hits)} method-concepts={len(m_hits)} ---",
              file=out)
        print(", ".join(token_strs), file=out)
        if r_hits or m_hits:
            print(f"    result: {r_hits}  |  method: {m_hits}", file=out)
        print(file=out)
        rows.append({
            "layer": layer_idx,
            "depth_pct": int(frac * 100),
            "result_concept_count": len(r_hits),
            "method_concept_count": len(m_hits),
            "result_hits": r_hits,
            "method_hits": m_hits,
            "top_tokens": token_strs,
        })
    return num_layers, rows


def run_probe(model_path, out=sys.stdout):
    print(f"Loading model from: {model_path}", file=out, flush=True)
    model, tokenizer = load_for_logit_lens(model_path)
    print(f"Loaded. dtype-check lm_head: {model.get_output_embeddings().weight.dtype}\n",
          file=out, flush=True)

    system_prompt = (PROJECT_ROOT / "prompts" / "system_prompt.md").read_text(
        encoding="utf-8"
    )
    probe = json.loads(
        (PROJECT_ROOT / "experiments" / "study_002" / "probe_set.json").read_text(
            encoding="utf-8"
        )
    )
    abstract_ids = probe["abstract_ids"][:3]

    record = {"model_path": model_path, "abstracts": []}

    for aid in abstract_ids:
        abstract = json.loads(
            (PROJECT_ROOT / "corpus" / "abstracts" / f"{aid}.json").read_text(
                encoding="utf-8"
            )
        )
        text = abstract.get("text") or abstract.get("abstract", "")
        print("=" * 70, file=out)
        print(f"ABSTRACT {aid}", file=out)
        print("=" * 70, file=out)
        n_layers, rows = sweep_layers(model, tokenizer, system_prompt, text, out=out)
        record["abstracts"].append({"id": aid, "n_layers": n_layers, "sweep": rows})

    synthetics = [
        ("SYNTHETIC RESULTS-ONLY", "synthetic_results_only",
         "Bilateral hippocampal activation increased significantly during "
         "encoding compared to baseline. Left prefrontal cortex showed greater "
         "activation for novel than repeated stimuli (p < 0.001)."),
        ("SYNTHETIC METHODS-ONLY", "synthetic_methods_only",
         "Fifteen healthy volunteers were recruited. fMRI was performed on a "
         "3T scanner with TR=2000ms. Voxel size was 3x3x3mm."),
    ]
    for header, sid, syn_text in synthetics:
        print("=" * 70, file=out)
        print(header, file=out)
        print("=" * 70, file=out)
        n_layers, rows = sweep_layers(model, tokenizer, system_prompt, syn_text, out=out)
        record["abstracts"].append({"id": sid, "n_layers": n_layers, "sweep": rows})

    return record


if __name__ == "__main__":
    # The logit-lens readout contains CJK/other non-latin tokens; force UTF-8 so
    # the console echo does not crash on a cp1252 stdout (Windows default).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    mp = sys.argv[1] if len(sys.argv) > 1 else default_model_path()
    out_dir = Path(__file__).resolve().parent
    txt_path = out_dir / "probe_output.txt"
    json_path = out_dir / "probe_output.json"
    with open(txt_path, "w", encoding="utf-8") as fh:
        rec = run_probe(mp, out=fh)
    json_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    # Echo to console as well for live inspection.
    print(txt_path.read_text(encoding="utf-8"))
    print(f"\nWrote {txt_path} and {json_path}")
