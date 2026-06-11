"""Standalone attention forward pass runner.

Loads the model, runs forward passes on probe abstracts, computes routing scores,
and writes results to a JSON file. Exits immediately, releasing all VRAM.

Usage:
    python -m protected.attention.forward_pass_runner \
        --study study_002 --iteration 0 --output experiments/study_002/attention_scores_0.json
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_abstract_text(abstract_id: str) -> str:
    af = PROJECT_ROOT / "corpus" / "abstracts" / f"{abstract_id}.json"
    data = json.loads(af.read_text(encoding="utf-8", errors="replace"))
    return data.get("abstract", data.get("text", ""))


def _load_probe_set(study_id: str) -> list[str]:
    probe_path = PROJECT_ROOT / "experiments" / study_id / "probe_set.json"
    data = json.loads(probe_path.read_text(encoding="utf-8"))
    return data["abstract_ids"]


def _read_system_prompt() -> str:
    prompts_dir = PROJECT_ROOT / "prompts"
    sp = prompts_dir / "system_prompt.md"
    examples = prompts_dir / "examples.md"
    prompt = sp.read_text(encoding="utf-8") if sp.exists() else ""
    ex = examples.read_text(encoding="utf-8").strip() if examples.exists() else ""
    if ex:
        prompt = prompt + "\n\n" + ex
    return prompt


def _load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False, local_files_only=True
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def _run_forward_pass(model, tokenizer, abstract_text: str, system_prompt: str, n_last_layers: int = 6):
    stored_weights = {}
    blocks = model.model.layers
    start = max(0, len(blocks) - n_last_layers)
    hooks = []

    def make_hook(layer_idx):
        def hook(module, args, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None:
                    stored_weights[layer_idx] = attn_weights[-1].detach().cpu()
        return hook

    for i in range(start, len(blocks)):
        attn_module = getattr(blocks[i], "self_attn", None)
        if attn_module is not None:
            hooks.append(attn_module.register_forward_hook(make_hook(i)))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": abstract_text},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    enc = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    try:
        with torch.no_grad():          # ← no gradient graph
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,       # ← no KV cache spike
                output_attentions=False,
            )
    finally:
        for hook in hooks:
            hook.remove()

    return stored_weights


def _compute_routing_score(abstract_id, abstract_text, attention_weights, segments, tokenizer, system_prompt):
    from protected.attention.scorer import compute_routing_score
    from protected.attention.analyzer import AttentionResult

    result = AttentionResult(
        abstract_id=abstract_id,
        abstract_text=abstract_text,
        attention_weights=attention_weights,
    )
    score = compute_routing_score(result, segments, tokenizer, system_prompt)
    return {
        "abstract_id": score.abstract_id,
        "score": score.score,
        "results_attention_fraction": score.results_attention_fraction,
        "methods_attention_fraction": score.methods_attention_fraction,
        "background_attention_fraction": score.background_attention_fraction,
        "n_results_tokens": score.n_results_tokens,
        "n_methods_tokens": score.n_methods_tokens,
        "n_background_tokens": score.n_background_tokens,
        "n_layers_used": score.n_layers_used,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_path = sys.argv[0]  # will be overridden by env
    import os
    model_path = os.environ.get("TRANSFORMERS_MODEL_PATH", "")
    if not model_path:
        print("ERROR: TRANSFORMERS_MODEL_PATH not set", flush=True)
        sys.exit(1)

    probe_ids = _load_probe_set(args.study)
    system_prompt = _read_system_prompt()

    print(f"  [FP] Loading model...", flush=True)
    t0 = time.time()
    model, tokenizer = _load_model(model_path)
    print(f"  [FP] Model loaded in {time.time()-t0:.0f}s ({torch.cuda.memory_allocated(0)/1e9:.1f}GB)", flush=True)

    from protected.attention.segmenter import segment_abstract, align_tokens

    scores = []
    for idx, aid in enumerate(probe_ids, 1):
        abstract_text = _load_abstract_text(aid)
        print(f"  [FP] Forward pass [{idx}/{len(probe_ids)}] {aid}...", flush=True)

        weights = _run_forward_pass(model, tokenizer, abstract_text, system_prompt)
        segments = segment_abstract(abstract_text)
        align_tokens(segments, tokenizer, abstract_text)
        score = _compute_routing_score(aid, abstract_text, weights, segments, tokenizer, system_prompt)
        scores.append(score)

        del weights
        gc.collect()
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"study_id": args.study, "iteration_n": args.iteration, "scores": scores}, indent=2),
        encoding="utf-8",
    )
    print(f"  [FP] Wrote {len(scores)} scores to {output_path}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  [FP] Done. GPU: {torch.cuda.memory_allocated(0)/1e9:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
