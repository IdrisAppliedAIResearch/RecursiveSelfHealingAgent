"""Prefill-only attention forward pass runner.

Loads the model, runs prefill-only forward passes on probe abstracts,
computes routing scores, and writes results to a JSON file.
Exits immediately, releasing all VRAM.

Usage:
    python -m protected.attention.forward_pass_runner \
        --study study_002 --iteration 0 --output experiments/study_002/attention_scores_0.json
"""
import argparse
import gc
import json
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_path = os.environ.get("TRANSFORMERS_MODEL_PATH", "")
    if not model_path:
        print("ERROR: TRANSFORMERS_MODEL_PATH not set", flush=True)
        sys.exit(1)

    probe_ids = _load_probe_set(args.study)
    system_prompt = _read_system_prompt()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from protected.attention.analyzer import load_attention_model, analyze_abstract

    print(f"  [FP] Loading model...", flush=True)
    t0 = time.time()
    model, tokenizer = load_attention_model(model_path)
    print(
        f"  [FP] Model loaded in {time.time()-t0:.0f}s "
        f"({torch.cuda.memory_allocated(next(model.parameters()).device)/1e9:.1f}GB)",
        flush=True,
    )

    scores = []
    for idx, aid in enumerate(probe_ids, 1):
        abstract_text = _load_abstract_text(aid)
        print(f"  [FP] Analyzing [{idx}/{len(probe_ids)}] {aid}...", flush=True)

        score = analyze_abstract(
            model, tokenizer,
            system_prompt=system_prompt,
            abstract_id=aid,
            abstract_text=abstract_text,
        )

        scores.append({
            "abstract_id": score.abstract_id,
            "score": score.score,
            "score_start": score.score_start,
            "score_end": score.score_end,
            "intra_generation_delta": score.intra_generation_delta,
            "results_attention_fraction": score.results_attention_fraction,
            "methods_attention_fraction": score.methods_attention_fraction,
            "background_attention_fraction": score.background_attention_fraction,
            "n_results_tokens": score.n_results_tokens,
            "n_methods_tokens": score.n_methods_tokens,
            "n_background_tokens": score.n_background_tokens,
            "n_layers_used": score.n_layers_used,
        })

        gc.collect()
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"study_id": args.study, "iteration_n": args.iteration, "scores": scores},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  [FP] Wrote {len(scores)} scores to {output_path}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  [FP] Done. GPU: {torch.cuda.memory_allocated(0)/1e9:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
