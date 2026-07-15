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


class _ZeroUsage:
    """Minimal token-usage stand-in so an extractor that unpacks (text, usage) and reads
    usage.* attributes does not crash under the capture shim."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0


class _CaptureProvider:
    """A009-1: a no-op provider that records the exact (system_prompt, text) the committed
    extractor passes to the model, and returns a minimal valid response so extract()
    completes without a real generation. Swapped in for the real provider during the routing
    capture only."""

    def __init__(self):
        self.calls = []

    def complete_with_usage(self, system_prompt=None, user_message=None, *args, **kwargs):
        self.calls.append((system_prompt, user_message))
        return ('{"claims": []}', _ZeroUsage())


def _capture_extractor_input(abstract_id: str, raw_text: str):
    """A009-1: run the committed extract() on this abstract with the capture shim and return
    the (system_prompt, text) of its FIRST model call — i.e. the actual input the extractor
    feeds the model, preprocessing included. Returns None if the extractor never called the
    model (nothing to measure). Records the first call even if extract() later raises."""
    import asyncio
    import playground.extractor as pg

    shim = _CaptureProvider()
    old = getattr(pg, "_provider", None)
    pg._provider = shim
    try:
        asyncio.run(pg.extract(abstract_id, raw_text))
    except Exception as e:  # a broken extractor still yields its first captured call
        print(f"  [FP] extractor raised during capture for {abstract_id}: "
              f"{type(e).__name__}: {e}", flush=True)
    finally:
        pg._provider = old

    if not shim.calls:
        return None
    return shim.calls[0]


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

    from protected.attention.analyzer import (
        load_attention_model, analyze_abstract, AbstractOffsetUnresolved,
    )
    from protected.harness.shared.anomaly_logger import log_anomaly

    print(f"  [FP] Loading model...", flush=True)
    t0 = time.time()
    model, tokenizer = load_attention_model(model_path)
    print(
        f"  [FP] Model loaded in {time.time()-t0:.0f}s "
        f"({torch.cuda.memory_allocated(next(model.parameters()).device)/1e9:.1f}GB)",
        flush=True,
    )

    def _null_score(abstract_id: str) -> dict:
        return {
            "abstract_id": abstract_id,
            "score": None, "score_start": None, "score_end": None,
            "intra_generation_delta": None,
            "results_attention_fraction": None,
            "methods_attention_fraction": None,
            "background_attention_fraction": None,
            "n_results_tokens": 0, "n_methods_tokens": 0, "n_background_tokens": 0,
            "n_layers_used": 0,
        }

    scores = []
    for idx, aid in enumerate(probe_ids, 1):
        raw_text = _load_abstract_text(aid)
        print(f"  [FP] Analyzing [{idx}/{len(probe_ids)}] {aid}...", flush=True)

        # A009-1: measure routing over the input the COMMITTED extractor actually feeds the
        # model (preprocessing included), not the raw abstract. Capture it via the shim; if
        # the extractor never called the model there is nothing to measure.
        captured = _capture_extractor_input(aid, raw_text)
        if captured is None:
            print(f"  [FP] {aid}: extractor made no model call — routing unmeasurable", flush=True)
            log_anomaly(args.study, args.iteration, "routing_unmeasurable",
                        {"abstract_id": aid, "reason": "no_model_call"})
            scores.append(_null_score(aid))
            gc.collect()
            torch.cuda.empty_cache()
            continue
        cap_system_prompt, cap_text = captured
        cap_system_prompt = cap_system_prompt if isinstance(cap_system_prompt, str) else system_prompt
        if not isinstance(cap_text, str) or not cap_text.strip():
            print(f"  [FP] {aid}: captured empty model input — routing unmeasurable", flush=True)
            log_anomaly(args.study, args.iteration, "routing_unmeasurable",
                        {"abstract_id": aid, "reason": "empty_input"})
            scores.append(_null_score(aid))
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # A004-10: one abstract's failure (OOM re-raised as RuntimeError, offset
        # resolution, etc.) must not terminate the pass. Log, record a null score,
        # and continue.
        try:
            score = analyze_abstract(
                model, tokenizer,
                system_prompt=cap_system_prompt,
                abstract_id=aid,
                abstract_text=cap_text,
            )
            # A009-1: if the (possibly transformed) input contains no locatable RESULTS
            # sentences, results-attention fidelity is undefined here — record a null rather
            # than a misleading 0 so the aggregate averages only over measurable abstracts.
            if score.n_results_tokens == 0:
                print(f"  [FP] {aid}: no RESULTS tokens located in extractor input — "
                      f"routing unmeasurable", flush=True)
                log_anomaly(args.study, args.iteration, "routing_unmeasurable",
                            {"abstract_id": aid, "reason": "no_results_tokens"})
                scores.append(_null_score(aid))
                gc.collect()
                torch.cuda.empty_cache()
                continue
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
        except AbstractOffsetUnresolved as e:  # A004-8
            print(f"  [FP] offset unresolved for {aid}: {e}", flush=True)
            log_anomaly(args.study, args.iteration, "abstract_offset_unresolved",
                        {"abstract_id": aid, "error": str(e)})
            scores.append(_null_score(aid))
        except Exception as e:  # includes OOM re-raised as RuntimeError
            print(f"  [FP] analysis failed for {aid}: {type(e).__name__}: {e}", flush=True)
            log_anomaly(args.study, args.iteration, "attention_abstract_failed",
                        {"abstract_id": aid, "error": f"{type(e).__name__}: {e}"})
            scores.append(_null_score(aid))

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
