# Amendment 006 (spec s2-009) — Edit Conditioning, Edit Sampling & No-Progress Breaker

**Status:** registered
**Date:** 2026-07-06
**Branch:** `experiment/study_002`
**Supersedes/extends:** builds on Amendment 005 (`s2-008`); narrows A004-3 for the
`edits` call only (see A006-2). No other A004/A005 decision changes.

## Motivation — the observed failure (post-A005 run, iters 0–25)

The A005 rerun did **not** reproduce the `no_match` dead-loop. It produced a *new mask
of the same fixed-point stall*:

- iters 0: baseline F1 = 0.603 ✓
- iters 1–3: the agent builds a 118-line "Anchor-Refine" pipeline; edits apply; F1 wanders
  0.60 → 0.66 → **0.48**.
- iters 4–25: **every iteration proposes 0 edits** (empty `[]`); `playground/extractor.py`
  is byte-identical across iters 3→25; F1 frozen at 0.484. 22 iterations, zero progress.

Anomalies: `203 context_truncated`, `1 field_call_failed`. No `apply_repair_*`, no halt.

### Root cause — the same greedy fixed point as the last run, different surface

1. **The `edits` field-call decodes greedily** (`agent_caller.invoke_edits`, A004-3).
2. **Head-first truncation preserves a constant tail.** `_budget_user_message`
   (`analyzer.py:609`) keeps `user_ids[-user_cap:]`. The user message is
   `assessment + episodic-memory + FILES + instruction`. From iter 3 the FILES are
   byte-identical and the schema instruction is fixed — that is the preserved tail. The
   *varying* content (assessment, episodic memory) is the head that gets trimmed
   (edits call: 5896 → 4716 tokens). So the effective edits prompt is near-constant.
3. **Constant prompt + greedy decode ⇒ one pinned output.** That output is `[]`.
4. **The edits generator is decoupled from the agent's own plan.** The `action` field
   (a *separate* call) says "I will refactor… remove the redundant preprocessing" every
   iteration, but `invoke_edits` never receives that plan — it re-decides from the
   assessment alone, which from iter 3 frames the pipeline as "successful," and declines.
5. **No detection.** `apply_edits([])` returns `applied=True` (a silent no-op), so no
   anomaly is logged; and the A005 circuit breaker only counts non-scanned iterations
   (`rationale is None`). A 0-edit iteration scans fine, so `consecutive_anomalies` resets
   to 0 every time and the breaker never fires. The stall is invisible and burns the budget.

Why it flipped from `no_match` (Amendment 005's run) to `[]` (this run): last run the file
stayed small enough to fit, so the pinned greedy output was a stale non-matching *edit*;
this run the agent built a large pipeline at iter 3, the context exceeded the 5120 budget,
truncation began discarding the varying head, and the pinned output became the *empty array*.

Story-1 (agent drives F1 down chasing the routing signal) reproduced again and remains a
kept soft-negative finding; it is the science, not the bug.

## Decisions

### A006-1 — Condition the edits call on the agent's stated plan (primary fix)
`invoke_edits` now receives the iteration's `hypothesis` + `action` text and appends it as
an explicit "STATED PLAN — implement exactly this" block at the **tail** of the user
message, immediately before the schema instruction. Two effects: (a) the structured edit is
generated to implement the plan the narrator already committed to, closing the
narrator/generator split; (b) because the plan varies every iteration and sits in the
truncation-preserved tail, the edits prompt is no longer near-constant — the greedy fixed
point is broken structurally, not just by sampling.

### A006-2 — Sample the edits call
`invoke_edits` decodes with `do_sample=True` (was greedy per A004-3). A006-2 narrows A004-3:
the primary *decision* prose and *extraction* remain greedy/aligned; only the `edits`
generation samples, so a near-constant prompt cannot pin to a single output. This mirrors
A005-5 (repair sampling). Retry turns already varied the prompt; they stay sampled.

### A006-3 — No-progress detection & circuit breaker
- Log a `no_edits_proposed` anomaly whenever the decision returns 0 edits (records whether
  the `action` text expressed edit intent, for later analysis).
- In `_run_study_async`, track `consecutive_no_change` — iterations that committed with
  **0 applied edits** (empty-edit no-ops *and* rolled-back anomalies both count as "no
  change to the playground"). Any iteration that applies ≥1 edit resets it. At
  `_MAX_CONSECUTIVE_NO_CHANGE` (= 5) log `study_halted_no_progress` and stop early. This
  caps a static stall at ~5 iterations instead of ~20, and also cleanly ends a genuinely
  converged study (operator inspects the anomaly and decides).

### A006-4 — Reduce the diagnostic-call truncation
The Call-1 diagnostic context dumps **all** prior episodes plus the full routing history,
reaching ~9164 tokens and getting cut nearly in half (→ ~4720). Window the diagnostic
episodes to the last 8 (mirroring the decision call's `[-5:]` windowing) so the assessment
is computed over intact context rather than a head-truncated fragment. **The input budget
`MAX_INPUT_TOKENS` (5120) is left unchanged** — it is validated against the RTX 5090 VRAM
ceiling, which is set by the O(seq²) eager attention-capture matrix (`use_cache=False`),
*not* by the KV cache. See "KV-cache note" below.

### KV-cache note (investigated, not adopted here)
Quantizing the K/V cache (as with llama.cpp `--cache-type-k/-v q8_0`) was considered to
raise the token budget. Findings:
- The VRAM ceiling on this stack is the **quadratic eager attention matrix** materialized
  in `AttentionAnalyzer.forward_pass` (the routing measurement, `use_cache=False`), not the
  KV cache — so cache quantization cannot raise the attention-path budget and cannot
  "double tokens."
- Generation (extraction + agent calls) runs on **SDPA** with a KV cache (`flash_attn` is
  not installed); there the cache is a real ~2.6–3 GB term, but that path does not bind the
  budget and does not need a bigger budget once A006-1/-4 land.
- Transformers 5.x `QuantizedCache` needs a backend (`optimum-quanto`/`hqq`) — **none is
  installed** — and it applies one bit-width to **both** K and V; separate per-K/V bits
  (the llama.cpp model) require a custom `Cache` subclass.
- Quantizing the cache on the routing-measurement generation would perturb attention
  weights and risk changing generated tokens, breaking the A004-6 identical-token invariant
  and contaminating the very signal the study measures.
Conclusion: cache quantization is the wrong lever for the truncation problem. If a larger
reasoning-call budget is wanted later, the correct path is a **separate, larger input budget
for the SDPA text-generation calls only** (leaving the attention-capture budget at 5120),
validated with a fresh `scratchpad/probe_context_budget.py` VRAM probe. Deferred.

## Files touched
- `protected/harness/study_002/agent_caller.py` — A006-1 (`invoke_edits` accepts + tails the
  plan; `invoke_decision` passes hypothesis+action), A006-2 (`do_sample=True` on edits),
  A006-4 (window diagnostic episodes).
- `protected/harness/study_002/study_runner.py` — A006-3 (`no_edits_proposed`,
  `consecutive_no_change` breaker, `_MAX_CONSECUTIVE_NO_CHANGE`).
- `protected/harness/shared/anomaly_logger.py` — register `no_edits_proposed`,
  `study_halted_no_progress`.

## Verification plan
Unit-level: (a) confirm `invoke_edits` places the plan at the tail and decodes sampled;
(b) confirm five consecutive 0-applied-edit iterations trigger `study_halted_no_progress`
and stop the loop, while an iteration applying an edit resets the counter. Then reset the
study to a fresh pre-run baseline (empty `metrics.jsonl`/`examples.md`, locked
`system_prompt.md` hash, playground = `__init__.py` + `extractor.py`) and hand back to the
operator for a clean full rerun.
