# Amendment 009 (spec s2-012) — Harness Audit Consolidation & Pre-Registration Reconciliation

**Status:** implemented (design approved by operator; code landed & unit-tested 2026-07-13)
**Date:** 2026-07-13
**Branch:** `experiment/study_002`
**Supersedes/extends:** builds on A004–A008. No prior *breaker* is reverted; A009 fixes
implementation bugs that diverge from the pre-registration, documents the breaker program
as a deviation, and re-scopes A008 to survive the #9a fix.

## Origin — a proactive full-harness audit (not a post-mortem)

After A004–A008 each fixed a new degenerate-loop discovered *during* a run, a background
subagent audited the whole harness for latent issues to stop the whack-a-mole. Ten findings
were returned and **all ten were verified against the code**; cross-referencing
`experiments/study_002/pre-registration.md` escalated two of them and surfaced a meta-issue
the audit missed: **several standing amendments have drifted from the pre-registered design.**

Decision (operator): **fix the true bugs to match the pre-registration; document the
breakers as justified deviations rather than reverting them.**

## The core tension the audit exposed

The pre-registration (DECISION 002-E, line 141) specifies the routing pass runs *"the
current extractor (`playground/extractor.py` and `prompts/`)"*. The implementation
(`forward_pass_runner.py:60,93,100`) instead runs `system_prompt + raw abstract` and never
invokes `extract()`. This is faithful **only** while the extractor is a plain
`prompt+abstract` model call — and diverges the moment the agent adds preprocessing,
filtering, or multi-call logic, which is exactly what success criterion 002-D #2 *rewards*.
The post-A007 run demonstrated the consequence: the agent edited `extractor.py` for four
iterations, correctly observed routing never responded, and (rationally, given a blind DV)
concluded "the metric is decoupled from my strategy." A008 caps the wasted compute but does
not fix the blindness.

## Decisions

### Tier A — fix to match the pre-registration + document deviations

**A009-1 — routing DV reflects the committed extractor (provider-capture shim).**
Replace the raw-abstract attention input with the extractor's *actual* model input. For
each of the 10 fixed control abstracts, run the committed `extract()` with `_provider`
swapped for a **capture shim** whose `complete_with_usage(system_prompt, text)` records the
`(system_prompt, text)` pair and returns a minimal valid JSON (so `extract()` completes
without a real generation). Feed the **captured** pair into the existing
`build_input → run_generation_attention → map_sentences_to_tokens → compute_routing_score`
pipeline. Multi-call extractors: **measure the first model call** (documented rule).

- *Label-alignment risk (must be resolved in implementation):* routing labels are assigned
  per-sentence inside `map_sentences_to_tokens(abstract_text, …)` and read as `sent["label"]`
  by `compute_routing_score` (`scorer.py:51`). If the extractor transforms the text, the
  labeler runs on the transformed text. Rule: measure on the captured text via the existing
  labeler; if it locates **zero results tokens** while the raw abstract *did* contain
  results sentences, emit a **null** routing score + new `routing_unmeasurable` anomaly for
  that abstract rather than a misleading 0. This keeps the DV honest without guessing.
- *DV integrity preserved:* the capture + attention pass still runs strictly **before** the
  corpus scan, on the committed code; no mid-iteration re-scan. The shim performs no real
  generation, so cost is ~unchanged.
- *Interaction:* with A009-1 in place, routing now *responds* to extractor edits, which is
  what makes the A008 re-scope (below) meaningful.

**A009-2 — impact set drawn per iteration (pre-reg line 104).** Replace the fixed
`random.seed(42)` in `_load_impact_abstracts` (`study_runner.py:92`) with a per-iteration
seed (`seed(42 + iteration_n)`) so the 15 impact abstracts are redrawn each iteration
"without replacement within the iteration," as pre-registered. The 10 control abstracts stay
fixed (they anchor the routing DV).

**A009-3 — config to match the pre-reg.** `N_ITERATIONS` 25 → **20** (aligns with the
existing `count >= 21` completion guard and pre-reg line 193); iteration-timeout default
`14400` → **1800** s (30 min, pre-reg line 193). `STUDY_ITERATION_TIMEOUT_S` still overrides.

**A009-4 — re-scope A008 to a corpus-independent stall signal (mandatory).** A009-2 makes
the corpus rotate, so `(micro_tp, micro_fp, micro_fn)` is no longer stable for a frozen
extractor — the A008 signature would never match and the breaker would silently die. Re-key
A008 on **`post_routing_score` alone** (computed on the *fixed* 10 control abstracts →
corpus-independent) plus `applied > 0`. With A009-1, a frozen `post_routing_score` despite
applied edits is a true inert-edit signal. Also (folds in audit #4) make it **cycle-aware**:
halt when the last K post-routing values contain ≤2 distinct states, not only on strict
consecutive equality, so a 2-state oscillation is caught too.

**A009-5 — document breakers/windowing as pre-reg deviations.** Add a "Deviations from the
original protocol (via Amendments)" section to `pre-registration.md` recording: the A005-2 /
A006-3 / A008(A009-4) early-halt breakers (vs "no hard stops," line 193) and A006-4's
8-episode diagnostic windowing (vs "no windowing," line 201), each with its justification
(degenerate-loop protection / context-budget truncation). These are kept, not reverted.

### Tier B — contamination & correctness

**A009-6 (audit #3).** After `rollback_playground()`'s `git checkout`, also
`git clean -fd -- playground/ prompts/` so untracked files the agent *created* in a
rolled-back iteration cannot be captured by the next `git add -A` and contaminate the
baseline / `_current_files()`.

**A009-7 (audit #6).** Make `apply_edits` transactional: apply to in-memory buffers,
re-validate each subsequent same-file edit against the running buffer (not the original
file, closing the silent-drop on overlapping edits), and write all files only after the full
batch validates. Wrap the `apply_edits` call sites so an apply exception degrades to an
anomaly + rollback instead of aborting the study.

**A009-8 (audit #2).** Call `git_ops.reset_partial_iteration()` at startup (per spec
`003:830`; study_001 already does) to discard uncommitted post-apply edits from a crashed
run, and gate `_verify_baseline_state` on `start_iter == 0` so a genuine resume is reachable
instead of forcing a full data-deletion restart.

### Tier C — hygiene

**A009-9 (audit #5).** Reset `metrics["agent_edits_applied"] = 0` on the scan-crash /
timeout / repair-exhausted rollback paths so the A006-3 no-change counter reflects reality
(its current comment claims this but the code does not do it).

**A009-10 (audit #7).** Enforce a per-abstract wall-clock/token cap inside the corpus runner
(run each `extract()` with a hard deadline) so the iteration timeout can actually preempt a
runaway extractor; the corpus path currently passes no token cap.

**A009-11 (audit #8).** Write edited files with `newline=""` in `edit_applier.py` (76/81/87)
so agent edits stay LF, and add a `.gitattributes` pinning `*.py`/`*.md` to LF — closing the
long-standing CRLF/hash-gate landmine at its source.

**A009-12 (audit #10).** Skip `append_routing` (or write an ignored sentinel) when the
attention pass produced no valid scores, so one failed pass doesn't null out the next
iteration's routing delta.

## Verification plan
Per-decision unit tests, no model required where possible: capture shim records the
transformed `(system_prompt, text)` and routing is computed over it (A009-1); impact set
differs across iterations (A009-2); A008 re-scope halts on frozen `post_routing_score` with
rotating corpus and on a 2-state oscillation, and does NOT halt on real routing movement
(A009-4); `git clean` removes an orphan created file after rollback (A009-6); overlapping
same-file edits no longer silently drop and a mid-batch failure rolls back (A009-7); resume
path reachable with `start_iter > 0` (A009-8); rollback zeroes `agent_edits_applied`
(A009-9); edited files are LF (A009-11). The A009-1 routing change additionally needs an
end-to-end smoke on the real model before a full run. Then reset to a fresh pre-run baseline
and hand to the operator.

## Implementation notes (refinements found during coding)
- **A009-10 scoped to observability.** The provider already caps every generation at
  `EXTRACTION_MAX_NEW_TOKENS` (`analyzer.py:685`), so a single call cannot run unbounded; the
  only residual slow path is a *multi-call* extractor, which cannot be hard-preempted
  in-process (the model call is synchronous). A009-10 therefore logs a `slow_abstract`
  anomaly + per-abstract wall clock; the hard bound is the now-30-min iteration timeout
  (A009-3). True preemption would require a subprocess-per-abstract — out of scope.
- **A009-1 fallback simplified.** The "label alignment" fallback is implemented as: if the
  captured input yields `n_results_tokens == 0`, record a null routing score +
  `routing_unmeasurable` for that abstract (rather than re-labelling the raw abstract). For
  the control set — real abstracts selected to contain results — a 0 means a transformation
  dropped them, which is exactly the unmeasurable case. Baseline (naive extractor) captures
  the raw abstract verbatim, so its routing is unchanged from the pre-A009 pipeline.
- **A008 test retired.** The old `(tp,fp,fn,routing)` signature (`_last_scan_signature`) is
  replaced by `_routing_window` + `_detect_routing_stall`; A009 unit tests cover frozen,
  oscillation, transition-not-firing, window filtering, per-iteration impact draw, and the
  transactional apply (overlap rejection, LF write, create-then-edit buffer visibility).

## Open implementation risks (resolve during coding, not design)
1. **A009-1 label alignment** on transformed extractor input (fallback: null +
   `routing_unmeasurable`, above).
2. **A009-1 VRAM**: the capture adds a lightweight `extract()` pass but no extra attention
   matrix; confirm no regression on the 5090 ceiling (the O(seq²) eager matrix is unchanged).
3. **A009-4 window size K** and the "≤2 distinct states" threshold need one calibration pass
   against the observed run's routing trajectory.
