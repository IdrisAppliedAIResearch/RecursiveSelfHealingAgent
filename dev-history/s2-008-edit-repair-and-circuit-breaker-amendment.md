# Amendment 005 (spec s2-008) — Edit-Apply Repair Loop & Anomaly Circuit Breaker

**Status:** registered
**Date:** 2026-07-04
**Branch:** `experiment/study_002`
**Supersedes/extends:** builds on Amendment 004 (`s2-007`); does not change any A004 decision.

## Motivation — the observed failure

The first post-A004 run (iterations 0–25) split cleanly into two stories.

**Story 1 (iters 0–5): the harness works.** Baseline F1 = 0.603 (up from 0.0 pre-A004),
5.4 claims/abstract, clean JSON extraction. Over iters 1–5 the agent — steering by the
attention-routing signal it can see — made the extractor progressively more conservative
(precision 0.62→0.77, recall 0.59→0.30, claims/abstract 5.4→2.64), driving F1 down to 0.42.
Its own intrinsic signal *also* fell (post-routing 0.023→0.015) and never once read
"improving". This is a legitimate (soft-negative) study observation and is **kept**.

**Story 2 (iters 6–25): a deterministic dead-loop.** Every one of iterations 6–25 is the
identical anomaly — the agent proposed **1 edit** to `playground/extractor.py`, it failed
to apply with reason `no_match` (the `old_string` was not found verbatim), the iteration was
marked anomalous, **the corpus scan was skipped**, and nothing changed. 20 iterations,
byte-for-byte the same failure. The run was effectively dead at iteration 6 but burned the
entire remaining budget spinning.

### Root cause

1. **No recovery on apply-failure.** When `apply_edits` returned `applied=False`, the
   iteration logged an anomaly and `return None` immediately (`study_runner._run_iteration`,
   the block at the old line ~611–621). The existing repair loop only engaged for
   *runtime*/interface failures **after** a successful apply — an edit that never applied got
   zero debugging turns. We are studying whether the agent can improve extraction, **not**
   whether it writes a correct search-and-replace one-shot.
2. **Greedy determinism + a stable prompt tail.** `edits` decode greedily (A004-3). The
   file the agent edits against sits at the *tail* of the decision context and is preserved
   by `_budget_user_message` (head-first trimming), so the effective prompt is near-constant
   iteration to iteration → the model re-emits the *same* non-matching `old_string` forever.
3. **No circuit breaker.** Nothing detected "N iterations in a row produced no scan" and
   stopped the study, so the loop ran to the full iteration budget.

Note (observation, **not** fixed here): each decision/edits call overflows the 5120-token
budget (194 `context_truncated` events, ~6267→~4716 tokens). Because the file is at the tail,
what gets trimmed is the **episodic memory** — the agent silently loses its own history each
call. This degrades memory but is not the cause of the dead-loop and is left for a future
tuning amendment.

## Decisions

### A005-1 — Edit-apply repair loop (the primary fix)
When the agent's proposed edits fail to **apply** (any `ApplyResult.reason`: `no_match`,
`ambiguous_match`, `file_not_found`, `allowlist_violation`, `missing_old_string`, etc.),
give the agent up to `_APPLY_REPAIR_ATTEMPTS` (= 3) debugging turns instead of aborting the
iteration. Each turn calls the existing `invoke_repair(error_message, current_files,
attempt_number)` with a **reason-specific, actionable** error message (A005-3) and the
current (unchanged) file contents, then re-applies the returned edits. On the first
successful apply, control falls through to the existing smoke-test / interface-validation
repair loop (unchanged) — so an edit that *applies* but is *runtime-broken* is still caught
there. If all apply-repair turns are exhausted, log `apply_repair_exhausted`,
`rollback_playground()` (defensive), mark the iteration anomalous, and `return None`.

### A005-2 — Consecutive-anomaly circuit breaker
In `_run_study_async`, track consecutive non-scanned iterations (`rationale is None`). A
successful iteration resets the counter to 0. When the counter reaches
`_MAX_CONSECUTIVE_ANOMALIES` (= 4), log `study_halted_consecutive_anomalies` and **stop the
study early** with a clear message. Once scans stop repeatedly, nothing productive can
happen; this caps the waste at ~4 iterations instead of ~20.

### A005-3 — Actionable apply-failure messages
Add `_describe_apply_failure(apply_result)` mapping each `ApplyResult.reason` to a concrete
instruction (e.g. `no_match` → "copy the exact text to replace verbatim, including all
whitespace and indentation, from the CURRENT FILE CONTENTS"). The repair prompt already
includes the full current file, so the agent has what it needs to correct itself.

### A005-4 — Metrics & anomaly registry
Add `apply_repair_attempts` (default 0) to the metrics base and record it on the iteration.
Register new anomaly types in `anomaly_logger.py`: `apply_repair_exhausted`,
`apply_repair_agent_failure`, `study_halted_consecutive_anomalies`.

### A005-5 — Sample on repair turns
`invoke_repair` decodes with `do_sample=True` (was greedy). Repair is best-effort recovery;
light sampling further guarantees successive repair turns cannot re-emit the identical broken
edit, so the loop can escape a fixed point within its attempt budget. (Extraction and the
primary decision path are unchanged — still greedy per A004-3.)

## Files touched
- `protected/harness/study_002/study_runner.py` — A005-1, A005-2, A005-3, A005-4 (constants,
  apply-repair loop, circuit breaker, `_describe_apply_failure`, metrics field).
- `protected/harness/study_002/agent_caller.py` — A005-5 (`invoke_repair` `do_sample=True`).
- `protected/harness/shared/anomaly_logger.py` — A005-4 (register anomaly types).

## Verification plan
Unit-level: (a) feed `_run_iteration` an edit with a non-matching `old_string` and confirm a
repair turn is issued and, when the repair matches, the iteration scans normally; (b) confirm
`_MAX_CONSECUTIVE_ANOMALIES` consecutive `None` returns halt the loop early. Then reset the
study to a fresh pre-run state (empty `metrics.jsonl`/`examples.md`, matching
`system_prompt.md` hash, playground = `__init__.py` + `extractor.py`) and hand back to the
operator for a clean full rerun.
