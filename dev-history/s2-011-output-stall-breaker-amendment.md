# Amendment 008 (spec s2-011) — Output-Stall Breaker (semantically inert edits)

**Status:** registered
**Date:** 2026-07-13
**Branch:** `experiment/study_002`
**Supersedes/extends:** builds on Amendment 006 (`s2-009`, no-progress breaker) and
Amendment 007 (`s2-010`). No prior decision is reverted; A008 closes a breaker hole the
post-A007 run exposed.

## Motivation — the observed failure (post-A007 run, iters 0–6)

Amendment 007 worked: no smoke false-pass, no `scan_failure` wave. The agent proposed
real, varying, contract-valid edits and every iteration scanned cleanly
(`anomaly=false`). It exposed the next layer — a **fixed point that evades every existing
breaker**.

From iteration 3 on, the hidden macro-F1 **and** the agent-visible routing score froze to
full float precision, while the agent kept applying non-empty edits:

| iter | macro_F1 | micro tp/fp/fn | routing (post) | Δrouting | edits applied |
|---|---|---|---|---|---|
| 3 | `0.55543194` | 62/29/93 | `0.00779061` | −0.008 | 2 |
| 4 | `0.55543194` | 62/29/93 | `0.00779061` | **0.0** | 2 |
| 5 | `0.55543194` | 62/29/93 | `0.00779061` | **0.0** | 1 |
| 6 | *(in-flight — same decision narrative)* | | | | ≥1 |

Identical `tp/fp/fn` (exact integers) means the extractor emitted a **byte-identical claim
set** every scan; the routing score identical to 8 decimals means the attention pass was
bit-identical too. The agent had locked onto a single meta-narrative ("the routing metric
is decoupled/misaligned with my strategy") and kept editing `extractor.py` to parse a
`reasoning`/`source_text` field and compute its *own* routing metric — greedy decoding on a
converged prompt reproduces the same output regardless, so each edit was **semantically
inert**. Both signals were also *below* baseline (F1 0.60→0.555, routing 0.023→0.008): the
agent degraded the extractor, then froze.

### Root cause — a nominally-applied edit resets the no-progress counter

The two existing breakers both miss this:

- **A005-2 consecutive-anomalies breaker** counts *non-scanned* iterations
  (`rationale is None`). Every iteration here scans cleanly, so the counter is reset to 0
  each pass (`study_runner.py:1109`) and never reaches 4.
- **A006-3 no-progress breaker** counts iterations that leave the playground unchanged,
  gated on **applied-edit count**: `if _last_applied_edit_count(study_id) > 0:
  consecutive_no_change = 0` (`study_runner.py:1113`). Here `agent_edits_applied` is 2, 2,
  1 — always > 0 — so the counter resets every iteration and never reaches 5.

The gap: a breaker keyed on *whether an edit was applied* cannot see an edit that applies
but **changes nothing about the extractor's behaviour**. Without A008 the run grinds to
`N_ITERATIONS=25` — ~19 further iterations × ~17.6 min ≈ 5.5 h of compute — reproducing the
same frozen numbers.

## Design constraint — do not corrupt the routing DV, do not leak F1

The breaker runs **at the harness level**, after the iteration's metrics record is written,
and only reads the metrics JSONL. It never touches agent-visible context, so keying it on
the hidden `micro_tp/fp/fn` is safe — those never reach the agent (no F1 contamination). It
does not re-scan or re-measure anything, so the routing-DV / committed-code binding
(A007's constraint) is untouched. A008 adds no agent-facing behaviour at all — it is a pure
termination condition.

## Decisions

### A008-1 — Output-stall breaker keyed on scan output, not edit count (primary fix)
New constant `_MAX_CONSECUTIVE_NO_OP = 3`. The runner tracks a per-iteration **scan
signature** = `(micro_tp, micro_fp, micro_fn, post_routing_score)` — the hidden extraction
DV plus the agent-visible routing DV. When a **scanned** iteration that **applied a
non-empty edit** produces a signature *exactly equal* to the previous scanned iteration's,
a `consecutive_no_op` counter increments; any real change (any differing field) resets it.
At 3 the study halts early and logs `study_halted_output_stall`.

- **Exact** equality, not a tolerance: a semantically inert edit yields bit-identical
  floats, whereas any genuine change — however small — differs and correctly *avoids* the
  halt (we never cut a slowly-progressing run short).
- **Gated on `applied > 0`**, making A008 the strict complement of A006-3: A006-3 handles
  frozen output with *zero* applied edits (empty-`[]` no-ops / rollbacks); A008 handles
  frozen output *despite* applied edits (inert edits). Together they cover both halves.
- Threshold 3 (vs A006-3's 5): needs two comparisons to confirm a stall and a third for
  safety margin. On the observed run this halts at iteration 6 (iter4=iter3 → 1,
  iter5=iter4 → 2, iter6=iter5 → 3), saving ~19 iterations. Set below A006-3's 5 because an
  inert-edit stall is a stronger signal than a frozen playground (the agent is actively
  trying and failing to move the extractor).

### A008-2 — signature helper + resume-safe seeding
New `_last_scan_signature(study_id)` reads the last metrics record and returns the signature
tuple, or `None` if that iteration did not scan (`scanned` is false / absent). The loop
seeds `prev_scan_sig` from this helper **before** the loop body so a resumed run compares
against its last committed scanned iteration rather than spuriously resetting. A
non-scanned (anomalous) iteration returns `None` and is skipped — it neither advances nor
resets the no-op counter (pure non-scanned stalls remain the A005-2 breaker's job).

### A008-3 — (scope note) no budget/flow change, one new anomaly type
No change to `MAX_INPUT_TOKENS`, decoding, repair budgets, or the iteration flow. One new
anomaly type `study_halted_output_stall` (parallel to `study_halted_no_progress` /
`study_halted_consecutive_anomalies`) registered in `anomaly_logger.py`.

## Files touched
- `protected/harness/study_002/study_runner.py` — A008-1/-2: `_MAX_CONSECUTIVE_NO_OP`
  constant, `_last_scan_signature` helper, signature tracking + breaker in the iteration
  loop.
- `protected/harness/shared/anomaly_logger.py` — register `study_halted_output_stall`.

## Verification plan
Unit-level, no model required: (a) `_last_scan_signature` returns the tuple for a scanned
record and `None` for a `scanned:false` record; (b) a synthetic three-record sequence with
identical `(tp,fp,fn,routing)` and `agent_edits_applied>0` drives `consecutive_no_op` to the
threshold, while flipping any one field on the middle record resets it and prevents the
halt; (c) an identical-output sequence with `agent_edits_applied=0` does **not** trip A008
(A006-3's domain). Then reset the study to a fresh pre-run baseline (empty `metrics.jsonl` /
`examples.md`, locked `system_prompt.md` hash `c6c43f5f…`, playground = `__init__.py` +
`extractor.py`) and hand back to the operator for a clean full rerun.
