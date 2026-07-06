# Amendment 007 (spec s2-010) — Smoke Contract Enforcement & DV-Safe Failure Feed-Forward

**Status:** registered
**Date:** 2026-07-06
**Branch:** `experiment/study_002`
**Supersedes/extends:** builds on Amendment 006 (`s2-009`) and Amendment 005 (`s2-008`).
No A004/A005/A006 decision is reverted; A007 closes a gate hole A006's run exposed.

## Motivation — the observed failure (post-A006 run, iters 0–7)

Amendment 006 worked exactly as intended and **fixed the empty-`[]` stall**:

- iters 1–7 the agent proposed real, *varying*, non-empty edits (`agent_edits_proposed`
  = 1, 1, 0, 2, 2, 2, 2) — the frozen greedy `[]` fixed point is gone.
- `no_edits_proposed` fired correctly at iter 3 (the one genuine no-op).
- The study **halted early at iter 7** via the A005-2 consecutive-anomalies breaker
  (4 non-scanned in a row) instead of grinding to 25.

It exposed the next layer. At **iter 4** the agent's sampled edit changed `extract()` to
return a **tuple** instead of `ExtractionResult`. Every abstract then died on
`result.claims`:

```
'tuple' object has no attribute 'claims'   ×100  (25 abstracts × 4 iters)
scan_failure                               ×4    (iters 4,5,6,7)
```

The harness rolled the edit back each time (the committed `extractor.py` is still the
healthy iter-2 version), the agent re-emitted the same class of broken edit, and the
breaker killed the study after ~3 real scanned iterations. F1 barely moved (0.603 → 0.612).

### Root cause — two gaps let a `return`-type bug reach the full corpus scan

1. **The smoke test false-passes a wrong return type.** `interface_validator.py:132`:
   ```python
   claim_count = len(result.claims) if hasattr(result, 'claims') else 0
   return ValidationResult(valid=True, smoke_test_passed=True, ...)
   ```
   A tuple has no `.claims`, so this silently records `claim_count=0` **and still returns
   `smoke_test_passed=True`**. The run confirms it: `smoke_test_passed=true,
   smoke_test_claim_count=0` at iters 4–7. The contract-violating edit clears the gate and
   is committed to the playground *before* the (expensive) attention pass and corpus scan.

2. **Scan-time crashes get no feedback.** The A005-1 apply-repair loop only fires on
   *apply* failures (edit won't apply to the file). A syntactically valid edit that applies
   cleanly and passes smoke, but crashes the corpus scan (`study_runner.py:955`), is
   silently rolled back — `apply_repair_attempts=0, repair_attempts=0`. The agent gets no
   signal and repeats the same bug until the breaker halts it.

Where `.claims` is accessed and crashes: `corpus_runner.py:93` (per-abstract) and `:105`
(`len(r.claims)`), the latter propagating out of `run_corpus` as the iteration-level
`scan_failure`.

## Design constraint — do not corrupt the routing DV

The obvious "fix" — on a scan crash, repair the code and re-scan within the same iteration —
is **rejected**. The post-modification attention pass (`study_runner.py:847–935`, the
routing-fidelity DV) runs *before* the scan on the edited code. Repairing the code and
re-scanning would mean the committed extractor no longer matches the code the attention pass
measured, silently decoupling the DV from the artifact. Recovery must therefore happen
either **before** the attention pass (the smoke/interface repair loop, which already re-runs
smoke after each fix — DV-safe) or **across the iteration boundary** (feed the failure
forward into the next diagnostic), never as a mid-iteration re-scan.

## Decisions

### A007-1 — The smoke test enforces the return contract (primary fix)
`run_smoke_test` (`interface_validator.py`) now asserts the extract result satisfies the
`ExtractionResult` contract: `isinstance(result, ExtractionResult)`, `result.claims` is a
`list`, and every element is a `Claim`. On violation it returns
`smoke_test_passed=False` with a precise error (e.g. *"extract() must return
ExtractionResult, got tuple"*) instead of masking it as 0 claims. Because the smoke test
runs at `study_runner.py:757` inside the existing 3-attempt repair loop (which re-runs smoke
after each `invoke_repair`), a type/shape-breaking edit is now:
- caught **before** the attention pass and corpus scan (saves ~6 min of scan compute), and
- routed into the **existing, DV-safe recovery path** — the agent is handed the exact
  contract error and gets up to 3 debugging turns, exactly as with an apply failure.

This alone fully catches the observed tuple bug: the return type is unconditional, so the
single smoke abstract triggers it deterministically.

### A007-2 — DV-safe feed-forward of runtime failures (`Episode.failure_note`)
A new optional `Episode.failure_note` field is persisted by the episode store and therefore
surfaced in the next iteration's diagnostic context (which dumps the last 8 episodes as
JSON, per A006-4). On the three rollback paths that currently discard the reason —
`scan_failure`, `iteration_timeout`, and smoke/interface `repair_exhausted` — the runner now
records a concrete note on the episode before persisting it, e.g.:

> *"SCAN CRASHED after your edits applied and was rolled back: 'tuple' object has no
> attribute 'claims'. extract() must return an ExtractionResult whose .claims is a list of
> Claim for every abstract. Do not repeat this change."*

This gives the agent feedback across the rollback boundary **without** re-scanning, so the
routing DV stays bound to the committed code. It converts the "silently repeat the same
crash 4×" behaviour into an informed next attempt.

### A007-3 — (scope note) no new breaker, no budget change
The A005-2 consecutive-anomalies breaker and the A006-3 no-progress breaker already halt the
run correctly (they did, at iter 7). A007 adds **no** new breaker and does **not** change
`MAX_INPUT_TOKENS` (5120). The residual "passes the single smoke abstract but crashes on a
specific corpus abstract" class is not eliminated (it cannot be caught pre-scan without
running the corpus), but A007-2 ensures even that case now informs the next iteration rather
than repeating blindly, and the breakers bound the cost.

## Files touched
- `protected/harness/shared/interface_validator.py` — A007-1 (`run_smoke_test` asserts the
  `ExtractionResult`/`Claim` contract; import `ExtractionResult`, `Claim`).
- `protected/harness/shared/edit_protocol.py` — A007-2 (`Episode.failure_note: str | None`).
- `protected/harness/shared/episode_store.py` — A007-2 (persist `failure_note`).
- `protected/harness/study_002/study_runner.py` — A007-2 (set `failure_note` on the
  `scan_failure`, `iteration_timeout`, and `repair_exhausted` rollback paths).
- `protected/harness/shared/anomaly_logger.py` — doc note only; A007 introduces no new
  anomaly types (it reuses `interface_smoke_test_failed`, `scan_failure`, `iteration_timeout`,
  `repair_exhausted`).

## Verification plan
Unit-level, no model required: (a) a fake `extract` returning a tuple makes
`run_smoke_test` return `smoke_test_passed=False` with a type error, while a valid
`ExtractionResult` still passes with the right claim count; (b) `Episode.failure_note`
round-trips through `episode_store.append`/`load_all` and appears in the JSON the diagnostic
context reads. Then reset the study to a fresh pre-run baseline (empty `metrics.jsonl` /
`examples.md`, locked `system_prompt.md` hash `c6c43f5f…`, playground = `__init__.py` +
`extractor.py`) and hand back to the operator for a clean full rerun.
