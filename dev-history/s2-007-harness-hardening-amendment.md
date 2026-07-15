# Study 002 — Amendment 004: Inference Wrapper and Harness Hardening (Study 002B, run-enabling correction)

**Organization:** Idris Applied AI Research
**Study ID:** study_002
**Date:** July 2026
**Author:** Muzaffer Ozen
**Status:** Pre-Registration Amendment
**Amends:** Study 002 Pre-Registration; Amendments 001, 002, 003
**Supersedes:** The invalid Study 002B run attempted under Amendment 003 (baseline F1=0.000, no valid iterations)

---

## Preamble

Amendment 003 committed a redesigned infrastructure for Study 002B but the run it
enabled never produced a single valid iteration. Iteration 0 recorded macro-F1 =
0.000 (25 of 25 abstracts returned zero claims) and iterations 1–10 were logged as
anomalies, dominated by empty `edits` field calls. The run was stopped by hand.

Root-cause analysis (below) shows these are not study-level findings. Most originate
in a single piece of infrastructure — the unified transformers inference wrapper
`AttentionAnalyzer.complete_with_usage()` — which the whole harness routes through for
extraction, for every agent field call, and while hosting the attention hooks; the
wrapper fails **silently** (returns empty text rather than raising), so a broken run is
recorded as a clean one. A second cluster lives in the surrounding control flow: the
interface smoke test invokes the extractor on a path where its provider was never
injected (so it fails every iteration), and a single attention-pass exception
propagates uncaught and terminates the whole study. Under these conditions the agent
was again modifying a broken system and receiving silence as feedback.

This amendment hardens that wrapper and the harness control flow around it. It changes
no scientific design decision. The research question, the thesis, the component under
test (intrinsic cost signal via attention routing fidelity), the corpus, the signal
definition, the evidence criteria, and the baseline state are all unchanged from
Amendment 003. The two signal-adjacent changes — matching the attention generation
budget to extraction (A004-11) and removing the degenerate pre/post routing fields
(A004-12) — make the measurement faithful to the already-registered signal rather than
redefining it. What changes is only the reliability and fidelity of the layer that
operationalizes the study's decisions.

The invalid Study 002B run is preserved in git history as a documented debugging
session. Study 002B begins fresh from the same clean baseline Amendment 003 locked.
This document is committed before any code change. Per the harness integrity rule,
no iteration may run while `protected/harness/` or `protected/attention/` differ
from HEAD, so the amendment commit necessarily precedes the code commit.

---

## Architectural Decision: Retain the Unified transformers Model

Amendment 003 unified extraction, agent generation, and attention capture onto one
transformers model. That unification is retained deliberately. It is the source of
the reliability problems corrected here, but it is also a scientific requirement:
the routing-fidelity thesis is a claim about the attention of the model that is
actually performing extraction. Splitting extraction and agent calls onto a
llama.cpp GGUF backend (as in Study 001) would give grammar-constrained JSON and
remove an entire class of failure, but it would measure attention on a 4-bit `nf4`
transformers model while extraction ran on a differently-quantized 6-bit GGUF — a
confound that undermines the study's central claim.

Therefore the correction is to harden the single transformers wrapper rather than to
split backends. Reliability is bought through deterministic decoding, thinking-mode
suppression, and instruction-preserving context budgeting — not through a second
backend.

Full single-pass unification (deriving the routing score from the same forward pass
that produces the extraction, rather than from a second pass) is a larger change
that would further tighten the science. It is **out of scope** for this amendment and
recorded as future work. Note, however, that A004-3 (greedy extraction), A004-6
(aligned inputs), and A004-11 (matched generation budget) together make the two passes
produce the *same* token sequence deterministically, so the second pass measures
attention over the identical extraction until single-pass unification is undertaken.

---

## Root Cause Analysis of the Invalid 002B Run

**Failure 1 — Thinking mode active during extraction → empty claims (baseline F1=0.000).**
The corpus extractor calls `complete_with_usage(system_prompt, abstract_text)` with
no token cap and no thinking suppression. Unlike `build_input()` (which correctly
passes `enable_thinking=False`), `complete_with_usage()` builds the chat template
without it, so extraction runs with Qwen thinking mode ON, sampling at
`temperature=0.7`, capped at 1024 new tokens. The model emits an unterminated
`<think>` monologue, the JSON parse fails, and `extractor.py` silently falls back to
`{"claims": []}`. Every abstract returned empty. The same naive prompt scored 0.467
in Study 001, which ran a different (llama.cpp) backend.

**Failure 2 — Tail truncation drops the instruction → empty agent field calls.**
Every agent call assembles `user = context + instruction + "/no_think"` and tokenizes
with `truncation=True, max_length=max_input_length`. HuggingFace truncates on the
right by default. The `edits` call carries the largest context (full current file
contents); once it exceeds the input budget, truncation removes the trailing
instruction and the `/no_think` marker, handing the model a wall of files with no
task. It returns nothing, and the empty completion is logged as
`field_call_failed: edits`. This was the dominant anomaly (42 occurrences).

**Failure 3 — Path separator mismatch → legal code edits rejected.**
`Edit.file_path` is taken verbatim from the agent's JSON and passed to the allowlist,
which compares against POSIX-slash literals. When the model emitted
`playground\extractor.py`, `is_allowed()` returned false and the edit was rejected as
an `allowlist_violation`. Beyond blocking a valid edit, this silently corrupts the
`code_changes_attempted` evidence criterion — the harness recorded no code change on
an iteration where the agent attempted one.

**Failure 4 — Silent degradation is invisible.**
Extraction parse failures fall back to empty claims with no anomaly, and the corpus
runner prints "done" for an empty result identically to a populated one. A systematic
extraction failure and a genuinely empty abstract are indistinguishable in the record.
The harness measured its own reliability, not agent behavior, and could not tell.

**Quiet issue 5 — Extraction and attention measured on divergent inputs.**
The extraction pass and the attention pass tokenize with different `max_length`
(6656+ vs 4096) and, per Failure 1, different thinking modes. The routing score can
therefore describe a different token window and generation mode than the extraction
it is meant to characterize.

**Quiet issue 6 — Abstract offset can silently collapse to zero.**
`build_input()` locates the abstract via `prompt.find(abstract_text)`; on any
whitespace/normalization mismatch this returns −1 and `abstract_start_token_idx`
stays 0, folding the system prompt into the region counted as "abstract" attention
and corrupting the routing score without any signal.

**Failure 7 — The smoke test can never pass; the provider is never injected.**
`run_smoke_test()` calls `reload_playground()`, which deletes the `playground.*`
modules; the fresh re-import resets the extractor's module-global `_provider` to
`None`, and the subsequent `extract()` invocation raises
`'NoneType' object has no attribute 'complete_with_usage'`. Provider injection lives
only inside `corpus_runner.run_corpus()`, so the smoke test — which reloads and then
invokes on a path that never injects — fails on every iteration. This burns all three
repair attempts (a harness defect no code edit can fix), reaches `repair_exhausted`,
rolls back, and logs an anomaly. This ran concurrently with Failure 2: two independent
iteration-killers fired at once. The underlying smell is that `_provider` is
module-global state that every `reload_playground()` destroys, and only one of the two
invocation sites re-establishes it.

**Failure 8 — A single attention-pass exception terminates the whole study.**
`_run_attention_subprocess()` is invoked unwrapped in both `_run_baseline` and
`_run_iteration`. `analyze_abstract` re-raises OOM as a `RuntimeError`; the subprocess
exits non-zero; `_run_attention_subprocess` raises; nothing catches it up the stack.
One OOM or one malformed abstract ends the entire run. On a single 32 GB GPU where OOM
is the dominant operational failure, this makes OOM not merely frequent but
unrecoverable.

**Quiet issue 9 — Degenerate pre/post routing fields.**
Only one attention pass runs per iteration (post-modification). The code assigns
`pre_scores = post_scores`, so the `pre_scores`/`post_scores` columns written to
`routing_history.jsonl` are identical every iteration. The consequence signal is
actually derived from the inter-iteration delta, not the within-iteration pre/post
delta the original DECISION 002-E described. The pre/post schema is now dead weight
that misrepresents what is measured.

**Quiet issue 10 — `score_end` is measured at a fixed short offset, not the true end.**
`run_generation_attention` caps at `max_new_tokens=20`, while extraction generates up
to its full budget. `score_end` and `intra_generation_delta` therefore describe
attention after 20 tokens, not attention when extraction actually completes — which is
not what the "does grounding hold through generation" thesis intends to measure.

---

## Amendment Decisions

**DECISION A004-1 — Suppress thinking mode at the template level, everywhere.**
`complete_with_usage()` builds its chat template with `enable_thinking=False`, matching
`build_input()`. The `/no_think` string appended to user messages is removed; it is
redundant once the template disables thinking and risks being the very text truncation
drops. All inference through the wrapper — extraction, diagnostic fields, episode
fields, edits, repair — runs with thinking suppressed.

**DECISION A004-2 — Instruction-preserving context budgeting.**
The wrapper no longer relies on right-truncation of the assembled prompt. Callers pass
the fixed framing (system prompt, task instruction) and the variable context
(file contents, episodic memory, routing history) as distinct parts. The wrapper
reserves a token budget for the framing plus generation headroom and truncates **only
the variable context, oldest content first**, so the task instruction is never lost.
When context is truncated, a non-blocking `context_truncated` anomaly is logged with
the pre- and post-truncation token counts. No call may be issued with its instruction
removed.

**DECISION A004-3 — Deterministic decoding for structured output.**
Extraction and the `edits` call decode greedily (`do_sample=False`). Greedy extraction
also aligns the extraction decoding with the attention pass, which already selects
tokens by `argmax`, so the routing score characterizes the same decoding the extractor
uses. Prose field calls (diagnostic and episode text) are unchanged; the failures were
confined to structured output. This strengthens the run's determinism; the pre-
registration's single-run nondeterminism limitation is unaffected.

**DECISION A004-4 — Extraction observability and a baseline guardrail.**
The corpus runner records, per iteration, the count of abstracts returning zero claims.
If **every** abstract in an iteration returns zero claims, a `zero_extraction_output`
anomaly is logged. At iteration 0 (baseline) this condition is a **hard stop**: the
harness refuses to proceed, mirroring the Amendment 003 baseline-verification
philosophy, because a broken baseline invalidates the entire run. After iteration 0 it
is a logged, non-blocking anomaly (a systematically empty extractor is a real, if
degenerate, agent state to be recorded).

**DECISION A004-5 — Normalize edit path separators before the allowlist.**
Edit file paths are normalized to POSIX separators (`\` → `/`) at the point the `Edit`
is constructed, before any allowlist or filesystem operation. This removes the false
`allowlist_violation` and restores the integrity of the `code_changes_attempted`
criterion.

**DECISION A004-6 — Align extraction and attention inputs.**
The extraction pass and the attention pass use the same tokenization `max_length` and
the same `enable_thinking=False` setting, so the routing score is computed over the
same token window and generation mode as the extraction it characterizes.

**DECISION A004-7 — Deterministic peak-memory bound (RTX 5090, 32 GB).**
Study 002B runs entirely on one local 32 GB GPU; out-of-memory has been the dominant
operational failure. A004-2's explicit context budget caps prefill length
deterministically, which bounds peak activation memory. The segmented attention
forward pass (run in a subprocess that releases and reloads VRAM) is retained, and the
explicit `gc.collect()` + `torch.cuda.empty_cache()` around generation are retained.
The input budget constant is set to a value validated to fit 32 GB with headroom and
recorded in code; it replaces the ad-hoc `13107` currently in use.

**DECISION A004-8 — Fail loudly on an unlocatable abstract offset.**
If `build_input()` cannot locate the abstract within the templated prompt
(`find()` returns −1), the abstract is skipped for routing and an
`abstract_offset_unresolved` anomaly is logged, rather than recording a routing score
computed over the whole sequence. A corrupted signal is worse than a missing one.

**DECISION A004-9 — Centralized provider injection shared by all invocation paths.**
Provider injection is lifted out of `corpus_runner` into a single shared context
manager (e.g. `provider_injected(analyzer)`) that sets the extractor's provider,
yields, and restores the prior value. Every path that invokes `extract()` — the corpus
run and the interface smoke test — enters through it, so the invoked extractor always
has a live provider regardless of prior `reload_playground()` calls. This eliminates
the `'NoneType' … complete_with_usage` smoke-test failure (Failure 7) and removes the
scattered, per-site injection logic that caused it.

**DECISION A004-10 — Fault-tolerant attention pass; one abstract cannot end the run.**
Per-abstract analysis in the attention subprocess is wrapped so that any exception —
including OOM re-raised as `RuntimeError` — is caught, logged as an
`attention_abstract_failed` anomaly, and recorded as a null routing score for that
abstract; the pass continues to the next abstract. The subprocess exits non-zero only
on a load-time or whole-pass failure, and the harness treats a non-zero attention
subprocess as a logged iteration anomaly rather than an uncaught exception that
terminates the study. OOM becomes recoverable and localized.

**DECISION A004-11 — Attention generation budget matches extraction; `score_end` is
the true end.** `run_generation_attention` generates up to the same token budget as
extraction (stopping at EOS), rather than a fixed 20-token cap, so `score_end` and
`intra_generation_delta` are measured at the actual end of extraction. Under A004-3
(greedy extraction) and A004-6 (aligned inputs), the argmax attention pass reproduces
the extraction token sequence exactly, so this makes `score_end` the grounding at the
real final token of the real extraction. The heavier decode cost is bounded (control
set only, greedy, EOS-terminated) and its peak memory is unchanged, since each captured
decode step remains `[1, heads, 1, seq_len]`.

**DECISION A004-12 — Remove the degenerate pre/post routing schema.**
`routing_history.jsonl` records a single `scores` list per iteration (the
end-of-generation routing scores on the control set). The inter-iteration delta is
computed against the previous entry's `scores`. The `pre_scores`/`post_scores` fields —
which were always identical (Quiet issue 9) — are removed, along with the code that
duplicated them. `format_for_agent` and the metrics delta logic read the single
`scores` field. This is a schema simplification, not a change to the measured signal.

**DECISION A004-13 — Single, centralized rollback-before-commit invariant.**
The requirement "the playground is clean or rolled back before `commit_iteration`" is
enforced in one place: the study loop rolls back the playground on any anomalous
(`None`) iteration return before committing, rather than relying on each of the ~6
early-return sites in `_run_iteration` to roll back individually. This removes the risk
that a future missed rollback silently commits broken playground state that becomes the
next iteration's baseline.

**DECISION A004-14 — Bounded smoke-test generation.**
The smoke test invokes `extract()` with a small hard token cap (≤ 64 new tokens). It
verifies the pipeline runs without raising, not extraction quality, so it must not risk
the 60 s timeout on a full-length generation once A004-9 lets it run at all.

**DECISION A004-15 — Scoped iteration commits.**
`commit_iteration` stages only the study's own artifacts and the mutable surface
(`experiments/<study>/`, `playground/`, `prompts/`) rather than `git add .`, and
transient outputs (`attention_scores_*.json`, `__pycache__`, working plan files) are
gitignored. Iteration commits capture the iteration state and nothing incidental.

---

## Baseline State (unchanged from Amendment 003)

No baseline files change. `prompts/system_prompt.md` remains the locked naive prompt
(hash verified by the existing `_verify_baseline_state` check), `prompts/examples.md`
remains empty, and `playground/` contains only `__init__.py` and `extractor.py`. Prior
002B iteration data in `experiments/study_002/` is cleared before the fresh run, as
under Amendment 003, and preserved in git history.

---

## Files Changed by This Amendment

| File | Change |
|---|---|
| `protected/attention/analyzer.py` | `complete_with_usage`: `enable_thinking=False`; instruction-preserving context budgeting (A004-2); greedy decoding path for structured calls (A004-3); aligned `max_length`/thinking with the attention pass (A004-6); replace `13107` with a validated 32 GB budget constant (A004-7). `run_generation_attention`: budget matches extraction, EOS-terminated (A004-11). `build_input`: raise/flag on unresolved abstract offset (A004-8). |
| `protected/harness/study_002/agent_caller.py` | Pass framing and variable context as distinct parts to the wrapper; remove `/no_think` appendage (A004-1); request greedy decoding for the `edits` call (A004-3). |
| `protected/harness/shared/corpus_runner.py` | Count zero-claim abstracts, emit `zero_extraction_output` (A004-4); expose the shared `provider_injected()` context manager and route the corpus run through it (A004-9). |
| `protected/harness/shared/interface_validator.py` | Smoke test enters through `provider_injected()` (A004-9); bounded ≤ 64-token generation (A004-14). |
| `protected/attention/forward_pass_runner.py` | Per-abstract try/except with `attention_abstract_failed` anomaly and null score; continue on failure (A004-10). |
| `protected/harness/shared/edit_applier.py` (or `edit_protocol.py`) | Normalize `file_path` separators before allowlist/apply (A004-5). |
| `protected/harness/study_002/routing_history.py` | Single `scores` field per iteration; remove pre/post; delta vs previous entry (A004-12). |
| `protected/harness/study_002/study_runner.py` | Hard-stop on baseline `zero_extraction_output` and surface new anomaly counts in metrics (A004-4); treat non-zero attention subprocess as a logged anomaly, not an uncaught crash (A004-10); centralized rollback-before-commit on any anomalous return (A004-13); drop pre/post routing bookkeeping (A004-12). |
| `protected/harness/shared/git_ops.py` | Scoped staging in `commit_iteration` instead of `git add .` (A004-15). |
| `protected/harness/shared/anomaly_logger.py` | Register `context_truncated`, `zero_extraction_output`, `abstract_offset_unresolved`, `attention_abstract_failed`. |
| `.gitignore` | Ignore `attention_scores_*.json`, `__pycache__`, working plan files (A004-15). |
| `experiments/study_002/` | Clear prior 002B iteration data; preserve in git history. |

No change to: the corpus, ground truth, the scorer, the routing-score definition, the
evidence criteria, the episode schema, the allowlist policy (only path normalization),
or the baseline files.

---

## Metrics Schema Additions

```json
{
  "extraction_zero_claim_abstracts": 0,
  "context_truncated_calls": 0,
  "abstract_offset_unresolved_count": 0,
  "attention_abstract_failed_count": 0
}
```

## Routing History Schema Change (A004-12)

The per-iteration record replaces the `pre_scores`/`post_scores` pair with a single
`scores` list:

```json
{
  "iteration_n": 3,
  "timestamp": "ISO8601",
  "scores": [{"abstract_id": "…", "score": 0.0, "score_start": 0.0, "score_end": 0.0,
              "intra_generation_delta": 0.0}],
  "aggregate_score": 0.0,
  "inter_iteration_delta": 0.0
}
```

---

## Amendment Record

| Amendment | Date | Description |
|---|---|---|
| 001 | June 2026 | Two-phase agent caller (diagnostic + decision split) |
| 002 | June 2026 | Disable thinking mode in attention forward pass and Call 1 |
| 003 | June 2026 | Infrastructure correction and full redesign for Study 002B |
| 004 | July 2026 | Inference wrapper hardening and harness robustness: thinking suppression, instruction-preserving budgeting, deterministic structured decoding, extraction observability, path normalization, offset safety, centralized provider injection, fault-tolerant attention pass, matched attention/extraction generation budget, routing-history schema simplification, centralized rollback-before-commit, bounded smoke test, scoped commits |

---

## Commit Instruction

This document is committed to `dev-history/s2-007-harness-hardening-amendment.md`
before any code change, data clearing, or iteration run. Its commit is the integrity
anchor for the Study 002B rerun under Amendment 004.
