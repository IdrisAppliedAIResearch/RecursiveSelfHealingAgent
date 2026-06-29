# Study 002 — Amendment 003: Infrastructure Correction and Redesign (Study 002B)

**Organization:** Idris Applied AI Research  
**Study ID:** study_002  
**Date:** June 2026  
**Author:** Muzaffer Ozen  
**Status:** Pre-Registration Amendment  
**Amends:** Study 002 Pre-Registration, Amendments 001 and 002  
**Supersedes:** All prior iteration data from study_002 runs prior to this amendment

---

## Preamble

Study 002's first execution (iterations 0-25) is invalidated as a scientific
record. The study ran on faulty infrastructure for the majority of its iterations:
the corpus extractor crashed silently producing "0 abstracts, 25 failures" in
most iterations, the baseline prompt was contaminated with Study 001's final
over-tightened state, the attention routing signal was operating below a useful
range, and late iterations produced garbage Call 2 output due to context overflow.

Under these conditions the agent was modifying a broken system and receiving
silence as feedback. No finding from that run is defensible.

This amendment commits a redesigned infrastructure before any new iteration runs.
The scientific question is unchanged. The research thesis is unchanged. The
component under test — intrinsic cost signal via attention routing fidelity —
is unchanged. What changes is the implementation that operationalizes each
design decision, and one methodological clarification to the episode persistence
rule.

The prior run is preserved in git history as a documented debugging session.
Study 002B begins fresh from a clean baseline. This document is committed before
any new code is written or any iteration runs.

---

## Root Cause Analysis of Prior Run Failures

**Failure 1 — Contaminated baseline prompt.**
The study began with `prompts/system_prompt.md` containing Study 001's final
over-tightened prompt rather than the original naive prompt. This produced
F1=0.000 at iteration 0. The agent spent the majority of the study attempting
to fix a problem that should not have existed at the start.

**Failure 2 — Silent corpus extractor crashes.**
Most iterations produced "0 abstracts, 25 failures" in 0 seconds. The extractor
was crashing on import errors (when the agent deleted modules that `extractor.py`
still referenced) before processing a single abstract. The interface validator
checked the function signature but never invoked the function, so import errors
were undetected until the corpus run began. The agent received silence as feedback
and continued modifying a broken system.

**Failure 3 — Attention routing signal ceiling.**
Despite Amendment 002's fix (disabling thinking mode), routing scores were stuck
in the 0.05-0.17 range across all iterations. The single-snapshot prefill-only
approach measured attention at one token position, which was insufficient to
distinguish between different extraction strategies. The signal was too narrow
to be informative.

**Failure 4 — Context overflow in Call 2.**
From iteration 20 onward, Call 2's context reached 13,107 tokens and the model
produced garbage output — `0]`, empty strings, partial code fragments. The
accumulated file contents, baseline correction sections, and growing assessment
were not windowed, causing coherent reasoning to collapse.

**Failure 5 — JSON brittleness causing non-blocking failures to become blockers.**
JSON parse failures in both the corpus extractor and the agent response calls
were treated as hard blockers. Iteration after iteration was lost to harness
failures rather than agent reasoning failures. The study measured harness
reliability, not agent self-modification behavior.

**Failure 6 — Real smoke test absent.**
The interface validator confirmed `extract()` existed and had the right signature
but never called it. A real invocation on a single test abstract would have
caught import errors and triggered the repair loop instead of burning a full
corpus run on broken code.

---

## Amendment Decisions

---

**DECISION A003-1 — Attention signal: generation start and generation end.**

The single prefill-only snapshot is replaced by two attention captures per
abstract: one at the first generated token (generation start) and one at the
last generated token immediately before EOS (generation end).

This produces three signals:
- `routing_score_start` — where the model is grounded when it begins generating
- `routing_score_end` — where it is grounded when it finishes
- `intra_generation_delta` — end minus start, measuring whether the model's
  attention to results content holds, improves, or drifts during generation

The inter-iteration delta (how scores change iteration to iteration) is
preserved as before.

**Technical implementation:** Disable `output_attentions=True` during prefill
to avoid the expensive `[seq_len, seq_len]` matrix. Enable it only for two
decode steps — the first generated token and the last. Each decode-step
attention tensor is `[1, heads, 1, seq_len]` rather than the full
`[seq_len, seq_len]` matrix, which substantially reduces memory pressure
compared to the prior approach. The model must generate tokens for this to
work — use `max_new_tokens=20` as a minimal generation budget sufficient
to capture a first and last token without running a full extraction.

This change better reflects the architectural inspiration of the signal:
the model's attention grounding is not a static property but a dynamic one
that evolves through generation. Measuring both endpoints reveals whether
attention stays anchored to evidence or drifts.

---

**DECISION A003-2 — Attention few-shot examples in Call 1 prompt.**

Call 1 (diagnostic assessment) is currently abstract — the agent receives
routing scores and is expected to interpret them correctly without guidance.
This was insufficient. The agent's assessments consistently demonstrated
confusion about what the routing signal means and how to reason from it.

Two to three concrete few-shot examples are added to the Call 1 prompt,
embedded in the system framing, demonstrating:

**Example 1 — Positive routing delta with correct interpretation:**
```
Scenario: routing_score_start=0.12, routing_score_end=0.31, 
          intra_delta=+0.19, iter_delta=+0.08
Correct interpretation: The model begins generation attending weakly to 
results content but strengthens that grounding during generation. 
Inter-iteration improvement suggests the last prompt change moved 
attention toward results sentences. The improvement is real but modest —
further changes should reinforce what worked rather than overhaul the 
approach.
```

**Example 2 — Flat routing with intra-generation drift:**
```
Scenario: routing_score_start=0.28, routing_score_end=0.11,
          intra_delta=-0.17, iter_delta=+0.01
Correct interpretation: The model starts with reasonable results grounding 
but loses it during generation — it begins writing anchored to findings 
but drifts toward background or methodology as the response extends.
The aggregate score is misleadingly flat. The real problem is sustained 
attention, not initial focus. Changes should address how the model 
maintains grounding through extended generation, not where it starts.
```

**Example 3 — Zero extraction with low scores:**
```
Scenario: routing_score_start=0.07, routing_score_end=0.08,
          intra_delta=+0.01, iter_delta=0.00, predicted_claims=[]
Correct interpretation: The model is not attending to results content at 
any point during generation. Empty output is a consequence of this, not 
the cause. Changes to the extraction prompt alone will not fix this — 
the model's attention needs to be redirected toward results sentences 
before it will extract them. Architectural preprocessing or structural 
prompt changes are likely needed.
```

These examples teach the agent what the signal means causally, not just
numerically. The agent should be able to read a routing profile and reason
about whether the model is starting, ending, or sustaining attention on
results content — and which of those is the binding constraint.

---

**DECISION A003-3 — Non-blocking field-by-field LLM calls replacing monolithic JSON.**

The existing architecture produces one large JSON object per agent call.
JSON parse failures block the entire call — if the model produces malformed
output, everything is lost. In Study 002's prior run, this caused iteration
after iteration to be discarded due to harness failures rather than agent
reasoning failures.

Replaced by: individual small LLM calls, one per field.

**Call 1 (Diagnostic) decomposition:**
- `invoke_diagnostic_routing_trend()` → single word: "improving", "declining", or "flat"
- `invoke_diagnostic_last_action_effect()` → plain prose paragraph
- `invoke_diagnostic_pattern_observed()` → plain prose paragraph
- `invoke_diagnostic_hypothesis()` → plain prose paragraph

**Call 2 (Decision) decomposition:**
- `invoke_episode_observation()` → plain prose
- `invoke_episode_hypothesis()` → plain prose
- `invoke_episode_action()` → plain prose
- `invoke_episode_expectation()` → plain prose
- `invoke_edits()` → JSON (the one place where structured output is required)
  - This is the only call that can fail with a JSON parse error
  - All other fields are already persisted before this call runs
  - Edit failures trigger the repair loop as before, but episode data is preserved

**Rationale:** Each individual call is small — a few hundred tokens of completion.
The prefill cost is paid multiple times per iteration (same context, multiple
completions) but completions were never the expensive part. The expensive part
was the prefill, which is now paid N times per iteration rather than once. At
current context sizes this adds latency but produces a study that can run
smoothly even when individual calls fail.

**Non-blocking guarantee:** If any field call fails, the failure is logged as
a specific anomaly (e.g., `field_call_failed: routing_trend`) and a default
value is used:
- Text fields default to `"[not available — call failed]"`
- `routing_trend` defaults to `"flat"`
- Edit call failure triggers repair loop as before

The iteration continues regardless. No single call failure can abort an iteration.

---

**DECISION A003-4 — Episode persistence rule amendment.**

**Prior rule:** Episodes are discarded if edit application fails. Rationale was
that an episode describing unapplied edits is a description without experience —
a phantom action the agent believes it took but didn't.

**Amended rule:** Episodes are always persisted. A boolean field
`edits_applied` on each episode record indicates whether the edit call
succeeded and edits were applied to disk.

**Rationale for change:** Under the new field-by-field architecture, episodes
are obtained independently of edits. The episode represents the agent's actual
reasoning — what it observed, hypothesized, decided, and expected. That
reasoning is real regardless of whether the harness subsequently applied the
edit. Discarding it destroys genuine cognitive data because of a downstream
harness issue.

**What this does NOT change:**

The calibration thesis is preserved by partitioning Sprint 005 analysis
explicitly:

- **Calibration analysis** (expectation vs outcome) uses only episodes where
  `edits_applied=true`. This is the only partition where a real consequence
  exists to compare against. The research thesis — does routing signal
  enable productive self-modification — is evaluated here.

- **Intent analysis** (what the agent wanted to do, independent of execution)
  uses the full episode set including `edits_applied=false` entries. This
  surfaces whether the agent's reasoning tendency was toward code changes vs
  prompt changes, independently of whether the harness let those changes
  through. This is a cleaner measure of agent behavior than what Study 001
  or the prior Study 002 run produced.

**Implication for research thesis:** Separating intent from execution allows
Study 002B to distinguish between two qualitatively different findings:

- If intent rate toward code changes is high but execution rate is low →
  harness finding (the agent was trying to make architectural changes but
  the infrastructure blocked it)
- If both intent rate and execution rate toward code changes are low →
  agent finding (the routing signal didn't drive the agent toward
  architectural thinking)

Study 001's zero code change finding couldn't make this distinction because
the capability framing was ambiguous. Study 002B can.

---

**DECISION A003-5 — Real smoke test in interface validator.**

The interface validator currently checks:
1. `playground/extractor.py` can be imported
2. `extract` function exists and is callable
3. `extract` is an async function
4. `extract` accepts at least 2 positional parameters

**Add step 5:** Actually invoke `extract()` on a hardcoded minimal test abstract
and verify it returns an `ExtractionResult` without raising. This step must
complete within 60 seconds (one provider call). If it fails, log
`interface_smoke_test_failed` anomaly and trigger the repair loop — not the
corpus run.

This catches import errors that only surface at call time (e.g., the agent
deleted a module that `extractor.py` references), which was the cause of
"0 abstracts, 25 failures" in the prior run.

The test abstract used for smoke testing is hardcoded in the harness and does
not change across iterations:

```
"Bilateral hippocampal activation was significantly increased during 
encoding compared to baseline (p < 0.001). Left prefrontal cortex showed 
greater activation for novel stimuli than repeated stimuli."
```

A valid smoke test result is any `ExtractionResult` with at least one claim
in `claims`. An empty `ExtractionResult` (zero claims) is also accepted —
the smoke test verifies the pipeline runs without error, not that it extracts
correctly. The harness logs the smoke test claim count for observability.

---

**DECISION A003-6 — Context windowing for Call 2.**

Prior run: Call 2 context grew unbounded, reaching 13,107 tokens by iteration
20 and producing garbage output from iterations 20-25.

**Amended rule:** Call 2 receives at most the last 5 episodes from episodic
memory, regardless of how many are available. Earlier episodes are dropped.

**Rationale:** N=20 with 5-episode window means the agent loses its earliest
memory by iteration 6. This is a trade-off between memory depth and context
coherence. The alternative — windowing on tokens rather than episode count —
is more principled but harder to implement reliably. Episode count is simple,
deterministic, and prevents the overflow that destroyed the final 5 iterations.

Call 1 (diagnostic assessment) receives the full routing history regardless of
length, since routing history is compact numerical data that does not grow
unboundedly.

If future studies require deeper memory, the episode window can be widened or
a summarization pass can compress older episodes. That is a Study 005 concern.

---

**DECISION A003-7 — Baseline state verification before study begins.**

Before iteration 0 runs, the harness verifies:

1. `prompts/system_prompt.md` contains the original naive prompt and nothing
   else. The harness computes a hash of the file and compares against the
   locked baseline hash committed with this amendment.

2. `prompts/examples.md` is empty (zero bytes or whitespace only).

3. `playground/extractor.py` is the original single-call naive extractor.
   No additional files exist in `playground/` at study start.

4. No prior Study 002B iteration data exists in `experiments/study_002/`
   (metrics.jsonl is empty or absent).

If any check fails, the harness refuses to run and prints which check failed.
This prevents the contaminated baseline problem that invalidated the prior run.

**Locked baseline hash for `prompts/system_prompt.md`:**
```
[PLACEHOLDER: compute SHA-256 of the original naive system prompt file
after it is restored and committed, and record here before running]
```

---

## Baseline State to Restore Before Study 002B

The following files must be restored exactly before the study begins. Commit
the restored state before running any verification or iteration.

**`prompts/system_prompt.md`** — restore to original naive prompt:
```
You are a scientific claim extractor. Given a neuroscience abstract,
extract all scientific claims the abstract explicitly makes.

A scientific claim is a declarative sentence asserting a specific,
testable finding that the abstract supports. Do not include background
statements, prior work references, or methodological descriptions.

Respond with a JSON object in this exact format:
{"claims": ["claim one", "claim two"]}

If no claims are present, return: {"claims": []}
```

**`prompts/examples.md`** — restore to empty file.

**`playground/extractor.py`** — restore to original single-call naive extractor:
```python
from protected.schema import Claim, ExtractionResult
import json
import re
from pathlib import Path

_provider = None

async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:
    prompts_dir = Path(__file__).parent.parent / "prompts"
    system_prompt = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")
    examples = (prompts_dir / "examples.md").read_text(encoding="utf-8").strip()
    if examples:
        system_prompt = system_prompt + "\n\n" + examples
    raw = _provider.complete_with_usage(system_prompt, abstract_text)[0]
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r'```(?:json)?\s*(\{.*?)\s*```', raw, re.DOTALL)
        data = json.loads(m.group(1)) if m else {"claims": []}
    claims = [Claim(claim_text=c) for c in data.get("claims", [])]
    return ExtractionResult(abstract_id=abstract_id, claims=claims)
```

**`playground/`** — delete `preprocessor.py`, `validator.py`, and any other
files created during the prior run. Only `__init__.py` and `extractor.py`
should exist at study start.

---

## Attention Signal Implementation Changes

### `protected/attention/analyzer.py` — complete replacement of `run_prefill`

The prefill-only forward pass is replaced by a minimal generation pass that
captures attention at first and last decode steps.

```python
def run_generation_attention(
    model,
    tokenizer,
    input_dict: dict,
    max_new_tokens: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs minimal generation and captures attention at first and last
    generated tokens.

    Returns:
        start_attn: [seq_len] tensor — last-6-layer averaged attention
                    at the first generated token, over abstract tokens
        end_attn:   [seq_len] tensor — same, at the last generated token
    """
    # Prefill — no output_attentions, just get the KV cache
    with torch.no_grad():
        prefill_out = model(
            input_ids=input_dict["input_ids"],
            attention_mask=input_dict["attention_mask"],
            use_cache=True,
            output_attentions=False,   # no attention during prefill
            return_dict=True,
        )

    past_key_values = prefill_out.past_key_values
    next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del prefill_out
    torch.cuda.empty_cache()

    start_attn_raw = None
    end_attn_raw = None
    generated_count = 0

    for step in range(max_new_tokens):
        with torch.no_grad():
            step_out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,   # capture at each decode step
                return_dict=True,
            )

        # Capture start attention at first decode step
        if step == 0:
            start_attn_raw = step_out.attentions[-6:]

        past_key_values = step_out.past_key_values
        next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_count += 1

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            end_attn_raw = step_out.attentions[-6:]
            del step_out
            break

        # Always update end attention
        end_attn_raw = step_out.attentions[-6:]
        del step_out
        torch.cuda.empty_cache()

    # If we hit max_new_tokens without EOS, end_attn_raw is the last step
    # Extract attention vectors from abstract token range
    start_attn = _extract_abstract_attn(
        start_attn_raw,
        input_dict["abstract_start_token_idx"],
    )
    end_attn = _extract_abstract_attn(
        end_attn_raw,
        input_dict["abstract_start_token_idx"],
    )

    return start_attn, end_attn


def _extract_abstract_attn(
    layer_attns: tuple,
    abstract_start_token_idx: int,
) -> torch.Tensor:
    """
    Average attention across last 6 layers and all heads.
    Slice to abstract token range.
    Returns 1D tensor over abstract tokens.
    """
    averaged = []
    for layer_attn in layer_attns:
        # layer_attn: [1, heads, 1, full_seq_len]
        # The decode step attends from the new token to all prior tokens
        # Squeeze batch and query dims: [heads, full_seq_len]
        attn = layer_attn.squeeze(0).squeeze(1)
        # Average over heads: [full_seq_len]
        averaged.append(attn.mean(dim=0))

    avg_over_layers = torch.stack(averaged).mean(dim=0)
    abstract_attn = avg_over_layers[abstract_start_token_idx:]
    return abstract_attn.cpu().float()
```

### `RoutingScore` dataclass extension

```python
@dataclass
class RoutingScore:
    abstract_id: str
    score: float                          # end routing score (primary signal)
    score_start: float                    # routing at first generated token
    score_end: float                      # routing at last generated token
    intra_generation_delta: float         # score_end - score_start
    results_attention_fraction: float     # from end pass
    methods_attention_fraction: float
    background_attention_fraction: float
    n_results_tokens: int
    n_methods_tokens: int
    n_background_tokens: int
    n_layers_used: int
```

`score` is set to `score_end` for backward compatibility with existing
routing history formatting.

### Natural language signal formatting update

`routing_history.format_for_agent()` is updated to surface all three signals:

```
ROUTING HISTORY — Attention to Results Sentences

Abstract     | Start | End  | Intra Δ | Iter 0 | Iter 1 | Current
-------------|-------|------|---------|--------|--------|--------
10355678     |  0.18 | 0.31 |  +0.13  |  0.22  |  0.31  |  0.31
10234026     |  0.12 | 0.09 |  -0.03  |  0.15  |  0.09  |  0.09
...
AGGREGATE    |  0.15 | 0.22 |  +0.07  |  0.19  |  0.22  |  0.22

Interpretation notes:
- Start score: where the model's attention is grounded before generating
- End score: where it is grounded as it completes generation
- Intra Δ: positive means attention improved during generation;
           negative means it drifted away from results content
- Iter delta: change from prior iteration's end score to this iteration's
```

---

## Agent Caller Changes

### Field-by-field call structure

Each field is obtained with a separate LLM call. The system framing and context
are identical across calls — the same prefill content, different completion
prompts appended at the end. In practice this means the tokenizer pays the
prefill cost once and the calls share the KV cache where the provider supports
it; where it doesn't, prefill is paid N times but completions are tiny.

**Call 1 — Diagnostic fields (4 calls):**

Each call ends with an explicit one-line instruction:

```
# routing_trend call ends with:
"Respond with exactly one word: improving, declining, or flat."

# last_action_effect call ends with:
"Respond with one paragraph describing what the prior modification did
to routing scores, referencing specific abstracts where scores moved notably."

# pattern_observed call ends with:
"Respond with one paragraph describing the pattern you observe across
your episode history and routing trajectory combined."

# hypothesis call ends with:
"Respond with one paragraph describing what direction you think
the system should move, without naming specific file changes."
```

**Call 2 — Episode fields (4 calls) + Edit call (1 JSON call):**

```
# observation call ends with:
"Respond with one paragraph describing what you observed in this
iteration's routing signal and extraction output."

# hypothesis call ends with:
"Respond with one paragraph describing your hypothesis about
what is causing the current pattern."

# action call ends with:
"Respond with one paragraph describing what you will change and why."

# expectation call ends with:
"Respond with one paragraph describing what you expect to observe
in the next iteration as a result of your changes."

# edits call ends with:
"Respond with ONLY a valid JSON array of edit objects matching the
schema below. No other text. No markdown. Raw JSON array only.
Schema: [{"file_path": "...", "operation": "...", ...}]"
```

The edits call returns a JSON array rather than a JSON object wrapping
the array. Simpler to parse, less nesting for the model to track.

**Failure handling per field:**

| Field | On failure | Iteration continues? |
|---|---|---|
| routing_trend | default "flat" | yes |
| last_action_effect | default "[unavailable]" | yes |
| pattern_observed | default "[unavailable]" | yes |
| hypothesis | default "[unavailable]" | yes |
| observation | default "[unavailable]" | yes |
| episode hypothesis | default "[unavailable]" | yes |
| action | default "[unavailable]" | yes |
| expectation | default "[unavailable]" | yes |
| edits | repair loop (3 attempts) | yes, with edits_applied=false |

Every field failure is logged as a specific anomaly. The iteration never aborts
due to a single field failure.

---

## Episode Schema Update

```json
{
  "iteration_n": 3,
  "observation": "string",
  "hypothesis": "string",
  "action": "string",
  "expectation": "string",
  "edits_applied": true,
  "edit_count": 2,
  "field_failures": []
}
```

`edits_applied` is `false` when the edits call failed all repair attempts.
`field_failures` is a list of field names that returned default values.
Episodes are always persisted regardless of `edits_applied`.

---

## Metrics Schema Update

New fields added to the metrics row:

```json
{
  "avg_routing_score_start": 0.0,
  "avg_routing_score_end": 0.0,
  "avg_intra_generation_delta": 0.0,
  "edits_applied": true,
  "episode_persisted": true,
  "field_failure_count": 0,
  "smoke_test_passed": true,
  "smoke_test_claim_count": 0
}
```

`episode_persisted` is now always `true` — episodes are always written.
`edits_applied` on the metrics row matches the episode's `edits_applied`.

---

## Sprint 005 Analysis Partition (Pre-Registered)

Study 002B Sprint 005 analysis explicitly partitions episodes into two sets:

**Partition A — Grounded episodes (`edits_applied=true`)**
Used for: calibration analysis, routing signal effectiveness, F1 trajectory.
Research question: does the routing signal enable productive self-modification?

**Partition B — Full episode set (all episodes)**
Used for: intent analysis, modification surface tendency (code vs prompt),
hypothesis quality assessment.
Research question: what was the agent trying to do, independent of execution?

Findings from Partition B that diverge from Partition A are themselves
significant: they indicate harness reliability affected what the study could
measure, and they inform Study 003 infrastructure requirements.

---

## Files Changed by This Amendment

| File | Change |
|---|---|
| `protected/attention/analyzer.py` | Replace `run_prefill` with `run_generation_attention`. Add `_extract_abstract_attn` helper. Extend `RoutingScore` dataclass with start/end/intra fields. |
| `protected/attention/scorer.py` | Update `compute_routing_score` to accept start and end attention tensors, return all three scores. |
| `protected/harness/study_002/agent_caller.py` | Replace monolithic `invoke_diagnostic` and `invoke_decision` with field-by-field call functions. Add few-shot examples to diagnostic system framing. |
| `protected/harness/study_002/routing_history.py` | Update `format_for_agent` to surface start/end/intra signals in the routing history table. |
| `protected/harness/study_002/study_runner.py` | Add DECISION A003-7 baseline verification checks. Add episode-always-persist logic. Add context windowing (last 5 episodes to Call 2). |
| `protected/harness/shared/interface_validator.py` | Add smoke test step 5: invoke `extract()` on test abstract, verify no exception. |
| `protected/harness/shared/episode_store.py` | Add `edits_applied` and `field_failures` fields to episode records. Remove episode-discard logic. |
| `protected/harness/shared/artifact_writer.py` | Add new metrics fields. Update routing metrics to capture start/end/intra. |
| `protected/harness/shared/anomaly_logger.py` | Add `field_call_failed`, `interface_smoke_test_failed`, `baseline_state_invalid` anomaly types. |
| `prompts/system_prompt.md` | Restore to original naive prompt. |
| `prompts/examples.md` | Restore to empty. |
| `playground/extractor.py` | Restore to original naive single-call extractor. |
| `playground/preprocessor.py` | Delete. |
| `playground/validator.py` | Delete. |
| `experiments/study_002/` | Clear all prior iteration data. Preserve in git history. |

---

## Amendment Record

| Amendment | Date | Description |
|---|---|---|
| 001 | June 2026 | Two-phase agent caller (diagnostic + decision split) |
| 002 | June 2026 | Disable thinking mode in attention forward pass and Call 1 |
| 003 | June 2026 | Infrastructure correction and full redesign for Study 002B rerun |

---

## Commit Instruction

This document is committed to `experiments/study_002/pre-registration-amendment-003.md`
before any code changes, baseline restoration, or data clearing occurs.

The commit SHA of this amendment is the integrity anchor for Study 002B.
No iteration of Study 002B may begin before this document is committed.