# Study 002 Pre-Registration — Intrinsic Cost Signal Isolation

**Organization:** Idris Applied AI Research  
**Study ID:** study_002  
**Date:** 2026-06-07  
**Author:** Muzaffer Ozen  
**Status:** Pre-Registered — Documentation Only  
**Depends on:** Study 001 complete, RDE Framework v1.0 published,  
Research Program Overview v1.0 published  

---

## Preamble

This document is the pre-registration for Study 002. It commits the study design — research question, component under test, corpus, signals, iteration structure, and evaluation criteria — to version control before any harness code is written and before any iteration runs.

Once committed, this document does not change. Subsequent sprints implement against it. Any deviation from the committed design is recorded in the implementation sprint's divergence log with explicit rationale. The discipline of pre-registration before implementation is what makes the study's findings defensible.

This document is documentation only. No harness code, no attention analysis infrastructure, no probe abstract selection, and no corpus runs occur in this sprint.

---

## Research Question

Does attention routing fidelity — an internal signal derived from model attention head distributions during extraction, computed without ground truth or external scoring — provide sufficient consequence signal to enable productive autonomous self-modification in a scientific claim extractor?

Productive self-modification is defined as: modification behavior that produces a measurably different trajectory from the Study 001 baseline on at least one of the primary evidence criteria defined in this document.

---

## Component Under Test

**Component: Intrinsic Cost Signal**

An intrinsic cost signal is an internal evaluation of the quality of the agent's processing state, computed without reference to external ground truth or scoring. Where external signals ask "was the output correct," an intrinsic cost signal asks "was the processing that produced the output in a good state."

In this study, the intrinsic cost signal is attention routing fidelity: a measure of whether the model's attention, during extraction, is routing through results-bearing sentences rather than methodology or background sentences. This signal is computed from the model's attention head distributions via transformers hooks applied to Qwen 3.6 27B during the extraction forward pass.

The hypothesis is that attention routing fidelity is a structurally meaningful proxy for extraction quality — that a model attending to the right parts of the input is more likely to extract correct claims than one attending to the wrong parts — and that an agent receiving this signal as feedback on its modifications can use it to guide productive self-improvement without access to ground truth.

**Why this component:**

Study 001 established that an agent with no consequence signal — one that can only observe its own outputs without any feedback about the quality of its processing — produces a consistent failure mode: internally coherent self-modification that drives quality monotonically downward. The agent optimizes what it can see and has no mechanism to detect that what it can see does not correspond to what matters.

An intrinsic cost signal addresses this failure mode by giving the agent something it did not have in Study 001: feedback about the quality of its own processing, not just the surface properties of its outputs. The signal is internal — derived from the model's own attention distributions — rather than external, making it available without ground truth, human annotation, or scoring infrastructure.

This is the first component identified in the Idris research program's systematic investigation of what an autonomous agent needs to improve itself productively.

---

## Baseline Correction

Study 001 produced a known confound: the agent's capability framing was ambiguous. The harness described the agent as having access to "a Python playground and a set of prompt files" without making clear that it could write, create, and delete Python files. The agent defaulted to prompt-only modification across all 13 iterations, never attempting code changes. It is not possible to determine from Study 001 data whether this behavior reflected a capability ceiling, an architectural limitation, or simply an orientation failure produced by ambiguous framing.

To prevent this confound from contaminating all subsequent studies, a **baseline correction** is applied to Study 002 and all subsequent studies in the program. The baseline correction is not a component under test — it is a repair to a known Study 001 defect. It is held constant across all studies from 002 onward so that cross-study comparisons are not confounded by capability framing differences.

**Baseline correction contents:**

**Explicit capability framing.** The agent is told in plain language, at the start of every iteration's harness prompt, what files it can change and what each file does operationally. Not "you have access to a Python playground" — but:

- `prompts/system_prompt.md` — the instruction the model receives before each abstract. Changing it changes how the model interprets the extraction task.
- `prompts/examples.md` — few-shot examples shown to the model. Changing it demonstrates correct extraction behavior.
- `playground/extractor.py` — the Python function that runs for every abstract. Changing it changes the architecture of extraction: preprocessing, multi-step logic, filtering, post-processing.
- `playground/` (new .py files) — any Python module the agent creates that `extractor.py` can import.

**Worked examples in the harness.** Two worked before/after examples are included in the harness prompt at every iteration:

Example A demonstrates a prompt change: a specific situation, the system prompt before, the system prompt after, and the edit instruction format.

Example B demonstrates an architecture change: a specific situation, the playground code before (a single-call extractor), the playground code after (a preprocessor that segments abstract sentences before extraction), and the edit instruction format. Example B includes complete, runnable Python code — not pseudocode — so the agent has a concrete model of what a code change looks like in this harness.

**Routing signal explanation.** Because Study 002 introduces a novel signal — attention routing fidelity — the baseline correction includes a plain language explanation of what the routing score means, how it is computed (without technical detail), and how to read the routing history across iterations.

The baseline correction harness sections are written once and committed before Study 002's harness is built. They are reused unchanged in Studies 003, 004, 005, and 006.

---

## Decisions

---

**DECISION 002-A — Pre-registration precedes all implementation.**

This document is committed to `experiments/study_002/pre-registration.md` before any harness code is written, before the attention analysis infrastructure is built, and before the 10 probe abstracts are selected. The commit SHA of this document is the study's integrity anchor. Implementation that precedes this commit is a protocol violation.

---

**DECISION 002-B — Study 002 isolates the intrinsic cost signal only.**

All other components are held at the corrected baseline:

- **Configurator:** Baseline correction only. No RDE mechanism. No hero statement. No five-dimension motivational framing. The agent receives a plain task description plus the baseline correction harness sections.
- **World model:** None. The agent does not receive a probe-set delta. It receives routing scores, which reflect the state of its processing — not a prediction of what its modifications will produce.
- **Enhanced memory:** Basic episodic memory identical to Study 001 structure. No pattern analysis, no calibration scoring, no strategy-level reflection.

Isolating the intrinsic cost signal means any behavioral difference from Study 001 is attributable to the routing signal plus the baseline correction. The baseline correction is held constant across all studies, so differences between Study 002 and subsequent studies are attributable to the component added in each.

---

**DECISION 002-C — Corpus structure: 25 per iteration, 200 at validation.**

Each iteration runs the extractor on 25 abstracts:
- 10 fixed control abstracts — the same 10 every iteration, selected randomly from the 200-abstract Study 001 corpus before the study begins and committed to `experiments/study_002/probe_set.json`. These provide longitudinal accountability — the agent can be observed improving or declining on the same abstracts across iterations.
- 15 random impact abstracts — drawn randomly from the remaining 190 each iteration without replacement within the iteration. These capture whether modifications generalize beyond the control set.

The full 200-abstract corpus runs once, at the end of the study, using the agent's final playground state. This is the validation run — it produces the F1 score that is comparable to Study 001's final state and to subsequent studies' validation scores.

No per-iteration F1 scoring occurs during the study. The agent never sees F1. The routing signal is the only feedback available during iteration.

**Rationale:** Study 002 is a signal detection study, not a performance measurement study. The question is whether routing-guided self-modification produces different behavior, not whether it produces a specific F1 score. Fast iteration on 25 abstracts — roughly 20-30 minutes per iteration — enables more modification cycles within the study window than the 80-minute full corpus runs of Study 001. The full corpus validation at study end provides the performance comparison.

---

**DECISION 002-D — Attention routing fidelity: signal design.**

The routing fidelity signal is computed as follows:

**Sentence segmentation:** Each abstract is segmented into sentences. Each sentence is classified as RESULTS, METHODS, or BACKGROUND using rule-based pattern matching. Results sentences contain empirical result language: activation verbs (showed, demonstrated, revealed, found), comparison language (greater than, less than, compared to), statistical markers (significantly, p <, r =), and neuroimaging outcome terms (activation, deactivation, correlation). Methods sentences contain procedural language (participants, were scanned, fMRI protocol, TR, voxel). Background sentences are the remainder. Segmentation uses the same rule set across all iterations and all studies — it does not change.

**Attention capture:** The extractor runs via `transformers` directly (not the llama.cpp server) on each probe abstract. Attention weights are captured from the last 6 layers of Qwen 3.6 27B via forward hooks registered on each attention module. The last 6 layers are selected because higher layers encode semantic content relevant to claim extraction decisions; this selection is fixed for the study.

**Routing score computation:** For each abstract, the routing score is computed as:

```
routing_score = attention_weight_to_results_tokens / 
                attention_weight_to_all_input_tokens
```

Attention weights are averaged across captured layers and across attention heads within each layer. The score ranges from 0 (all attention on non-results tokens) to 1 (all attention on results tokens). A score above 0.5 indicates the model is routing more than half of its attention through results-bearing content when generating claims.

**Routing history:** The agent receives its full routing score history at every iteration — one row per prior iteration showing the routing score per control abstract and the aggregate score. The agent can observe whether its modifications are producing consistent movement in routing scores across iterations, which is the longitudinal signal that distinguishes a productive strategy from an oscillating one.

**Natural language translation:** The routing signal is delivered to the agent as natural language alongside the numerical scores. The harness translates the delta between the prior iteration's routing score and the current iteration's routing score into a plain statement: which control abstracts showed increased routing toward results sentences, which showed decreased routing, and what the aggregate direction was. The agent reasons from the language; the numbers are in the record for Sprint 005 analysis.

---

**DECISION 002-E — Two-call iteration structure.**

Each iteration consists of two sequential LLM calls.

**Call 1 — Routing analysis pass:** The current extractor (current `playground/extractor.py` and `prompts/`) runs on all 25 abstracts. Routing scores are computed for all 25. Results are formatted as natural language: per-abstract routing scores for the 10 control abstracts (with prior iteration scores for comparison), aggregate routing direction, and identification of which abstracts showed the largest routing shifts since the last iteration.

Call 1 does not involve a separate LLM call — it is a computation pass. The output is the routing signal that feeds into Call 2.

**Call 2 — Modification decision:** The agent receives the full harness prompt including:
- Baseline correction sections (capability framing, worked examples, routing signal explanation)
- Episodic memory (all prior episodes in chronological order)
- Current file contents (all playground and prompt files)
- Routing history (full trajectory across all prior iterations)
- Current iteration routing signal (natural language translation of routing results on 25 abstracts)
- Response schema

The agent produces an episode and a set of edit instructions. The harness validates and applies edits, runs interface validation, and repeats the extraction pass for the post-modification routing check.

**Post-modification routing check:** After edits are applied, the extractor runs again on the 10 fixed control abstracts only. The post-modification routing score is computed and recorded alongside the pre-modification score. The delta is the consequence signal for this iteration — what the modification actually did to routing, not what the agent predicted it would do.

The post-modification routing check result is included in the next iteration's routing history, giving the agent a tight feedback loop: modification → observed routing consequence → next modification decision.

---

**DECISION 002-F — No RDE mechanism in Study 002.**

The configurator component is held at the corrected baseline. The agent receives a plain task description:

*"You are an autonomous scientific claim extractor. Your task is to modify your own extraction system — both the Python code in `playground/` and the prompt files in `prompts/` — to improve the quality of claims you extract from neuroscience abstracts. You have access to an attention routing signal that shows you what parts of each abstract your model attends to when extracting claims. Use this signal to guide your modifications."*

No identity framing. No consequence framing. No relationship, standard, or agency framing. The task description is functional, not motivational.

The rationale: Study 004 will test the configurator component in isolation. Introducing motivational framing in Study 002 would confound the intrinsic cost signal with the configurator, making it impossible to attribute behavioral differences to either component alone.

---

**DECISION 002-G — Mutable surface: both playground Python and prompts.**

The agent may modify any file in `playground/` (Python only) and `prompts/system_prompt.md` and `prompts/examples.md`. The allowlist from Study 001 is carried forward unchanged. The baseline correction explicitly informs the agent of this surface in plain language.

No additional restrictions are imposed. Isolating the intrinsic cost signal does not require restricting the modification surface — it requires holding the configurator and world model constant, which is accomplished through harness design, not surface restriction.

---

**DECISION 002-H — Model: Qwen 3.6 27B.**

Qwen 3.6 27B is held constant from Study 001. For the attention analysis pass (Call 1 and the post-modification routing check), the model runs via `transformers` directly to enable forward hook access to attention weights. For the full 200-abstract validation run at study end, the model runs via llama.cpp server as in Study 001.

This dual-backend approach is a pragmatic choice: `transformers` provides attention access for signal computation; llama.cpp provides faster inference for the validation corpus. The extraction behavior is identical across backends for the same model weights.

The model is fixed for the duration of Study 002 and across all isolation studies. Cross-study comparisons are only interpretable if the model is held constant.

---

**DECISION 002-I — N=20 unconditional.**

Twenty iterations with no hard stops. Anomalies are logged as structured events and do not halt the run. The 3-attempt repair loop from Study 001 is carried forward. The iteration timeout is 30 minutes (configurable via `STUDY_ITERATION_TIMEOUT_S`), reflecting the faster per-iteration structure.

Single run (n=1). Model nondeterminism is acknowledged as a limitation in the analysis.

---

**DECISION 002-J — Episodic memory: basic structure from Study 001.**

Each episode contains `observation`, `hypothesis`, `action`, and `expectation`. Final state only — repair attempts are not surfaced in episodic memory. All prior episodes are passed to the agent at every iteration. No windowing at N=20.

The `expectation` field commits a falsifiable prediction before the next iteration's routing scores are seen. Post-hoc calibration analysis (predicted routing direction vs. observed routing direction) is a Study 005 analysis task.

---

**DECISION 002-K — Primary evidence criteria.**

Study 002 produces a positive signal if any of the following are observed:

1. **Routing improvement:** Average routing score on the 10 control abstracts increases monotonically or shows a net positive trend across 20 iterations.

2. **Behavioral differentiation:** The agent's self-modification trajectory is qualitatively different from Study 001's — specifically, if the agent attempts code changes in the playground at any point during the study. Study 001 produced zero code changes across 13 iterations. Any code change in Study 002 is a behavioral difference attributable to the routing signal or the baseline correction.

3. **Self-correction:** The agent reverses a modification direction after observing negative routing consequences. Study 001 showed no self-correction across 13 iterations. Self-correction requires a consequence signal — its presence in Study 002 would implicate the routing signal as the enabling mechanism.

4. **Calibration:** The agent's `expectation` field predicts routing outcomes with above-chance accuracy. If the agent says "I expect routing toward results sentences to increase" and it does, that is evidence the agent is developing an accurate model of how its modifications affect routing.

5. **Validation F1:** The full 200-abstract validation run at study end produces F1 above Study 001's final state (0.142). A higher validation F1 would suggest routing-guided self-modification produced better extraction quality than unguided self-modification.

A positive signal on any single criterion warrants full-scale investigation in a subsequent study. A null result on all criteria is itself a significant finding: if attention routing fidelity does not differentiate agent behavior from a no-signal baseline, that constrains the class of intrinsic signals likely to be effective.

---

**DECISION 002-L — Artifacts and storage.**

All study artifacts persist to `experiments/study_002/` at the repo root. The agent cannot read or write this directory.

```
experiments/
  study_002/
    pre-registration.md       # this document — immutable after commit
    probe_set.json            # 10 fixed control abstract IDs — committed 
                              # before study begins
    episodes.jsonl            # append-only episodic memory ledger
    agent-rationale.jsonl     # free-form rationale per iteration
    anomalies.jsonl           # structured anomaly events
    metrics.jsonl             # programmatic metrics per iteration
    routing_history.jsonl     # routing scores per iteration per abstract
    model_performance.jsonl   # cumulative token and timing metrics
    iterations/               # per-iteration extraction output + snapshots
```

**Routing history format** (one line per iteration):

```json
{
  "iteration_n": 3,
  "timestamp": "ISO8601",
  "pre_modification": {
    "control_scores": {"abstract_id": 0.0},
    "aggregate_score": 0.0
  },
  "post_modification": {
    "control_scores": {"abstract_id": 0.0},
    "aggregate_score": 0.0
  },
  "delta": 0.0
}
```

**Metrics per iteration** extends the Study 001 metrics schema with routing fields:

```json
{
  "iteration_n": 3,
  "timestamp": "ISO8601",
  "pre_routing_score": 0.0,
  "post_routing_score": 0.0,
  "routing_delta": 0.0,
  "routing_direction": "positive | negative | neutral",
  "control_abstracts_improved": 0,
  "control_abstracts_declined": 0,
  "agent_prompt_tokens": 0,
  "agent_completion_tokens": 0,
  "agent_tokens_per_second": 0.0,
  "agent_edits_proposed": 0,
  "agent_edits_applied": 0,
  "playground_files_changed": [],
  "code_changes_attempted": false,
  "repair_attempts": 0,
  "anomaly": false,
  "episode_persisted": false,
  "scanned": false
}
```

`code_changes_attempted` is a boolean that is `true` if any edit in the iteration targeted a `.py` file in `playground/`. This field exists specifically to track whether the baseline correction and routing signal together produce code modification attempts absent in Study 001.

---

## Scope Boundaries

Explicitly out of scope for this pre-registration sprint:

- Attention analysis infrastructure (transformers hooks, routing score computation)
- Sentence segmentation implementation
- Probe abstract selection (committed in first implementation sprint)
- Two-call harness implementation
- Baseline correction harness section drafting
- Any iteration execution
- Study 002 analysis and writeup

All of the above are implementation work. This sprint commits the design. Implementation begins in the next sprint.

---

## Relationship to Study 001

Study 002 differs from Study 001 in exactly two ways:

1. **Baseline correction** — explicit capability framing and worked examples. Applied to all subsequent studies. Not a component under test.

2. **Intrinsic cost signal** — attention routing fidelity. The component under test.

All other design elements are held constant: same model, same corpus (200 abstracts, same ground truth), same N, same episodic memory structure, same edit protocol, same allowlist, same mutable surface, same anomaly handling, same git commit structure.

This deliberate constancy is what makes Study 002's results interpretable against Study 001. Any behavioral difference between the two studies is attributable to the routing signal or the baseline correction. Since the baseline correction is applied identically to all subsequent studies, differences between Study 002 and Study 003 are attributable to the routing signal alone.

---

## Relationship to RDE Framework

Study 002 does not apply the RDE Framework. The configurator component — which the RDE Framework addresses — is held at the corrected baseline in this study. Study 004 is the first study to apply the RDE Framework in a controlled isolation design, using the REFLECTION mechanism through the five-dimension rubric.

The RDE Framework is referenced here as a published methodology artifact that Study 004 will apply. Its existence does not constrain Study 002's design.

---

## Files Created This Sprint

| File | Purpose |
|---|---|
| `experiments/study_002/pre-registration.md` | This document — locked study design |
| `experiments/study_002/episodes.jsonl` | Empty placeholder |
| `experiments/study_002/agent-rationale.jsonl` | Empty placeholder |
| `experiments/study_002/anomalies.jsonl` | Empty placeholder |
| `experiments/study_002/metrics.jsonl` | Empty placeholder |
| `experiments/study_002/routing_history.jsonl` | Empty placeholder |
| `experiments/study_002/model_performance.jsonl` | Empty placeholder |
| `experiments/study_002/iterations/.gitkeep` | Placeholder |

`probe_set.json` is not created in this sprint. It is committed in the first implementation sprint, after random selection from the Study 001 corpus, before any harness code runs.

**No code. No infrastructure. No corpus runs.**

---

*Study 002 Pre-Registration — Idris Applied AI Research*  
*Commit this document before any implementation begins.*  
*The commit SHA of this document is the study's integrity anchor.*

---

# Amendment Addendum — Documented Deviations (A009-5, 2026-07-13)

The original protocol above is unchanged. The following deviations were introduced by
harness amendments during implementation and are logged here for transparency, per
standard pre-registration practice. Each is a response to a degenerate condition the
original protocol did not anticipate; none alters the research question, the corpus, the
signals, or the evaluation criteria (002-D).

**D1 — Early-halt circuit breakers (deviates from "Twenty iterations with no hard stops",
line 193).** Runs may stop before 20 iterations when the loop enters a provably unproductive
state, to avoid burning compute on a fixed point:
- `study_halted_consecutive_anomalies` (A005-2): ≥4 consecutive non-scanned iterations.
- `study_halted_no_progress` (A006-3): ≥5 consecutive iterations that leave the playground
  unchanged (empty-edit no-ops / rollbacks).
- `study_halted_output_stall` (A008 / re-scoped A009-4): the routing signal is a fixed point
  (identical or a 2-cycle) across scanned iterations that nonetheless applied edits.
An early halt is itself a recorded result; it does not fabricate or omit iteration data.

**D2 — Diagnostic-context windowing (deviates from "All prior episodes… No windowing at
N=20", line 201).** The agent's *diagnostic* call receives the last 8 episodes rather than
all prior (A006-4), because the full history overflowed the model's input budget and was
truncated arbitrarily. Episodic memory is still persisted in full; only the per-call context
is windowed.

**D3 — Routing measured on the committed extractor (implementation correction, A009-1).**
The routing pass now measures attention over the input the committed `extractor.py` actually
feeds the model (via a capture shim), as 002-E line 141 specifies ("the current extractor…
runs"), rather than over the raw abstract. This *restores* fidelity to the pre-registration;
it was an implementation bug that the routing pass ignored extractor code changes.

**D4 — Config alignment (A009-3).** `N_ITERATIONS` and the iteration timeout are set to the
pre-registered 20 iterations and 30 minutes; a prior build had drifted to 25 / 4 hours.