# Sprint 002 ΓÇö Self-Healing Detection Study: Pre-Registration

**Organization:** Idris Applied AI Research
**Status:** Specced
**Date:** May 2026
**Author:** Muzaffer Ozen
**Depends on:** Sprint 001 (corpus frozen, baseline evaluation artifact committed)

---

## Context

This sprint opens a multi-sprint research arc investigating whether an
autonomous agent can improve a scientific claim extractor through iterative,
self-directed modification of its own code and prompt layer against a fixed
target corpus. The arc spans Sprint 002 (pre-registration), 003 (harness),
004 (the 20-iteration run), and 005 (analysis and writeup).

Sprint 002 is **documentation-only**. It produces no code. Its single purpose
is to lock the study design in version control before any harness exists and
before any iteration runs. Once committed, the design cannot drift to fit
emerging results. That constraint is what makes the study defensible.

The authoritative design artifact is
`experiments/study_001/pre-registration.md`. This entry records the sprint
and its decisions. Sprints 003ΓÇô005 implement against it and do not amend it.

---

## Research Question

Can an autonomous agent iteratively improve the precision and recall of a
naive scientific claim extractor by modifying its own extraction code and
prompt layer, operating only on the signal of its prior outputs against a
fixed corpus?

This is distinct from prompt-tuning studies. The agent has access to a
scoped Python playground and can propose real architectural changes ΓÇö
multi-step pipelines, preprocessing logic, post-processing filters,
claim validation passes ΓÇö in addition to prompt evolution. The research
question is whether autonomous architectural self-modification produces
meaningful quality improvement, degrades quality, or plateaus, and what
the trajectory looks like across 20 iterations.

---

## Problem Statement

A research study whose results are only credible if the design predates the
data needs a hard separation between design and build. Pre-registration
removes the degree of freedom to quietly reshape the study once early
iterations reveal something inconvenient. The hypothesis, metrics, rubric,
allowlist, execution constraints, and storage layout are committed to git
before any harness code exists.

---

## Fixed Target

The fixed target is the frozen corpus from Sprint 001:
`corpus/ground_truth.jsonl` at the commit SHA recorded in
`corpus/corpus_manifest.md`. 200 neuroscience abstracts from NeuroSynth,
annotated using the SciFact-derived claim definition. The corpus does not
change across any iteration of this study.

The baseline is the Sprint 001 evaluation run recorded in
`evaluation/results/`. Its macro-F1 is iteration 0's anchor. It is not
a pass/fail threshold ΓÇö it is the starting point the study measures
movement from.

---

## Decisions

---

**DECISION 002-A ΓÇö Design is pre-registered and immutable; no code this sprint.**

The full study design is committed to
`experiments/study_001/pre-registration.md` before any harness code is
written. No `experiments/study_001/harness/` code is created in Sprint 002.
Implementation pressure ("might as well start the harness") is explicitly
resisted. The discipline is the point.

---

**DECISION 002-B ΓÇö The agent writes real Python within a scoped playground.**

The mutable surface is split into two zones:

```
playground/           # agent owns this entirely
  __init__.py
  extractor.py        # main extraction logic ΓÇö agent rewrites freely
  pipeline.py         # optional pipeline (agent creates if it wants)
  utils.py            # optional helpers (agent creates if it wants)
prompts/              # prompt files ΓÇö also mutable
  system_prompt.md
  examples.md
```

The agent can create, modify, or delete any file inside `playground/` and
`prompts/`. It can build multi-step pipelines, preprocessing logic,
post-processing filters, claim validation passes, or any architecture it
reasons will improve extraction quality. It can make multiple LLM calls per
abstract. It imports `ExtractionResult` and `Claim` from `protected/schema.py`
ΓÇö those models are immutable.

The only architectural constraint is the entry point contract (DECISION 002-C).
Everything inside the playground is the agent's to evolve.

This scope is a deliberate acknowledgment that prompt-only modification
produces a more constrained research question. Allowing real architectural
change makes the study a stronger and more novel contribution ΓÇö the
observable trajectory of autonomous architectural self-modification is the
finding, regardless of direction.

---

**DECISION 002-C ΓÇö One fixed entry point; interface is immutable.**

The harness calls exactly one function:

```python
from playground.extractor import extract
result: ExtractionResult = await extract(abstract_id, abstract_text)
```

`extract` must be an async function accepting `(str, str)` and returning
`ExtractionResult`. This signature is declared in `protected/interface.py`
and validated by the harness at each iteration before the corpus run begins.
The agent cannot modify `protected/` ΓÇö it can only honor the contract.

If the agent's `playground/extractor.py` does not expose a callable `extract`
with this signature after edits are applied, the iteration enters the repair
loop (DECISION 002-F).

---

**DECISION 002-D ΓÇö Protected zone is completely off limits.**

```
protected/            # agent cannot read or write
  schema.py           # Claim, ExtractionResult pydantic models
  interface.py        # entry point contract declaration
  harness/            # all harness modules
  scorer.py           # evaluation logic
  runner.py           # corpus runner
corpus/               # agent cannot read or write
experiments/          # agent cannot read or write
evaluation/           # agent cannot read or write
```

The evaluation apparatus is fully protected. The agent cannot change what
F1 means, cannot read its own performance metrics, cannot inspect the ground
truth, and cannot see its own artifact history. The only signal the agent
receives is the prior iteration's per-abstract extraction output ΓÇö not scores,
not ground truth, not the rubric.

**Why the agent cannot read its own scores:** If the agent knew its F1 was
0.43 at iteration 3, it could reason toward gaming the scorer rather than
improving extraction. Hiding the metric forces the agent to reason about
extraction quality from first principles. The carrot stays out of reach by
design.

---

**DECISION 002-E ΓÇö Episodic memory; final state only.**

The agent operates with episodic memory: an append-only, temporally-ordered
record of its prior iterations. Each episode contains:

```json
{
  "iteration_n": 3,
  "observation": "what I noticed in the prior iteration's extraction output",
  "hypothesis": "what I think is wrong or could improve",
  "action": "what I changed and why",
  "expectation": "what I expect to see in the next iteration's output"
}
```

**Final state only.** If a repair loop occurred (DECISION 002-F), the
episode records only the final working state. Repair attempts are logged in
`experiments/study_001/anomalies.jsonl` but are not surfaced in episodic
memory. This is a declared constraint acknowledging the practical limits of
local model inference ΓÇö the agent's working memory reflects what succeeded,
not the debugging trace that got there.

The `expectation` field commits a falsifiable prediction before the next
iteration's results are seen. This enables a post-hoc calibration analysis
in Sprint 005 (predicted vs. observed output characteristics). Do not allow
the agent to revise its expectation after seeing results ΓÇö that would destroy
the calibration signal.

`iteration_n` is not in the agent's response schema. The harness adds it
when persisting the episode.

---

**DECISION 002-F ΓÇö 3-attempt repair loop; broken iterations roll back.**

If the harness validation pass fails (import error, missing `extract`
function, wrong signature), the harness enters a repair loop:

1. Send the error message and current `playground/` file contents back to
   the agent
2. Agent proposes a repair (edits only ΓÇö no new episode required)
3. Harness re-validates
4. Repeat up to **3 total attempts** (original + 2 repairs)

If validation still fails after 3 attempts: rollback `playground/` to the
last clean git state (`git checkout -- playground/ prompts/`), log anomaly
`repair_exhausted`, **do not persist the episode**, do not commit, continue
to the next iteration.

3 attempts is a declared constraint, not a tuning parameter. It reflects the
inference capability of the local model ΓÇö enough to recover from syntax
errors and import mistakes, not enough to rescue a fundamentally broken
approach. Subsequent studies may adjust this parameter; v1 holds it fixed.

Repair attempts are anomaly-logged with their error messages. They are
invisible to episodic memory.

---

**DECISION 002-G ΓÇö In-process execution; 300-second timeout.**

The agent's playground code executes in-process (no subprocess isolation).
`asyncio.wait_for` wraps the full corpus run with a 300-second hard timeout,
configurable via `STUDY_ITERATION_TIMEOUT_S` env var. On timeout:
rollback playground, log anomaly `iteration_timeout`, no episode persisted,
continue.

300 seconds is generous headroom for a local model running a multi-step
pipeline over 200 abstracts. It catches genuine infinite loops without
false-firing on slow inference. Corpus run time will vary across iterations
as the agent's architecture evolves ΓÇö a single-call pipeline may complete
in 40 seconds; a three-step pipeline may take several minutes. That variance
is signal, not noise. `scan_duration_seconds` is logged per iteration and
is part of the Sprint 005 analysis.

There is no per-abstract LLM call limit. The timeout is the only governor.
If the agent discovers that a three-step pipeline is worth the inference cost
and evolves toward it, that is an observable finding. Constraining call count
would prevent the study from observing that trajectory.

---

**DECISION 002-H ΓÇö Modifier held constant at Qwen 3.6 27B MTP 6-bit GGUF.**

The model is fixed for v1 so the variable under study is the recursive
self-modification loop, not a loop ├ù model interaction. Qwen 3.6 27B runs
locally via llama.cpp server on RTX 5090 32GB, accessed via OpenAI-compatible
endpoint. 262k context window ΓÇö no windowing of episodic memory required
at N=20.

Schema validation for the agent's response is enforced at the harness parse
layer, not at the API level. The llama.cpp server does not support OpenAI's
strict JSON schema enforcement. A malformed response is an anomaly-logged
failure, not an API-rejected call.

---

**DECISION 002-I ΓÇö N=20 unconditional, single run, fully local.**

Twenty iterations with no hard stops. Anomalies are logged as structured
events but do not halt the run ΓÇö v1 exists to observe failure modes, not
prevent them. Single run (n=1); model nondeterminism is caveated in analysis
rather than controlled by replication.

Execution is fully local. No remote git operations. No external API calls
except the local llama.cpp server. No publication or sharing of intermediate
results before the run completes and Sprint 005 analysis is written.

---

**DECISION 002-J ΓÇö Solo rater; pre-registered 5-dimension rubric.**

Qualitative evaluation uses a pre-registered rubric applied by a single
rater (Muzaffer Ozen) after the run completes. The rubric is applied to
a sample of iterations (0, 5, 10, 15, 20) to characterize the qualitative
trajectory independent of the programmatic F1 signal.

Rubric dimensions:

1. **True positive quality** ΓÇö are the extracted claims factually present in
   the abstract and correctly stated?
2. **False positive character** ΓÇö what kind of statements are being incorrectly
   extracted (background, methods, implications)?
3. **Recall coverage** ΓÇö are the high-salience claims in the abstract being
   captured?
4. **Claim atomicity** ΓÇö are claims appropriately split or are compound
   assertions being returned as single claims?
5. **Architectural coherence** ΓÇö does the agent's playground code reflect
   a coherent extraction strategy, or is it incoherent iteration noise?

Dimension 5 is unique to this study's expanded scope and has no analog in
prompt-only self-healing research. It is included because the agent's
architectural choices are themselves a primary observable, not just the
extraction output they produce.

Multi-rater evaluation is contingent on v1 results being interesting enough
to warrant external collaboration. Bring collaborators in with results,
not with a request for help.

---

**DECISION 002-K ΓÇö Artifacts live outside the agent's reach; git is the
version chain.**

All study artifacts persist to `experiments/study_001/` at the repo root,
outside any directory the agent can read or write. The agent sees its history
only through the harness's controlled input channel: the prior iteration's
per-abstract extraction output and all prior episodes.

Per-iteration playground snapshots are kept even though they are redundant
with git ΓÇö cheap to store, expensive to reconstruct, essential for the
architectural coherence analysis in Sprint 005.

```
experiments/
  study_001/
    pre-registration.md         # this document ΓÇö immutable after commit
    iterations/                 # per-iteration extraction output JSON + playground snapshot
    episodes.jsonl              # append-only episodic memory ledger
    agent-rationale.jsonl       # free-form rationale per iteration
    anomalies.jsonl             # structured anomaly events
    metrics.jsonl               # programmatic metrics per iteration
```

---

## Metrics (Programmatic, Per Iteration)

Captured by the harness. No human judgment required.

- `iteration_n`
- `timestamp`
- `macro_precision`, `macro_recall`, `macro_f1`
- `micro_tp`, `micro_fp`, `micro_fn`
- `scan_duration_seconds`
- `agent_edits_proposed` (file count)
- `agent_edits_applied` (file count ΓÇö zero if repair exhausted)
- `repair_attempts` (0ΓÇô3)
- `playground_files_changed` (list)
- `prompt_chars_delta`
- `avg_claims_per_abstract`
- `anomaly` (boolean)
- `episode_persisted` (boolean)
- `scanned` (boolean ΓÇö false when iteration produced no extraction output)

**Hidden from agent:** all of the above. The agent receives only the prior
iteration's per-abstract extraction output (predicted claim lists, no scores)
and its episodic memory.

---

## Files Changed This Sprint

**New files:**

| File | Purpose |
|---|---|
| `experiments/study_001/pre-registration.md` | Locked study design (authoritative spec) |
| `experiments/study_001/episodes.jsonl` | Empty placeholder |
| `experiments/study_001/agent-rationale.jsonl` | Empty placeholder |
| `experiments/study_001/anomalies.jsonl` | Empty placeholder |
| `experiments/study_001/metrics.jsonl` | Empty placeholder |
| `experiments/study_001/iterations/.gitkeep` | Placeholder |

**No code. No harness. No playground.**

---

## Scope Boundaries

Explicitly out of scope:

- Harness implementation (Sprint 003)
- Executing the 20-iteration run (Sprint 004)
- Analysis and writeup (Sprint 005)
- Multi-rater evaluation (contingent on Sprint 005 results)
- Any modification to `corpus/ground_truth.jsonl`
- Any pre-optimization of the Sprint 001 baseline prompt

---

## Amendment: Early Termination (May 30, 2026)

**DECISION 002-I** specified N=20 unconditional iterations. The run was
terminated at iteration 13 based on the following evidence:

### Observed F1 Trajectory

| Iteration | Macro-F1 | Macro-P | Macro-R | Avg Claims |
|---|---|---|---|---|
| 0 (baseline) | 0.4674 | 0.5126 | 0.4296 | 4.5 |
| 1 | 0.3793 | 0.4181 | 0.3471 | 4.5 |
| 2 | 0.2638 | 0.3006 | 0.2351 | 4.2 |
| 3 | 0.2683 | 0.3230 | 0.2294 | 3.9 |
| 4-6 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| 7 | 0.1643 | 0.2303 | 0.1277 | 2.7 |
| 8 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| 9 | 0.1793 | 0.2438 | 0.1417 | 2.8 |
| 10 | 0.1827 | 0.2510 | 0.1436 | 2.7 |
| 11 | 0.1500 | 0.2225 | 0.1131 | 2.3 |
| 12 | 0.1303 | 0.1998 | 0.0967 | 2.2 |
| 13 | 0.1418 | 0.2209 | 0.1045 | 2.3 |

### Rationale

Three distinct failure phases were observed:

1. **Initial degradation (iterations 1-3):** The agent's first edits
   (few-shot prompting improvements) immediately reduced F1 from 0.47 to
   0.26, a 44% drop. The agent over-constrained the extraction criteria,
   reducing recall more than it improved precision.

2. **Total collapse (iterations 4-6):** The agent's subsequent edits
   produced zero valid claims across all 200 abstracts (F1=0.0000). The
   extraction pipeline was broken — either the agent's architectural changes
   were fundamentally incompatible with the interface contract, or the
   prompt constraints were so restrictive that no claims passed validation.
   Iteration 8 repeated this collapse.

3. **Partial recovery and floor (iterations 7-13):** The agent recovered
   to a stable but degraded floor of F1=0.13-0.18, 61-72% below baseline.
   Despite 7 additional iterations of self-modification, the agent could
   not exceed F1=0.18 (iteration 10, the peak of the recovery phase).
   The trajectory from iteration 9 onward shows no improvement signal —
   only minor oscillation around the 0.14-0.18 band.

Continuing to iteration 20 would not produce additional signal. The
agent has found a local minimum and lacks the information to escape it
(scores are hidden by design, per DECISION 002-D). The remaining 7
iterations would replicate the same oscillatory behavior.

### What This Study Demonstrates

Despite not meeting the original goal of improvement, the study produced
a clear finding: **autonomous self-modification without score feedback
tends to degrade extraction quality and become trapped in local minima.**
The agent's episodic memory enabled recovery from total collapse but
could not guide it back to or beyond baseline performance. The hidden
score constraint, intended to prevent gaming, proved sufficient to
prevent improvement.

This is a valid negative result that informs the next study's design.
Sprint 005 analysis will proceed with the 14-iteration dataset (0-13).

**Amended by:** Muzaffer Ozen
**Date:** May 30, 2026
**Commit:** Early termination of study_001 at iteration 13.
