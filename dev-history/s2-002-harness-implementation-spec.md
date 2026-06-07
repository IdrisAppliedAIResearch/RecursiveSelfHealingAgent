# Study 002 — Harness Implementation Spec

**Organization:** Idris Applied AI Research  
**Study ID:** study_002  
**Date:** June 2026  
**Author:** Muzaffer Ozen  
**Status:** Specced  
**Depends on:** Study 002 pre-registration committed before this sprint begins

---

## Summary

Builds the harness that executes Study 002's isolation of the intrinsic cost signal component. The new infrastructure in this sprint centers on the attention analysis pipeline — transformers-based forward hooks, rule-based sentence segmentation, and routing fidelity score computation. The study runner, agent caller, and artifact writer are adapted from Study 001's harness with targeted extensions.

This sprint also selects and commits the 10 fixed control abstracts before any harness code runs against them. That ordering is non-negotiable: probe set selection is the first act of this sprint, committed before the attention infrastructure is built.

The acceptance bar: harness executes two iterations end-to-end, routing scores are computed correctly on probe abstracts, the baseline correction harness sections are present in the agent call, and the routing history is correctly injected and persisted.

---

## Repository Layout (Post-Sprint)

```
protected/
  attention/
    __init__.py
    analyzer.py           # transformers model loading + forward hook management
    segmenter.py          # rule-based sentence classification
    scorer.py             # routing fidelity score computation
  harness/
    study_001/            # Study 001 harness — untouched, reference only
      ...
    study_002/
      __init__.py
      study_runner.py     # Study 002 iteration loop
      agent_caller.py     # Two-call structure with routing signal injection
      baseline_correction.py  # Committed harness sections — capability framing,
                              #   worked examples, routing signal explanation
      routing_history.py  # Routing score persistence and retrieval
    shared/               # Modules reused across studies
      edit_protocol.py    # Unchanged from Study 001
      edit_applier.py     # Unchanged from Study 001
      allowlist.py        # Unchanged from Study 001
      interface_validator.py  # Unchanged from Study 001
      episode_store.py    # Unchanged from Study 001
      anomaly_logger.py   # Extended with Study 002 anomaly types
      git_ops.py          # Unchanged from Study 001
      artifact_writer.py  # Extended with routing metrics fields
      model_performance.py    # Unchanged from Study 001
      corpus_runner.py    # Unchanged from Study 001
  schema.py               # Unchanged from Study 001
  interface.py            # Unchanged from Study 001
  scorer.py               # Unchanged from Study 001
experiments/
  study_002/
    pre-registration.md   # Committed in pre-registration sprint
    probe_set.json        # Committed this sprint — first act before any code
    episodes.jsonl
    agent-rationale.jsonl
    anomalies.jsonl
    metrics.jsonl
    routing_history.jsonl
    model_performance.jsonl
    iterations/
```

**Study 001 harness is not modified.** All shared logic is refactored into `protected/harness/shared/` and imported by both study runners. If a shared module requires extension for Study 002 (anomaly logger, artifact writer), the extension is additive — no existing behavior changes.

---

## Pre-Implementation Step: Probe Set Selection

Before any code is written, the 10 fixed control abstracts are selected and committed.

```python
# scripts/select_probe_set.py
# Randomly selects 10 abstract IDs from corpus/abstracts/
# Writes experiments/study_002/probe_set.json
# Commits the file
# Must be run before any harness implementation begins
```

`probe_set.json` format:

```json
{
  "study_id": "study_002",
  "selection_method": "random",
  "selected_at": "ISO8601",
  "corpus_size": 200,
  "probe_count": 10,
  "abstract_ids": ["12345678", "..."]
}
```

The script commits `probe_set.json` with message `[study_002] commit probe set — 10 fixed control abstracts`. This commit precedes all harness implementation commits. The ordering is verifiable in git log.

---

## Critical Design: Two Inference Backends

Study 002 uses two inference backends for the same model weights:

**Backend 1 — transformers (attention analysis)**
Used exclusively for the attention analysis pass. Provides access to attention weight tensors via forward hooks. Slower than llama.cpp. Runs on the 10 control abstracts only (pre-modification) and 10 control abstracts (post-modification). Does not run the 15 random impact abstracts — routing scores are computed only for the control set.

**Backend 2 — llama.cpp (extraction)**
Used for the 25-abstract mini-corpus extraction pass (same as Study 001). Provides the extraction output the agent reasons from. Does not provide attention weights. Fast inference.

The two backends produce independent outputs that are combined by the study runner:
- llama.cpp extraction output → what claims were extracted (sent to agent)
- transformers routing scores → how attention was distributed (also sent to agent)

The extraction output and routing scores come from separate forward passes on the same abstracts. They are not the same pass. The coding agent must not conflate them.

**Model weight compatibility:** The same Qwen 3.6 27B weights are used by both backends. The transformers pass loads the model in the format transformers requires. The llama.cpp pass uses the GGUF quantized version. If outputs differ due to quantization differences, this is noted as a limitation — it does not change the design.

---

## Component Specifications

---

### `protected/attention/analyzer.py`

Manages transformers model loading and forward hook registration. The model is loaded once at study startup and held in memory for the duration of the run. Loading a 27B model is expensive — never reload between iterations.

```python
class AttentionAnalyzer:
    def __init__(self, model_path: str, n_last_layers: int = 6):
        """
        Loads Qwen 3.6 27B via transformers.
        Registers forward hooks on the last n_last_layers attention modules.
        model_path: path to transformers-compatible model weights
        n_last_layers: number of layers from which to capture attention (default 6)
        """

    def forward_pass(
        self,
        abstract_text: str,
        system_prompt: str,
    ) -> AttentionResult:
        """
        Runs a single forward pass on system_prompt + abstract_text.
        Returns AttentionResult containing per-layer attention weight tensors
        for the last n_last_layers layers.
        Clears stored hook outputs before each pass.
        Does not generate tokens — prefill only.
        """

    def close(self) -> None:
        """
        Releases model from GPU memory.
        Called once at study end.
        """
```

**Forward hook implementation:**

Following the pattern established in Idris's prior attention gateway work, hooks are registered using `register_forward_hook` on each targeted attention module:

```python
def _register_hooks(self) -> None:
    blocks = self.model.model.layers
    start = max(0, len(blocks) - self.n_last_layers)
    for i, block in enumerate(blocks[start:], start=start):
        block.self_attn.register_forward_hook(
            self._make_hook(layer_idx=i)
        )

def _make_hook(self, layer_idx: int):
    def hook(module, input, output):
        # output is (attn_output, attn_weights, ...)
        # attn_weights shape: [batch, heads, seq_len, seq_len]
        if isinstance(output, tuple) and len(output) > 1:
            self._stored_weights[layer_idx] = output[1].detach().cpu()
    return hook
```

**Memory management:** Attention weight tensors for a 27B model on 25 abstracts can be large. Store only the last-token attention row — shape `[heads, seq_len]` — rather than the full `[seq_len, seq_len]` matrix. The last token's attention distribution is what the model uses when deciding what to extract, matching the approach validated in the attention gateway project.

**Qwen architecture note:** Qwen uses grouped-query attention (GQA). Attention weight shapes may differ from standard multi-head attention. The hook implementation must handle GQA correctly — verify the output tuple structure against the actual Qwen 3.6 27B architecture before finalizing the hook. Log the attention weight shapes at study startup for verification.

---

### `protected/attention/segmenter.py`

Rule-based sentence classification. Deterministic, fast, requires no model calls.

```python
def segment_abstract(abstract_text: str) -> list[Sentence]
```

`Sentence` dataclass:

```python
@dataclass
class Sentence:
    text: str
    label: str          # "RESULTS" | "METHODS" | "BACKGROUND"
    char_start: int     # character offset in original abstract
    char_end: int
    token_start: int    # token position (set by scorer after tokenization)
    token_end: int
```

**Sentence boundary detection:** Split on `. ` with handling for common neuroscience abbreviations that contain periods: `e.g.`, `i.e.`, `vs.`, `Fig.`, `Eq.`, `et al.`, `vol.`, `no.`. A period followed by a lowercase letter is not a sentence boundary unless the preceding word is in the abbreviation list.

**Classification rules:**

RESULTS — sentence contains at least one match from the results pattern set:
```python
RESULTS_PATTERNS = [
    r'\b(showed?|demonstrated?|revealed?|found|observed|identified|detected)\b',
    r'\b(significantly|greater than|less than|more than|compared to|relative to)\b',
    r'\b(activation|deactivation|correlation|increase[sd]?|decrease[sd]?|reduction)\b',
    r'\b(p\s*[<>=]\s*0\.\d+|t\s*\(\d+\)|F\s*\(\d+|r\s*=\s*[-\d.])\b',
    r'\b(higher|lower|larger|smaller|stronger|weaker)\b.{0,40}\b(than|compared)\b',
    r'\b(bilateral|unilateral|left|right)\b.{0,30}\b(cortex|gyrus|sulcus|area|region)\b',
]
```

METHODS — sentence contains at least one match from the methods pattern set and no results pattern matches, or methods pattern count exceeds results pattern count:
```python
METHODS_PATTERNS = [
    r'\b(participants?|subjects?|volunteers?|patients?)\b.{0,20}\b(were|had|completed)\b',
    r'\b(fMRI|MRI|PET|EEG|MEG)\b.{0,30}\b(scanner|session|protocol|study)\b',
    r'\b(TR|TE|voxel|slice|mm|tesla|T)\b',
    r'\b(we used|study (examined|investigated|aimed|used)|designed to)\b',
    r'\b(informed consent|ethics|IRB|approved)\b',
    r'\b(\d+\s*(male|female|men|women|participants|subjects))\b',
]
```

BACKGROUND — all sentences not classified as RESULTS or METHODS. This includes introductory context, prior work references, and objective statements.

**Classification decision:** If a sentence matches both RESULTS and METHODS patterns, count the matches. The label with more matches wins. On a tie, classify as RESULTS (erring toward inclusion in the routing target).

**Token alignment:** After tokenization by the transformers tokenizer, map each sentence's character offsets to token positions. This mapping is required by the scorer to sum attention weights by sentence type. The mapping is computed once per abstract and cached.

---

### `protected/attention/scorer.py`

Computes routing fidelity scores from attention weights and sentence segmentation.

```python
def compute_routing_score(
    attention_result: AttentionResult,
    segments: list[Sentence],
    tokenizer,
) -> RoutingScore
```

`RoutingScore` dataclass:

```python
@dataclass
class RoutingScore:
    abstract_id: str
    score: float                    # 0.0 to 1.0
    results_attention_fraction: float
    methods_attention_fraction: float
    background_attention_fraction: float
    n_results_tokens: int
    n_methods_tokens: int
    n_background_tokens: int
    n_layers_used: int
```

**Computation:**

1. For each captured layer's attention weights (shape `[heads, seq_len]` for the last token):
   - Average across all heads: shape `[seq_len]`
   - This is the last token's attention distribution — how much it attends to each prior token

2. For each sentence type, sum the attention weights corresponding to tokens in that sentence type's spans.

3. Normalize by total attention to abstract tokens only (exclude system prompt tokens from the denominator — we care about attention within the abstract, not attention to the prompt):

```
results_fraction = sum(attn[results_token_positions]) / 
                   sum(attn[all_abstract_token_positions])
```

4. Average across all captured layers.

5. `routing_score = results_fraction`

**Edge cases:**
- If an abstract has no RESULTS sentences (all methods or background): routing score is 0.0. This is valid — the segmenter may have classified all sentences as non-results. Log as `no_results_sentences` in the routing record.
- If tokenization produces no abstract tokens (pathological case): routing score is null, log anomaly `routing_score_null`.
- If attention weights are None for a layer (GQA variant that doesn't expose weights): skip that layer and note in `n_layers_used`.

---

### `protected/harness/study_002/routing_history.py`

Persists and retrieves routing scores across iterations.

```python
def append(
    study_id: str,
    iteration_n: int,
    pre_scores: list[RoutingScore],
    post_scores: list[RoutingScore],
) -> None

def load_all(study_id: str) -> list[dict]

def format_for_agent(
    history: list[dict],
    current_pre_scores: list[RoutingScore],
) -> str
```

**`format_for_agent` output:**

Produces a natural language + tabular summary for injection into the agent call:

```
ROUTING HISTORY — Attention to Results Sentences

Abstract       | Iter 0 | Iter 1 | Iter 2 | Current
---------------|--------|--------|--------|--------
10220229       |  0.41  |  0.38  |  0.43  |  0.51
10234026       |  0.55  |  0.52  |  0.48  |  0.44
...
AGGREGATE      |  0.46  |  0.43  |  0.45  |  0.49

Current iteration routing signal:
After your iteration 2 modification, routing toward results sentences 
increased on 6 of 10 control abstracts (aggregate: +0.04). 
The largest increase was on abstract 10220229 (+0.08). 
The largest decrease was on abstract 10234026 (-0.04).

Trend: Your last 3 modifications have produced a net positive routing 
shift of +0.03. The aggregate routing score has not yet crossed 0.5 
(the threshold at which the model attends more to results than to 
non-results content on average).
```

The trend statement is computed from the routing history. If fewer than 3 prior iterations exist, the trend statement is omitted. If routing scores have not moved more than 0.02 in either direction across the last 3 iterations, the statement reads: "Your last 3 modifications have produced no meaningful routing change. Consider a different modification approach."

That last sentence is the consequence signal that may drive the agent toward code changes. It is not a recommendation — it is an observation. The agent decides what to do with it.

---

### `protected/harness/study_002/baseline_correction.py`

A module of committed text constants. Not generated per-iteration — written once in this sprint and committed. Included in every agent call verbatim.

```python
CAPABILITY_FRAMING: str     # Plain language tool description
WORKED_EXAMPLE_A: str       # Prompt change example
WORKED_EXAMPLE_B: str       # Architecture change example with real Python
ROUTING_SIGNAL_EXPLANATION: str  # What routing score means, how to read it

def compose_baseline_correction() -> str:
    """Concatenates all sections in order."""
```

**CAPABILITY_FRAMING content:**

```
YOUR TOOLS

You have four types of files you can change. Here is what each one 
does and how changing it affects the extraction system:

prompts/system_prompt.md
  This is the instruction the model reads before seeing each abstract.
  Changing it changes how the model interprets the extraction task —
  what it looks for, what it excludes, and how it formats its output.

prompts/examples.md
  These are example abstracts with correct extractions shown alongside.
  Changing them teaches the model by demonstration. An empty file means
  no examples are shown.

playground/extractor.py
  This is the Python function that runs for every abstract. It is called
  as: result = await extract(abstract_id, abstract_text)
  Changing it changes the architecture of extraction — you can add
  preprocessing that runs before the model call, post-processing that
  filters results after, or multiple model calls in sequence.

playground/ (new .py files you create)
  You can create any Python file in playground/ and import it from
  extractor.py. This lets you build modular components — a preprocessor,
  a validator, a claim filter — as separate modules.

WHAT YOU CANNOT CHANGE
The evaluation system, the corpus, the ground truth, the harness, and
the scoring infrastructure are off limits. You can only change what is
listed above.
```

**WORKED_EXAMPLE_A content:**

```
EXAMPLE A — Changing the system prompt

Situation: The extractor is including methodology descriptions as claims.
Abstracts that describe fMRI protocols are producing claims like
"The study used a 3T scanner with TR=2000ms" which are not findings.

Change: Add an explicit exclusion to the system prompt.

Edit instruction:
{
  "file_path": "prompts/system_prompt.md",
  "operation": "replace_string",
  "old_string": "Do not include background statements, prior work references,
or methodological descriptions.",
  "new_string": "Do not include background statements, prior work references,
methodological descriptions, scanner parameters, participant counts,
or any statement that describes how the study was conducted rather than
what it found.",
  "new_content": null
}
```

**WORKED_EXAMPLE_B content (includes complete Python):**

```
EXAMPLE B — Adding a preprocessing step

Situation: The model is attending to methodology sentences when extracting
claims. The routing signal shows low scores (below 0.4) on abstracts with
long methodology sections. You want to filter the abstract to results-only
content before passing it to the extraction model.

Change: Create a preprocessor module and rewire extractor.py to use it.

Step 1 — Create playground/preprocessor.py:
{
  "file_path": "playground/preprocessor.py",
  "operation": "create_file",
  "old_string": null,
  "new_string": null,
  "new_content": "import re\n\nRESULTS_PATTERNS = [\n    r'\\b(showed?|demonstrated?|revealed?|found|observed)\\b',\n    r'\\b(significantly|greater than|less than|compared to)\\b',\n    r'\\b(activation|deactivation|correlation|increase|decrease)\\b',\n    r'\\b(p\\s*[<>=]\\s*0\\.\\d+|t\\s*\\(\\d+\\))\\b',\n]\n\nMETHODS_PATTERNS = [\n    r'\\b(participants?|subjects?|were recruited|were scanned)\\b',\n    r'\\b(fMRI|scanner|TR|voxel|mm|tesla)\\b',\n    r'\\b(we used|study examined|designed to)\\b',\n]\n\ndef score_sentence(sentence: str) -> str:\n    results_hits = sum(1 for p in RESULTS_PATTERNS if re.search(p, sentence, re.IGNORECASE))\n    methods_hits = sum(1 for p in METHODS_PATTERNS if re.search(p, sentence, re.IGNORECASE))\n    if results_hits >= methods_hits and results_hits > 0:\n        return 'RESULTS'\n    elif methods_hits > results_hits:\n        return 'METHODS'\n    return 'BACKGROUND'\n\ndef filter_to_results(abstract_text: str) -> str:\n    sentences = re.split(r'(?<=[.!?])\\s+', abstract_text)\n    results = [s for s in sentences if score_sentence(s) == 'RESULTS']\n    return ' '.join(results) if results else abstract_text\n"
}

Step 2 — Rewire extractor.py to use the preprocessor:
{
  "file_path": "playground/extractor.py",
  "operation": "replace_string",
  "old_string": "async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:",
  "new_string": "from playground.preprocessor import filter_to_results\n\nasync def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:\n    abstract_text = filter_to_results(abstract_text)",
  "new_content": null
}
```

**ROUTING_SIGNAL_EXPLANATION content:**

```
YOUR ROUTING SIGNAL

When the model reads an abstract to extract claims, it distributes attention
across the abstract's sentences. Some of that attention goes to sentences
reporting results — findings, activations, correlations. Some goes to
methodology sentences — how the study was run, what equipment was used.
Some goes to background — prior work, objectives, context.

The routing score measures what fraction of the model's attention goes to
results sentences when it is deciding what to extract. A score of 1.0 means
all attention is on results sentences. A score of 0.0 means none is.

A higher routing score does not guarantee better extractions. But a model
that attends primarily to results sentences when extracting claims is more
likely to extract actual findings than one attending to methodology or
background. The signal tells you about the model's processing, not its output.

You see this signal in two forms:
- ROUTING HISTORY: your routing scores across all prior iterations
- ROUTING DELTA: what your last modification did to routing scores

If your modifications are not moving routing scores, that is information.
It may mean the prompt is not the right lever for changing where the model
attends. It may be worth considering architectural changes instead.
```

---

### `protected/harness/study_002/agent_caller.py`

Extends Study 001's agent caller with routing signal injection and the two-call structure. Study 001's `agent_caller.py` is not modified — this is a new module in `study_002/`.

```python
async def invoke(
    prior_output: list[dict],
    prior_output_iteration: int,
    current_files: dict[str, str],
    objective: str,
    prior_episodes: list[dict],
    routing_history_text: str,       # formatted by routing_history.format_for_agent
    current_routing_scores: list[RoutingScore],  # pre-modification scores this iteration
) -> AgentResponse | AgentFailure

async def invoke_repair(
    error_message: str,
    current_files: dict[str, str],
    attempt_number: int,
) -> RepairResponse | AgentFailure
```

**Prompt structure for `invoke`:**

Sections assembled in order:

1. Task description (plain, no mechanism framing per DECISION 002-F)
2. `baseline_correction.CAPABILITY_FRAMING`
3. `baseline_correction.WORKED_EXAMPLE_A`
4. `baseline_correction.WORKED_EXAMPLE_B`
5. `baseline_correction.ROUTING_SIGNAL_EXPLANATION`
6. Episodic memory (all prior episodes, chronological)
7. `routing_history_text` (formatted routing history table + trend statement)
8. Current file contents (playground/ and prompts/, keyed by path)
9. Prior extraction output (abstract_id, abstract_text, predicted_claims for 25 abstracts)
10. Response schema

The baseline correction sections (2, 3, 4, 5) are included at every iteration, not just iteration 1. The agent needs the capability framing and worked examples available at every modification decision, not just as an onboarding artifact.

**What the agent does not receive:**
- F1 scores
- Ground truth claims
- Routing scores from the current iteration before its modification decision (it sees prior-iteration routing history, not current pre-modification scores — the pre-modification scores are computed but withheld until post-modification comparison)

Wait — this needs clarification. The routing signal the agent receives is: what routing looked like at the end of the prior iteration (post-modification scores from iteration N-1). Not the current pre-modification state. The current pre-modification scores are computed to establish the baseline for the post-modification delta, but the agent's signal is the historical record, not a real-time reading.

This preserves the consequence structure: the agent sees what its prior modification produced (the post-modification score from last time), decides on a new modification, and the harness then checks what that new modification did (post-modification score this time). The delta between these two is the consequence signal for next iteration.

---

### `protected/harness/study_002/study_runner.py`

Study 002 iteration loop. Adapts Study 001's runner with the two-call structure, routing analysis passes, and 25-abstract mini-corpus.

**Iteration structure:**

```
PRE-RUN CHECKS (same as Study 001 plus):
  - experiments/study_002/probe_set.json exists and is committed
  - AttentionAnalyzer can load model (attempt load at startup, fail fast)
  - transformers and torch available

ITERATION 0 (baseline):
  Run 25-abstract mini-corpus (10 control + 15 random) via llama.cpp
  Run attention analysis on 10 control abstracts via transformers
  Record routing scores as iteration 0 baseline
  Write artifacts + metrics
  Commit

ITERATIONS 1 through 20:

  a. Load prior extraction output (most recent scanned iteration)
     Load all prior episodes
     Load routing history via routing_history.load_all()
     Format routing history via routing_history.format_for_agent()

  b. Call agent_caller.invoke() with routing history text
     If AgentFailure → log anomaly, write metrics, continue

  c. Apply edits via edit_applier
     If failed → log anomaly, write metrics, continue

  d. Repair loop (3 attempts, identical to Study 001)

  e. POST-MODIFICATION ATTENTION PASS:
     Run transformers forward pass on 10 control abstracts
     Compute post-modification routing scores
     Compute delta vs. last iteration's post-modification scores
     This delta is the consequence record for this iteration
     (NOT shown to agent this iteration — shown next iteration
      as part of routing history)

  f. Snapshot playground + prompts

  g. Run 25-abstract mini-corpus (10 control + 15 random) via llama.cpp
     asyncio.wait_for with ITERATION_TIMEOUT_S timeout
     If timeout/failure → rollback, log anomaly, continue

  h. Score 25 abstracts vs ground truth (hidden from agent)
     Record mini-corpus scores in metrics (not shown to agent)

  i. Persist episode via episode_store.append()

  j. Append routing_history entry (pre + post scores for this iteration)

  k. Write metrics row (extended schema with routing fields)

  l. Append model_performance snapshot

  m. Commit iteration
```

**On routing score timing:**

The routing history the agent sees at iteration N contains:
- Post-modification scores from iterations 0 through N-1
- The trend computed from those scores

The pre-modification scores computed at step (e) are recorded in `routing_history.jsonl` as `pre_scores` for that iteration but are not surfaced in the agent's routing history until iteration N+1 — at which point they appear as the prior iteration's post-modification consequence.

This maintains the correct consequence structure: the agent sees what its actions produced, not what the current state is before it acts.

---

### Extended `artifact_writer.py`

Adds routing fields to the metrics row. The Study 001 metrics schema is extended, not replaced:

```python
def append_metrics_study_002(
    iteration_n: int,
    study_id: str,
    metrics: dict,
    routing_pre: float | None,
    routing_post: float | None,
    routing_delta: float | None,
    control_improved: int,
    control_declined: int,
    code_changes_attempted: bool,
) -> None
```

New fields added to metrics row:
- `pre_routing_score` — aggregate routing score before this iteration's modification
- `post_routing_score` — aggregate routing score after modification
- `routing_delta` — post minus pre
- `routing_direction` — "positive" if delta > 0.02, "negative" if delta < -0.02, "neutral" otherwise
- `control_abstracts_improved` — count of control abstracts with positive routing delta
- `control_abstracts_declined` — count with negative routing delta
- `code_changes_attempted` — boolean, true if any edit targeted a `.py` file in `playground/`

---

### Extended `anomaly_logger.py`

New anomaly types for Study 002 (added without modifying Study 001 anomaly handling):

- `routing_score_null` — routing score could not be computed for an abstract
- `no_results_sentences` — segmenter found no RESULTS sentences in an abstract
- `attention_weights_missing` — forward hook returned no weights for one or more layers
- `transformers_load_failure` — AttentionAnalyzer failed to load model at startup
- `gqa_shape_mismatch` — attention weight tensor shape unexpected (GQA handling issue)

---

### Full 200-Abstract Validation Runner

A standalone script run once at study end:

```
python scripts/study_002_validate.py
```

Loads the final playground state (current `playground/` and `prompts/` after iteration 20), runs the full 200-abstract corpus via llama.cpp, scores against ground truth, writes results to `experiments/study_002/validation_run.json`. The F1 from this run is the study's performance comparison point against Study 001.

This script is not part of the iteration harness. It runs once, manually, after iteration 20 is committed.

---

## Pre-Registration Alignment Check

| This spec | Pre-registration decision |
|---|---|
| Probe set selection before harness code | DECISION 002-A |
| Intrinsic cost signal only, other components baseline | DECISION 002-B |
| 25 per iteration, 200 validation at end | DECISION 002-C |
| Attention hooks, last 6 layers, routing fidelity formula | DECISION 002-D |
| Two-call structure with routing pass | DECISION 002-E |
| No RDE mechanism, plain task description | DECISION 002-F |
| Both playground and prompts mutable | DECISION 002-G |
| Qwen 3.6 27B, dual backend | DECISION 002-H |
| N=20 unconditional | DECISION 002-I |
| Basic episodic memory from Study 001 | DECISION 002-J |
| code_changes_attempted field | DECISION 002-K |
| routing_history.jsonl schema | DECISION 002-L |

---

## Tasks

- Run `scripts/select_probe_set.py` — select 10 random control abstracts, commit `experiments/study_002/probe_set.json` before any other implementation task
- Implement `protected/attention/segmenter.py` with rule-based sentence classification and token alignment
- Implement `protected/attention/analyzer.py` with transformers model loading, hook registration, and `forward_pass()` — verify attention weight shapes against actual Qwen 3.6 27B architecture at load time and log them
- Implement `protected/attention/scorer.py` with routing fidelity computation, edge case handling, and `RoutingScore` dataclass
- Write and commit `protected/harness/study_002/baseline_correction.py` with all four committed text sections — `CAPABILITY_FRAMING`, `WORKED_EXAMPLE_A`, `WORKED_EXAMPLE_B`, `ROUTING_SIGNAL_EXPLANATION` — verify `WORKED_EXAMPLE_B` Python code runs correctly before committing
- Implement `protected/harness/study_002/routing_history.py` with `append`, `load_all`, and `format_for_agent` including trend statement logic
- Implement `protected/harness/study_002/agent_caller.py` with full prompt assembly and correct routing signal timing (agent sees prior-iteration post-modification scores, not current pre-modification scores)
- Refactor Study 001 shared modules into `protected/harness/shared/` — no behavior changes, import paths updated in Study 001 runner
- Implement `protected/harness/study_002/study_runner.py` with full iteration loop including pre/post modification attention passes and 25-abstract mini-corpus
- Extend `artifact_writer.py` with `append_metrics_study_002` (additive — Study 001 function unchanged)
- Extend `anomaly_logger.py` with Study 002 anomaly types (additive)
- Implement `scripts/study_002_validate.py` standalone validation runner
- **Attention hook verification test:** Run `analyzer.forward_pass` on one control abstract, print attention weight shapes, confirm non-null weights are captured from at least 4 of the 6 target layers. If GQA produces unexpected shapes, fix hook before proceeding.
- **Routing score sanity test:** Run full routing score pipeline on 3 control abstracts manually. Inspect `results_attention_fraction` vs `methods_attention_fraction`. Confirm scores are in [0, 1] and that a synthetic abstract with only results sentences scores above 0.7 while a synthetic abstract with only methods sentences scores below 0.3.
- **Routing signal timing test:** Run iteration 1, confirm the routing history injected into the agent call contains only iteration 0's post-modification scores and not the current pre-modification scores.
- **Baseline correction test:** Confirm `baseline_correction.compose_baseline_correction()` output contains all four sections and that `WORKED_EXAMPLE_B`'s Python code block parses without syntax errors.
- **code_changes_attempted test:** Submit a test edit targeting `playground/preprocessor.py` (create_file), confirm `code_changes_attempted=true` in metrics row.
- **Two-iteration end-to-end test:** Run iterations 0 and 1 against study_002 corpus. Confirm `routing_history.jsonl` has 2 entries, `episodes.jsonl` has 1 entry (iteration 1), metrics has 2 rows, and the iteration 1 commit bundles all artifacts.

---

## Acceptance Criteria

- `experiments/study_002/probe_set.json` committed before any harness implementation commits (verifiable in git log by commit timestamp)
- `AttentionAnalyzer` loads Qwen 3.6 27B, registers hooks on last 6 layers, and captures non-null attention weights on forward pass — verified by attention hook verification test
- Routing score pipeline produces values in [0, 1] with correct directional behavior — verified by routing score sanity test
- `baseline_correction.py` contains all four committed text sections; `WORKED_EXAMPLE_B` Python is syntactically valid and runnable
- Routing history timing is correct: agent at iteration N sees post-modification scores from iteration N-1, not current pre-modification scores — verified by routing signal timing test
- `code_changes_attempted` field is populated correctly in metrics rows
- Harness executes iterations 0 and 1 end-to-end: routing scores computed, agent called with correct prompt structure, edits applied, mini-corpus run, episode persisted, commit made
- Study 001 harness is unmodified and still runs correctly after shared module refactor
- No remote git operations; experiment branch has no upstream
- `scripts/study_002_validate.py` exists and runs without error on final playground state

---

## What This Sprint Deliberately Omits

- Executing the full 20-iteration run (separate sprint)
- Analysis and writeup (Study 002 Sprint 005 equivalent)
- Study 003, 004, or 005 infrastructure
- RDE mechanism harness sections (Study 004)
- World model / delta engine (Study 003)
- Enhanced memory structures (Study 005)
- Web monitoring UI
- Multi-worker inference