# Sprint 003 — Self-Healing Study Harness Implementation

**Organization:** Idris Applied AI Research
**Status:** Implemented (Sprint 003 validation running)
**Date:** May 2026
**Author:** Muzaffer Ozen
**Depends on:** Sprint 002 (pre-registration committed before this sprint begins)

**Post-Sprint Divergence Log:** All deviations from the original spec are
recorded at the bottom of this document under "Divergence Log". Each entry
is tagged with the commit that introduced it and the rationale.

---

## Summary

Builds the harness that executes the self-healing detection study committed
to in Sprint 002's pre-registration. The harness is the chokepoint: every
byte the agent changes flows through validated, allowlisted writes. The agent
never touches disk directly, never sees its own scores, and never has access
to protected infrastructure.

This sprint produces code only — no study execution. The 20-iteration run
is Sprint 004. The acceptance bar for Sprint 003 is: harness executes two
iterations end-to-end, the repair loop is verified by deliberate violation
tests, episodic memory plumbing works correctly across both iterations, and
the baseline (iteration 0) is captured.

---

## Architectural Context

This is the **playground + protected split** described in Sprint 002's
pre-registration. The agent is not a coding agent with filesystem access —
it receives a structured prompt containing its prior extraction output and
episodic memory, responds with a JSON object containing a new episode and
edit instructions, and the harness validates and applies the edits while
enforcing the architectural boundary.

The key distinction from a prompt-only harness: the agent can create, modify,
or delete Python files inside `playground/`. The mutable surface is larger,
the enforcement boundary is the `protected/` directory and the fixed entry
point contract, not a small list of `.md` files. Allowing real architectural
change is the study's research bet — the harness exists to make that change
observable and controlled.

---

## Full Repository Layout (Post-Sprint 003)

```
idris-neuro-extract/
  protected/
    schema.py                     # Claim, ExtractionResult (Sprint 001, immutable)
    interface.py                  # entry point contract declaration (immutable)
    scorer.py                     # evaluation logic (Sprint 001, immutable)
    harness/
      __init__.py
      study_runner.py             # entry point: N-iteration loop
      agent_caller.py             # LLM invocation with episodic memory
      edit_protocol.py            # response schema, Edit/Episode/AgentResponse dataclasses
      edit_applier.py             # applies validated edits to disk
      allowlist.py                # allowlist definition and enforcement
      interface_validator.py      # playground module reload + entry point validation
      episode_store.py            # episode persistence and retrieval
      artifact_writer.py          # per-iteration artifacts
      anomaly_logger.py           # structured anomaly events
      git_ops.py                  # local-only git operations
      corpus_runner.py            # runs full corpus through playground entry point
      model_performance.py        # cumulative token/timing/wall-clock metrics
  playground/                     # agent owns this entirely
    __init__.py
    extractor.py                  # initial naive extractor (Sprint 001 logic, rewired)
  prompts/                        # agent can modify
    system_prompt.md              # from Sprint 001
    examples.md                   # from Sprint 001 (empty)
  corpus/                         # frozen — agent cannot touch
    abstracts/
    ground_truth.jsonl
    corpus_manifest.md
    annotation_prompt.md
    review_notes.md
  evaluation/                     # frozen — agent cannot touch
    results/
  experiments/
    study_001/
      pre-registration.md
      iterations/
      episodes.jsonl
      agent-rationale.jsonl
      anomalies.jsonl
      metrics.jsonl
      model_performance.jsonl   # cumulative performance snapshot per iteration
      .perf_start_ts            # wall-clock start timestamp (internal)
  scripts/                        # frozen — agent cannot touch
  requirements.txt
  README.md
```

---

## Critical Invariant: Module Reload Per Iteration

**This is the most important implementation detail in the harness.**

The agent writes Python files to `playground/`. The harness executes them
in-process. Python caches imported modules in `sys.modules`. Without explicit
cache invalidation between iterations, the corpus run silently uses stale
code — the edits the agent just applied are ignored.

Before every corpus run, `interface_validator.py` must purge all
`playground.*` entries from `sys.modules` before importing fresh:

```python
def reload_playground():
    to_remove = [key for key in sys.modules if key.startswith("playground")]
    for key in to_remove:
        del sys.modules[key]
```

This is called both during interface validation and immediately before the
corpus run. A future "optimization" that skips the purge would silently run
the prior iteration's code — the test suite would still pass (it constructs
fresh instances), but the study would be measuring a loop that never
actually mutates. Treat this purge as an invariant, not an implementation
detail.

The module reload test (acceptance criteria) exists specifically to catch
this failure mode.

---

## Component Specifications

---

### `protected/interface.py`

Declares the entry point contract. Never modified after Sprint 003.

```python
# The harness calls this function on every abstract, every iteration.
# The agent must expose this signature in playground/extractor.py.
# Signature: async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult

ENTRY_POINT_MODULE = "playground.extractor"
ENTRY_POINT_FUNCTION = "extract"
ENTRY_POINT_PARAMS = 2          # positional args: abstract_id, abstract_text
ITERATION_TIMEOUT_S = int(os.environ.get("STUDY_ITERATION_TIMEOUT_S", 14400))
# Default raised to 4 hours to accommodate ~125 t/s generation speed for 200 abstracts.
# Original 300s spec value retained as DECISION 002-G code spec; override is for local
# hardware speed only (see Divergence Log entry D01).
```

---

### `playground/extractor.py` (initial state)

The initial playground extractor rewires the Sprint 001 extraction logic
under the harness entry point contract. It is behavior-identical to Sprint
001's `extractor/extractor.py` — same prompt loading, same provider call,
same output schema. The agent will overwrite this.

```python
from protected.schema import Claim, ExtractionResult
from extractor.provider import LlamaCppProvider
import json
from pathlib import Path

_provider = LlamaCppProvider()

async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:
    prompts_dir = Path(__file__).parent.parent / "prompts"
    system_prompt = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")
    examples = (prompts_dir / "examples.md").read_text(encoding="utf-8").strip()
    if examples:
        system_prompt = system_prompt + "\n\n" + examples
    raw = _provider.complete(system_prompt, abstract_text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"claims": []}
    claims = [Claim(claim_text=c) for c in data.get("claims", [])]
    return ExtractionResult(abstract_id=abstract_id, claims=claims)
```

**Note:** The initial extractor uses `encoding="utf-8"` on all prompt file reads
and wraps `json.loads()` in a try/except to gracefully handle malformed LLM
responses. These hardening measures were added during implementation (see
Divergence Log entries D02 and D03).

**Note:** The initial extractor loads prompts at call time, not at module
import time, because the module is reloaded fresh each iteration — import-time
loading would be equivalent here. The agent may restructure this however it
chooses.

---

### `protected/harness/edit_protocol.py`

Defines the agent response schema and dataclasses.

**Agent response schema** (validated at parse layer, not API level):

```json
{
  "episode": {
    "observation": "string",
    "hypothesis": "string",
    "action": "string",
    "expectation": "string"
  },
  "rationale": "string",
  "edits": [
    {
      "file_path": "string",
      "operation": "replace_string | replace_file | create_file | delete_file",
      "old_string": "string | null",
      "new_string": "string | null",
      "new_content": "string | null"
    }
  ]
}
```

**Repair response schema** (no episode required — repair calls only):

```json
{
  "edits": [...]
}
```

**Operations:**

- `replace_string` — `old_string` must match exactly once in the file.
  `new_string` is the replacement. `new_content` must be null.
- `replace_file` — `new_content` is the full new file contents. `old_string`
  and `new_string` must be null. File must already exist.
- `create_file` — `new_content` is the full file contents. File must not
  already exist. `old_string` and `new_string` must be null. Path must be
  under `playground/` and end in `.py`.
- `delete_file` — Removes the file. All three content fields must be null.
  Path must be under `playground/`. Cannot delete `playground/__init__.py`
  or `playground/extractor.py` (core files protected by allowlist).

**Dataclasses:**

```python
@dataclass
class Edit:
    file_path: str
    operation: str  # replace_string | replace_file | create_file | delete_file
    old_string: str | None
    new_string: str | None
    new_content: str | None

@dataclass
class Episode:
    observation: str
    hypothesis: str
    action: str
    expectation: str

@dataclass
class AgentResponse:
    episode: Episode
    rationale: str
    edits: list[Edit]

@dataclass
class RepairResponse:
    edits: list[Edit]

@dataclass
class AgentFailure:
    reason: str
    raw_response: str | None = None
```

`iteration_n` is not in the schema. The harness adds it on persistence.
Empty edits list is valid — represents the agent choosing not to change
anything this iteration. The episode is still required and persisted.

---

### `protected/harness/allowlist.py`

```python
# Files the agent can write. Closed by default — anything not listed is denied.
ALLOWED_FILE_EXACT = [
    "prompts/system_prompt.md",
    "prompts/examples.md",
    "playground/extractor.py",    # core file — replace_file and replace_string only
    "playground/__init__.py",     # core file — replace_file and replace_string only
]

ALLOWED_DIR_PREFIX = "playground/"   # agent may create_file here (*.py only)

# These can never be written regardless of other rules.
EXCLUDED_PREFIXES = [
    "protected/",
    "corpus/",
    "experiments/",
    "evaluation/",
    "scripts/",
]

CORE_FILES = [
    "playground/extractor.py",
    "playground/__init__.py",
]

def is_allowed(file_path: str, operation: str) -> bool:
    """
    Returns True if the operation on file_path is permitted.

    Rules (evaluated in order):
    1. If path matches any EXCLUDED_PREFIXES → deny.
    2. If operation is create_file:
       - Path must be under ALLOWED_DIR_PREFIX
       - Path must end in .py
       - File must not already exist on disk
       → allow if all three hold, else deny.
    3. If operation is delete_file:
       - Path must be under ALLOWED_DIR_PREFIX
       - Path must not be in CORE_FILES
       → allow if both hold, else deny.
    4. If path is in ALLOWED_FILE_EXACT → allow.
    5. If path is under ALLOWED_DIR_PREFIX and ends in .py → allow.
    6. Default → deny.
    """
```

**Why `playground/extractor.py` is in ALLOWED_FILE_EXACT but not deletable:**
The entry point contract requires `playground/extractor.py` to exist and
expose `extract()`. If the agent deletes it, the interface validation fails
immediately on the next iteration. The delete prohibition on core files
prevents the agent from accidentally destroying the entry point and consuming
a repair attempt on a trivially avoidable error.

---

### `protected/harness/edit_applier.py`

```python
def apply_edits(edits: list[Edit]) -> ApplyResult
```

**Validation pass (all edits validated before any write):**

1. For each edit, call `allowlist.is_allowed(file_path, operation)`. If any
   edit is denied → return `ApplyResult(applied=False, reason="allowlist_violation",
   offending_path=path)`. Apply nothing.
2. For `replace_string`: verify `old_string` is not null, `new_content` is
   null, and `old_string` matches exactly once in the current file. Zero or
   multiple matches → `ApplyResult(applied=False, reason="ambiguous_match")`.
3. For `replace_file`: verify `new_content` is not null and non-empty.
   `old_string`/`new_string` must be null.
4. For `create_file`: verify file does not already exist on disk.
5. For `delete_file`: verify file exists on disk.

**Write pass (only if all validations pass):**

Edits are applied in order. Each write is atomic: write to a `.tmp` file,
then `os.replace()`. Deletes use `os.unlink()`.

Returns `ApplyResult(applied=True, files_changed=[...])`.

**All-or-nothing is a hard invariant.** Partial application leaves the
playground in an inconsistent state that the module reload will faithfully
reflect — a partially-applied iteration is worse than a rolled-back one
because it may import successfully but behave incorrectly without logging
a failure. If any edit in the batch is invalid, apply nothing.

---

### `protected/harness/interface_validator.py`

```python
async def validate_interface() -> ValidationResult
```

**Steps:**

1. Call `reload_playground()` — purge all `playground.*` from `sys.modules`.
2. Attempt `import playground.extractor`. On `ImportError`, `SyntaxError`,
   or any exception → return `ValidationResult(valid=False, error=str(e))`.
3. Retrieve `playground.extractor.extract`. If not present or not callable
   → `ValidationResult(valid=False, error="extract not found or not callable")`.
4. Verify it is an `asyncio.iscoroutinefunction` → if not, fail.
5. Verify it accepts at least 2 positional parameters → inspect signature.
6. Return `ValidationResult(valid=True)`.

```python
def reload_playground() -> None:
    """
    Purges playground.* from sys.modules.
    Must be called before every import of playground code.
    Skipping this causes the harness to silently run stale code.
    """
    to_remove = [key for key in sys.modules if key.startswith("playground")]
    for key in to_remove:
        del sys.modules[key]
```

`reload_playground()` is also called by `corpus_runner.py` immediately
before the corpus run — redundant with the validation call, but explicit.
Belt and suspenders: the reload must happen before execution, and it costs
nothing to call twice.

---

### `protected/harness/corpus_runner.py`

```python
async def run_corpus(study_id: str) -> CorpusRunResult
```

**Behavior:**

1. Call `reload_playground()`.
2. Import `playground.extractor.extract`.
3. Load all 200 abstracts from `corpus/abstracts/`. Each abstract file is read
   with `encoding="utf-8", errors="replace"` to handle non-UTF-8 bytes from
   the NeuroSynth corpus (see Divergence Log entry D02).
4. For each abstract, call `extract(abstract_id, abstract_text)`.
   Per-abstract failures are caught and logged as `CorpusAbstractFailure`
   events — they do not halt the corpus run. A failed abstract contributes
   zero claims to the output.
5. After each successful extraction, print progress to terminal with cumulative
   elapsed time: `Corpus: {idx}/200 done (abstract {id}) ({elapsed}s)`. This
   provides visibility without impacting performance (see Divergence Log entry D06).
6. Return `CorpusRunResult(results=list[ExtractionResult], failures=list[CorpusAbstractFailure], duration_seconds=float, corpus_token_usage=CorpusTokenUsage)`.

`CorpusTokenUsage` aggregates token data across all 200 abstract calls:

```python
@dataclass
class CorpusTokenUsage:
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_tokens_per_abstract: float      # total / abstracts attempted
    avg_tokens_per_second: float        # aggregate throughput
```

Token data is collected from `complete_with_usage` on the provider inside
the playground's `extract()` call — but the playground is agent-owned code
and may not use `complete_with_usage`. To avoid requiring the agent to
instrument its own calls, corpus token tracking uses a lightweight provider
wrapper: `corpus_runner.py` wraps the `LlamaCppProvider` singleton with a
counting proxy before each corpus run that intercepts `complete()` calls and
accumulates usage from `complete_with_usage` transparently. The playground
code calls `complete()` as normal; the proxy captures the usage data without
the agent knowing. The proxy is reset at the start of each corpus run.

**Timeout is applied by `study_runner.py`** via `asyncio.wait_for`, not
inside `corpus_runner.py`. The runner does not know about the timeout — it
runs to completion or is cancelled by the outer wait_for. On cancellation,
the `asyncio.TimeoutError` propagates to `study_runner.py`. Note: the
`asyncio.shield()` that was originally used to protect the corpus run from
cancellation was removed (see Divergence Log entry D05) because it prevented
the timeout from reaching the event loop.

**Scoring happens in `study_runner.py`** after the corpus run returns, using
`protected/scorer.py`. The score is never passed to the agent — it goes
directly to metrics.

**What the agent receives from the corpus run:** the raw per-abstract
extraction output — `abstract_id`, `abstract_text`, and `predicted_claims`
for each abstract. No scores, no ground truth, no comparison. This is the
signal the agent reasons from.

---

### `protected/harness/agent_caller.py`

**Provider token surface prerequisite.** The llama.cpp server returns token
counts and timing in every response: `usage.prompt_tokens`,
`usage.completion_tokens`, and `timings.predicted_per_second`. The Sprint 001
`LlamaCppProvider.complete()` currently discards this data. Before the harness
is built, extend the provider with an additive method that returns both the
completion text and a `TokenUsage` object:

```python
@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float        # timings.predicted_per_second from llama.cpp
    context_window: int             # declared at provider init via env var
```

Keep `complete()` unchanged for existing callers. Add
`complete_with_usage(system_prompt, user_message) -> tuple[str, TokenUsage]`
alongside it. `agent_caller` uses `complete_with_usage`; the corpus extractor
uses `complete` (per-abstract token tracking is aggregated at the
`CorpusRunResult` level, not per-call, to avoid instrumentation overhead in
the playground — see `corpus_runner.py`).

**HTTP timeout.** The `openai.OpenAI()` client is initialized with a timeout
defaulting to 600s (configurable via `LLM_HTTP_TIMEOUT_S` env var). This
prevents the provider from hanging indefinitely if the llama.cpp server
becomes unresponsive.

**Async threading.** Both `invoke` and `invoke_repair` wrap the provider's
synchronous `complete_with_usage()` call in `asyncio.to_thread()`. This is
required because `complete_with_usage()` makes a blocking HTTP call that would
freeze the event loop, preventing `asyncio.wait_for` timeouts from being
delivered. Without this, the 4-hour iteration timeout would never fire.

```python
async def invoke(
    prior_output: list[dict],          # per-abstract: abstract_id, abstract_text, predicted_claims
    prior_output_iteration: int,       # which iteration produced this output
    current_files: dict[str, str],     # playground/ and prompts/ files, keyed by path
    objective: str,
    prior_episodes: list[dict],
) -> AgentResponse | AgentFailure

async def invoke_repair(
    error_message: str,
    current_files: dict[str, str],     # current playground/ state (post-failed-edit)
    attempt_number: int,               # 1, 2, or 3
) -> RepairResponse | AgentFailure
```

Both `invoke` and `invoke_repair` capture `TokenUsage` from
`complete_with_usage` and attach it to the response dataclass. Token usage
from repair calls is accumulated separately and reported in metrics as
`repair_prompt_tokens` / `repair_completion_tokens` (summed across all repair
attempts this iteration).

**`invoke` behavior:**

Constructs a prompt containing, in order:

1. System framing: the agent is an autonomous AI researcher modifying a
   scientific claim extractor to improve its precision and recall. It has
   access to a Python playground and a set of prompt files. It must respond
   with a JSON object matching the response schema exactly.
2. The objective verbatim (from pre-registration).
3. Episodic memory: all prior episodes in chronological order as a JSON array.
   If empty (iteration 1), communicated explicitly: "This is your first
   iteration; you have no prior episodes."
4. Current file contents: all files under `playground/` and `prompts/`,
   keyed by relative path. These are the exact bytes the agent is editing.
5. Prior extraction output: the full per-abstract list (abstract_id,
   abstract_text, predicted_claims) from the most recent successful iteration.
   The agent is told which iteration this is from.
6. Response schema: the exact JSON structure required, with field descriptions.

Calls `LlamaCppProvider.complete()` with
`response_format={"type": "json_object"}`. Parses response. The parser first
strips any markdown code fences (````json ... ````) that the LLM may wrap
around its JSON output, then extracts the outermost JSON object using regex.
On any failure: return `AgentFailure`.

**`invoke_repair` behavior:**

Constructs a shorter prompt:

1. System framing: the agent's previous edits produced a Python error. It
   must propose repair edits to fix the broken playground code.
2. The error message verbatim (import error, syntax error, etc.).
3. Current playground file contents (post-failed-edit state — the agent sees
   exactly what it produced and what broke).
4. Repair response schema (edits only, no episode).
5. Attempt number and remaining attempts.

**Context window management:** At N=20 with 200 abstracts, the full prior
output could be approximately 50,000 tokens. Total prompt including episodes,
current files, and output is estimated at 80,000–120,000 tokens — well within
Qwen 3.6 27B's 262k context. No windowing is needed for v1. If context
limits are hit at runtime, the anomaly is logged as `context_limit_exceeded`
and the iteration fails cleanly.

---

### `protected/harness/episode_store.py`

```python
def append(study_id: str, iteration_n: int, episode: Episode) -> None
def load_all(study_id: str) -> list[dict]
def count(study_id: str) -> int
```

Line format in `episodes.jsonl`:

```json
{"iteration_n": 3, "observation": "...", "hypothesis": "...", "action": "...", "expectation": "..."}
```

`load_all` returns episodes in `iteration_n` ascending order. The ordering
is canonical — it is the episodic memory the agent sees. If episodes are
missing (because iterations failed), the agent's memory has gaps. This is
expected and acceptable behavior, not a bug.

---

### `protected/harness/git_ops.py`

```python
def ensure_branch(branch_name: str) -> None
def verify_no_remote_push(branch_name: str) -> None
def reset_partial_iteration() -> None       # git reset --hard HEAD
def rollback_playground() -> None           # git checkout -- playground/ prompts/
def commit_iteration(iteration_n: int, study_id: str, rationale: str) -> str
def last_committed_iteration(study_id: str) -> int   # -1 if none
```

`rollback_playground()` is distinct from `reset_partial_iteration()`.
`reset_partial_iteration()` is called at startup to discard any uncommitted
state left by a crash — it resets the entire working tree. `rollback_playground()`
is called mid-run when an iteration fails after edits have been applied — it
restores only `playground/` and `prompts/` to HEAD, leaving the ledger files
(`metrics.jsonl`, `anomalies.jsonl`, etc.) intact so the failure is recorded.

`verify_no_remote_push` inspects `.git/config` and refuses if the experiment
branch has any configured remote upstream. The harness also never calls
`git push`, `git fetch`, or `git pull` anywhere.

Commit message format:

```
[study_001] iteration NN

Rationale: <first 500 chars of agent rationale>

Files changed: <list>

Playground files at this commit are the exact state used for this
iteration's corpus run. See experiments/study_001/pre-registration.md.
```

The commit bundles playground edits + prompt edits + all artifacts and
ledger appends for this iteration. A commit is one complete, self-contained
iteration. `reset_partial_iteration()` at startup discards anything
uncommitted, which can only be a partial iteration.

---

### `protected/harness/artifact_writer.py`

```python
def write_iteration_artifacts(iteration_n: int, study_id: str,
                               corpus_result: CorpusRunResult) -> None
def snapshot_playground(iteration_n: int, study_id: str) -> None
def append_metrics(iteration_n: int, study_id: str, metrics: dict) -> None
def append_rationale(iteration_n: int, study_id: str, rationale: str) -> None
```

Filename conventions:

- Extraction output JSON: `experiments/study_001/iterations/iteration_{NN}_{ISO8601}.json`
- Playground snapshot: `experiments/study_001/iterations/iteration_{NN}_playground/`
  (copies of all files under `playground/` and `prompts/` at the time of
  the snapshot — taken after edits are applied and validation passes, so
  the snapshot matches exactly what the corpus run used)
- Metrics: append to `experiments/study_001/metrics.jsonl`
- Rationale: append to `experiments/study_001/agent-rationale.jsonl`

Metrics row shape:

```json
{
  "iteration_n": 3,
  "timestamp": "ISO8601",
  "macro_precision": 0.0,
  "macro_recall": 0.0,
  "macro_f1": 0.0,
  "micro_tp": 0,
  "micro_fp": 0,
  "micro_fn": 0,
  "avg_claims_per_abstract": 0.0,
  "scan_duration_seconds": 0.0,
  "agent_edits_proposed": 0,
  "agent_edits_applied": 0,
  "repair_attempts": 0,
  "playground_files_changed": [],
  "prompt_chars_delta": 0,
  "anomaly": false,
  "episode_persisted": false,
  "scanned": false,
  "agent_prompt_tokens": 0,
  "agent_completion_tokens": 0,
  "agent_total_tokens": 0,
  "agent_tokens_per_second": 0.0,
  "agent_context_window": 0,
  "corpus_total_prompt_tokens": 0,
  "corpus_total_completion_tokens": 0,
  "corpus_avg_tokens_per_abstract": 0.0,
  "corpus_avg_tokens_per_second": 0.0,
  "repair_prompt_tokens": 0,
  "repair_completion_tokens": 0
}
```

**Null rules:**
- `scanned=false` rows: all score fields, `scan_duration_seconds`, and all
  `corpus_*` token fields are `null`. Agent token fields are still populated
  if the agent call succeeded before the failure.
- Iteration 0: all agent and repair token fields are `null` (no agent call).
  Corpus token fields are populated from the baseline run.
- `repair_prompt_tokens` / `repair_completion_tokens`: `null` if
  `repair_attempts=0`, summed across all repair calls otherwise.

---

### `protected/harness/anomaly_logger.py`

```python
def log_anomaly(study_id: str, iteration_n: int,
                anomaly_type: str, details: dict) -> None
```

Anomaly types:

- `allowlist_violation` — edit targeted a protected path
- `ambiguous_match` — `old_string` matched zero or multiple times
- `empty_file_replacement` — `replace_file` with empty content
- `create_file_exists` — `create_file` targeting an existing path
- `delete_file_missing` — `delete_file` targeting a non-existent path
- `agent_response_malformed` — response failed schema validation
- `agent_api_failure` — provider call raised
- `interface_validation_failed` — playground failed import/signature check
  (one entry per attempt, includes error message)
- `repair_exhausted` — 3 repair attempts failed; playground rolled back
- `iteration_timeout` — corpus run exceeded timeout; playground rolled back
- `corpus_abstract_failure` — individual abstract raised during extraction
  (one entry per failed abstract, does not halt the run)
- `scan_failure` — corpus run raised for a non-timeout reason
- `episode_discarded` — iteration failed before episode could be persisted
- `empty_edits` — agent returned empty edits array (informational, not flagged
  as anomaly in metrics boolean)
- `context_limit_exceeded` — prompt exceeded model context window
- `no_prior_output` — iteration could not load prior iteration output
  (added during implementation, see D04)
- `repair_agent_failure` — repair loop agent call returned AgentFailure
  (added during implementation, see D04)

---

### `protected/harness/model_performance.py`

Cumulative performance tracking. Appended once per iteration after metrics are
written. Pure file I/O — zero model impact.

```python
def append_after_iteration(study_id: str, iteration_n: int) -> dict
def summarize(study_id: str) -> None
```

Wall-clock timer: at study startup, the current timestamp is written to
`experiments/{study_id}/.perf_start_ts`. Each subsequent `append_after_iteration`
call reads this file to compute wall-clock elapsed time.

Metrics row appended to `model_performance.jsonl`:

```json
{
  "iteration_n": 3,
  "timestamp": "ISO8601",
  "iterations_completed": 4,
  "iterations_successful": 3,
  "iterations_anomalous": 1,
  "cumulative_total_tokens": 1234567,
  "cumulative_prompt_tokens": 987654,
  "cumulative_completion_tokens": 246913,
  "cumulative_agent_prompt_tokens": 10000,
  "cumulative_agent_completion_tokens": 5000,
  "cumulative_repair_prompt_tokens": 2000,
  "cumulative_repair_completion_tokens": 1000,
  "cumulative_scan_duration_seconds": 3600.5,
  "wall_clock_seconds": 7200.0,
  "avg_tokens_per_second": 171.3,
  "avg_corpus_tokens_per_second": 274.4,
  "avg_abstracts_per_second": 0.028,
  "total_abstracts_processed": 800,
  "total_repair_attempts": 2,
  "peak_macro_f1": 0.4521,
  "cumulative_micro_tp": 500,
  "cumulative_micro_fp": 100,
  "cumulative_micro_fn": 50,
  "cumulative_micro_precision": 0.8333,
  "cumulative_micro_recall": 0.9091
}
```

`summarize()` prints a table at study completion:

```
Iter  Wall(s)     Tokens       TPS  Corpus TPS   Abst/s   Scan(s)    Abst      F1    OK   Err
--------------------------------------------------------------------------------
   0    1800.0   500,000     277.8       277.8     0.11   1800.0     200  0.3100    1    0
   1    3600.0   900,000     250.0       250.0     0.11   3600.0     400  0.3800    2    0
```

All visibility is post-iteration file I/O + print. Zero model impact.


### `protected/harness/study_runner.py`

Entry point. Drives the N-iteration loop.

```python
def run_study(study_id: str = "study_001", n_iterations: int = 20) -> None
```

Sync wrapper calling `asyncio.run(_run_study_async(...))`.

**Iteration loop behavior:**

**Pre-run checks (any failure → refuse to run, exit non-zero):**

1. `experiments/{study_id}/pre-registration.md` exists and is committed.
2. `corpus/ground_truth.jsonl` exists and is committed.
3. `corpus/corpus_manifest.md` exists with a `commit_sha` field.
4. Every file in `ALLOWED_FILE_EXACT` exists on disk.
5. Experiment branch exists locally with no remote upstream.
6. `LLM_PROVIDER` env var is set and `LLAMA_CPP_BASE_URL` is reachable.
7. If `metrics.jsonl` has 21 entries (iteration 0 + 20) → raise
   `StudyAlreadyComplete`.
8. `protected/harness/` is unmodified relative to HEAD on the experiment
   branch (the harness checks its own integrity).
9. Consistency check: `episodes.jsonl` episode count matches the count of
   `episode_persisted=true` rows in `metrics.jsonl`.

**On startup:** call `git_ops.reset_partial_iteration()` to discard any
uncommitted working-tree state, then resume at
`last_committed_iteration + 1`.

**Iteration 0 (baseline):**
If not committed: run corpus (no agent call, no episode), snapshot playground,
write artifacts + metrics (`scanned=true`, all agent fields zero,
`episode_persisted=false`), commit. No agent invocation.

The baseline corpus run is wrapped in try/except for both `asyncio.TimeoutError`
and general `Exception`. On timeout or failure, logs anomaly, writes metrics
with `anomaly=true` and `scanned=false`, and continues to iteration 1
(see Divergence Log entry D04).

**Iterations 1 through N:**

```
a. Load prior output — most recent iteration where scanned=true.
   Load all prior episodes via episode_store.load_all().
   Tell the agent which iteration the output is from.

b. Call agent_caller.invoke() → AgentResponse | AgentFailure.
   If AgentFailure → log anomaly, write metrics (scanned=false), continue.

c. Call edit_applier.apply_edits() → ApplyResult.
   If not applied → log anomaly, write metrics (scanned=false), continue.
   (No rollback needed — nothing was written.)

d. REPAIR LOOP (up to 3 total attempts including the original):
   attempt = 1
   while attempt <= 3:
     result = await interface_validator.validate_interface()
     if result.valid: break
     log_anomaly("interface_validation_failed", {error, attempt})
     if attempt == 3:
       rollback_playground()
       log_anomaly("repair_exhausted", {})
       log_anomaly("episode_discarded", {})
       write metrics (scanned=false, repair_attempts=3)
       continue to next iteration
     repair = await agent_caller.invoke_repair(result.error, current_files, attempt)
     if isinstance(repair, AgentFailure):
       log anomaly, rollback_playground(), write metrics, continue
     apply_result = edit_applier.apply_edits(repair.edits)
     if not apply_result.applied:
       log anomaly, rollback_playground(), write metrics, continue
     attempt += 1

e. Snapshot playground + prompts (post-edit, post-validation state).

f. Run corpus:
   try:
     corpus_result = await asyncio.wait_for(
       corpus_runner.run_corpus(study_id),
       timeout=ITERATION_TIMEOUT_S
     )
   except asyncio.TimeoutError:
     rollback_playground()
     log_anomaly("iteration_timeout")
     log_anomaly("episode_discarded")
     write metrics (scanned=false)
     continue
   except Exception as e:
     rollback_playground()
     log_anomaly("scan_failure", {error: str(e)})
     log_anomaly("episode_discarded")
     write metrics (scanned=false)
     continue

g. Score corpus_result against ground_truth (hidden from agent).

h. Persist episode via episode_store.append().

i. Write artifacts: extraction output JSON, metrics (scanned=true), rationale.

j. Append model_performance cumulative snapshot.

k. Commit iteration (edits + snapshot + artifacts + ledger appends).

**Post-study summary:** After all N iterations complete, `model_performance.summarize()`
prints a cumulative performance table to terminal.
```

**No hard stops.** The only exceptions that halt the run are the pre-run
state checks. Anomalies are logged, never raised mid-run.

---

## Pre-Registration Alignment Check

Every decision in this spec maps to a committed pre-registration decision.
Implementers must not deviate without acknowledging the misalignment:

| This spec | Pre-registration |
|---|---|
| Playground/protected split | DECISION 002-B |
| Fixed `extract()` entry point | DECISION 002-C |
| Protected zone off limits | DECISION 002-D |
| Final-state-only episodes | DECISION 002-E |
| 3-attempt repair loop | DECISION 002-F |
| In-process timeout (4h default) | DECISION 002-G (default raised, see D01) |
| Qwen 3.6 27B, llama.cpp | DECISION 002-H |
| N=20 unconditional | DECISION 002-I |
| Hidden metrics | DECISION 002-D |

---

## Tasks

- Extend `extractor/provider.py` with `TokenUsage` dataclass and additive
  `complete_with_usage()` method surfacing `usage.prompt_tokens`,
  `usage.completion_tokens`, and `timings.predicted_per_second` from the
  llama.cpp response; keep `complete()` unchanged
- Implement counting proxy in `corpus_runner.py` that wraps `LlamaCppProvider`
  to intercept `complete()` calls and accumulate `CorpusTokenUsage` transparently
- Implement `protected/interface.py` with entry point constants
- Implement `protected/harness/edit_protocol.py` with schema,
  `Edit`/`Episode`/`AgentResponse`/`RepairResponse`/`AgentFailure` dataclasses
- Implement `protected/harness/allowlist.py` with `is_allowed` enforcing all
  four operation types
- Implement `protected/harness/edit_applier.py` with all-or-nothing validation
  and atomic writes for all four operations
- Implement `protected/harness/interface_validator.py` with `reload_playground()`
  and full signature validation
- Implement `protected/harness/episode_store.py` with append, load_all, count
- Implement `protected/harness/corpus_runner.py` with per-abstract failure
  isolation (failures logged, run continues)
- Implement `protected/harness/agent_caller.py` with `invoke` and
  `invoke_repair`, full episodic memory injection, prior output injection
- Implement `protected/harness/git_ops.py` with `rollback_playground` distinct
  from `reset_partial_iteration`, remote push prohibition, structured commits
- Implement `protected/harness/anomaly_logger.py` with all declared anomaly types
- Implement `protected/harness/artifact_writer.py` with playground snapshots
  and all metrics fields
- Implement `protected/harness/study_runner.py` with full iteration loop
  including repair loop, timeout handling, and all pre-run checks
- Rewire `playground/extractor.py` to match the entry point contract
- **Module reload test:** write a sentinel string into `playground/extractor.py`
  via `edit_applier.apply_edits`, call `reload_playground()`, import
  `playground.extractor`, assert the sentinel is present in the imported
  module's source — confirms the cache invalidation path works end-to-end
- **Allowlist violation test:** attempt to write to `protected/schema.py`,
  `corpus/ground_truth.jsonl`, and `experiments/study_001/pre-registration.md`
  — confirm all three are rejected with `allowlist_violation`, no write occurs
- **Core file deletion test:** attempt `delete_file` on
  `playground/extractor.py` — confirm rejection
- **Repair loop test:** inject a syntax error into `playground/extractor.py`,
  run one iteration, confirm the repair loop fires, confirm 3-attempt
  exhaustion rolls back correctly and logs `repair_exhausted`
- **Episodic memory plumbing test:** run iterations 1 and 2 end-to-end,
  verify iteration 2's agent call received iteration 1's episode as input
- Capture baseline (iteration 0) for `study_001`

---

## Acceptance Criteria

- All harness modules exist with specified surfaces
- `reload_playground()` is verified end-to-end: a sentinel written to
  `playground/extractor.py` is present in a freshly-imported module, not
  the prior cached version
- `allowlist.is_allowed` correctly handles all four operations; protected
  paths are denied; core file deletion is denied
- `edit_applier.apply_edits` is all-or-nothing: a batch with one invalid
  edit applies nothing
- Repair loop fires on interface validation failure; exhaustion after 3
  attempts rolls back playground and does not persist episode
- Corpus run timeout rolls back playground and does not persist episode
- Harness executes iterations 0, 1, and 2 end-to-end against `study_001`'s
  corpus
- `metrics.jsonl` has 3 entries (iterations 0, 1, 2)
- `episodes.jsonl` has 2 entries (iterations 1 and 2), each schema-valid
- Iteration 2's agent call received iteration 1's episode (verified via
  logged trace or test harness)
- No remote git operations performed; experiment branch has no upstream
- Playground snapshot for each completed iteration exists in
  `experiments/study_001/iterations/`
- Token metrics are populated in every metrics row: agent call fields on
  iterations 1+, corpus fields on all scanned iterations, repair fields on
  iterations where repairs occurred; all null rules apply correctly
- Scores are never surfaced to the agent (verified by code review: no score
  field appears in agent_caller.invoke() parameters)

---

## What This Sprint Deliberately Omits

- Executing the full 20-iteration run (Sprint 004)
- Analysis, calibration scoring, or architectural coherence review (Sprint 005)
- Multi-rater tooling (contingent on Sprint 005)
- Web UI or monitoring dashboard
- Retry logic on LLM API failures — one failure is one anomaly, move on
- Memory windowing — all episodes passed every iteration; N=20 fits in context
- Subprocess isolation — in-process with timeout is the declared approach

---

## Divergence Log

Every deviation from the original spec. Each entry records the commit, the
change, the rationale, and whether it affects the pre-registration contract.

### D01: Raised iteration timeout default to 4 hours (14400s)
**Commit:** acddbfa
**Spec impact:** `protected/interface.py` default changed from 300 to 14400.
**Rationale:** At ~125 t/s generation speed with ~1500 tokens per abstract,
200 abstracts take approximately 2.4 hours. The 300s default would cause
every iteration to time out. The env var override (`STUDY_ITERATION_TIMEOUT_S`)
remains available to restore 300s.
**Pre-reg impact:** None. DECISION 002-G specifies "300s timeout" as the code
spec. The runtime override is for local hardware speed only and does not
affect the study's methodology or results.

### D02: UTF-8 encoding hardening across all file I/O
**Commit:** 408d56e, 9d879c7, 48a9f0a
**Spec impact:** All `read_text()` calls now use `encoding="utf-8"` explicitly.
Abstract file reads use `errors="replace"` to handle non-UTF-8 bytes from the
NeuroSynth corpus (file `14749289.json` contains byte 0xef at position 2065).
**Rationale:** The original spec assumed clean UTF-8. The NeuroSynth corpus
contains at least one file with invalid continuation bytes. Without
`errors="replace"`, the corpus run crashes at abstract 173/200 with
`UnicodeDecodeError`.
**Pre-reg impact:** None. This is a robustness fix, not a methodology change.

### D03: JSON parsing hardening
**Commit:** 998a14f
**Spec impact:** `playground/extractor.py`, `study_runner.py`, `episode_store.py`,
`git_ops.py` all wrap `json.loads()` in try/except. Malformed lines are skipped
or produce empty output rather than crashing.
**Rationale:** LLM responses may contain markdown fences, trailing whitespace,
or corrupted JSON. Without defensive parsing, a single malformed response
crashes the entire iteration.
**Pre-reg impact:** None. Matches the existing spec intent for `agent_caller.py`
to return `AgentFailure` on parse errors; this extends the pattern to other
modules that read JSON artifacts.

### D04: Baseline error handling
**Commit:** 998a14f
**Spec impact:** `_run_baseline` now wraps the corpus run in try/except for
both `asyncio.TimeoutError` and general `Exception`. On failure, logs anomaly,
writes metrics with `anomaly=true` and `scanned=false`, and returns without
crashing. New anomaly types: `no_prior_output`, `repair_agent_failure`.
**Rationale:** The original spec described iteration-level error handling but
omitted baseline error handling. Without it, a baseline timeout crashes the
entire study before any iterations run.
**Pre-reg impact:** None. Graceful failure is consistent with the "no hard
stops" invariant of the study runner.

### D05: Removed asyncio.shield from corpus_runner
**Commit:** 9fb9f47
**Spec impact:** `asyncio.shield()` removed from corpus runner loop.
**Rationale:** The shield was preventing `asyncio.wait_for` timeout cancellations
from reaching the event loop, making the timeout ineffective. The corpus run
now runs bare, allowing the timeout to fire properly.
**Pre-reg impact:** None. Restores the intended timeout behavior.

### D06: Terminal visibility for corpus progress
**Commit:** e6ad691, fb18b68, 998a14f, bea2a6d
**Spec impact:** `corpus_runner.py` prints per-abstract progress with cumulative
elapsed time: `Corpus: {idx}/200 done (abstract {id}) ({elapsed}s)`.
`study_runner.py` prints phase-level progress for each step of the iteration
loop. `model_performance.py` prints a summary table at study completion.
**Rationale:** The original spec had no visibility requirements. With 200
abstracts per corpus run taking 2-3 hours each, there was no way to tell
if the run was progressing or stuck. All visibility is `print()` only —
zero model impact.
**Pre-reg impact:** None. Purely observational.

### D07: model_performance module
**Commit:** 5eef54e, bea2a6d
**Spec impact:** New module `model_performance.py` with cumulative token,
timing, and wall-clock metrics. Output file: `model_performance.jsonl`.
**Rationale:** The metrics.jsonl tracks per-iteration data but requires manual
aggregation to see cumulative trends. This module provides running totals
of tokens, wall-clock time, throughput, and peak performance.
**Pre-reg impact:** None. Purely observational, no effect on agent behavior.

### D08: async agent_caller with asyncio.to_thread
**Commit:** 48a9f0a
**Spec impact:** `agent_caller.invoke()` and `agent_caller.invoke_repair()`
wrap provider calls in `asyncio.to_thread()`.
**Rationale:** The llama.cpp provider makes synchronous HTTP calls. Without
`asyncio.to_thread()`, the event loop is blocked and `asyncio.wait_for`
timeouts cannot fire. This is required for the timeout to work correctly.
**Pre-reg impact:** None. Restores the intended async behavior.

### D09: HTTP timeout on provider
**Commit:** 48a9f0a
**Spec impact:** `openai.OpenAI()` initialized with `timeout=600s` (configurable
via `LLM_HTTP_TIMEOUT_S`).
**Rationale:** Without an HTTP timeout, a stalled llama.cpp server would block
the provider call indefinitely, defeating the iteration-level timeout.
**Pre-reg impact:** None. Defensive measure.

### D10: Markdown fence stripping in agent_caller
**Commit:** 48a9f0a
**Spec impact:** `_parse_response()` in `agent_caller.py` strips markdown fences
and extracts the outermost JSON object using regex before parsing.
**Rationale:** The LLM occasionally wraps JSON responses in markdown code fences
even when `response_format={"type": "json_object"}` is set. Without stripping,
`json.loads()` fails with `JSONDecodeError`.
**Pre-reg impact:** None. Robustness fix for LLM output formatting.

### D11: Git ops hardening
**Commit:** 48a9f0a, 998a14f
**Spec impact:** `git_ops.py` checks for empty commit history before `git reset`.
Uses `check=False` on `git checkout --` for paths that may not exist.
Sanitizes newlines in commit message rationale.
**Rationale:** On a fresh repository with no commits, `git reset --hard HEAD`
fails with "fatal: ambiguous argument 'HEAD'". Similarly, `git checkout --`
fails if the target paths don't exist yet.
**Pre-reg impact:** None. Defensive measure for edge cases.

### D12: Deprecated asyncio API fix
**Commit:** 48a9f0a
**Spec impact:** `interface_validator.py` uses `asyncio.get_running_loop().run_in_executor()`
instead of deprecated `asyncio.get_event_loop()`.
**Rationale:** Python 3.13 deprecated `asyncio.get_event_loop()` when no
event loop is running. The fix uses the modern API.
**Pre-reg impact:** None. Compatibility fix.