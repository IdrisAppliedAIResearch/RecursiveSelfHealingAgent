# Study 002 — Attention Analyzer Amendment 002
## Disable Thinking Mode in Forward Pass

**Organization:** Idris Applied AI Research  
**Study ID:** study_002  
**Date:** June 2026  
**Author:** Muzaffer Ozen  
**Status:** Pre-Registration Amendment  
**Amends:** Study 002 Harness Implementation Spec, `protected/attention/analyzer.py`

---

## Problem Statement

Routing scores computed across iterations 0-4 were consistently 0.07-0.11 regardless
of prompt or architectural changes made by the agent. Investigation of
`attention_scores_0.json` revealed that approximately 90% of attention at the last
token position was going to unclassified tokens rather than classified abstract
content.

The root cause: with `add_generation_prompt=True` in the Qwen3 chat template,
the last token in the sequence is `<think>` — the opening of Qwen3's thinking
block. This token is trained to attend heavily to chat structural tokens
(`<|im_start|>`, `user`, `<|im_end|>`, `<|im_start|>assistant`) rather than
abstract content. These structural tokens fall within the abstract token range
(after `abstract_start_token_idx`) but are not assigned to any sentence
classification, causing them to absorb 90% of the abstract attention budget while
carrying no routing signal.

The routing score is technically computed correctly but measures the wrong thing:
attention from the `<think>` position to structural tokens, not attention from a
content-generating position to results sentences. This signal is insensitive to
prompt or architectural changes because the structural token attention pattern is
fixed by Qwen3's training, not by the extraction prompt.

**Evidence:** Routing scores 0.066-0.114 across all 10 control abstracts across
all 4 iterations, with zero meaningful variation despite the agent making real
prompt and code changes.

---

## Fix: Disable Thinking Mode in Attention Analysis Pass

Qwen3 thinking mode is controlled by the chat template. When `enable_thinking=True`
(default), the template appends `<|im_start|>assistant\n<think>\n` and the last
token is `<think>`. When `enable_thinking=False`, the template appends a normal
assistant turn start and the last token is a content-generation token that attends
to abstract content rather than structural markers.

The fix is to pass `enable_thinking=False` to `apply_chat_template` inside
`build_input` when called from the attention analyzer. The extraction model (llama.cpp
via the playground) is unaffected — this change applies only to the transformers
forward pass used for routing score computation.

---

## Changes Required

### `protected/attention/analyzer.py` — `build_input` function

**Change:** Pass `enable_thinking=False` to `apply_chat_template`.

**Current code:**
```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

**Replace with:**
```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

Apply the same change to the `system_only` template call for consistency:

**Current code:**
```python
system_only = tokenizer.apply_chat_template(
    [{"role": "system", "content": system_prompt}],
    tokenize=False,
    add_generation_prompt=False,
)
```

**Replace with:**
```python
system_only = tokenizer.apply_chat_template(
    [{"role": "system", "content": system_prompt}],
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False,
)
```

**Why both calls:** The `system_only` length is used to compute
`abstract_start_token_idx`. Both calls must use identical template parameters
or the offset computation will be wrong.

---

### `protected/attention/analyzer.py` — `verify_attention_pipeline` function

Add a last-token identity check to the verification test. This confirms the fix
is working before the study runs:

Add after `input_dict = build_input(...)`:

```python
# Verify last token is NOT <think>
last_token_id = input_dict["input_ids"][0, -1].item()
last_token_str = tokenizer.decode([last_token_id])
print(f"Last token: '{last_token_str}' (id={last_token_id})")
if "<think>" in last_token_str or "think" in last_token_str.lower():
    raise AssertionError(
        f"Thinking mode is still active — last token is '{last_token_str}'. "
        f"Pass enable_thinking=False to apply_chat_template."
    )
print("Confirmed: thinking mode disabled, last token is content-generation token.")
```

This check must pass before the study proceeds. If it raises, the template
parameter is not being respected and the routing scores will remain uninformative.

---

### `protected/attention/analyzer.py` — `run_prefill` function

No changes required. The forward pass itself is correct. The only change needed
is in how the input sequence is constructed.

---

## Verification Protocol

After implementing the fix, run the full `verify_attention_pipeline` function
from the implementation brief with the two synthetic abstracts. The pass criteria
are:

1. **Last token check passes** — `<think>` is not the last token.

2. **Fraction sum is meaningful** — Results + methods + background fractions should
   sum closer to 0.5-0.9 (not 0.10 as before). The remaining fraction reflects
   genuine attention to unclassified tokens (sentence boundaries, punctuation,
   template delimiters that are now a small minority of the range).

3. **Directional sanity check passes:**
   - Synthetic results-only abstract scores above 0.5
   - Synthetic methods-only abstract scores below 0.3
   - Results score exceeds methods score

4. **Control abstract scores are meaningfully higher** — Rerun on the 10 control
   abstracts from `experiments/study_002/probe_set.json`. Scores should now be
   in the range 0.3-0.7 rather than 0.07-0.11. Exact values are not
   pre-specified — what matters is that they are meaningfully above the previous
   floor and show variation across abstracts.

If all four checks pass, the routing signal is repaired and the study can restart.

---

## Study Restart Protocol

After the fix is verified:

1. Clear all iteration artifacts from the stopped run:
   - Delete `experiments/study_002/iterations/iteration_*` files
   - Clear `experiments/study_002/episodes.jsonl` (truncate to empty)
   - Clear `experiments/study_002/agent-rationale.jsonl`
   - Clear `experiments/study_002/anomalies.jsonl`
   - Clear `experiments/study_002/metrics.jsonl`
   - Clear `experiments/study_002/assessments.jsonl`
   - Clear `experiments/study_002/routing_history.jsonl`
   - Delete `experiments/study_002/attention_scores_*.json`

2. Reset `playground/` to the original baseline state:
   - Delete `playground/preprocessor.py` and `playground/validator.py`
   - Restore `playground/extractor.py` to the original single-call naive extractor
   - Restore `prompts/system_prompt.md` to the original naive prompt from Sprint 001
   - Restore `prompts/examples.md` to empty

3. Commit the clean reset state before running iteration 0.

4. Run the study from iteration 0.

---

## Baseline Prompt Issue

The stopped run revealed a second problem: the baseline extractor (iteration 0)
produced F1=0.000 because `prompts/system_prompt.md` contained Study 001's
final over-tightened prompt rather than the original naive prompt.

The restart protocol above includes restoring the original naive prompt.
The original text is:

```
You are a scientific claim extractor. Given a neuroscience abstract, extract
all scientific claims the abstract explicitly makes.

A scientific claim is a declarative sentence asserting a specific, testable
finding that the abstract supports. Do not include background statements,
prior work references, or methodological descriptions.

Respond with a JSON object in this exact format:
{"claims": ["claim one", "claim two"]}

If no claims are present, return: {"claims": []}
```

And `prompts/examples.md` should be empty at iteration 0.

This ensures the baseline F1 reflects the naive extractor's natural performance,
not the output of Study 001's modification history.

---

## Call 1 Thinking Mode Issue

A separate but related issue: Call 1 diagnostic assessments were also failing
because Qwen3 thinking mode was producing extended chain-of-thought that
consumed the token budget before the JSON output completed. Three of four
Call 1 assessments failed with `assessment_malformed`.

**Fix for Call 1:** In `agent_caller.py`, add `/no_think` to the end of the
Call 1 user message, or pass `enable_thinking=False` in the tokenizer call if
using the transformers backend. For the llama.cpp provider, append the following
to the Call 1 user message:

```
/no_think
```

Qwen3 honors this directive in the user message and suppresses the thinking block,
producing direct JSON output. This is the recommended approach for structured
output calls where the thinking block adds latency without value.

Alternatively, increase Call 1 max tokens from 1024 to 4096 to give the thinking
block room to complete before the JSON. This is less clean but avoids modifying
the prompt structure.

The recommended fix is `/no_think` appended to the Call 1 user message since
Call 1 is a structured data extraction task, not a reasoning task.

---

## Amendment Record

| Amendment | Date | Description |
|---|---|---|
| 001 | June 2026 | Two-phase agent caller (diagnostic + decision split) |
| 002 | June 2026 | Disable thinking mode in attention forward pass + Call 1 |

---

## Files Changed by This Amendment

| File | Change |
|---|---|
| `protected/attention/analyzer.py` | Add `enable_thinking=False` to both `apply_chat_template` calls in `build_input`. Add last-token identity check to `verify_attention_pipeline`. |
| `protected/harness/study_002/agent_caller.py` | Append `/no_think` to Call 1 user message. |
| `experiments/study_002/` | Clear all iteration artifacts per restart protocol. |
| `playground/` | Reset to original baseline state. |
| `prompts/` | Reset to original naive prompt and empty examples. |

No changes to `edit_protocol.py`, `edit_applier.py`, `allowlist.py`,
`interface_validator.py`, `episode_store.py`, `routing_history.py`,
`git_ops.py`, `baseline_correction.py`, `scorer.py`, or `corpus_runner.py`.