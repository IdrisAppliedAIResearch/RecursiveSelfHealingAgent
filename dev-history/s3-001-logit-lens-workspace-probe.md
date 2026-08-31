# Logit Lens Workspace Probe — Verification Script

**Organization:** Idris Applied AI Research  
**Purpose:** Feasibility probe, not a study. Determine whether Qwen 3.6 27B
surfaces result-relevant concepts in a logit-lens-readable "workspace" at
middle-band layers, before committing to a study design around this signal.  
**Status:** Exploratory — run once, inspect output, no pre-registration required  
**Date:** June 2026

---

## Context

Anthropic's "Verbalizable Representations Form a Global Workspace in Language
Models" (Transformer Circuits, 2026) identifies a privileged set of
representations — a workspace — that a model can report on and reason with,
living in a middle band of layers (roughly 38-92% depth). The paper notes the
logit lens captures much of this structure at low cost: apply the unembedding
matrix directly to the intermediate residual stream and read which tokens the
model has "verbalizable" at that position.

This probe tests one narrow question: **when Qwen 3.6 27B processes a
neuroscience abstract and is about to extract claims, do result-relevant
concepts appear in the logit-lens readout at middle-band layers?**

If yes, this could become an alternative intrinsic cost signal — workspace
grounding — to compare against attention routing fidelity in a future study.
If no, we learn that Qwen's workspace is not cleanly logit-lens-readable for
this task and we do not pursue it.

This is a feasibility check. It does not modify any study. It does not touch
the running Study 002B. Run it in a separate scratch script.

---

## What the Logit Lens Does

The logit lens takes a hidden state from an intermediate layer — the residual
stream at some layer L and token position P — and applies the model's final
unembedding matrix (the same matrix that converts the final layer's hidden
state to output logits) directly to that intermediate state. The result is a
distribution over the vocabulary showing which tokens the model is "leaning
toward" at that intermediate point in its processing.

At early layers this readout is noisy or reflects input tokens. At middle
layers, per the workspace paper, it reflects concepts the model has loaded and
can verbalize. At final layers it reflects the actual next-token prediction.

We are looking for the middle band where result-concepts appear.

---

## The Probe

### Setup

Reuse the existing `AttentionAnalyzer` model loading from
`protected/attention/analyzer.py`. The model is already loaded via transformers
with 4-bit quantization. We do NOT need attention hooks for this probe — we need
hidden states, which are simpler and cheaper.

```python
import torch

def load_for_logit_lens(model_path: str):
    """
    Reuse the same 4-bit load as the attention analyzer.
    We need output_hidden_states, not output_attentions.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant,
        attn_implementation="eager",   # not strictly needed here but keep consistent
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
```

### Step 1 — Capture hidden states across all layers

```python
def get_hidden_states(model, tokenizer, system_prompt: str, abstract_text: str):
    """
    Single forward pass, capture hidden states at all layers.
    Returns hidden_states tuple and the input token info.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": abstract_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # per Amendment 002
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=4096).to(model.device)

    with torch.no_grad():
        out = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    # out.hidden_states is a tuple of (num_layers + 1) tensors
    # each: [1, seq_len, hidden_dim]
    # index 0 is the embedding output, index i is after layer i
    return out.hidden_states, inputs["input_ids"]
```

### Step 2 — Apply the logit lens at the generation position

The position we care about is the last token — the assistant-turn-start position
where the model is about to begin generating its answer. This is the moment its
workspace should contain what it is about to extract.

```python
def logit_lens_at_position(
    model,
    tokenizer,
    hidden_states,
    layer_idx: int,
    position: int = -1,
    top_k: int = 30,
):
    """
    Apply the unembedding matrix to the hidden state at (layer_idx, position).
    Return the top_k tokens the model has 'verbalizable' at that point.
    """
    # Get the residual stream at this layer and position
    h = hidden_states[layer_idx][0, position, :]   # [hidden_dim]

    # Apply the model's final norm then unembedding (lm_head)
    # Qwen applies a final RMSNorm before the lm_head — replicate that
    normed = model.model.norm(h)
    logits = model.lm_head(normed)   # [vocab_size]

    # Top-k tokens
    top_vals, top_ids = torch.topk(logits, top_k)
    tokens = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
    return list(zip(tokens, top_vals.tolist()))
```

**Note on the final norm:** Qwen applies a final RMSNorm (`model.model.norm`)
before `lm_head`. The logit lens must replicate this to produce a faithful
readout — applying `lm_head` to a raw intermediate hidden state without the
norm produces distorted results. If `model.model.norm` is not the correct
attribute path for this Qwen version, inspect the model structure with
`print(model)` and find the final norm layer applied before `lm_head`.

### Step 3 — Sweep the middle-band layers

```python
def sweep_layers(model, tokenizer, system_prompt, abstract_text):
    """
    Run the logit lens at multiple depths and print the top tokens at each.
    We are looking for the layer band where result-concepts appear.
    """
    hidden_states, input_ids = get_hidden_states(
        model, tokenizer, system_prompt, abstract_text
    )
    num_layers = len(hidden_states) - 1   # subtract embedding layer

    # Sample layers at 20%, 40%, 50%, 60%, 70%, 80%, 90% depth
    depth_fractions = [0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    print(f"Model has {num_layers} layers")
    print(f"Abstract (first 200 chars): {abstract_text[:200]}...\n")

    for frac in depth_fractions:
        layer_idx = int(num_layers * frac)
        top_tokens = logit_lens_at_position(
            model, tokenizer, hidden_states, layer_idx, position=-1, top_k=30
        )
        token_strs = [t.strip() for t, _ in top_tokens]
        print(f"--- Layer {layer_idx} ({int(frac*100)}% depth) ---")
        print(", ".join(token_strs))
        print()
```

### Step 4 — Score result-concept presence

```python
# Concept word lists — what we're hoping to see in the workspace
RESULT_CONCEPT_WORDS = {
    "activation", "activated", "increase", "increased", "decrease", "decreased",
    "correlation", "correlated", "significant", "greater", "reduced", "region",
    "cortex", "gyrus", "hippocampus", "amygdala", "prefrontal", "response",
    "bilateral", "showed", "found", "revealed", "difference", "effect",
}

METHOD_CONCEPT_WORDS = {
    "scanner", "fMRI", "MRI", "participants", "subjects", "voxel", "protocol",
    "acquired", "recruited", "session", "tesla", "sequence", "analysis",
    "measured", "recorded", "task", "paradigm", "design",
}

def score_workspace_grounding(model, tokenizer, system_prompt, abstract_text,
                               layer_idx):
    """
    At a given layer, count how many of the top-30 logit-lens tokens are
    result-concepts vs method-concepts. Returns a grounding ratio.
    """
    hidden_states, _ = get_hidden_states(
        model, tokenizer, system_prompt, abstract_text
    )
    top_tokens = logit_lens_at_position(
        model, tokenizer, hidden_states, layer_idx, position=-1, top_k=30
    )
    token_strs = [t.strip().lower() for t, _ in top_tokens]

    result_hits = sum(1 for t in token_strs if t in RESULT_CONCEPT_WORDS)
    method_hits = sum(1 for t in token_strs if t in METHOD_CONCEPT_WORDS)

    return {
        "layer": layer_idx,
        "result_concept_count": result_hits,
        "method_concept_count": method_hits,
        "top_tokens": token_strs,
    }
```

### Step 5 — Run the probe

```python
def run_probe(model_path):
    model, tokenizer = load_for_logit_lens(model_path)

    system_prompt = open("prompts/system_prompt.md").read()

    # Use 3 abstracts from the Study 002 probe set
    import json
    probe = json.load(open("experiments/study_002/probe_set.json"))
    abstract_ids = probe["abstract_ids"][:3]

    for aid in abstract_ids:
        abstract = json.load(open(f"corpus/abstracts/{aid}.json"))
        text = abstract["text"] if "text" in abstract else abstract.get("abstract", "")

        print("=" * 70)
        print(f"ABSTRACT {aid}")
        print("=" * 70)
        sweep_layers(model, tokenizer, system_prompt, text)

    # Also run on the two synthetic controls
    print("=" * 70)
    print("SYNTHETIC RESULTS-ONLY")
    print("=" * 70)
    results_only = (
        "Bilateral hippocampal activation increased significantly during "
        "encoding compared to baseline. Left prefrontal cortex showed greater "
        "activation for novel than repeated stimuli (p < 0.001)."
    )
    sweep_layers(model, tokenizer, system_prompt, results_only)

    print("=" * 70)
    print("SYNTHETIC METHODS-ONLY")
    print("=" * 70)
    methods_only = (
        "Fifteen healthy volunteers were recruited. fMRI was performed on a "
        "3T scanner with TR=2000ms. Voxel size was 3x3x3mm."
    )
    sweep_layers(model, tokenizer, system_prompt, methods_only)


if __name__ == "__main__":
    import sys
    run_probe(sys.argv[1])   # pass model path as argument
```

---

## What to Look For in the Output

**The probe succeeds (workspace signal is viable) if:**

1. At some middle-band layer (likely 50-75% depth), the logit-lens top tokens
   for a results-heavy abstract include recognizable result-concept words —
   region names, activation/correlation terms, directional words. The model
   has these concepts "loaded" before it generates.

2. The synthetic results-only abstract shows MORE result-concepts in its
   workspace readout than the synthetic methods-only abstract, at the same
   layer. This is the directional check — the workspace reflects the content
   type.

3. There is a readable band — early layers show input echoes or noise, middle
   layers show concepts, final layers show output-prediction tokens (JSON
   structure like `{`, `"claims"`, `[`). This layer structure matching the
   workspace paper's description is strong confirmation the phenomenon
   generalizes to Qwen.

**The probe fails (do not pursue this signal) if:**

1. The logit-lens readout is noise at every layer — no recognizable concept
   words, just fragments, punctuation, or high-frequency tokens.

2. Results-only and methods-only abstracts produce indistinguishable workspace
   readouts — the workspace doesn't reflect content type.

3. There is no readable middle band — the readout jumps from noise directly to
   output-prediction tokens with no concept-bearing layers between.

---

## Interpreting a Mixed Result

If some abstracts show clean workspace signal and others don't, that is
informative but not disqualifying. It may mean the signal works for
results-heavy abstracts but degrades on ambiguous ones. Note which abstracts
worked and inspect whether there is a pattern (abstract length, results density,
etc.). A partially working signal may still be worth pursuing with refinement.

---

## What This Probe Does NOT Do

- It does not run the full Jacobian lens (too expensive for 32GB VRAM).
- It does not modify Study 002B.
- It does not constitute a pre-registered study.
- It does not commit to using this signal — it only tests feasibility.

If the probe succeeds, the next step is a proper pre-registered study design
comparing workspace grounding against attention routing fidelity as alternative
instantiations of the intrinsic cost component. That design is a separate
document written only after this probe produces positive results.

---

## Dependencies

Same as the attention analyzer — no new dependencies. Uses `output_hidden_states`
instead of `output_attentions`, both native to transformers.

---

## Time Estimate

Five abstracts (3 probe + 2 synthetic), 7 layers each, one forward pass per
abstract. On the RTX 5090 this is a few minutes total. Run it in a scratch
script, inspect the printed output manually, and decide whether the signal
is worth pursuing.