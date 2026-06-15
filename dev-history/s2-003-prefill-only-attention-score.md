# Implementation Brief: Prefill-Only Attention Routing Score

**For:** Coding agent  
**Study:** Idris Applied AI Research — Study 002  
**Component:** `protected/attention/analyzer.py`  
**Date:** June 2026

---

## What You Are Building

A function that runs a single forward pass through Qwen 3.6 27B via transformers (no token generation), captures attention weights from the last 6 layers at the final token position, maps those weights to sentence types in the abstract, and returns a routing fidelity score between 0 and 1.

**Why prefill-only:** Generation requires storing a growing KV cache across every generated token, which exhausts VRAM. A prefill-only pass computes one forward pass on the input sequence and stops. Attention weights are available at the last input token position — which is exactly the position the model uses to decide what to extract. No generation needed.

---

## Step 1 — Model Loading

This is the most critical step. Two requirements that are non-negotiable.

**Requirement 1: `attn_implementation="eager"`**

Qwen3 by default uses Flash Attention or PyTorch SDPA (scaled dot product attention). Both of these fused kernels do not return attention weight tensors — they discard them for memory efficiency. You cannot call `output_attentions=True` and get anything back unless you force the eager (unfused) attention implementation.

**Requirement 2: 4-bit quantization via bitsandbytes**

At 4-bit quantization, Qwen3 27B weights occupy approximately 14GB. That leaves ~18GB of headroom on a 32GB card for activations, attention tensors, and overhead. Without quantization the weights alone exceed 32GB.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_attention_model(model_path: str):
    """
    Load Qwen3 27B base model for attention analysis.
    model_path: local path to the base model (HuggingFace format, not GGUF).
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        attn_implementation="eager",   # REQUIRED — flash attn discards weights
        device_map="auto",             # places layers on GPU automatically
        torch_dtype=torch.float16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer
```

**Load once at study startup. Never reload between iterations or between abstracts.** Loading a 27B model takes 2-4 minutes. The model stays resident in VRAM for the entire study run.

---

## Step 2 — Input Construction

The routing score measures what the model attends to in the abstract when it is about to generate output. The input to the forward pass must match what the extraction model sees during actual extraction — same system prompt, same abstract text, same chat template formatting.

```python
def build_input(
    tokenizer,
    system_prompt: str,
    abstract_text: str,
    device: str = "cuda",
) -> dict:
    """
    Build tokenized input matching the extraction model's chat format.
    Returns dict with input_ids, attention_mask, and abstract_start_token_idx.
    abstract_start_token_idx tells the scorer where the abstract begins
    in the full token sequence.
    """
    # Build the chat-formatted prompt exactly as the extractor does
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": abstract_text},
    ]

    # Apply the model's chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # adds the assistant turn start token
    )

    # Tokenize system prompt alone to find where abstract starts
    system_only = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}],
        tokenize=False,
        add_generation_prompt=False,
    )
    system_tokens = tokenizer(system_only, return_tensors="pt")
    system_len = system_tokens["input_ids"].shape[1]

    # Tokenize full prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,  # cap to prevent OOM on very long abstracts
    ).to(device)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "abstract_start_token_idx": system_len,
        "total_seq_len": inputs["input_ids"].shape[1],
    }
```

**Why `add_generation_prompt=True`:** This appends the assistant turn start token. The last token in the sequence is then the position where the model would begin generating — exactly the position whose attention distribution you want. The last token is the model's "ready to generate" state.

---

## Step 3 — Forward Pass

```python
def run_prefill(model, input_dict: dict) -> tuple:
    """
    Single forward pass. No generation. Returns attention weights from last 6 layers.
    Returns: tuple of 6 tensors, each shape [1, num_heads, seq_len, seq_len]
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_dict["input_ids"],
            attention_mask=input_dict["attention_mask"],
            output_attentions=True,   # REQUIRED — returns attention weight tensors
            use_cache=False,          # REQUIRED — no KV cache, saves memory
            return_dict=True,
        )

    # outputs.attentions is a tuple of length num_layers
    # each element: [batch=1, num_heads, seq_len, seq_len]
    # Take the last 6 layers
    last_6_layers = outputs.attentions[-6:]

    # Free everything except the attention weights we need
    del outputs
    torch.cuda.empty_cache()

    return last_6_layers
```

**Critical memory note:** `outputs` contains the full model output including logits and intermediate states. Delete it immediately after extracting the attention weights. Call `torch.cuda.empty_cache()` to release CUDA memory back to the allocator.

**What `outputs.attentions` contains:** A tuple of `num_layers` tensors. Each tensor shape is `[batch_size, num_attention_heads, sequence_length, sequence_length]`. The `[i, j, q, k]` value is how much position `q` attends to position `k` in head `j`.

**GQA note:** Qwen3 uses Grouped Query Attention. The number of key-value heads is smaller than the number of query heads. When `output_attentions=True` and `attn_implementation="eager"`, transformers expands the KV heads to match the Q heads before returning — so the tensor you receive has shape `[1, num_q_heads, seq_len, seq_len]`. You do not need to handle GQA manually. Verify this by checking `last_6_layers[0].shape` on a test pass and logging it.

---

## Step 4 — Extract Last-Token Attention

You want the attention FROM the last token in the sequence TO all earlier tokens. This is the `[-1, :]` slice of the sequence dimension — the last query position attending to all key positions.

```python
def extract_last_token_attention(
    last_6_layers: tuple,
    abstract_start_token_idx: int,
    total_seq_len: int,
) -> torch.Tensor:
    """
    Extracts the last token's attention distribution over abstract tokens.

    Returns: 1D tensor of length (total_seq_len - abstract_start_token_idx)
             representing attention weight on each abstract token,
             averaged across all heads and the 6 captured layers.
    """
    layer_attns = []

    for layer_attn in last_6_layers:
        # layer_attn shape: [1, num_heads, seq_len, seq_len]
        # Get last token's attention to all positions
        # Result shape: [1, num_heads, seq_len]
        last_token_attn = layer_attn[:, :, -1, :]

        # Average across heads
        # Result shape: [1, seq_len]
        avg_over_heads = last_token_attn.mean(dim=1)

        # Remove batch dimension → shape: [seq_len]
        layer_attns.append(avg_over_heads.squeeze(0))

    # Average across layers → shape: [seq_len]
    avg_over_layers = torch.stack(layer_attns, dim=0).mean(dim=0)

    # Slice to abstract tokens only — discard system prompt attention
    abstract_attn = avg_over_layers[abstract_start_token_idx:]

    # Move to CPU and convert to float32 for numpy compatibility
    return abstract_attn.cpu().float()
```

---

## Step 5 — Sentence Segmentation and Token Mapping

This step maps character-level sentence boundaries to token positions.

```python
import re

RESULTS_PATTERNS = [
    r'\b(showed?|demonstrated?|revealed?|found|observed|identified|detected)\b',
    r'\b(significantly|greater than|less than|more than|compared to|relative to)\b',
    r'\b(activation|deactivation|correlation|increase[sd]?|decrease[sd]?|reduction)\b',
    r'\b(p\s*[<>=]\s*0\.\d+|t\s*\(\d+\)|F\s*\(\d+|r\s*=\s*[-\d.])\b',
    r'\b(higher|lower|larger|smaller|stronger|weaker)\b.{0,40}\b(than|compared)\b',
    r'\b(bilateral|unilateral|left|right)\b.{0,30}\b(cortex|gyrus|sulcus|area|region)\b',
]

METHODS_PATTERNS = [
    r'\b(participants?|subjects?|volunteers?|patients?)\b.{0,20}\b(were|had|completed)\b',
    r'\b(fMRI|MRI|PET|EEG|MEG)\b.{0,30}\b(scanner|session|protocol|study)\b',
    r'\b(TR|TE|voxel|slice|mm|tesla)\b',
    r'\b(we used|study (examined|investigated|aimed)|designed to)\b',
    r'\b(informed consent|ethics|IRB|approved)\b',
    r'\b(\d+\s*(male|female|men|women|participants|subjects))\b',
]

ABBREVIATIONS = {'e.g', 'i.e', 'vs', 'Fig', 'Eq', 'et al', 'vol', 'no', 'Dr'}


def classify_sentence(sentence: str) -> str:
    results_hits = sum(
        1 for p in RESULTS_PATTERNS
        if re.search(p, sentence, re.IGNORECASE)
    )
    methods_hits = sum(
        1 for p in METHODS_PATTERNS
        if re.search(p, sentence, re.IGNORECASE)
    )
    if results_hits > 0 and results_hits >= methods_hits:
        return "RESULTS"
    elif methods_hits > results_hits:
        return "METHODS"
    return "BACKGROUND"


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """
    Returns list of (sentence_text, char_start, char_end).
    Handles common abbreviations to avoid false sentence boundaries.
    """
    pattern = r'(?<!\b(?:' + '|'.join(ABBREVIATIONS) + r'))\. (?=[A-Z])'
    boundaries = [0] + [m.end() for m in re.finditer(pattern, text)] + [len(text)]
    sentences = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        sent = text[start:end].strip()
        if sent:
            sentences.append((sent, start, end))
    return sentences


def map_sentences_to_tokens(
    abstract_text: str,
    tokenizer,
    abstract_start_token_idx: int,
) -> list[dict]:
    """
    Returns list of dicts:
    {
        "text": sentence text,
        "label": "RESULTS" | "METHODS" | "BACKGROUND",
        "token_positions": list of int (positions within abstract token range,
                           i.e. 0 = first abstract token, not full sequence position)
    }
    """
    sentences = split_sentences(abstract_text)
    result = []

    for sent_text, char_start, char_end in sentences:
        label = classify_sentence(sent_text)

        # Tokenize prefix up to sentence boundaries to find token positions
        prefix_before = tokenizer(
            abstract_text[:char_start],
            add_special_tokens=False
        )["input_ids"]
        prefix_after = tokenizer(
            abstract_text[:char_end],
            add_special_tokens=False
        )["input_ids"]

        token_start = len(prefix_before)
        token_end = len(prefix_after)
        token_positions = list(range(token_start, token_end))

        if token_positions:
            result.append({
                "text": sent_text,
                "label": label,
                "token_positions": token_positions,
            })

    return result
```

---

## Step 6 — Compute Routing Score

```python
def compute_routing_score(
    abstract_attn: torch.Tensor,
    sentence_map: list[dict],
) -> dict:
    """
    Returns routing score dict:
    {
        "routing_score": float,           # fraction of attention on RESULTS tokens
        "results_fraction": float,
        "methods_fraction": float,
        "background_fraction": float,
        "n_results_tokens": int,
        "n_methods_tokens": int,
        "n_background_tokens": int,
        "total_abstract_tokens": int,
    }
    """
    total_attn = abstract_attn.sum().item()

    if total_attn == 0:
        return {
            "routing_score": 0.0,
            "results_fraction": 0.0,
            "methods_fraction": 0.0,
            "background_fraction": 0.0,
            "n_results_tokens": 0,
            "n_methods_tokens": 0,
            "n_background_tokens": 0,
            "total_abstract_tokens": len(abstract_attn),
        }

    results_attn = 0.0
    methods_attn = 0.0
    background_attn = 0.0
    n_results = n_methods = n_background = 0

    for sent in sentence_map:
        label = sent["label"]
        positions = [p for p in sent["token_positions"] if p < len(abstract_attn)]
        if not positions:
            continue

        sent_attn = abstract_attn[positions].sum().item()

        if label == "RESULTS":
            results_attn += sent_attn
            n_results += len(positions)
        elif label == "METHODS":
            methods_attn += sent_attn
            n_methods += len(positions)
        else:
            background_attn += sent_attn
            n_background += len(positions)

    return {
        "routing_score": results_attn / total_attn,
        "results_fraction": results_attn / total_attn,
        "methods_fraction": methods_attn / total_attn,
        "background_fraction": background_attn / total_attn,
        "n_results_tokens": n_results,
        "n_methods_tokens": n_methods,
        "n_background_tokens": n_background,
        "total_abstract_tokens": len(abstract_attn),
    }
```

---

## Step 7 — Top-Level Analysis Function

This is the public interface that `protected/harness/study_002/study_runner.py` calls.

```python
from protected.attention.scorer import RoutingScore

def analyze_abstract(
    model,
    tokenizer,
    system_prompt: str,
    abstract_id: str,
    abstract_text: str,
    device: str = "cuda",
) -> RoutingScore:
    """
    Full pipeline: input → forward pass → attention extraction → routing score.
    Runs on a single abstract. Call in a loop for the probe set.
    Cleans up GPU memory after each call.
    """
    input_dict = None

    try:
        input_dict = build_input(tokenizer, system_prompt, abstract_text, device)
        last_6_layers = run_prefill(model, input_dict)

        abstract_attn = extract_last_token_attention(
            last_6_layers,
            input_dict["abstract_start_token_idx"],
            input_dict["total_seq_len"],
        )

        del last_6_layers
        torch.cuda.empty_cache()

        sentence_map = map_sentences_to_tokens(
            abstract_text,
            tokenizer,
            input_dict["abstract_start_token_idx"],
        )

        score_dict = compute_routing_score(abstract_attn, sentence_map)

        return RoutingScore(
            abstract_id=abstract_id,
            score=score_dict["routing_score"],
            results_attention_fraction=score_dict["results_fraction"],
            methods_attention_fraction=score_dict["methods_fraction"],
            background_attention_fraction=score_dict["background_fraction"],
            n_results_tokens=score_dict["n_results_tokens"],
            n_methods_tokens=score_dict["n_methods_tokens"],
            n_background_tokens=score_dict["n_background_tokens"],
            n_layers_used=6,
        )

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        seq_len = input_dict["total_seq_len"] if input_dict else "unknown"
        raise RuntimeError(
            f"OOM during attention analysis of abstract {abstract_id}. "
            f"Sequence length: {seq_len}. "
            f"Consider reducing max_length in build_input."
        ) from e
```

---

## Step 8 — Verification Test

Run this before any study iterations begin. It must pass before Study 002 can proceed.

```python
def verify_attention_pipeline(model, tokenizer, sample_abstract: str) -> None:
    """
    Diagnostic verification of the full attention pipeline.
    Call once at study startup before any iteration runs.
    Raises AssertionError if sanity checks fail.
    """
    system_prompt = open("prompts/system_prompt.md").read()

    print("=== Attention Pipeline Verification ===")
    print(f"Model layers: {model.config.num_hidden_layers}")
    print(f"Num attention heads (Q): {model.config.num_attention_heads}")
    print(f"Num KV heads: {model.config.num_key_value_heads}")

    # Run on sample abstract
    result = analyze_abstract(
        model, tokenizer,
        system_prompt=system_prompt,
        abstract_id="verification_sample",
        abstract_text=sample_abstract,
    )

    print(f"\nSample abstract routing score: {result.score:.4f}")
    print(f"Results attention fraction:    {result.results_attention_fraction:.4f}")
    print(f"Methods attention fraction:    {result.methods_attention_fraction:.4f}")
    print(f"Background attention fraction: {result.background_attention_fraction:.4f}")
    print(f"Results tokens: {result.n_results_tokens}")
    print(f"Methods tokens: {result.n_methods_tokens}")
    print(f"Background tokens: {result.n_background_tokens}")

    # Fraction sum check
    total = (
        result.results_attention_fraction
        + result.methods_attention_fraction
        + result.background_attention_fraction
    )
    print(f"\nFraction sum (should be ~1.0): {total:.4f}")
    assert abs(total - 1.0) < 0.05, f"Fractions do not sum to 1.0: {total}"

    # Directional sanity check — results-only abstract
    results_only = (
        "Bilateral hippocampal activation increased significantly during "
        "encoding compared to baseline. Left prefrontal cortex showed "
        "greater activation for novel than repeated stimuli (p < 0.001). "
        "Memory performance correlated positively with hippocampal BOLD signal."
    )
    score_results = analyze_abstract(
        model, tokenizer,
        system_prompt=system_prompt,
        abstract_id="synthetic_results",
        abstract_text=results_only,
    )
    print(f"\nSynthetic results-only score: {score_results.score:.4f} (expect > 0.5)")

    # Directional sanity check — methods-only abstract
    methods_only = (
        "Fifteen healthy volunteers were recruited. fMRI was performed on a "
        "3T scanner with TR=2000ms and TE=30ms. Voxel size was 3x3x3mm. "
        "Statistical maps were thresholded at p<0.001 uncorrected."
    )
    score_methods = analyze_abstract(
        model, tokenizer,
        system_prompt=system_prompt,
        abstract_id="synthetic_methods",
        abstract_text=methods_only,
    )
    print(f"Synthetic methods-only score:  {score_methods.score:.4f} (expect < 0.3)")

    assert score_results.score > score_methods.score, (
        f"Directional check failed: results score ({score_results.score:.4f}) "
        f"should exceed methods score ({score_methods.score:.4f})"
    )

    print("\n=== Verification passed ===")
```

---

## Step 9 — Post-Modification Sensitivity Check

After wiring the pipeline into the harness, verify this one additional property: the routing score changes when the system prompt changes. Run `analyze_abstract` on one control abstract before applying an agent edit, apply the edit, run it again, confirm the score moved.

```python
def verify_sensitivity_to_prompt_change(
    model,
    tokenizer,
    abstract_id: str,
    abstract_text: str,
) -> None:
    """
    Confirms routing score responds to prompt changes.
    Run once after the first iteration applies an edit.
    """
    prompt_v1 = open("prompts/system_prompt.md").read()

    score_before = analyze_abstract(
        model, tokenizer,
        system_prompt=prompt_v1,
        abstract_id=abstract_id,
        abstract_text=abstract_text,
    )

    # Apply a test edit to the prompt temporarily
    test_prompt = prompt_v1 + "\nFocus exclusively on activation findings."

    score_after = analyze_abstract(
        model, tokenizer,
        system_prompt=test_prompt,
        abstract_id=abstract_id,
        abstract_text=abstract_text,
    )

    print(f"Score before prompt change: {score_before.score:.4f}")
    print(f"Score after prompt change:  {score_after.score:.4f}")
    print(f"Delta: {score_after.score - score_before.score:+.4f}")

    if abs(score_after.score - score_before.score) < 0.001:
        print("WARNING: Routing score did not respond to prompt change.")
        print("Check that system_prompt is being passed correctly to build_input.")
    else:
        print("Sensitivity confirmed — routing score responds to prompt changes.")
```

---

## Dependencies

```
pip install transformers>=4.40.0 bitsandbytes>=0.43.0 accelerate>=0.29.0
```

All three are required. `bitsandbytes` provides 4-bit quantization. `accelerate` is required for `device_map="auto"`. Install before attempting model load.

---

## Study Validity Note

This implementation uses the same model (Qwen 3.6 27B) for both extraction and attention analysis. This is a non-negotiable requirement. A secondary or proxy model would measure a different model's attention patterns, breaking the causal chain between agent modifications and routing score changes. The signal is only meaningful because it is computed from the same weights that do the extraction.

If OOM persists after implementing this spec, reduce `max_length` in `build_input` incrementally (4096 → 2048 → 1024) until the forward pass fits. Most neuroscience abstracts tokenize to under 400 tokens so truncation at 1024 should not affect results in practice.