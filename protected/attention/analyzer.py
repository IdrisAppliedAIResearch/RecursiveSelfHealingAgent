from dataclasses import dataclass
from typing import Tuple

import torch

from protected.attention.scorer import RoutingScore, compute_routing_score
from protected.attention.segmenter import map_sentences_to_tokens

# A004-7: input token budget validated to fit the RTX 5090 (32 GB) with headroom.
# Calibrated by scratchpad/probe_context_budget.py (2026-07-04): input alone stays in
# VRAM and fast up to ~6000 tokens (27.2 GB peak, 17.7 s); at 8000+ generation spills
# to WDDM shared memory (31-94 s) and hard-OOMs by 14000. The prior ad-hoc 13107 sat
# deep in the spill/OOM zone. Set to 5120 to leave headroom for decode-time KV growth
# (edits/repair generate up to 2048 tokens) plus fragmentation and the registered hooks.
MAX_INPUT_TOKENS = 5120
# A004-11: the attention pass generates on the same budget as extraction so that
# score_end is measured at the true end of extraction (EOS-terminated). Under greedy
# decoding (A004-3) and aligned inputs (A004-6) the two passes produce the same tokens.
EXTRACTION_MAX_NEW_TOKENS = 1024


class AbstractOffsetUnresolved(RuntimeError):
    """A004-8: raised when the abstract cannot be located within the templated
    prompt, so a routing score would otherwise be computed over the whole sequence."""


def _resolve_hf_cache_path(path: str) -> str:
    """Resolve HuggingFace cache refs to the actual snapshot directory."""
    from pathlib import Path
    p = Path(path)
    if p.joinpath("config.json").exists():
        return str(p)
    refs_dir = p / "refs"
    if refs_dir.exists():
        main_ref = refs_dir / "main"
        if main_ref.exists():
            commit = main_ref.read_text(encoding="utf-8").strip()
            snapshot = p / "snapshots" / commit
            if snapshot.joinpath("config.json").exists():
                return str(snapshot)
    return str(p)


def load_attention_model(model_path: str):
    """
    Load Qwen3 27B base model for attention analysis.
    model_path: local path to the base model (HuggingFace format, not GGUF).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    resolved_path = _resolve_hf_cache_path(model_path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        quantization_config=quantization_config,
        attn_implementation="eager",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path, trust_remote_code=True, use_fast=False, local_files_only=True
    )

    return model, tokenizer


def build_input(
    tokenizer,
    system_prompt: str,
    abstract_text: str,
    device: str = "cuda",
) -> dict:
    """
    Build tokenized input matching the extraction model's chat format.
    Returns dict with input_ids, attention_mask, and abstract_start_token_idx.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": abstract_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,  # A004-6: align with the extraction input budget
    )

    # A004-8: fail loudly rather than silently attributing whole-sequence attention
    # to the abstract when the abstract cannot be located in the templated prompt.
    abstract_char_start = prompt.find(abstract_text)
    if abstract_char_start == -1:
        raise AbstractOffsetUnresolved(
            "Abstract text not found in templated prompt; cannot locate token offset."
        )
    offsets = inputs["offset_mapping"][0]
    system_len = None
    for i, (s, e) in enumerate(offsets):
        if s >= abstract_char_start:
            system_len = i
            break
    if system_len is None:
        # The abstract's char offset fell beyond the (truncated) token range.
        raise AbstractOffsetUnresolved(
            "Abstract offset beyond tokenized range (likely truncated out by a "
            "large system prompt)."
        )

    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "abstract_start_token_idx": system_len,
        "total_seq_len": inputs["input_ids"].shape[1],
    }


def run_prefill(model, input_dict: dict) -> tuple:
    """
    Single forward pass. No generation. Returns attention weights from last 6 layers.
    DEPRECATED: Use run_generation_attention instead. Kept for backward compat.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_dict["input_ids"],
            attention_mask=input_dict["attention_mask"],
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    last_6_layers = outputs.attentions[-6:]

    del outputs
    torch.cuda.empty_cache()

    return last_6_layers


def extract_last_token_attention(
    last_6_layers: tuple,
    abstract_start_token_idx: int,
    total_seq_len: int,
) -> torch.Tensor:
    """
    Extracts the last token's attention distribution over abstract tokens.
    DEPRECATED: Use run_generation_attention instead. Kept for backward compat.
    """
    layer_attns = []

    for layer_attn in last_6_layers:
        last_token_attn = layer_attn[:, :, -1, :]
        avg_over_heads = last_token_attn.mean(dim=1)
        layer_attns.append(avg_over_heads.squeeze(0))

    avg_over_layers = torch.stack(layer_attns, dim=0).mean(dim=0)
    abstract_attn = avg_over_layers[abstract_start_token_idx:]

    return abstract_attn.cpu().float()


def run_generation_attention(
    model,
    tokenizer,
    input_dict: dict,
    max_new_tokens: int = EXTRACTION_MAX_NEW_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs generation and captures attention at the first and last generated
    tokens. A004-11: the budget matches extraction and generation stops at EOS,
    so score_end is the grounding at the true final token of the extraction.

    Returns:
        start_attn: [seq_len] tensor — last-6-layer averaged attention
                    at the first generated token, over abstract tokens
        end_attn:   [seq_len] tensor — same, at the last generated token
    """
    with torch.no_grad():
        prefill_out = model(
            input_ids=input_dict["input_ids"],
            attention_mask=input_dict["attention_mask"],
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )

    past_key_values = prefill_out.past_key_values
    next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del prefill_out
    torch.cuda.empty_cache()

    start_attn_raw = None
    end_attn_raw = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            step_out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        if step == 0:
            start_attn_raw = step_out.attentions[-6:]

        past_key_values = step_out.past_key_values
        next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if next_token.item() == tokenizer.eos_token_id:
            end_attn_raw = step_out.attentions[-6:]
            del step_out
            break

        end_attn_raw = step_out.attentions[-6:]
        del step_out
        torch.cuda.empty_cache()

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
        averaged.append(layer_attn.squeeze(0))

    avg_over_layers = torch.stack(averaged).mean(dim=0)
    avg_over_heads = avg_over_layers.mean(dim=0)
    avg_over_queries = avg_over_heads.mean(dim=0)
    abstract_attn = avg_over_queries[abstract_start_token_idx:]
    return abstract_attn.cpu().float()


def analyze_abstract(
    model,
    tokenizer,
    system_prompt: str,
    abstract_id: str,
    abstract_text: str,
    device: str = "cuda",
) -> RoutingScore:
    """
    Full pipeline: input -> generation attention -> routing score.
    Runs on a single abstract. Call in a loop for the probe set.
    """
    input_dict = None

    try:
        input_dict = build_input(tokenizer, system_prompt, abstract_text, device)

        start_attn, end_attn = run_generation_attention(
            model, tokenizer, input_dict, max_new_tokens=EXTRACTION_MAX_NEW_TOKENS
        )

        sentence_map = map_sentences_to_tokens(
            abstract_text,
            tokenizer,
            input_dict["abstract_start_token_idx"],
        )

        score_dict = compute_routing_score(start_attn, end_attn, sentence_map)

        return RoutingScore(
            abstract_id=abstract_id,
            score=score_dict["routing_score"],
            score_start=score_dict["score_start"],
            score_end=score_dict["score_end"],
            intra_generation_delta=score_dict["intra_generation_delta"],
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


def verify_attention_pipeline(model, tokenizer, sample_abstract: str) -> None:
    """
    Diagnostic verification of the full attention pipeline.
    Call once at study startup before any iteration runs.
    """
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    system_prompt = (project_root / "prompts" / "system_prompt.md").read_text(
        encoding="utf-8"
    )

    print("=== Attention Pipeline Verification ===")
    print(f"Model layers: {model.config.num_hidden_layers}")
    print(f"Num attention heads (Q): {model.config.num_attention_heads}")
    print(f"Num KV heads: {model.config.num_key_value_heads}")

    test_input = build_input(tokenizer, system_prompt, sample_abstract)
    last_token_id = test_input["input_ids"][0, -1].item()
    last_token_str = tokenizer.decode([last_token_id])
    print(f"Last token: '{last_token_str}' (id={last_token_id})")
    if "think" in last_token_str.lower():
        raise AssertionError(
            f"Thinking mode is still active — last token is '{last_token_str}'. "
            f"Pass enable_thinking=False to apply_chat_template."
        )
    print("Confirmed: thinking mode disabled, last token is content-generation token.")

    result = analyze_abstract(
        model, tokenizer,
        system_prompt=system_prompt,
        abstract_id="verification_sample",
        abstract_text=sample_abstract,
    )

    print(f"\nSample abstract routing score: {result.score:.4f}")
    print(f"  score_start: {result.score_start:.4f}")
    print(f"  score_end: {result.score_end:.4f}")
    print(f"  intra_generation_delta: {result.intra_generation_delta:+.4f}")
    print(f"Results attention fraction:    {result.results_attention_fraction:.4f}")
    print(f"Methods attention fraction:    {result.methods_attention_fraction:.4f}")
    print(f"Background attention fraction: {result.background_attention_fraction:.4f}")
    print(f"Results tokens: {result.n_results_tokens}")
    print(f"Methods tokens: {result.n_methods_tokens}")
    print(f"Background tokens: {result.n_background_tokens}")

    total = (
        result.results_attention_fraction
        + result.methods_attention_fraction
        + result.background_attention_fraction
    )
    print(f"\nFraction sum (should be ~1.0): {total:.4f}")
    assert abs(total - 1.0) < 0.05, f"Fractions do not sum to 1.0: {total}"

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
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    prompt_v1 = (project_root / "prompts" / "system_prompt.md").read_text(
        encoding="utf-8"
    )

    score_before = analyze_abstract(
        model, tokenizer,
        system_prompt=prompt_v1,
        abstract_id=abstract_id,
        abstract_text=abstract_text,
    )

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


@dataclass
class AttentionResult:
    abstract_id: str
    abstract_text: str
    attention_weights: dict[int, torch.Tensor | None]
    abstract_token_offset: int = 0


class AttentionAnalyzer:
    def __init__(self, model_path: str, n_last_layers: int = 6):
        self.model_path = self._resolve_model_path(model_path)
        self.n_last_layers = n_last_layers
        self._stored_weights: dict[int, torch.Tensor | None] = {}
        self._hooks = []
        self.model = None
        self.tokenizer = None

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        from pathlib import Path

        p = Path(path)
        if p.joinpath("config.json").exists():
            return str(p)
        refs_dir = p / "refs"
        if refs_dir.exists():
            main_ref = refs_dir / "main"
            if main_ref.exists():
                commit = main_ref.read_text(encoding="utf-8").strip()
                snapshot = p / "snapshots" / commit
                if snapshot.joinpath("config.json").exists():
                    return str(snapshot)
        return str(p)

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False, local_files_only=True
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=True,
        )

        print(f"  Model device: {self.model.device}")
        gpu_mem = torch.cuda.memory_allocated(0) / 1e9
        print(f"  GPU VRAM used: {gpu_mem:.1f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")

        self.model.eval()
        self._register_hooks()
        self._log_shapes()

        print("  Running generation smoke test...")
        try:
            test_text, test_usage = self.complete_with_usage(
                "You are a helpful assistant.", "Reply with exactly: OK", 16
            )
            print(f"  Smoke test output: {repr(test_text[:80])}")
            if not test_text.strip():
                print("  WARNING: Smoke test produced empty output!")
            else:
                print("  Smoke test passed.")
        except Exception:
            import traceback
            traceback.print_exc()
            print("  WARNING: Smoke test FAILED (see traceback above)")

    def _register_hooks(self) -> None:
        blocks = self.model.model.layers
        start = max(0, len(blocks) - self.n_last_layers)
        for i in range(start, len(blocks)):
            block = blocks[i]
            attn_module = getattr(block, "self_attn", None)
            if attn_module is not None:
                hook = attn_module.register_forward_hook(self._make_hook(i))
                self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None:
                    stored = attn_weights[-1].detach().cpu()
                    self._stored_weights[layer_idx] = stored

        return hook

    def _log_shapes(self) -> None:
        blocks = self.model.model.layers
        start = max(0, len(blocks) - self.n_last_layers)
        print(
            f"  AttentionAnalyzer: registered hooks on layers "
            f"{start} to {len(blocks) - 1} ({len(blocks) - start} layers)"
        )

    def forward_pass(
        self,
        abstract_text: str,
        system_prompt: str,
        abstract_id: str = "",
    ) -> AttentionResult:
        import gc

        self._stored_weights.clear()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": abstract_text},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        enc = self.tokenizer(
            chat_text, return_tensors="pt", truncation=True, max_length=4096
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)

        with torch.no_grad():
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
            )

        captured = dict(self._stored_weights)
        self._stored_weights.clear()

        gc.collect()
        torch.cuda.empty_cache()

        return AttentionResult(
            abstract_id=abstract_id,
            abstract_text=abstract_text,
            attention_weights=captured,
        )

    def _budget_user_message(self, system_prompt, user_message, max_input_length, reserve=384):
        """A004-2: trim the variable user message so it fits the input budget while
        the instruction at its tail is preserved. The system prompt is a separate
        message and is left intact; only the user message's head (oldest context)
        is dropped. Logs a context_truncated anomaly when trimming occurs."""
        sys_ids = self.tokenizer(system_prompt)["input_ids"]
        user_ids = self.tokenizer(user_message)["input_ids"]
        user_cap = max_input_length - len(sys_ids) - reserve
        if user_cap < 256:
            user_cap = 256
        if len(user_ids) <= user_cap:
            return user_message
        kept = user_ids[-user_cap:]  # keep the tail (instruction), drop the head
        trimmed = self.tokenizer.decode(kept, skip_special_tokens=True)
        try:
            from protected.harness.shared.anomaly_logger import log_anomaly
            log_anomaly("study_002", -1, "context_truncated", {
                "original_user_tokens": len(user_ids),
                "kept_user_tokens": len(kept),
                "max_input_length": max_input_length,
            })
        except Exception:
            pass
        return trimmed

    def complete_with_usage(self, system_prompt, user_message, max_tokens=None,
                            max_input_length=None, do_sample=False):
        import gc
        from extractor.provider import TokenUsage

        if max_input_length is None:
            max_input_length = MAX_INPUT_TOKENS

        # A004-2: preserve the instruction; trim only the variable context.
        user_message = self._budget_user_message(
            system_prompt, user_message, max_input_length
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        # A004-1: suppress thinking mode at the template level.
        chat_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        enc = self.tokenizer(
            chat_text, return_tensors="pt", truncation=True, max_length=max_input_length
        )
        input_ids = enc["input_ids"].to(self.model.device)
        prompt_len = input_ids.shape[1]

        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        original_attn = self.model.config._attn_implementation_internal
        gen_attn = original_attn
        try:
            self.model.set_attn_implementation("flash_attention_2")
            gen_attn = "flash_attention_2"
        except Exception:
            try:
                self.model.set_attn_implementation("sdpa")
                gen_attn = "sdpa"
            except Exception:
                gen_attn = f"fallback ({original_attn})"

        print(f"  [generate] attention={gen_attn} ctx={prompt_len}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

        # A004-3: greedy by default for structured output (extraction, edits);
        # prose field calls opt into sampling via do_sample=True.
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_tokens or EXTRACTION_MAX_NEW_TOKENS,
            use_cache=True,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = 0.7
            gen_kwargs["top_p"] = 0.9

        try:
            with torch.no_grad():
                output_sequences = self.model.generate(**gen_kwargs)

            output_ids = output_sequences[0][prompt_len:].clone()
            del output_sequences
            gc.collect()
            torch.cuda.empty_cache()

        finally:
            self.model.set_attn_implementation(original_attn)
            self._register_hooks()

        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        completion_tokens = len(output_ids)
        total_tokens = prompt_len + completion_tokens

        return text, TokenUsage(
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tokens_per_second=0.0,
            context_window=131072,
        )

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._stored_weights.clear()
        if self.model is not None:
            import gc
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
