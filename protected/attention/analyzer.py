from dataclasses import dataclass
from typing import Tuple

import torch

from protected.attention.scorer import RoutingScore, compute_routing_score
from protected.attention.segmenter import map_sentences_to_tokens


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
        max_length=4096,
    )

    # Find where abstract text starts in the rendered prompt
    abstract_char_start = prompt.find(abstract_text)
    offsets = inputs["offset_mapping"][0]
    system_len = 0
    for i, (s, e) in enumerate(offsets):
        if s >= abstract_char_start:
            system_len = i
            break

    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "abstract_start_token_idx": system_len,
        "total_seq_len": inputs["input_ids"].shape[1],
    }


def run_prefill(model, input_dict: dict) -> tuple:
    """
    Single forward pass. No generation. Returns attention weights from last 6 layers.
    Returns: tuple of 6 tensors, each shape [1, num_heads, seq_len, seq_len]
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

    Returns: 1D tensor of length (total_seq_len - abstract_start_token_idx)
    """
    layer_attns = []

    for layer_attn in last_6_layers:
        last_token_attn = layer_attn[:, :, -1, :]
        avg_over_heads = last_token_attn.mean(dim=1)
        layer_attns.append(avg_over_heads.squeeze(0))

    avg_over_layers = torch.stack(layer_attns, dim=0).mean(dim=0)
    abstract_attn = avg_over_layers[abstract_start_token_idx:]

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
    Full pipeline: input -> forward pass -> attention extraction -> routing score.
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

    # Verify last token is NOT a thinking token
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

        # Hooks stay registered — they fire during the forward pass below
        with torch.no_grad():                         # ← prevents gradient graph
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,                      # ← no KV cache allocation
                output_attentions=False,              # hooks handle capture
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

    def complete_with_usage(self, system_prompt, user_message, max_tokens=None, max_input_length=6656):
        import gc
        from extractor.provider import TokenUsage

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        enc = self.tokenizer(
            chat_text, return_tensors="pt", truncation=True, max_length=max_input_length
        )
        input_ids = enc["input_ids"].to(self.model.device)
        prompt_len = input_ids.shape[1]

        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        # Switch attention backend for cheaper generation — hooks are
        # already removed so eager attention capture is not needed.
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

        try:
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens or 1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=True,
                )

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
