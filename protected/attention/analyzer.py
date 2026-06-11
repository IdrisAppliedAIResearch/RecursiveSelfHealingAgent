from dataclasses import dataclass
from typing import Tuple

import torch


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
