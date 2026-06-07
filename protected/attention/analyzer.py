from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class AttentionResult:
    abstract_id: str
    abstract_text: str
    attention_weights: dict[int, torch.Tensor | None]


class AttentionAnalyzer:
    def __init__(self, model_path: str, n_last_layers: int = 6):
        self.model_path = model_path
        self.n_last_layers = n_last_layers
        self._stored_weights: dict[int, torch.Tensor | None] = {}
        self._hooks = []
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self._register_hooks()
        self._log_shapes()

    def _register_hooks(self) -> None:
        blocks = self.model.model.layers
        start = max(0, len(blocks) - self.n_last_layers)
        for i in range(start, len(blocks)):
            block = blocks[i]
            attn_module = block.self_attn
            hook = attn_module.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                last_token_idx = -1
                stored = attn_weights[last_token_idx].detach().cpu()
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
        self._stored_weights.clear()

        full_input = system_prompt + "\n" + abstract_text
        inputs = self.tokenizer(
            full_input, return_tensors="pt", truncation=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        return AttentionResult(
            abstract_id=abstract_id,
            abstract_text=abstract_text,
            attention_weights=dict(self._stored_weights),
        )

    def complete_with_usage(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int | None = None,
    ) -> Tuple[str, "TokenUsage"]:
        """Generate a completion using the loaded transformers model."""
        from extractor.provider import TokenUsage

        full_prompt = system_prompt + "\n" + user_message
        inputs = self.tokenizer(
            full_prompt, return_tensors="pt", truncation=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        prompt_len = input_ids.shape[1]

        self._stored_weights.clear()
        generated = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens or 8192,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True,
        )

        output_ids = generated.sequences[0][prompt_len:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        completion_tokens = len(output_ids)
        total_tokens = prompt_len + completion_tokens

        self._stored_weights.clear()

        token_usage = TokenUsage(
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tokens_per_second=0.0,
            context_window=4096,
        )
        return text, token_usage

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
