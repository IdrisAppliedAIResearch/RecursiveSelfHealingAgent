import json
import os

from dataclasses import dataclass

import openai


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float
    context_window: int


class LlamaCppProvider:
    def __init__(self):
        http_timeout = float(os.environ.get("LLM_HTTP_TIMEOUT_S", 600))
        self.client = openai.OpenAI(
            base_url=os.environ["LLAMA_CPP_BASE_URL"],
            api_key=os.environ.get("LLAMA_CPP_API_KEY", "no-key"),
            timeout=http_timeout,
        )
        self.model = os.environ.get("LLAMA_CPP_MODEL_ID", "qwen3-27b-mtp-6bit")
        self.context_window = int(os.environ.get("LLAMA_CPP_CONTEXT_WINDOW", "262144"))

    def complete(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def complete_with_usage(
        self, system_prompt: str, user_message: str, max_tokens: int | None = None
    ) -> tuple[str, TokenUsage]:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        usage_data = response.usage if response.usage else None
        prompt_tokens = usage_data.prompt_tokens if usage_data else 0
        completion_tokens = usage_data.completion_tokens if usage_data else 0
        total_tokens = (prompt_tokens + completion_tokens)

        tokens_per_second = 0.0
        if hasattr(response, "timings") and response.timings:
            if hasattr(response.timings, "predicted_per_second"):
                tokens_per_second = response.timings.predicted_per_second
        elif hasattr(response, "model_extra") and response.model_extra:
            extra = response.model_extra
            if "timings" in extra and isinstance(extra["timings"], dict):
                tokens_per_second = extra["timings"].get("predicted_per_second", 0.0)

        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            context_window=self.context_window,
        )
        return content, token_usage
