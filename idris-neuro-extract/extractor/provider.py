import json
import os

import openai


class LlamaCppProvider:
    def __init__(self):
        self.client = openai.OpenAI(
            base_url=os.environ["LLAMA_CPP_BASE_URL"],
            api_key=os.environ.get("LLAMA_CPP_API_KEY", "no-key"),
        )
        self.model = os.environ.get("LLAMA_CPP_MODEL_ID", "qwen3-27b-mtp-6bit")

    def complete(self, system_prompt: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
