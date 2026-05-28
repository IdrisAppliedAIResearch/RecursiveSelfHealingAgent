import json
from pathlib import Path

from extractor.provider import LlamaCppProvider
from extractor.schema import Claim, ExtractionResult


class Extractor:
    def __init__(self):
        self.provider = LlamaCppProvider()
        prompts_dir = Path(__file__).parent.parent / "prompts"
        system_prompt = (prompts_dir / "system_prompt.md").read_text()
        examples = (prompts_dir / "examples.md").read_text().strip()
        if examples:
            system_prompt = system_prompt + "\n\n" + examples
        self.system_prompt = system_prompt

    def extract(self, abstract_id: str, abstract_text: str) -> ExtractionResult:
        raw = self.provider.complete(self.system_prompt, abstract_text)
        data = json.loads(raw)
        claims = [Claim(claim_text=c) for c in data.get("claims", [])]
        return ExtractionResult(abstract_id=abstract_id, claims=claims)
