from protected.schema import Claim, ExtractionResult
from extractor.provider import LlamaCppProvider
import json
from pathlib import Path

_provider = LlamaCppProvider()


async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:
    prompts_dir = Path(__file__).parent.parent / "prompts"
    system_prompt = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")
    examples = (prompts_dir / "examples.md").read_text(encoding="utf-8").strip()
    if examples:
        system_prompt = system_prompt + "\n\n" + examples
    raw = _provider.complete(system_prompt, abstract_text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"claims": []}
    claims = [Claim(claim_text=c) for c in data.get("claims", [])]
    return ExtractionResult(abstract_id=abstract_id, claims=claims)
