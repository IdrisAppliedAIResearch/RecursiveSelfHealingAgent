from protected.schema import Claim, ExtractionResult
import json
import re
from pathlib import Path

_provider = None

async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:
    prompts_dir = Path(__file__).parent.parent / "prompts"
    system_prompt = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")
    examples = (prompts_dir / "examples.md").read_text(encoding="utf-8").strip()
    if examples:
        system_prompt = system_prompt + "\n\n" + examples
    raw = _provider.complete_with_usage(system_prompt, abstract_text)[0]
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r'```(?:json)?\s*(\{.*?)\s*```', raw, re.DOTALL)
        data = json.loads(m.group(1)) if m else {"claims": []}
    
    # Handle both legacy format (just claims) and new format (claims + reasoning)
    claims_list = data.get("claims", [])
    
    # If reasoning is present, we could potentially use it for logging or debugging,
    # but for now we just ensure claims are extracted correctly.
    # The prompt change ensures the model outputs reasoning to help routing metrics.
    
    claims = [Claim(claim_text=c) for c in claims_list]
    return ExtractionResult(abstract_id=abstract_id, claims=claims)
