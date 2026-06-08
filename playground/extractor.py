from protected.schema import Claim, ExtractionResult
import json
from pathlib import Path

_provider = None


def _get_provider():
    global _provider
    if _provider is not None:
        return _provider
    from protected.harness.study_002.study_runner import _analyzer_instance

    if _analyzer_instance is not None:
        _provider = _analyzer_instance
        return _provider
    raise RuntimeError(
        "No model provider available. corpus_runner must inject the analyzer before calling extract()."
    )


async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:
    prompts_dir = Path(__file__).parent.parent / "prompts"
    system_prompt = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")
    examples = (prompts_dir / "examples.md").read_text(encoding="utf-8").strip()
    if examples:
        system_prompt = system_prompt + "\n\n" + examples
    raw = _get_provider().complete_with_usage(system_prompt, abstract_text)[0]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"claims": []}
    claims = [Claim(claim_text=c) for c in data.get("claims", [])]
    return ExtractionResult(abstract_id=abstract_id, claims=claims)
