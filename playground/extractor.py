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
    data = {"claims": []}

    # Try direct parse
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Try stripping markdown code fences
    if not data.get("claims"):
        m = re.search(r'```(?:json)?\s*(\{.*?)\s*```', raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    # Try finding valid JSON object via brace-depth
    if not data.get("claims"):
        for m in re.finditer(r'\{', raw):
            start = m.start()
            depth = 0
            in_string = False
            escape_next = False
            for i in range(start, len(raw)):
                ch = raw[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(raw[start:i+1])
                            if data.get("claims"):
                                break
                        except json.JSONDecodeError:
                            continue

    # Filter out placeholder and empty claims
    placeholder_re = re.compile(
        r'^(?:claim\s*\d+|example\s*claim|sample\s*claim|placeholder'
        r'|insert\s*claim|claim\s*here|no\s*claims|none'
        r'|not\s*applicable|n\.?a\.?)$', re.IGNORECASE
    )
    claims = []
    for c in data.get("claims", []):
        text = c.strip()
        if text and not placeholder_re.match(text):
            claims.append(Claim(claim_text=text))

    return ExtractionResult(abstract_id=abstract_id, claims=claims)
