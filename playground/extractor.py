from protected.schema import Claim, ExtractionResult
import json
import re
from pathlib import Path

_provider = None


# Patterns that indicate a sentence is reporting a result/finding
RESULT_PATTERNS = [
    r'\b(showed?|demonstrated?|revealed?|found|observed|indicated|suggested|detected|identified)\b',
    r'\b(significantly|greater than|less than|increased|decreased|enhanced|reduced|elevated)\b',
    r'\b(activation|deactivation|correlation|enhancement|reduction|modulation)\b',
    r'\b(connected|connectivity)\b',
    r'\b(associated with|correlated with|predicts|mediated)\b',
    r'\b(p\s*[<>=]\s*0\.\d+|t\s*\(\d+|F\s*\(\d+|p\s*<\s*0\.\d+|p\s*=\s*0\.\d+)\b',
    r'\b(no difference|not significant|did not differ|failed to show)\b',
    r'\b(activation in|deactivation in|recruited|engaged|involved in)\b',
]


def extract_sentences(text: str) -> list:
    """Split text into sentences and return list."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def is_result_sentence(sentence: str) -> bool:
    """Check if a sentence appears to report a finding."""
    lower = sentence.lower()
    for pattern in RESULT_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True
    return False


def fallback_extract_sentences(abstract_text: str) -> list:
    """Fallback: extract sentences that look like results."""
    sentences = extract_sentences(abstract_text)
    results = [s for s in sentences if is_result_sentence(s)]
    return results


async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:
    from playground.validator import validate_claims

    prompts_dir = Path(__file__).parent.parent / "prompts"
    system_prompt = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")
    examples = (prompts_dir / "examples.md").read_text(encoding="utf-8").strip()
    if examples:
        system_prompt = system_prompt + "\n\n" + examples

    # Pass full abstract text to model (no preprocessing)
    raw = _provider.complete_with_usage(system_prompt, abstract_text)[0]
    data = {"claims": []}

    # Try direct JSON parse
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
    raw_claims = []
    for c in data.get("claims", []):
        text = c.strip()
        if text and not placeholder_re.match(text):
            raw_claims.append(text)

    # CRITICAL FALLBACK: If JSON parsing produced no claims, extract result sentences directly
    if not raw_claims:
        raw_claims = fallback_extract_sentences(abstract_text)

    # Post-process: validate claims to remove methodology descriptions
    claims = [Claim(claim_text=c) for c in validate_claims(raw_claims)]

    return ExtractionResult(abstract_id=abstract_id, claims=claims)
