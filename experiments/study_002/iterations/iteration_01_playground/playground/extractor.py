from protected.schema import Claim, ExtractionResult
import json
import re
from pathlib import Path

_provider = None

def isolate_results_section(abstract_text: str) -> str:
    """
    Attempts to isolate the Results section from the abstract.
    If explicit headers are found, it extracts the text between 'Results' and the next header or end.
    If no headers are found, it returns the full text but warns the model via prompt context.
    """
    # Case-insensitive search for common section headers
    # We look for 'Results' specifically
    lines = abstract_text.split('\n')
    results_start = None
    results_end = len(lines)
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Check for Results header
        if re.match(r'^results$', stripped, re.IGNORECASE) or re.match(r'^results[:\.]$', stripped, re.IGNORECASE):
            results_start = i + 1
        # Check for subsequent headers (e.g., Conclusion, Discussion, Background)
        elif results_start is not None and re.match(r'^(conclusion|discussion|background|methods|introduction|acknowledgements)$', stripped, re.IGNORECASE):
            results_end = i
            break
            
    if results_start is not None:
        return '\n'.join(lines[results_start:results_end]).strip()
    
    # Fallback: If no headers, return full text but we will prompt differently
    return abstract_text

async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:
    prompts_dir = Path(__file__).parent.parent / "prompts"
    system_prompt = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")
    examples = (prompts_dir / "examples.md").read_text(encoding="utf-8").strip()
    if examples:
        system_prompt = system_prompt + "\n\n" + examples
    
    # Isolate results to constrain attention and reduce drift
    results_text = isolate_results_section(abstract_text)
    
    # If we isolated results, append a specific instruction to focus on the provided text
    if results_text != abstract_text:
        system_prompt += "\n\nIMPORTANT: The following text is strictly the Results section. Extract claims ONLY from this text. Ignore any background or methodology context not present here."
        input_text = results_text
    else:
        # If we couldn't isolate, we rely on the original prompt but add a constraint
        system_prompt += "\n\nIMPORTANT: Focus exclusively on the empirical findings and results presented in the abstract. Do not extract background or methods."
        input_text = abstract_text

    raw = _provider.complete_with_usage(system_prompt, input_text)[0]
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r'```(?:json)?\s*(\{.*?)\s*```', raw, re.DOTALL)
        data = json.loads(m.group(1)) if m else {"claims": []}
    claims = [Claim(claim_text=c) for c in data.get("claims", [])]
    return ExtractionResult(abstract_id=abstract_id, claims=claims)
