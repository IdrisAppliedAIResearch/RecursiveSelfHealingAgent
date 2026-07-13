from protected.schema import Claim, ExtractionResult
import json
import re
from pathlib import Path

_provider = None

async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:
    prompts_dir = Path(__file__).parent.parent / "prompts"
    system_prompt = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")
    examples = (prompts_dir / "examples.md").read_text(encoding="utf-8").strip()
    
    # STAGE 1: Structural Classification (Forcing Grounding)
    # We modify the prompt to request ONLY the XML-tagged sentences for this step.
    # This forces the model to attend to sentence boundaries and types before generating claims.
    stage1_prompt = system_prompt.replace(
        "Output Format:",
        "STAGE 1 OUTPUT FORMAT:\nYou must output ONLY the XML-tagged sentences as described in STAGE 1. Do not output JSON yet.\n\nOutput Format:"
    )
    if examples:
        stage1_prompt = stage1_prompt + "\n\n" + examples
    
    raw_classification = _provider.complete_with_usage(stage1_prompt, abstract_text)[0]
    
    # Parse the XML-tagged results to identify grounded sentences
    results_sentences = []
    # Regex to capture text inside <sentence type="results">...</sentence>
    sentence_pattern = re.compile(r'<sentence\s+type="results">(.*?)</sentence>', re.DOTALL)
    matches = sentence_pattern.findall(raw_classification)
    
    # Clean up the matched sentences (remove extra whitespace/newlines if any)
    for match in matches:
        clean_text = " ".join(match.split())
        if clean_text:
            results_sentences.append(clean_text)
    
    # If no results sentences found, return empty
    if not results_sentences:
        return ExtractionResult(abstract_id=abstract_id, claims=[])
    
    # STAGE 2: Claim Extraction from Grounded Sentences
    # Construct a new prompt that provides the pre-identified sentences and asks for JSON claims
    grounded_context = "\n".join([f"- {s}" for s in results_sentences])
    stage2_instruction = f"""
Based on the following pre-identified results sentences from the abstract, generate the final JSON output.

IDENTIFIED RESULTS SENTENCES:
{grounded_context}

Instructions:
1. For each sentence above, formulate a concise, declarative claim.
2. The source_quote must be the exact text of the sentence.
3. Output ONLY the JSON object with the 'claims' array.
"""
    
    stage2_system_prompt = system_prompt + "\n\n" + stage2_instruction
    raw_claims_response = _provider.complete_with_usage(stage2_system_prompt, grounded_context)[0]
    
    # Parse the JSON response from Stage 2
    try:
        data = json.loads(raw_claims_response.strip())
    except json.JSONDecodeError:
        m = re.search(r'```(?:json)?\s*(\{.*?)\s*```', raw_claims_response, re.DOTALL)
        data = json.loads(m.group(1)) if m else {"claims": []}
        
    # Handle both old format (list of strings) and new format (list of objects)
    raw_claims = data.get("claims", [])
    claims = []
    for c in raw_claims:
        if isinstance(c, dict):
            # New format: {"claim": "...", "source_quote": "..."}
            claim_text = c.get("claim", c.get("claim_text", ""))
            source_quote = c.get("source_quote", "")
            claims.append(Claim(claim_text=claim_text, source_quote=source_quote))
        else:
            # Old format: string
            claims.append(Claim(claim_text=c))
            
    return ExtractionResult(abstract_id=abstract_id, claims=claims)
