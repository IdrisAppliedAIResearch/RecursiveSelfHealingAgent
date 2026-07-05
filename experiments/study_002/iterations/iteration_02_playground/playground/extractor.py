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
    
    # Step 1: Pre-process text to extract and prioritize results sentences
    # This addresses the attentional collapse by explicitly anchoring the model to results content
    sentences = re.split(r'(?<=[.!?]) +', abstract_text.replace('\n', ' '))
    results_sentences = []
    other_sentences = []
    
    # Heuristic to identify results sentences based on common keywords
    results_keywords = ['result', 'found', 'show', 'demonstrate', 'indicate', 'reveal', 'conclude', 'significant', 'p <', 'effect', 'increase', 'decrease', 'improve', 'reduce']
    
    for sentence in sentences:
        if any(kw in sentence.lower() for kw in results_keywords):
            results_sentences.append(sentence)
        else:
            other_sentences.append(sentence)
    
    # Construct a prioritized context: Results first, then other context
    # This forces the model to attend to results sentences early and maintain focus
    prioritized_text = "\n".join(results_sentences)
    if other_sentences:
        prioritized_text += "\n\n[Background/Methods Context]\n" + "\n".join(other_sentences)
    
    # Step 2: Generate initial extraction using prioritized text
    raw = _provider.complete_with_usage(system_prompt, prioritized_text)[0]
    
    # Step 3: Parse and validate claims
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r'```(?:json)?\s*(\{.*?)\s*```', raw, re.DOTALL)
        data = json.loads(m.group(1)) if m else {"claims": []}
    
    claims = [Claim(claim_text=c) for c in data.get("claims", [])]
    
    # Step 4: Grounding Check - Verify claims against original text to ensure no hallucination
    # This acts as a secondary attention anchor, ensuring extracted claims are supported
    if claims:
        grounding_prompt = f"""You are a verification assistant. Given a list of extracted claims and the original abstract text, verify if each claim is explicitly supported by the text. Return a JSON object with a 'verified_claims' list containing only the claims that are directly supported.

Original Abstract:
{abstract_text}

Extracted Claims:
{[c.claim_text for c in claims]}

Respond with ONLY a valid JSON object: {{"verified_claims": ["claim 1", "claim 2"]}}"""
        
        verification_raw = _provider.complete_with_usage(grounding_prompt, "")[0]
        try:
            verification_data = json.loads(verification_raw.strip())
            verified_texts = verification_data.get("verified_claims", [])
            # Filter claims based on verification
            claims = [Claim(claim_text=c) for c in verified_texts]
        except json.JSONDecodeError:
            # Fallback to original claims if verification fails
            pass
            
    return ExtractionResult(abstract_id=abstract_id, claims=claims)
