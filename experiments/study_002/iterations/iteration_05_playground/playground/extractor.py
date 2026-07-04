from protected.schema import Claim, ExtractionResult
import json
import re
from pathlib import Path

_provider = None

def _isolate_results_sentences(abstract_text: str) -> str:
    """
    Pre-processes the abstract to isolate sentences likely belonging to the Results section.
    This addresses the attentional drift by removing background/methods distractors from the context window.
    Implements structural exclusion of methods/background and aggressive statistical capture.
    """
    # Split into sentences based on common terminators
    sentences = re.split(r'(?<=[.!?])\s+', abstract_text)
    
    results_sentences = []
    
    # Structural markers to explicitly exclude method/background sections
    exclusion_markers = [
        'methodology', 'methods', 'method', 'background', 'introduction', 'introduction:', 'background:',
        'methods:', 'methodology:', 'participants', 'materials', 'procedure', 'data collection',
        'statistical analysis', 'analysis', 'design', 'study design', 'ethical approval', 'consent'
    ]
    
    # Heuristic: Identify results sentences by keywords and structural position
    # We look for sentences that contain result-indicative verbs or nouns
    # Using word boundaries to prevent false positives (e.g., 'mean' in 'meaning')
    result_indicators = [
        r'\bfound\b', r'\bshowed\b', r'\brevealed\b', r'\bindicated\b', r'\bdemonstrated\b', r'\bobserved\b', 
        r'\bincreased\b', r'\bdecreased\b', r'\bimproved\b', r'\bworsened\b', r'\bsignificant\b', r'\bp\s*<', 
        r'\bp<', r'\bmean\b', r'\bmedian\b', r'\bstandard\s+deviation\b', r'\bconfidence\s+interval\b',
        r'\bcorrelation\b', r'\beffect\s+size\b', r'\bodds\s+ratio\b', r'\bhazard\s+ratio\b',
        r'\bstatistically\s+significant\b', r'\bsignificantly\s+higher\b', r'\bsignificantly\s+lower\b',
        r'\bassociation\b', r'\brelationship\b', r'\bdifference\b', r'\bcomparison\b', r'\bresult\b',
        r'\boutcome\b', r'\bimpact\b', r'\binfluence\b', r'\bpredict\b', r'\bpredicted\b',
        r'\bconcluded\b', r'\bsuggest\b', r'\bsuggests\b', r'\bsuggesting\b', r'\bimplies\b',
        r'\bimplies\b', r'\bimplies\b', r'\bimplies\b'
    ]
    
    # Filter sentences that likely contain results
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Explicitly exclude sentences starting with or containing strong method/background markers
        # This prevents attentional drift into non-result sections
        is_excluded = any(marker in sentence_lower for marker in exclusion_markers)
        
        if is_excluded:
            continue
            
        # Check if sentence contains any result indicators using regex for word boundaries
        if any(re.search(indicator, sentence_lower) for indicator in result_indicators):
            results_sentences.append(sentence)
        # Also include sentences that start with typical result headers if present
        elif sentence_lower.startswith(('results', 'findings', 'conclusion')):
            results_sentences.append(sentence)
            
    # If no specific results sentences found, return the last 20% of the abstract
    # as a fallback, assuming results are often at the end
    if not results_sentences:
        start_index = int(len(sentences) * 0.8)
        results_sentences = sentences[start_index:] if start_index < len(sentences) else sentences
        
    return " ".join(results_sentences)

async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:
    prompts_dir = Path(__file__).parent.parent / "prompts"
    system_prompt = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")
    examples = (prompts_dir / "examples.md").read_text(encoding="utf-8").strip()
    if examples:
        system_prompt = system_prompt + "\n\n" + examples
    
    # Architectural Input Isolation: Filter context to only results sentences
    # This removes competing narrative structures (background/methods) that cause attentional drift
    isolated_context = _isolate_results_sentences(abstract_text)
    
    raw = _provider.complete_with_usage(system_prompt, isolated_context)[0]
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r'```(?:json)?\s*(\{.*?)\s*```', raw, re.DOTALL)
        data = json.loads(m.group(1)) if m else {"claims": []}
    claims = [Claim(claim_text=c) for c in data.get("claims", [])]
    return ExtractionResult(abstract_id=abstract_id, claims=claims)
