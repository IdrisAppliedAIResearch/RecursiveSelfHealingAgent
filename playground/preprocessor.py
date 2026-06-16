import re

RESULT_KEYWORDS = [
    r'\b(showed?|demonstrated?|revealed?|found|observed|indicated|suggested)\b',
    r'\b(significantly|greater than|less than|compared to|versus|relative to)\b',
    r'\b(activation|deactivation|correlation|increase|decrease|enhanced|reduced)\b',
    r'\b(p\s*[<>=]\s*0\.\d+|t\s*\(\d+\)|F\s*\(\d+|χ\s*\(|p\s*<\s*0\.\d+|p\s*=\s*0\.\d+)\b',
    r'\b(hippocampus|cortex|cortical|prefrontal|frontal|parietal|temporal|occipital|cingulate|amygdala|insula|cerebellum|thalamus|basal ganglia)\b',
    r'\b(BOLD|fMRI|MRI|EEG|MEG|PET|DTI|VBM|voxel|cluster)\b',
    r'\b(connected|connectivity|functional connectivity|structural connectivity)\b',
    r'\b(associated with|correlated with|predicts|mediated|moderated)\b',
    r'\b(activation in|deactivation in|recruited|engaged|involved in)\b',
    r'\b(results|findings|outcome)\b',
]

METHOD_KEYWORDS = [
    r'\b(participants?|subjects?|volunteers?|patients?|were recruited|were scanned|were enrolled)\b',
    r'\b(scanner|TR|TE|voxel|mm|tesla|field strength|gradient)\b',
    r'\b(we used|we employed|study examined|designed to|aimed to|objective was|purpose was)\b',
    r'\b(protocol|procedure|method|approach|acquisition|sequence|parameters)\b',
    r'\b(included|excluded|criteria|inclusion|exclusion|age range|mean age)\b',
    r'\b(randomized|assigned|allocated|matched|sample size|n\s*=|N\s*=)\b',
    r'\b(statistical|threshold|correction|FDR|Bonferroni|ANOVA|regression|GLM|SPM|FSL|AFNI)\b',
    r'\b(consented|consent|IRB|ethics|approved by|approved from)\b',
]

BACKGROUND_KEYWORDS = [
    r'\b(previous studies?|prior work|literature|previously reported|earlier studies?)\b',
    r'\b(it is known|it has been shown|it is well established|it is widely accepted)\b',
    r'\b(background|context|introduction|hereby|herein|in this study we|in the present study)\b',
]


def score_sentence(sentence: str) -> str:
    """Classify a sentence as RESULTS, METHODS, or BACKGROUND."""
    sentence_lower = sentence.lower()
    
    result_hits = sum(1 for p in RESULT_KEYWORDS if re.search(p, sentence_lower))
    method_hits = sum(1 for p in METHOD_KEYWORDS if re.search(p, sentence_lower))
    background_hits = sum(1 for p in BACKGROUND_KEYWORDS if re.search(p, sentence_lower))
    
    if result_hits >= method_hits and result_hits > 0:
        return 'RESULTS'
    elif method_hits > result_hits:
        return 'METHODS'
    elif background_hits > result_hits:
        return 'BACKGROUND'
    return 'RESULTS'


def filter_to_results(abstract_text: str) -> str:
    """Filter abstract text to only results-focused sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', abstract_text)
    results = [s for s in sentences if score_sentence(s) == 'RESULTS']
    return ' '.join(results) if results else abstract_text
