import re

# Patterns that indicate methodology rather than findings
METHOD_PATTERNS = [
    r'\b(participants?|subjects?|volunteers?|patients?)\s*(were|n\s*=|N\s*=|total|count|number)',
    r'\b(scanner|TR|TE|voxel|mm|tesla|field strength)',
    r'\b(we used|we employed|we collected|we acquired|we performed)',
    r'\b(protocol|procedure|method|approach|acquisition)',
    r'\b(included|excluded|criteria|inclusion|exclusion)',
    r'\b(randomized|assigned|allocated|matched)',
    r'\b(statistical|threshold|correction|FDR|Bonferroni|ANOVA)',
    r'\b(consented|consent|IRB|ethics|approved)',
    r'\b(sample size|n\s*=\s*\d|N\s*=\s*\d)',
    r'\b(we aimed|we investigated|we examined|we studied|we sought)',
    r'\b(designed to|aimed to|objective was|purpose was|goal was)',
    r'\b(previous studies?|prior work|literature|previously reported)',
    r'\b(it is known|it has been shown|it is well established)',
]


def is_methodology_claim(claim_text: str) -> bool:
    """Check if a claim is actually a methodology description."""
    for pattern in METHOD_PATTERNS:
        if re.search(pattern, claim_text, re.IGNORECASE):
            return True
    return False


def validate_claims(claims: list) -> list:
    """Filter out methodology claims from extracted claims."""
    return [c for c in claims if not is_methodology_claim(c.claim_text)]
