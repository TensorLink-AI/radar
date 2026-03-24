"""
Code similarity for provenance queries.

Used by shared/provenance.py for observational similarity analysis —
not used as a submission gate.
"""

import re
from collections import Counter


def code_similarity(code_a: str, code_b: str) -> float:
    """
    Compute structural similarity between two code strings.
    Uses normalized token overlap.
    Returns: float in [0, 1], where 1 = identical.
    """
    tokens_a = _tokenize_code(code_a)
    tokens_b = _tokenize_code(code_b)

    if not tokens_a or not tokens_b:
        return 0.0

    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)

    intersection = sum((counter_a & counter_b).values())
    union = sum((counter_a | counter_b).values())

    return intersection / union if union > 0 else 0.0



def _tokenize_code(code: str) -> list[str]:
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

    # Capture identifiers AND numeric literals (int, float, scientific notation)
    tokens = re.findall(r'[a-zA-Z_]\w*|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', code)

    stopwords = {
        'self', 'def', 'class', 'return', 'import', 'from', 'if', 'else',
        'elif', 'for', 'while', 'in', 'not', 'and', 'or', 'is', 'None',
        'True', 'False', 'with', 'as', 'try', 'except', 'finally', 'raise',
        'pass', 'break', 'continue', 'lambda', 'yield', 'assert', 'global',
    }
    return [t for t in tokens if t.lower() not in stopwords and len(t) > 1]
