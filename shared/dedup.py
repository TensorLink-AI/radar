"""
Code similarity and parent identity checking.

The only objective deduplication gate: did the miner actually change the code
from the parent they were given? Cross-miner similarity is gameable (rename
variables, add dead code, reorder functions), so we don't use it as a gate.

The Pareto front + metrics handle the rest — two identical models produce
identical metrics, and only the first one joins the front.
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


def is_unchanged_from_parent(code: str, parent_code: str, threshold: float = 0.95) -> bool:
    """
    Check if submitted code is essentially unchanged from the parent.

    This is the hard gate: miners MUST modify the code they receive.
    Threshold of 0.95 catches copy-paste with trivial whitespace changes
    while allowing submissions that make real modifications.

    Returns: True if code is too similar to parent (should be rejected).
    """
    if not code or not parent_code:
        return False
    return code_similarity(code, parent_code) >= threshold


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
