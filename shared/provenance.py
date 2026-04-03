"""Provenance: pure-Python helpers for component detection and similarity.

The ProvenanceQuery class has been moved to shared/pg_provenance.py
(async Postgres-backed PgProvenanceQuery).

This module retains the stateless helper functions that have no DB dependency.
"""

import difflib
import re
from typing import Optional

from shared.dedup import code_similarity

# Default ML component patterns (deterministic regex)
DEFAULT_COMPONENT_PATTERNS: dict[str, str] = {
    "RMSNorm": r"\bRMSNorm\b",
    "LayerNorm": r"\bLayerNorm\b",
    "GELU": r"\bGELU\b",
    "SwiGLU": r"\bSwiGLU\b",
    "RotaryEmbedding": r"\b(?:rotary|RoPE|RotaryEmbedding)\b",
    "PatchEmbedding": r"\b(?:PatchEmbed|patch_embed|PatchEmbedding)\b",
    "FlashAttention": r"\b(?:flash_attn|FlashAttention|flash_attention)\b",
    "AdamW": r"\bAdamW\b",
    "CosineSchedule": r"\b(?:CosineAnnealing|cosine_schedule|CosineSchedule)\b",
    "TransformerEncoder": r"\bTransformerEncoder\b",
    "QuantileHead": r"\b(?:QuantileHead|quantile_head|quantile_output)\b",
    "MoE": r"\b(?:MixtureOfExperts|MoE|mixture_of_experts)\b",
}


def detect_components(
    code: str, patterns: Optional[dict[str, str]] = None,
) -> list[str]:
    """Regex scan for known ML components. Returns component names."""
    pats = patterns or DEFAULT_COMPONENT_PATTERNS
    return [name for name, pattern in pats.items() if re.search(pattern, code)]


def compute_similarity(code_a: str, code_b: str) -> dict:
    """Return multiple similarity signals. No thresholds — consumer decides."""
    jaccard = code_similarity(code_a, code_b)

    lines_a = code_a.splitlines(keepends=True)
    lines_b = code_b.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
    diff_ratio = matcher.ratio()

    total_lines = max(len(lines_a), len(lines_b), 1)
    ops = matcher.get_opcodes()
    changed = sum(max(j2 - j1, i2 - i1) for op, i1, i2, j1, j2 in ops if op != "equal")
    diff_size = changed / total_lines

    return {"jaccard": jaccard, "diff_ratio": diff_ratio, "diff_size": diff_size}
