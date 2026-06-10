"""Empirical eval-noise floor from replicate rounds.

A *replicate* round re-trains a frontier member's exact code with a new
seed (validator-owned, no miner involved — see ``local/special_rounds``).
Each (original, replicate) pair is two independent draws of the same
configuration, so the paired metric differences measure the combined
seed + eval noise that every frontier comparison is subject to.

The estimator: for paired diffs ``d_i = m_orig - m_repl`` (mean ~0 by
construction), ``Var(d) = 2·σ²`` ⇒ ``σ = sqrt(mean(d²) / 2)``. The
relative floor divides by the mean metric level so it transfers across
metric regimes. Pure functions over experiment dicts.
"""

from __future__ import annotations

import math
from typing import Optional

# One-sided 95% band: a delta smaller than NOISE_K·σ is "within noise".
NOISE_K = 1.645


def replicate_pairs(experiments: list[dict],
                    task: Optional[str] = None) -> list[tuple[float, float]]:
    """(original_metric, replicate_metric) pairs joined on
    ``objectives['replicate_of']``."""
    by_id = {e["id"]: e for e in experiments if e.get("id") is not None}
    pairs: list[tuple[float, float]] = []
    for e in experiments:
        if e.get("mode") != "replicate" or not e.get("success"):
            continue
        if task is not None and e.get("task") != task:
            continue
        src_id = (e.get("objectives", {}) or {}).get("replicate_of")
        if src_id is None:
            continue
        src = by_id.get(int(src_id))
        if src is None or not src.get("success"):
            continue
        m_orig, m_repl = src.get("metric"), e.get("metric")
        if m_orig is None or m_repl is None:
            continue
        m_orig, m_repl = float(m_orig), float(m_repl)
        if math.isfinite(m_orig) and math.isfinite(m_repl):
            pairs.append((m_orig, m_repl))
    return pairs


def noise_floor(experiments: list[dict],
                task: Optional[str] = None) -> dict:
    """Pooled σ estimate over all replicate pairs (empty dict fields when
    no pairs exist yet)."""
    pairs = replicate_pairs(experiments, task=task)
    if not pairs:
        return {"n_pairs": 0, "sigma_metric": None, "sigma_rel": None,
                "threshold": None}
    diffs_sq = [(a - b) ** 2 for a, b in pairs]
    sigma = math.sqrt(sum(diffs_sq) / len(diffs_sq) / 2.0)
    level = sum((a + b) / 2.0 for a, b in pairs) / len(pairs)
    sigma_rel = sigma / level if level > 0 else None
    return {
        "n_pairs": len(pairs),
        "sigma_metric": sigma,
        "sigma_rel": sigma_rel,
        # Minimum |Δmetric| that clears the one-sided 95% noise band.
        "threshold": NOISE_K * sigma,
    }


def within_noise(delta: float, floor: Optional[dict]) -> Optional[bool]:
    """True/False when a floor estimate exists, None when it doesn't."""
    if not floor or floor.get("threshold") is None:
        return None
    return abs(float(delta)) < float(floor["threshold"])


def noise_summary(store, task: Optional[str] = None) -> dict:
    """``GET /experiments/noise`` payload."""
    exps = store.recent_experiments(n=10_000)
    if task:
        exps = [e for e in exps if e.get("task") == task]
    floor = noise_floor(exps, task=None)
    floor["task"] = task
    floor["note"] = (
        "sigma estimated from replicate-round pairs; deltas below "
        "'threshold' are within the one-sided 95% noise band"
    )
    return floor
