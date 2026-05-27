"""Phase-C scoring for the local stack.

Mirrors the shape of ``shared/scoring.py`` (size gate, frontier
comparison, Pareto bonus) on the smaller surface the local trainer
emits. Pure functions, no torch.
"""

from __future__ import annotations

import math
from typing import Optional


SIZE_GATE_TOLERANCE = 0.1   # 10%, matches Config.SIZE_GATE_TOLERANCE


def passes_size_gate(objectives: dict, min_flops: int, max_flops: int) -> bool:
    flops = objectives.get("flops_equivalent_size", 0)
    lo = int(min_flops * (1 - SIZE_GATE_TOLERANCE))
    hi = int(max_flops * (1 + SIZE_GATE_TOLERANCE))
    return lo <= flops <= hi


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_against_frontier(
    metric: float, frontier_metrics: list[float],
) -> float:
    """Bootstrapping ranking when no frontier yet, sigmoid improvement
    otherwise. Lower metric is better (MSE)."""
    if not frontier_metrics:
        # No frontier: 0.5 baseline, will be normalized across miners.
        return 0.5
    best = min(frontier_metrics)
    # Positive when current beats best.
    denom = max(abs(best), 1e-8)
    delta = (best - metric) / denom
    return _sigmoid(5.0 * delta)


def pareto_bonus(
    objectives: dict, metric: float, frontier: list[dict],
) -> float:
    """1.5x if this point dominates any frontier member (both lower
    metric AND lower FLOPs), 1.0x otherwise."""
    flops = objectives.get("flops_equivalent_size", 0)
    for f in frontier:
        f_metric = f.get("metric")
        f_flops = f.get("objectives", {}).get("flops_equivalent_size", 0)
        if f_metric is None:
            continue
        if metric <= f_metric and flops <= f_flops and (
            metric < f_metric or flops < f_flops
        ):
            return 1.5
    return 1.0


def compute_pareto(experiments: list[dict]) -> list[dict]:
    """Non-dominated set on (metric, flops). Both lower is better."""
    pts = [
        e for e in experiments
        if e.get("success") and e.get("metric") is not None
    ]
    front: list[dict] = []
    for e in pts:
        em = e["metric"]
        ef = e["objectives"].get("flops_equivalent_size", 0)
        dominated = False
        for o in pts:
            if o is e:
                continue
            om = o["metric"]
            of = o["objectives"].get("flops_equivalent_size", 0)
            if om <= em and of <= ef and (om < em or of < ef):
                dominated = True
                break
        if not dominated:
            front.append(e)
    return front


def score_round(
    proposals_with_metrics: list[dict],
    min_flops: int,
    max_flops: int,
    frontier: list[dict],
) -> list[dict]:
    """One scoring pass for a round. Mutates the input list with
    ``score`` and ``analysis`` fields and returns it.

    ``proposals_with_metrics`` items must have keys ``success``,
    ``metric``, ``objectives``. Failures and size-gate violations score
    zero.
    """
    out = []
    feasible_metrics = [
        f.get("metric") for f in frontier
        if f.get("metric") is not None
        and passes_size_gate(f.get("objectives", {}), min_flops, max_flops)
    ]

    for p in proposals_with_metrics:
        if not p.get("success") or p.get("metric") is None:
            p["score"] = 0.0
            p["analysis"] = (p.get("analysis") or "") + " [score=0: failed]"
            out.append(p)
            continue
        if not passes_size_gate(p["objectives"], min_flops, max_flops):
            p["score"] = 0.0
            p["analysis"] = (
                (p.get("analysis") or "")
                + f" [score=0: outside bucket {min_flops}-{max_flops}]"
            )
            out.append(p)
            continue

        base = score_against_frontier(p["metric"], feasible_metrics)
        bonus = pareto_bonus(p["objectives"], p["metric"], frontier)
        p["score"] = base * bonus
        p["analysis"] = (
            (p.get("analysis") or "")
            + f" [score={p['score']:.3f} base={base:.3f} bonus={bonus:.2f}]"
        )
        out.append(p)
    return out
