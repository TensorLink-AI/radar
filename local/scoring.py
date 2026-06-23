"""Phase-C scoring for the local stack.

Mirrors the shape of ``shared/scoring.py`` (size gate, frontier
comparison, Pareto bonus) on the smaller surface the local trainer
emits. Pure functions, no torch.
"""

from __future__ import annotations

import math
from typing import Optional

from local.continuation import score_continuation


SIZE_GATE_TOLERANCE = 0.1   # 10%, matches Config.SIZE_GATE_TOLERANCE


def passes_size_gate(objectives: dict, min_flops: int, max_flops: int) -> bool:
    # Both bounds 0 = gate disabled (e.g. the ts_data_pipeline task where
    # the architecture is frozen and every miner reports identical flops).
    if min_flops <= 0 and max_flops <= 0:
        return True
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
    """Non-dominated set on (metric, flops). Both lower is better.

    Replicate rounds are excluded — they re-run an existing frontier
    member's code to measure eval noise (``local/noise.py``) and must
    never compete with it for frontier membership.
    """
    pts = [
        e for e in experiments
        if e.get("success") and e.get("metric") is not None
        and e.get("mode") != "replicate"
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


def compute_pareto_by_bucket(experiments: list[dict]) -> list[dict]:
    """Per-bucket Pareto union across a (possibly mixed-task) list.

    Each task uses its own size-bucket scheme; within every bucket the
    non-dominated set on ``(metric, flops)`` is computed independently and
    the results are unioned (an experiment near a tolerance boundary may
    sit on two adjacent buckets' frontiers). Computing Pareto *per bucket*
    — rather than once globally and then filtering by size gate — keeps the
    strongest model in a large bucket on the frontier even when a cheaper,
    lower-metric model in a smaller bucket would dominate it globally.
    """
    from local.task import SIZE_BUCKETS, buckets_for, make_spec

    by_task: dict[Optional[str], list[dict]] = {}
    for e in experiments:
        by_task.setdefault(e.get("task"), []).append(e)

    out: list[dict] = []
    seen: set[int] = set()
    for tname, group in by_task.items():
        try:
            buckets = buckets_for(make_spec(tname))
        except ValueError:
            buckets = SIZE_BUCKETS
        for lo, hi in buckets.values():
            members = [
                e for e in group
                if passes_size_gate(e.get("objectives", {}), lo, hi)
            ]
            for e in compute_pareto(members):
                if id(e) not in seen:
                    seen.add(id(e))
                    out.append(e)
    return out


def score_round(
    proposals_with_metrics: list[dict],
    min_flops: int,
    max_flops: int,
    frontier: list[dict],
    continuation_frontier: Optional[list[dict]] = None,
    noise_threshold: float = 0.0,
    novelty_bonus_weight: float = 0.0,
) -> list[dict]:
    """One scoring pass for a round. Mutates the input list with
    ``score`` and ``analysis`` fields and returns it.

    ``proposals_with_metrics`` items must have keys ``success``,
    ``metric``, ``objectives``. Failures and size-gate violations score
    zero.

    Continuation runs — items the validator marked with
    ``mode == "continue"`` and a ``parent_metric`` — are scored on the
    continuation frontier (GIFT-eval Δ vs cumulative compute) instead of
    the absolute initial frontier. Validation loss is never read here.
    ``noise_threshold`` (k·σ from the replicate noise floor) gates the
    continuation Δ: a sub-noise improvement scores zero.
    """
    out = []
    cont_frontier = continuation_frontier or []
    feasible_metrics = [
        f.get("metric") for f in frontier
        if f.get("metric") is not None
        and passes_size_gate(f.get("objectives", {}), min_flops, max_flops)
    ]
    # Structural-diversity bonus (Task 5): reward fresh designs that use
    # techniques the frontier under-explores. Additive and opt-in — the
    # frontier's source code is the comparison baseline.
    frontier_codes = [f.get("code", "") for f in frontier if f.get("code")]

    for p in proposals_with_metrics:
        if not p.get("success") or p.get("metric") is None:
            p["score"] = 0.0
            p["analysis"] = (p.get("analysis") or "") + " [score=0: failed]"
            out.append(p)
            continue
        if p.get("mode") == "replicate":
            # Noise probes are control measurements, never rewarded.
            p["score"] = 0.0
            p["analysis"] = (
                (p.get("analysis") or "") + " [score=0: replicate noise probe]"
            )
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

        if p.get("mode") == "continue" and p.get("parent_metric") is not None:
            paired = p.get("objectives", {}).get("paired")
            score, delta = score_continuation(
                metric=p["metric"],
                parent_metric=p["parent_metric"],
                cumulative_compute=float(
                    p.get("objectives", {}).get("cumulative_compute", 0.0)
                ),
                frontier=cont_frontier,
                paired=paired,
                noise_threshold=noise_threshold,
            )
            p["score"] = score
            # Persist Δ so the continuation frontier
            # (continuation_frontier reads objectives['delta']) actually
            # accumulates points — previously nothing wrote it.
            if isinstance(p.get("objectives"), dict):
                p["objectives"]["delta"] = delta
            paired_tag = ""
            if paired is not None:
                paired_tag = (
                    f" paired={paired.get('n_improved')}/{paired.get('n')}"
                    f" sig={int(bool(paired.get('significant')))}"
                )
            p["analysis"] = (
                (p.get("analysis") or "")
                + f" [score={score:.3f} continuation Δ={delta:.4f}{paired_tag}]"
            )
            out.append(p)
            continue

        base = score_against_frontier(p["metric"], feasible_metrics)
        bonus = pareto_bonus(p["objectives"], p["metric"], frontier)
        nov_mult, nov = 1.0, 0.0
        if novelty_bonus_weight > 0.0:
            from local.diversity import novelty_multiplier
            nov_mult, nov = novelty_multiplier(
                p.get("code", ""), frontier_codes, novelty_bonus_weight,
            )
        p["score"] = base * bonus * nov_mult
        nov_tag = f" novelty={nov:.2f}×{nov_mult:.2f}" if nov_mult != 1.0 else ""
        p["analysis"] = (
            (p.get("analysis") or "")
            + f" [score={p['score']:.3f} base={base:.3f} bonus={bonus:.2f}"
            + f"{nov_tag}]"
        )
        out.append(p)
    return out
