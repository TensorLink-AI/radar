"""Continuation-frontier scoring for the local stack.

Fresh runs (``mode == "new"``, ``n_rounds == 1``) are scored on absolute
metric by ``local/scoring.py`` against the *initial* frontier. Runs that
warm-start from a parent checkpoint (``mode == "continue"``,
``n_rounds >= 2``) are scored here instead, on a second frontier:

    x = cumulative_compute  (Σ training FLOPs over the lineage)
    y = Δ = parent.metric − this.metric   (positive = progress)

A continuation only earns a Pareto bonus when it sits on the
compute-efficiency frontier — best Δ for its compute. Validation loss is
**never** read here; the score is GIFT-eval Δ only.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# Steepness of the improvement sigmoid (mirrors scoring.score_against_frontier).
_DELTA_K = 5.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def is_continuation(exp: dict) -> bool:
    """True when an experiment warm-started from a parent checkpoint."""
    if exp.get("mode") == "continue":
        return True
    return int(exp.get("n_rounds", 1) or 1) >= 2 and exp.get("parent_index") is not None


def continuation_frontier(experiments: list[dict]) -> list[dict]:
    """Non-dominated set over (cumulative_compute ↓, delta ↑).

    Operates on successful continuations carrying a finite ``delta`` and
    ``cumulative_compute`` in ``objectives``. A point dominates another
    when it reaches at least as large a Δ for no more compute, strictly
    better on one axis.
    """
    pts = [
        e for e in experiments
        if e.get("success") and is_continuation(e)
        and _delta(e) is not None and _compute(e) is not None
    ]
    front: list[dict] = []
    for e in pts:
        ed, ec = _delta(e), _compute(e)
        dominated = False
        for o in pts:
            if o is e:
                continue
            od, oc = _delta(o), _compute(o)
            if oc <= ec and od >= ed and (oc < ec or od > ed):
                dominated = True
                break
        if not dominated:
            front.append(e)
    return front


def _delta(exp: dict) -> float | None:
    d = (exp.get("objectives", {}) or {}).get("delta")
    if d is None or not math.isfinite(float(d)):
        return None
    return float(d)


def _compute(exp: dict) -> float | None:
    c = exp.get("cumulative_compute")
    if c is None:
        c = (exp.get("objectives", {}) or {}).get("cumulative_compute")
    if c is None or not math.isfinite(float(c)):
        return None
    return float(c)


def _dominates_frontier(delta: float, compute: float, frontier: list[dict]) -> bool:
    """True if (compute, delta) is not dominated by any frontier member."""
    for f in frontier:
        fd, fc = _delta(f), _compute(f)
        if fd is None or fc is None:
            continue
        if fc <= compute and fd >= delta and (fc < compute or fd > delta):
            return False
    return True


def prepare_continuation(
    store,
    ckpt_store,
    *,
    payload: dict,
    task_name: str,
    min_flops: int,
    max_flops: int,
    pool: list[str],
    shards_per_round: int,
    seed: int,
) -> dict:
    """Resolve a proposal's continuation request into ``run_training`` kwargs.

    Validates the requested parent (must be eligible + have a resolvable
    checkpoint); on any failure the run degrades to a fresh run and the
    reason is reported in ``note`` so the validator can record
    ``continuation_status``. Also computes the lineage-disjoint shard
    assignment.
    """
    from local.shards import assign_shards, lineage_shards

    mode = payload.get("mode", "new")
    parent_index = payload.get("parent_index")
    prep: dict = {
        "mode": "new",
        "parent_index": parent_index if isinstance(parent_index, int) else None,
        "parent_metric": None,
        "parent_checkpoint_path": None,
        "compute_offset": 0.0,
        "step_offset": 0,
        "n_rounds": 1,
        "shard_paths": None,
        "shard_reuse": False,
        "note": "",
    }
    lineage_used: set[str] = set()

    if mode == "continue" and isinstance(parent_index, int):
        parent = store.get_experiment(parent_index)
        reason = _parent_reject_reason(parent, min_flops, max_flops)
        ckpt_path = (
            ckpt_store.resolve(parent.get("checkpoint_ref"))
            if parent is not None and reason is None else None
        )
        if reason is not None:
            prep["note"] = f"continuation rejected: {reason}"
        elif ckpt_path is None:
            prep["note"] = "continuation rejected: checkpoint unresolvable"
        else:
            prep.update(
                mode="continue",
                parent_metric=parent["metric"],
                parent_checkpoint_path=ckpt_path,
                compute_offset=float(parent.get("cumulative_compute", 0.0) or 0.0),
                n_rounds=int(parent.get("n_rounds", 1) or 1) + 1,
            )
            lineage_used = lineage_shards(store.lineage(parent_index))

    if pool:
        keys, reused = assign_shards(
            pool, lineage_used=lineage_used, n=shards_per_round, seed=seed,
        )
        prep["shard_paths"] = keys
        prep["shard_reuse"] = reused
    return prep


def _parent_reject_reason(parent, min_flops: int, max_flops: int):
    """Return a human reason a parent can't be continued, or None if OK."""
    if parent is None:
        return "parent not found"
    if not parent.get("success") or parent.get("metric") is None:
        return "parent not fully evaluated"
    if parent.get("checkpoint_ref") is None:
        return "parent has no saved checkpoint"
    flops = (parent.get("objectives", {}) or {}).get("flops_equivalent_size", 0)
    if not (int(min_flops * 0.9) <= flops <= int(max_flops * 1.1)):
        return "parent outside size bucket"
    return None


def score_continuation(
    *,
    metric: float,
    parent_metric: float,
    cumulative_compute: float,
    frontier: list[dict],
) -> tuple[float, float]:
    """Score one continuation. Returns ``(score, delta)``.

    Δ ≤ 0 (no improvement over the parent) scores zero. Otherwise the
    base is a sigmoid of the normalized improvement, with a 1.5× bonus
    when the point lands on the continuation frontier.
    """
    if parent_metric is None or metric is None:
        return 0.0, 0.0
    delta = float(parent_metric) - float(metric)
    if delta <= 0 or not math.isfinite(delta):
        return 0.0, delta
    denom = max(abs(float(parent_metric)), 1e-8)
    base = _sigmoid(_DELTA_K * delta / denom)
    bonus = 1.5 if _dominates_frontier(delta, cumulative_compute, frontier) else 1.0
    return base * bonus, delta
