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
import random

logger = logging.getLogger(__name__)

# Steepness of the improvement sigmoid (mirrors scoring.score_against_frontier).
_DELTA_K = 5.0


def continuation_rate(
    attempted_rounds: int,
    *,
    equilibrium: float = 0.7,
    warmup_rounds: int = 20,
    step_pct: float = 2.0,
    step_every: int = 3,
) -> float:
    """Target continuation fraction given the count of attempted rounds.

    Stays at **0** until ``warmup_rounds`` rounds have been attempted (any
    experiment row, success or failure — failed rounds count too), then
    climbs as a staircase — ``step_pct`` percentage points every
    ``step_every`` rounds — up to ``equilibrium``. With the defaults
    (warmup 20, +2%/3 rounds, eq 0.70) it reaches 70% about 105 rounds
    after the warmup.
    """
    eq = max(0.0, min(1.0, equilibrium))
    if attempted_rounds < warmup_rounds:
        return 0.0
    if step_every <= 0:
        return eq
    steps = (attempted_rounds - warmup_rounds) // step_every
    return min(eq, max(0.0, (step_pct / 100.0) * steps))


def is_continuation_round(
    round_id: int,
    attempted_rounds: int,
    *,
    equilibrium: float = 0.7,
    warmup_rounds: int = 20,
    step_pct: float = 2.0,
    step_every: int = 3,
) -> bool:
    """Deterministic per-round coin flip at the scheduled rate.

    Seeded by ``round_id`` so the schedule is reproducible and independent
    of the training seed; the *rate* is driven by ``attempted_rounds``.
    Whether a scheduled continuation round actually runs as one still
    depends on eligible parents existing (the validator downgrades to a
    fresh round when none do).
    """
    rate = continuation_rate(
        attempted_rounds, equilibrium=equilibrium, warmup_rounds=warmup_rounds,
        step_pct=step_pct, step_every=step_every,
    )
    if rate <= 0.0:
        return False
    return random.Random(f"continuation-schedule-{round_id}").random() < rate


def is_extend_round(round_id: int, extend_pct: float = 0.5) -> bool:
    """Deterministic per-round coin flip splitting a continuation round into
    ``extend`` (re-train the parent's *own* generator on the warm-started
    weights — more compute on identical data) vs ``modify`` (train the miner's
    freshly submitted generator).

    Seeded by ``round_id`` in a namespace distinct from the new-vs-continuation
    flip so the two schedules are independent. Only meaningful on a round that
    is already a continuation; the validator gates the call accordingly.
    """
    pct = max(0.0, min(1.0, extend_pct))
    if pct <= 0.0:
        return False
    if pct >= 1.0:
        return True
    return random.Random(f"continuation-extend-{round_id}").random() < pct



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
    current_epoch: dict | None = None,
    force_continuation: bool = False,
    extend: bool = False,
    eligible_parent_ids: list[int] | None = None,
    miner_id: str = "",
) -> dict:
    """Resolve a proposal's continuation request into ``run_training`` kwargs.

    Validates the requested parent (must be eligible + have a resolvable
    checkpoint). When ``force_continuation`` is true (the validator owns
    the cadence and has declared a continuation round), a miner that
    didn't pick a usable parent gets one auto-assigned at seeded random
    from ``eligible_parent_ids`` — the miner only chooses *which* parent,
    not *whether* to continue. Falls back to a fresh run only when no
    eligible parent's checkpoint is resolvable.

    ``current_epoch`` is a dict of ``objectives``-keys → required values
    (e.g. ``{"frozen_arch_version": 3}`` for ts_data_pipeline). A parent
    whose objectives don't match is rejected, because Δ is only honest
    inside a fixed context epoch.

    When ``extend`` is true (the validator declared an *extend* continuation
    round) and a parent resolves, ``prep["train_code"]`` is set to the
    parent's own generator code so the run literally keeps training the same
    data — ``prep["continuation_kind"]`` is then ``"extend"`` rather than
    ``"modify"`` (the miner's submitted code is ignored). The split is
    validator-owned; the miner only ever picks *which* parent.
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
        "continuation_kind": "",
        "train_code": None,
        "note": "",
    }
    lineage_used: set[str] = set()
    parent_code: str | None = None

    def _try_parent(pid: int) -> tuple[bool, str]:
        parent = store.get_experiment(pid)
        reason = _parent_reject_reason(
            parent, min_flops, max_flops, current_epoch=current_epoch,
        )
        if reason is not None:
            return False, reason
        ckpt_path = ckpt_store.resolve(parent.get("checkpoint_ref"))
        if ckpt_path is None:
            return False, "checkpoint unresolvable"
        prep.update(
            mode="continue",
            parent_index=pid,
            parent_metric=parent["metric"],
            parent_checkpoint_path=ckpt_path,
            compute_offset=float(parent.get("cumulative_compute", 0.0) or 0.0),
            n_rounds=int(parent.get("n_rounds", 1) or 1) + 1,
        )
        nonlocal lineage_used, parent_code
        lineage_used = lineage_shards(store.lineage(pid))
        parent_code = parent.get("code")
        return True, ""

    if mode == "continue" and isinstance(parent_index, int):
        ok, reason = _try_parent(parent_index)
        if not ok:
            prep["note"] = f"continuation rejected: {reason}"

    if force_continuation and prep["mode"] != "continue" and eligible_parent_ids:
        rng = random.Random(f"force-continuation-{seed}-{miner_id}")
        order = list(eligible_parent_ids)
        rng.shuffle(order)
        prior_note = prep["note"]
        for pid in order:
            if pid == parent_index:
                continue  # already tried above
            ok, reason = _try_parent(pid)
            if ok:
                why = (
                    "miner did not request continuation"
                    if not prior_note else prior_note
                )
                prep["note"] = f"continuation auto-assigned parent {pid} ({why})"
                break
        else:
            # Every eligible parent failed to resolve — keep prep as "new".
            prep["note"] = (
                f"{prior_note}; force-continuation could not resolve any parent"
                if prior_note
                else "force-continuation could not resolve any parent"
            )

    # Split a resolved continuation into extend vs modify. ``extend`` reuses
    # the parent's own generator (more compute on identical data); it falls
    # back to ``modify`` if the parent carries no usable code.
    if prep["mode"] == "continue":
        if extend and parent_code:
            prep["continuation_kind"] = "extend"
            prep["train_code"] = parent_code
        else:
            prep["continuation_kind"] = "modify"
            if extend and not parent_code:
                prefix = (prep["note"] + "; ") if prep["note"] else ""
                prep["note"] = prefix + "extend requested but parent has no code"

    if pool:
        keys, reused = assign_shards(
            pool, lineage_used=lineage_used, n=shards_per_round, seed=seed,
        )
        prep["shard_paths"] = keys
        prep["shard_reuse"] = reused
    return prep


def _parent_reject_reason(
    parent, min_flops: int, max_flops: int,
    *, current_epoch: dict | None = None,
):
    """Return a human reason a parent can't be continued, or None if OK."""
    if parent is None:
        return "parent not found"
    if not parent.get("success") or parent.get("metric") is None:
        return "parent not fully evaluated"
    if parent.get("checkpoint_ref") is None:
        return "parent has no saved checkpoint"
    objs = parent.get("objectives", {}) or {}
    # Size-gate check is meaningful only when bounds are set (e.g. the
    # ts_data_pipeline task disables the gate via 0/0 because the frozen
    # arch fixes FLOPs for every miner in a round).
    if min_flops > 0 or max_flops > 0:
        flops = objs.get("flops_equivalent_size", 0)
        if not (int(min_flops * 0.9) <= flops <= int(max_flops * 1.1)):
            return "parent outside size bucket"
    if current_epoch:
        for k, v in current_epoch.items():
            pv = objs.get(k)
            if pv != v:
                return f"parent in different epoch ({k}={pv} vs {v})"
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
