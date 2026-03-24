"""Size-gated Pareto frontier scoring for Phase C evaluation.

Uses validator-computed metrics (from Phase C evaluate_checkpoint) rather than
trainer-reported metrics. This is the trust anchor — every validator computes
identical scores from identical checkpoints.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.pareto import ParetoFront
    from shared.task import Objective


def passes_size_gate(metrics: dict, challenge) -> bool:
    """Hard gate with configurable tolerance for FLOPs measurement variance.

    Default 10% — analytical FLOPs counting (torch.utils.flop_counter) is
    exact for standard ops; wallclock calibration fallback for custom ops.
    """
    from config import Config
    tolerance = Config.SIZE_GATE_TOLERANCE
    flops = metrics.get("flops_equivalent_size", 0)
    effective_min = int(challenge.min_flops_equivalent * (1 - tolerance))
    effective_max = int(challenge.max_flops_equivalent * (1 + tolerance))
    return effective_min <= flops <= effective_max


def score_round(
    eval_results: dict[int, dict],
    challenge,
    frontier: "ParetoFront",
    objectives: list["Objective"],
    penalties: dict[int, float],
) -> dict[int, float]:
    """Score all miners in a round using Phase C eval results.

    1. Size gate: reject out-of-range (hard, score zero)
    2. Filter frontier to feasible region (this round's size bucket)
    3. No feasible frontier -> pure relative ranking (bootstrapping)
    4. Feasible frontier -> rank by improvement beyond frontier
    5. Someone always wins
    6. Apply penalties
    """
    from shared.database import DataElement

    scores: dict[int, float] = {}
    feasible: dict[int, dict] = {}

    # 1. Size gate
    for uid, metrics in eval_results.items():
        if not metrics.get("passed_size_gate", False):
            scores[uid] = 0.0
        elif metrics.get("crps") is None or not math.isfinite(metrics.get("crps", float("inf"))):
            scores[uid] = 0.0
        else:
            feasible[uid] = metrics

    if not feasible:
        return scores

    # 2. Get feasible frontier members
    feasible_front = frontier.get_feasible(
        challenge.min_flops_equivalent, challenge.max_flops_equivalent,
    )

    # 3/4. Rank by CRPS (primary metric, lower is better)
    ranked = sorted(feasible.items(), key=lambda x: x[1].get("crps", float("inf")))

    if not feasible_front:
        # Bootstrapping: pure relative ranking
        n = len(ranked)
        for rank, (uid, metrics) in enumerate(ranked):
            scores[uid] = 1.0 - (rank / max(n, 1))
    else:
        # Rank by improvement beyond frontier best CRPS
        best_front_crps = min(
            (c.element.objectives.get("crps", float("inf")) for c in feasible_front),
            default=float("inf"),
        )
        for uid, metrics in ranked:
            crps = metrics.get("crps", float("inf"))
            if best_front_crps > 0:
                improvement = (best_front_crps - crps) / max(abs(best_front_crps), 1e-8)
            else:
                improvement = 0.0
            scores[uid] = _sigmoid(improvement, steepness=20.0)

            # Pareto dominance bonus
            temp = DataElement(
                metric=crps, success=True,
                objectives=metrics,
            )
            if frontier.count_dominated_by(temp) > 0:
                scores[uid] *= 1.5

    # 6. Apply penalties
    for uid, penalty in penalties.items():
        if uid in scores:
            scores[uid] *= max(0.0, 1.0 - penalty)

    return scores


def compute_penalties(
    training_metas: dict[int, dict],
    eval_results: dict[int, dict],
) -> dict[int, float]:
    """Compute per-miner penalties.

    - Trainer claimed FLOPs doesn't match validator-measured FLOPs -> penalty on trainer
    - Trainer returned failure/timeout -> reliability penalty on trainer
    """
    penalties: dict[int, float] = {}

    for uid, meta in training_metas.items():
        status = meta.get("status", "")
        if status == "attestation_failed":
            trainer_uid = meta.get("trainer_uid", uid)
            penalties[trainer_uid] = penalties.get(trainer_uid, 0.0) + 1.0
        elif status in ("failed", "timeout", "build_failed"):
            trainer_uid = meta.get("trainer_uid", uid)
            penalties[trainer_uid] = penalties.get(trainer_uid, 0.0) + 0.5

        if uid in eval_results:
            ev = eval_results[uid]
            if not ev.get("flops_verified", True):
                trainer_uid = meta.get("trainer_uid", uid)
                penalties[trainer_uid] = penalties.get(trainer_uid, 0.0) + 0.3

    return penalties


def scores_to_weights(
    scores: dict[int, float],
    temperature: float = 0.1,
) -> tuple[list[int], list[float]]:
    """Softmax-normalize raw scores across miners.

    Low temperature skews rewards sharply toward top performers.
    Miners with raw score 0 remain at 0.

    Returns (uids, weights) where weights sum to 1.0.
    """
    positive = {uid: s for uid, s in scores.items() if s > 0}
    if not positive:
        uids = sorted(scores.keys())
        return uids, [0.0] * len(uids)

    max_s = max(positive.values())
    exp_scores = {
        uid: math.exp((s - max_s) / max(temperature, 1e-8))
        for uid, s in positive.items()
    }
    total = sum(exp_scores.values())

    uids = sorted(scores.keys())
    weights = []
    for uid in uids:
        if uid in exp_scores:
            weights.append(exp_scores[uid] / total)
        else:
            weights.append(0.0)
    return uids, weights


def ema_update(
    ema_scores: dict[int, float],
    round_scores: dict[int, float],
    all_uids: list[int],
    alpha: float = 0.3,
) -> dict[int, float]:
    """Update EMA scores with this round's results."""
    for uid in all_uids:
        score = round_scores.get(uid, 0.0)
        if uid in ema_scores:
            ema_scores[uid] = alpha * score + (1 - alpha) * ema_scores[uid]
        else:
            ema_scores[uid] = score
    return ema_scores


def _sigmoid(x: float, steepness: float = 20.0) -> float:
    """Sigmoid squash."""
    return 1.0 / (1.0 + math.exp(-steepness * x))
