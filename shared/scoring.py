"""Size-gated Pareto frontier scoring for Phase C evaluation.

Uses validator-computed metrics (from Phase C evaluate_checkpoint) rather than
trainer-reported metrics. This is the trust anchor — every validator computes
identical scores from identical checkpoints.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.pareto import ParetoFront
    from shared.task import Objective

logger = logging.getLogger(__name__)


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


def _primary_objective(objectives: list["Objective"]) -> "Objective | None":
    """Pick the primary Objective (or first, if none marked primary)."""
    if not objectives:
        return None
    for obj in objectives:
        if obj.primary:
            return obj
    return objectives[0]


def _signed_improvement(current: float, best: float, lower_is_better: bool) -> float:
    """Fractional improvement of `current` over `best`, direction-aware.

    Positive means `current` is better than `best`.
    """
    denom = max(abs(best), 1e-8)
    delta = (best - current) if lower_is_better else (current - best)
    return delta / denom


def score_round(
    eval_results: dict[int, dict],
    challenge,
    frontier: "ParetoFront",
    objectives: list["Objective"],
    penalties: dict[int, float],
    training_metas: dict[int, dict] | None = None,
) -> dict[int, float]:
    """Score all miners in a round using Phase C eval results.

    Drives off TaskSpec.objectives — the primary Objective supplies the
    metric name and direction (lower/higher is better).

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

    primary = _primary_objective(objectives)
    if primary is None:
        # Nothing to rank against — everyone gets 0.
        for uid in eval_results:
            scores[uid] = 0.0
        return scores

    primary_name = primary.name
    lower_is_better = primary.lower_is_better
    worst = primary.default

    # 1. Size gate
    for uid, metrics in eval_results.items():
        if not metrics.get("passed_size_gate", False):
            scores[uid] = 0.0
            logger.info(
                "UID %d scored 0: failed size gate "
                "(flops=%d, allowed=[%d, %d], error=%s)",
                uid, metrics.get("flops_equivalent_size", 0),
                challenge.min_flops_equivalent,
                challenge.max_flops_equivalent,
                metrics.get("error", ""),
            )
        elif (
            metrics.get(primary_name) is None
            or not math.isfinite(metrics.get(primary_name, worst))
        ):
            scores[uid] = 0.0
            logger.info(
                "UID %d scored 0: invalid %s=%s error=%s",
                uid, primary_name, metrics.get(primary_name),
                metrics.get("error", ""),
            )
        else:
            feasible[uid] = metrics

    if not feasible:
        return scores

    # 2. Get feasible frontier members
    feasible_front = frontier.get_feasible(
        challenge.min_flops_equivalent, challenge.max_flops_equivalent,
    )

    # 3/4. Rank by primary metric; flip sign so "best first" is always ascending.
    sort_sign = 1.0 if lower_is_better else -1.0
    ranked = sorted(
        feasible.items(),
        key=lambda x: x[1].get(primary_name, worst) * sort_sign,
    )

    if not feasible_front:
        # Bootstrapping: pure relative ranking
        n = len(ranked)
        for rank, (uid, metrics) in enumerate(ranked):
            scores[uid] = 1.0 - (rank / max(n, 1))
    else:
        # Rank by improvement beyond frontier best
        from config import Config
        threshold = Config.FRONTIER_IMPROVEMENT_THRESHOLD
        front_values = [
            c.element.objectives.get(primary_name, worst) for c in feasible_front
        ]
        best_front = min(front_values) if lower_is_better else max(front_values)
        for uid, metrics in ranked:
            current = metrics.get(primary_name, worst)
            if abs(best_front) > 1e-9:
                improvement = _signed_improvement(current, best_front, lower_is_better)
            else:
                improvement = 0.0

            # Gate: must beat the feasible frontier's best by at least
            # `threshold` (fractional) to earn a score. Ties and regressions → 0.
            if improvement < threshold:
                scores[uid] = 0.0
                logger.info(
                    "UID %d scored 0: %s %.6f vs frontier best %.6f "
                    "(improvement %.4f%% < threshold %.4f%%)",
                    uid, primary_name, current, best_front,
                    improvement * 100.0, threshold * 100.0,
                )
                continue

            scores[uid] = _sigmoid(improvement, steepness=20.0)

            # Pareto dominance bonus — merge training meta for full objective vector
            merged_objectives = dict(metrics)
            if training_metas and uid in training_metas:
                tm = training_metas[uid]
                merged_objectives.setdefault("exec_time", tm.get("training_time_seconds", float("inf")))
                merged_objectives.setdefault("memory_mb", tm.get("peak_vram_mb", float("inf")))
            temp = DataElement(
                metric=current, success=True,
                objectives=merged_objectives,
            )
            if frontier.count_dominated_by(temp) > 0:
                scores[uid] *= 1.5

    # 6. Apply penalties
    for uid, penalty in penalties.items():
        if uid in scores:
            scores[uid] *= max(0.0, 1.0 - penalty)

    return scores


def apply_round_metadata(
    scores: dict[int, float],
    round_metadata: dict | None,
) -> dict[int, float]:
    """Apply Targon migration's hybrid-fallback policy to per-miner scores.

    ``round_metadata`` is the dict produced by
    ``TrainingCoordinator.round_metadata`` — it contains:

      - ``targon_unavailable``: list of UIDs verified during a Targon
        outage. Their score gets multiplied by
        ``Config.TARGON_UNAVAILABLE_SCORE_MULTIPLIER`` (default 0.5)
        so the round still progresses without fully rewarding miners
        we couldn't attest in real-time.
      - ``compromised``: list of UIDs that failed mid-round
        re-verification. Excluded from scoring entirely (score → 0).

    Returns a NEW dict; the input ``scores`` is not mutated.
    """
    if not round_metadata:
        return dict(scores)
    from config import Config
    multiplier = float(Config.TARGON_UNAVAILABLE_SCORE_MULTIPLIER)
    targon_unavailable = set(round_metadata.get("targon_unavailable") or [])
    compromised = set(round_metadata.get("compromised") or [])

    out: dict[int, float] = {}
    for uid, score in scores.items():
        if uid in compromised:
            out[uid] = 0.0
            continue
        if uid in targon_unavailable:
            out[uid] = score * multiplier
            continue
        out[uid] = score
    return out


def compute_penalties(
    training_metas: dict[int, dict],
    eval_results: dict[int, dict],
) -> dict[int, float]:
    """Compute per-miner penalties keyed by arch_owner UID.

    Penalties apply to the architecture owner (the miner being scored),
    not the trainer (who may be a different miner in cross-eval).
    """
    penalties: dict[int, float] = {}

    for uid, meta in training_metas.items():
        status = meta.get("status", "")
        if status == "attestation_failed":
            penalties[uid] = min(1.0, penalties.get(uid, 0.0) + 1.0)
        elif status in ("failed", "timeout", "build_failed", "size_violation"):
            penalties[uid] = min(1.0, penalties.get(uid, 0.0) + 0.5)

        if uid in eval_results:
            ev = eval_results[uid]
            if not ev.get("flops_verified", True):
                penalties[uid] = min(1.0, penalties.get(uid, 0.0) + 0.3)

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
