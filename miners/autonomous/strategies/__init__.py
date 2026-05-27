"""Strategy framework for the autonomous agent.

Each strategy is a personality prompt — NOT a separate code path. It
returns four fields (identity, kickoff_additions, workflow_guidance,
temperature) that are blended into the generic system prompt. The
validation loop, tools, fallback chain, and scratchpad are all shared.

``select_strategy`` picks a default per round via a simple priority
chain. Override by setting ``state['strategy_override']`` in the
scratchpad.

Strategies that want richer context (like previous code for a bucket)
should store what they need in the shared scratchpad state — they never
modify the loop itself.
"""

from core.history import (
    extract_flops_budget,
    get_history,
    identify_bucket,
)

from . import (
    ablation_scientist,
    bucket_specialist,
    ensemble_distiller,
    frontier_sniper,
    pareto_hunter,
    reliable_baseline,
    simple_modeler,
    training_optimizer,
)


STRATEGY_BUILDERS = {
    "bucket_specialist":   bucket_specialist.build_strategy,
    "frontier_sniper":     frontier_sniper.build_strategy,
    "pareto_hunter":       pareto_hunter.build_strategy,
    "simple_modeler":      simple_modeler.build_strategy,
    "reliable_baseline":   reliable_baseline.build_strategy,
    "ablation_scientist":  ablation_scientist.build_strategy,
    "training_optimizer":  training_optimizer.build_strategy,
    "ensemble_distiller":  ensemble_distiller.build_strategy,
}


def _current_bucket(challenge: dict) -> str:
    flops_min, flops_max = extract_flops_budget(challenge)
    return identify_bucket(flops_min, flops_max)


def _metric_plateaued(bucket_entries: list[dict], window: int = 3,
                      eps: float = 0.02) -> bool:
    """True when the last ``window`` bucket entries show <eps relative gain.

    History entries don't record the primary metric directly (to avoid
    leaking private scoring), but they record a ``flops_target`` per
    round. We proxy plateau by checking whether the last ``window``
    entries came from the same strategy with near-identical targets — a
    strong signal that the architecture search has converged.
    """
    if len(bucket_entries) < window:
        return False
    recent = bucket_entries[-window:]
    strategies = {e.get("strategy", "") for e in recent}
    if len(strategies) > 1:
        return False
    targets = [int(e.get("flops_target") or 0) for e in recent]
    if not targets or not targets[0]:
        return False
    spread = (max(targets) - min(targets)) / max(1, targets[0])
    return spread <= eps


def select_strategy(challenge: dict, state: dict) -> str:
    """Pick the strategy for this round.

    Priority chain (first match wins):
      0. explicit override in state['strategy_override']
      1. no frontier → reliable_baseline
      2. >=5 entries in bucket AND plateaued → training_optimizer
      3. >=5 entries in bucket → ablation_scientist
      4. frontier >=3 → ensemble_distiller
      5. have prior code in bucket → frontier_sniper
      6. fallback → simple_modeler

    ``bucket_specialist`` and ``pareto_hunter`` are opt-in via
    ``strategy_override`` — they are not default choices.
    """
    override = state.get("strategy_override", "")
    if override in STRATEGY_BUILDERS:
        return override

    frontier = (
        challenge.get("feasible_frontier")
        or challenge.get("pareto_frontier")
        or []
    )
    history_entries = get_history(state)
    bucket = _current_bucket(challenge)
    bucket_entries = [e for e in history_entries if e.get("bucket") == bucket]

    if not frontier:
        return "reliable_baseline"
    if len(bucket_entries) >= 5:
        if _metric_plateaued(bucket_entries):
            return "training_optimizer"
        return "ablation_scientist"
    if len(frontier) >= 3:
        return "ensemble_distiller"
    if any(e.get("code_length", 0) for e in bucket_entries):
        return "frontier_sniper"
    return "simple_modeler"


def build_strategy(name: str, challenge: dict, state: dict) -> dict:
    """Build the strategy dict for ``name``. Returns a 4-field dict."""
    builder = STRATEGY_BUILDERS.get(name)
    if builder is None:
        builder = simple_modeler.build_strategy
    result = builder(challenge, state) or {}
    result.setdefault("identity", "")
    result.setdefault("kickoff_additions", "")
    result.setdefault("workflow_guidance", "")
    result.setdefault("temperature", 0.7)
    return result
