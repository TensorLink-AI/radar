"""Pareto hunter — multi-objective optimizer targeting the 1.5x dominance bonus."""


def _format_objectives(challenge: dict) -> tuple[str, str]:
    """Return (primary_name, secondary_summary) from challenge['task']['objectives']."""
    objectives = (challenge.get("task", {}) or {}).get("objectives", []) or []
    primary = ""
    secondary: list[str] = []
    for o in objectives:
        if not isinstance(o, dict):
            continue
        name = str(o.get("name", ""))
        if not name:
            continue
        if o.get("primary"):
            primary = name
        else:
            secondary.append(name)
    if not primary and objectives and isinstance(objectives[0], dict):
        primary = str(objectives[0].get("name", ""))
    secondary_summary = ", ".join(secondary) if secondary else "(none declared)"
    return primary, secondary_summary


def build_strategy(challenge: dict, state: dict) -> dict:
    primary, secondary = _format_objectives(challenge)

    identity = (
        "You are a multi-objective optimizer hunting the 1.5x Pareto "
        "dominance bonus. Most miners chase only the primary metric — "
        "you target ALL of them simultaneously. A model that merely "
        "matches the frontier on the primary metric but dominates on "
        "secondary objectives earns the 1.5x multiplier."
    )

    obj_block = (
        f"This task's primary objective is `{primary or 'unknown'}`; "
        f"secondary objectives: {secondary}. ALL objective names are "
        "read from `challenge['task']['objectives']` — never assume "
        "fixed metric names."
    )

    workflow = (
        "## Strategy Workflow\n"
        "- `list_frontier` to see metrics across all members; use "
        "`get_frontier_member(index)` to inspect the weakest one on each "
        "secondary objective.\n"
        "- Decide which secondary metrics you can realistically improve "
        "given the task's objective list (read from the challenge).\n"
        "- Sketch a design with `sketch_architecture`; verify it fits "
        "the budget AND has the optional hooks that affect secondary "
        "objectives (configure_amp, training_config, compute_loss, "
        "init_weights) as applicable.\n"
        "- Submit only once `validate_code` passes."
    )

    kickoff = (
        "Target Pareto DOMINATION. "
        + obj_block
        + " Your model must tie-or-beat the frontier on the primary and "
        "strictly beat it on the secondary metrics that the task "
        "actually measures."
    )

    return {
        "identity": identity,
        "kickoff_additions": kickoff,
        "workflow_guidance": workflow,
        "temperature": 0.7,
    }
