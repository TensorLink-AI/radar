"""Frontier sniper — copy the best frontier code, make ONE surgical change."""


def build_strategy(challenge: dict, state: dict) -> dict:
    frontier = (
        challenge.get("feasible_frontier")
        or challenge.get("pareto_frontier")
        or []
    )

    if not frontier:
        # Degrade gracefully to a bootstrap persona.
        return {
            "identity": (
                "You are a careful code reviewer. No frontier exists yet, "
                "so treat this round as bootstrapping: submit a strong, "
                "conservative baseline that later sniper rounds can iterate on."
            ),
            "kickoff_additions": (
                "No frontier available. Build a clean baseline so the "
                "next round has something to snipe."
            ),
            "workflow_guidance": (
                "## Strategy Workflow\n"
                "- `analyze_task` → `sketch_architecture` → `validate_code` → submit.\n"
                "- Favor simplicity; reserve cleverness for rounds where a "
                "frontier exists."
            ),
            "temperature": 0.5,
        }

    identity = (
        "You are a code reviewer, not an architect. The frontier already "
        "contains competitive code. Your job is to find the single most "
        "impactful improvement and make that one change. Copy the best "
        "frontier code almost entirely. Change ONE thing — a better "
        "optimizer config, init scheme, normalization, LR schedule, or "
        "regularization. Minimal diff, maximum impact."
    )

    workflow = (
        "## Strategy Workflow\n"
        "- Use `list_frontier` to see all members, then "
        "`get_frontier_member(index)` to read the best one's code.\n"
        "- Identify the ONE thing you will change. Write it down in a "
        "code comment.\n"
        "- Sketch it with `sketch_architecture` if your change alters "
        "FLOPs or shapes. Otherwise skip straight to `validate_code`.\n"
        "- Keep the model class IDENTICAL in structure. Only tune "
        "hyperparameters or training-dynamics hooks.\n"
        "- NEVER redesign the architecture. One change. That's it."
    )

    kickoff = (
        f"Frontier has {len(frontier)} member(s). Pick the best one on "
        "the primary metric, copy its code, and make one surgical "
        "improvement. Explain your single change in a code comment."
    )

    return {
        "identity": identity,
        "kickoff_additions": kickoff,
        "workflow_guidance": workflow,
        "temperature": 0.4,
    }
