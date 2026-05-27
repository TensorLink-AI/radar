"""Ensemble distiller — synthesize a model from the best parts of every frontier member."""


def build_strategy(challenge: dict, state: dict) -> dict:
    frontier = (
        challenge.get("feasible_frontier")
        or challenge.get("pareto_frontier")
        or []
    )

    identity = (
        "You read every frontier member's code deeply and identify what "
        "makes each one strong on its best metric. You synthesize ONE "
        "new model that combines the best operational choice per role — "
        "the encoder from the best primary-metric model, the head from "
        "the best secondary-metric model, etc. You are a distiller, not "
        "a mixer of parallel inferences — your output is a single "
        "coherent model."
    )

    workflow = (
        "## Strategy Workflow\n"
        "- `list_frontier` to see all members at a glance, then "
        "`get_frontier_member(index)` on each one you want to study.\n"
        "- For each member, identify: (a) which metric it wins on, and "
        "(b) the specific architectural choice responsible.\n"
        "- Sketch a synthesis with `sketch_architecture`, cherry-picking "
        "one choice per role. Check that the combined FLOPs still fit.\n"
        "- Expand into full code and `validate_code`.\n"
        "- Record which choices came from which member in a code "
        "comment so future rounds can trace the lineage."
    )

    kickoff = (
        f"Frontier has {len(frontier)} member(s). Study ALL of them. "
        "Your output should combine the best operational choices across "
        "members into a single coherent model."
    )

    return {
        "identity": identity,
        "kickoff_additions": kickoff,
        "workflow_guidance": workflow,
        "temperature": 0.8,
    }
