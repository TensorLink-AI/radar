"""Reliable baseline — always submit something that passes validation.

This strategy is the default when no frontier exists. It prioritizes
correctness over cleverness: a working baseline in every bucket is what
lets later strategies iterate.
"""


def build_strategy(challenge: dict, state: dict) -> dict:
    identity = (
        "You are the safety net. Your job is to ALWAYS submit code that "
        "passes validate_code. Favor simple, proven designs over clever "
        "ones. A submitted model that trains is worth infinitely more "
        "than a failed attempt at something sophisticated."
    )

    workflow = (
        "## Strategy Workflow\n"
        "- Call `analyze_task` first to understand shapes and budget.\n"
        "- Use `sketch_architecture` early with a minimal design — don't "
        "over-engineer.\n"
        "- Validate EARLY. Submit as soon as something passes. Polish "
        "only if time remains.\n"
        "- Never submit code that hasn't passed `validate_code`."
    )

    kickoff = (
        "No frontier exists yet for this bucket / task combination, so "
        "you are setting the first position. Aim for a reliable baseline "
        "that later rounds can build on."
    )

    return {
        "identity": identity,
        "kickoff_additions": kickoff,
        "workflow_guidance": workflow,
        "temperature": 0.5,
    }
