"""Ablation scientist — one controlled experiment per round."""

from core.history import (
    extract_flops_budget,
    get_history,
    identify_bucket,
)


def _summarize_lab_notebook(state: dict, bucket: str) -> str:
    """Render a compact summary of past ablations for THIS bucket."""
    experiments = [
        e for e in get_history(state)
        if e.get("bucket") == bucket and e.get("strategy") == "ablation_scientist"
    ]
    notebook = state.get("ablation_notebook", {}).get(bucket, {})
    tested = list(notebook.get("tested", []))
    not_tried = list(notebook.get("not_tried", []))

    lines: list[str] = []
    if experiments:
        lines.append(f"Prior ablations in `{bucket}`: {len(experiments)}.")
        last = experiments[-5:]
        for e in last:
            motiv = e.get("motivation", "")
            name = e.get("name", "?")
            lines.append(f"  - {name}: {motiv}")
    else:
        lines.append(f"No prior ablations recorded for `{bucket}`.")
    if tested:
        lines.append("Tested variables: " + ", ".join(tested[-10:]))
    if not_tried:
        lines.append("Still untried: " + ", ".join(not_tried[:10]))
    return "\n".join(lines)


def build_strategy(challenge: dict, state: dict) -> dict:
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = identify_bucket(flops_min, flops_max)

    identity = (
        "You are an experimental scientist. Your scratchpad is your lab "
        "notebook. Each round, you formulate a hypothesis about ONE "
        "component change, test it, and record the result. You NEVER "
        "change more than one thing at a time — this lets you attribute "
        "success or failure to the specific change you made."
    )

    workflow = (
        "## Strategy Workflow\n"
        "- `read_scratchpad` — load the lab notebook and find your "
        "current baseline.\n"
        "- Pick ONE variable to change. State the hypothesis explicitly "
        "(what, why, expected direction of effect).\n"
        "- Apply the change to the baseline code.\n"
        "- Call `estimate_model_flops` on the changed version to confirm "
        "FLOPs didn't drift before running full validation.\n"
        "- `validate_code` → `submit`.\n"
        "- Record the change and its hypothesis via `write_scratchpad` "
        "as part of the round's notes, so future rounds know it was "
        "tested."
    )

    kickoff = (
        "Run ONE controlled experiment this round. Lab notebook summary:\n"
        + _summarize_lab_notebook(state, bucket)
        + "\n\nChange exactly one variable. Record the hypothesis. "
        "Submit."
    )

    return {
        "identity": identity,
        "kickoff_additions": kickoff,
        "workflow_guidance": workflow,
        "temperature": 0.7,
    }
