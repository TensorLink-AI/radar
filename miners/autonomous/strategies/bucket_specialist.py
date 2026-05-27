"""Bucket specialist — evolve the best-known code for this FLOPs bucket."""

from core.history import extract_flops_budget, identify_bucket


def build_strategy(challenge: dict, state: dict) -> dict:
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = identify_bucket(flops_min, flops_max)

    saved_code = (state.get("templates") or {}).get(bucket, "")
    saved_metrics = (state.get("template_metrics") or {}).get(bucket, {})

    identity = (
        "You are a bucket specialist. You treat each FLOPs range as its "
        "own domain and maintain a best-known architecture for it. You "
        "don't start from scratch — you evolve your best previous code "
        "for THIS bucket, making targeted refinements that compound over "
        "rounds."
    )

    workflow = (
        "## Strategy Workflow\n"
        "- Call `read_scratchpad` to load your bucket archive.\n"
        "- If a prior best exists for this bucket, start there. Make "
        "small, measurable changes.\n"
        "- If no prior exists, build a careful baseline and save it.\n"
        "- Use `sketch_architecture` to keep FLOPs centered on target.\n"
        "- Save what worked to the scratchpad before submitting."
    )

    kickoff_lines = [
        f"Specialist focus: the `{bucket}` bucket "
        f"([{flops_min:,}, {flops_max:,}] FLOPs)."
    ]
    if saved_code:
        metric_str = (
            ", ".join(f"{k}={v}" for k, v in saved_metrics.items())
            if saved_metrics else "(no metrics recorded)"
        )
        kickoff_lines.append(
            f"You have a saved best for this bucket — prior metrics: "
            f"{metric_str}. Evolve it rather than replacing it wholesale."
        )
    else:
        kickoff_lines.append(
            "No saved best for this bucket yet — build a careful baseline "
            "and save it via `write_scratchpad` for future rounds."
        )

    return {
        "identity": identity,
        "kickoff_additions": "\n".join(kickoff_lines),
        "workflow_guidance": workflow,
        "temperature": 0.6,
    }
