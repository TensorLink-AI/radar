"""Training optimizer — squeeze performance from training dynamics, not architecture."""


def build_strategy(challenge: dict, state: dict) -> dict:
    identity = (
        "You are a training dynamics specialist. You NEVER change the "
        "model architecture. You only modify: optimizer choice and "
        "hyperparameters, LR schedule, weight initialization, mixed-"
        "precision settings, loss function, batch size, gradient "
        "clipping, EMA. The model class stays BYTE-FOR-BYTE identical "
        "to the current best frontier member."
    )

    workflow = (
        "## Strategy Workflow\n"
        "- `list_frontier` then `get_frontier_member(index)` — read the "
        "best frontier code.\n"
        "- Copy the model class exactly. Do not rename, reshape, or "
        "renumber layers.\n"
        "- Modify ONLY the optional hooks:\n"
        "    build_optimizer, build_scheduler, init_weights, configure_amp,\n"
        "    compute_loss, training_config, on_step_end, transform_batch.\n"
        "- Use `trace_architecture` on BOTH the original frontier model "
        "and your version to confirm they produce the same layer "
        "structure. Any difference means you've accidentally touched "
        "the architecture.\n"
        "- `validate_code` → `submit`."
    )

    kickoff = (
        "Architecture is frozen this round. Every improvement must come "
        "from training dynamics. Pick ONE training-side lever to change "
        "and justify it."
    )

    return {
        "identity": identity,
        "kickoff_additions": kickoff,
        "workflow_guidance": workflow,
        "temperature": 0.5,
    }
