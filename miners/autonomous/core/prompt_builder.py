"""Build rich context prompts for the LLM.

All harness parameters are read from the challenge dict generically.
No task-specific parameter names are hardcoded — the prompt adapts to
whatever task_params the challenge provides.
"""

from core.history import extract_flops_budget, format_history


def _format_task_params(tp: dict) -> str:
    """Format task_params as key=value pairs for display."""
    if not tp:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in tp.items())


def _compute_bucket_strategy(challenge: dict, bucket: str) -> str:
    """Generate architecture-agnostic sizing guidance from the challenge budget.

    Instead of prescribing specific architectures, computes budget-derived
    constraints the LLM can use to size whatever model it builds.
    """
    flops_min, flops_max = extract_flops_budget(challenge)
    target = int(flops_max * 0.6) if flops_max else 0

    return (
        f"For this '{bucket}' bucket ({flops_min:,}-{flops_max:,} FLOPs), "
        f"target ~{target:,} FLOPs (60% of max). "
        f"Choose whatever architecture best fits this budget — the estimator "
        f"measures actual forward-pass FLOPs, so any standard PyTorch ops work. "
        f"More complex architectures must budget FLOPs across their layers."
    )


def _compute_sizing_guidance(challenge: dict) -> str:
    """Build architecture-agnostic FLOPs calculator and compliance protocol."""
    tp = challenge.get("task", {}).get("task_params", {})
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    flops_min, flops_max = extract_flops_budget(challenge)
    target = int(flops_max * 0.6) if flops_max else 0
    gate_min = int(flops_min * 0.9) if flops_min else 0
    gate_max = int(flops_max * 1.1) if flops_max else 0

    lines = [
        "## FLOPs Compliance Protocol (CRITICAL — read this carefully)",
        "",
        "Your model WILL be measured with `torch.utils.flop_counter.FlopCounterMode`. "
        "This is the EXACT same measurement the validator uses. The `validate_code` tool "
        "runs this exact measurement. If it says your model is too large or too small, "
        "you MUST fix it before submitting — otherwise the validator will reject it "
        "with score=0.",
        "",
        "**Rules:**",
        "1. Use standard nn ops (nn.Linear, nn.Conv1d, nn.MultiheadAttention, etc.) "
        "so the FLOPs counter can see them",
        "2. ALWAYS call `validate_code` before `submit` — it runs the real FLOPs check",
        "3. If validate_code says FLOPs are off, follow its resize suggestion",
        f"4. Target ~{target:,} FLOPs (60% of max). Hard gate: [{gate_min:,}, {gate_max:,}]",
        "",
        "## FLOPs Calculator",
        "",
        "**FLOPs formulas for common ops:**",
        "- `nn.Linear(in, out)` on input `(batch, seq, in)` -> `2 * in * out * seq` FLOPs",
        "- `nn.Conv1d(C_in, C_out, K)` on length L -> `2 * C_in * K * C_out * L_out` FLOPs",
        "- `nn.MultiheadAttention(d, heads)` on seq S -> `8 * S * d^2 + 2 * S^2 * d` FLOPs",
        "- Note: if your model reshapes to `(batch * V, ...)`, V is already in the batch "
        "-- do NOT double-count",
        "",
        "**Budget summary:**",
        f"- Target FLOPs: {target:,} (60% of max)",
        f"- Hard gate: [{gate_min:,}, {gate_max:,}]",
        f"- Task params: {_format_task_params(tp)}",
        "",
        "## Self-Sizing Pattern (MANDATORY — embed this in your build_model)",
        "",
        "Your `build_model()` MUST compute layer dimensions dynamically from an embedded "
        "FLOPs budget constant. Do NOT hardcode hidden dims — derive them from the budget. "
        "This is the #1 reason models get rejected: hardcoded dimensions that don't match "
        "the FLOPs budget.",
        "",
        "```python",
        f"def build_model({param_str}):",
        f"    TARGET_FLOPS = {target}  # 60% of max budget",
        "",
        "    # 1. Compute input/output features from your task_params",
        "    # 2. Estimate FLOPs per hidden unit: flops_per_h = 2 * (in_feat + out_feat)",
        "    # 3. Derive: hidden_dim = max(4, TARGET_FLOPS // flops_per_h)",
        "    hidden_dim = max(4, TARGET_FLOPS // flops_per_hidden)",
        "",
        f"    return MyModel({param_str}, hidden_dim)",
        "```",
        "",
        "This ensures the model auto-sizes to any budget. "
        "For multi-layer models, divide the budget across layers accordingly.",
    ]
    return "\n".join(lines)


def build_system_prompt(challenge: dict) -> str:
    """Build system prompt with domain context and hard constraints."""
    task = challenge.get("task", {})
    tp = task.get("task_params", {})
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    task_name = task.get("name", "unknown")
    parts: list[str] = []

    parts.append(
        f"You are an expert ML researcher designing models for the '{task_name}' task "
        "in a competitive evaluation.\n\n"
        "Your code runs inside a frozen training harness. You MUST define:\n"
        f"- def build_model({param_str}) -> nn.Module\n"
        "- def build_optimizer(model) -> Optimizer\n\n"
        f"The harness calls build_model with these parameters: {_format_task_params(tp)}.\n"
        "See the task description and constraints below for I/O shape requirements."
    )

    domain = task.get("domain_system_prompt", "")
    if domain:
        parts.append(f"### Domain Context\n{domain}")

    constraints = task.get("constraints", [])
    if constraints:
        parts.append("### Constraints\n" + "\n".join(f"- {c}" for c in constraints))

    objectives = task.get("objectives", [])
    if objectives:
        parts.append("### Objectives\n" + "\n".join(f"- {o}" for o in objectives))

    anti_patterns = task.get("anti_patterns", [])
    if anti_patterns:
        parts.append("### Anti-Patterns to Avoid\n"
                      + "\n".join(f"- {a}" for a in anti_patterns))

    parts.append(
        "### ABSOLUTE RULES\n"
        "- Respond with a SINGLE ```python code block — nothing else\n"
        "- build_model and build_optimizer MUST be top-level functions "
        "(not inside a class)\n"
        "- Use only standard PyTorch (torch, torch.nn, torch.optim, "
        "torch.nn.functional)\n"
        "- Do NOT import subprocess, socket, or ftplib\n"
        "- Use standard nn ops (nn.Linear, nn.Conv1d, nn.TransformerEncoderLayer, "
        "nn.MultiheadAttention, nn.LayerNorm) so FLOPs counter works"
    )

    return "\n\n".join(parts)


def build_user_prompt(challenge: dict, context: dict) -> str:
    """Build the user prompt with budget, frontier, DB context, and history."""
    parts: list[str] = []

    tp = challenge.get("task", {}).get("task_params", {})
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    call_args = ", ".join(f"{k}={v!r}" for k, v in tp.items()) if tp else "..."
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = context.get("bucket", "unknown")
    target = context.get("target_flops", int(flops_max * 0.6))
    gate_min = int(flops_min * 0.9) if flops_min else 0
    gate_max = int(flops_max * 1.1) if flops_max else 0

    parts.append(
        f"## FLOPs Budget\n"
        f"Bucket: {bucket} [{flops_min:,} - {flops_max:,}]\n"
        f"Target: ~{target:,} FLOPs (60% of max)\n"
        f"Hard gate: [{gate_min:,}, {gate_max:,}] (10% tolerance)"
    )

    # Required interface with values from the challenge
    parts.append(
        "## Required Output Structure\n"
        "Your code MUST define these two top-level functions (not inside a class):\n\n"
        f"1. `def build_model({param_str}):`\n"
        f"   - Called as: build_model({call_args})\n"
        "   - Must return an nn.Module\n"
        "   - See the task description and constraints for I/O shape details\n\n"
        "2. `def build_optimizer(model):`\n"
        "   - Takes the model returned by build_model()\n"
        "   - Must return a torch.optim.Optimizer\n\n"
        "Optional hooks: training_config(), init_weights(), configure_amp(), "
        "transform_batch(), on_step_end(), build_scheduler(), compute_loss()\n\n"
        "Minimal skeleton:\n"
        "```python\n"
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class YourModel(nn.Module):\n"
        f"    def __init__(self, {param_str}):\n"
        "        super().__init__()\n"
        "        # ... your architecture ...\n"
        "    def forward(self, x):\n"
        "        # See task description/constraints for I/O shapes\n"
        "        ...\n\n"
        f"def build_model({param_str}):\n"
        f"    return YourModel({param_str})\n\n"
        "def build_optimizer(model):\n"
        "    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)\n"
        "```\n\n"
        f"Task parameters: {_format_task_params(tp)}"
    )

    # Dynamic FLOPs calculator
    parts.append(_compute_sizing_guidance(challenge))

    # Frontier context
    frontier = context.get("frontier", [])
    if frontier:
        parts.append(format_frontier(frontier, max_entries=3))
    else:
        parts.append("## Current Frontier\nNo frontier exists yet — you are bootstrapping.")

    # DB context
    db_ctx = format_db_context(
        context.get("recent_experiments", {}),
        context.get("failures", {}),
        context.get("component_stats", {}),
        context.get("dead_ends", {}),
    )
    if db_ctx != "No DB context available.":
        parts.append(f"## DB Insights\n{db_ctx}")

    # History
    history_entries = context.get("history", [])
    hist_text = format_history(history_entries, max_entries=5)
    if hist_text != "No previous submissions.":
        parts.append(f"## Your Previous Submissions\n{hist_text}")

    # Tool-assisted research findings
    tool_analysis = context.get("tool_analysis", "")
    if tool_analysis:
        parts.append(f"## Database Research Findings\n{tool_analysis}")

    # Strategy — computed from the challenge, not a hardcoded dict
    strategy = _compute_bucket_strategy(challenge, bucket)
    if strategy:
        parts.append(f"## Strategy\n{strategy}")

    parts.append(
        "## Output Format\n"
        "Return ONLY a single ```python code block with the complete code. "
        "Include optional hooks (training_config, build_scheduler, init_weights) "
        "for best training dynamics."
    )

    return "\n\n".join(parts)


def format_frontier(frontier: list[dict], max_entries: int = 3) -> str:
    """Format frontier members for inclusion in prompt."""
    lines: list[str] = ["## Current Frontier (models you must beat)"]
    for i, member in enumerate(frontier[:max_entries]):
        metrics = member.get("objectives", {})
        metric_parts = ", ".join(f"{k}={v}" for k, v in metrics.items())
        lines.append(f"**Member {i + 1}**: {metric_parts}")
        code = member.get("code", "")
        if code:
            if len(code) > 4000:
                code = code[:4000] + "\n# ... (truncated)"
            lines.append(f"```python\n{code}\n```")
    return "\n\n".join(lines)


def format_db_context(recent: dict | list, failures: dict | list,
                      component_stats: dict | list,
                      dead_ends: dict | list) -> str:
    """Format DB query results into a compact context string."""
    parts: list[str] = []

    if recent:
        items = recent if isinstance(recent, list) else recent.get("experiments", [])
        if items:
            parts.append("**Recent experiments:**")
            for exp in items[:5]:
                m = exp.get("metrics", {})
                metric_str = ", ".join(f"{k}={v}" for k, v in m.items()) if m else "no metrics"
                flops = exp.get('flops', '')
                name = exp.get('name', '?')
                parts.append(f"- {name}: {metric_str}" + (f", flops={flops}" if flops else ""))

    if failures:
        items = failures if isinstance(failures, list) else failures.get("failures", [])
        if items:
            parts.append("**Recent failures:**")
            for f in items[:3]:
                parts.append(f"- {f.get('name', '?')}: {f.get('reason', '?')}")

    if component_stats:
        items = (component_stats if isinstance(component_stats, list)
                 else component_stats.get("components", []))
        if items:
            parts.append("**Component correlations:**")
            for c in (items[:5] if isinstance(items, list) else []):
                parts.append(
                    f"- {c.get('name', '?')}: "
                    f"success_rate={c.get('success_rate', '?')}")

    if dead_ends:
        items = (dead_ends if isinstance(dead_ends, list)
                 else dead_ends.get("dead_ends", []))
        if items:
            parts.append("**Dead ends:**")
            for d in (items[:3] if isinstance(items, list) else []):
                parts.append(
                    f"- {d.get('pattern', '?')}: {d.get('reason', '?')}")

    return "\n".join(parts) if parts else "No DB context available."
