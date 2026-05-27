"""Per-subagent system / user prompt builders.

Ported from ``agents/openai_sdk/prompts.py`` and restructured for the
subagent split. The FLOPs / sizing / code-requirements sections are
copied verbatim because they're well-tuned; the researcher and critic
prompts are bespoke (the openai_sdk single-loop prompt is too broad
to reuse for a researcher whose only output is a JSON brief).

Size targets (loosely enforced by tests, not asserted here):
  * researcher: ~3-4k chars  (small tool surface, plan-only output)
  * designer:   ~6-8k chars  (full code-shape + hooks + critic context)
  * critic:     ~500 chars   (three-line template)
"""
from __future__ import annotations

import json

from core.history import extract_flops_budget, identify_bucket
from core.prompt_builder import _compute_sizing_guidance, _format_task_params


# ── Brief schema (researcher output contract) ─────────────────────────

BRIEF_SCHEMA_EXAMPLE = {
    "relevant_prior_work": [
        "PatchTST (Nie 2022) — patch-level transformer with shared "
        "channel head",
    ],
    "frontier_gaps": [
        "current frontier members all use full attention; depthwise "
        "convs are absent",
    ],
    "ideas_to_try": [
        "depthwise-separable conv1d backbone with linear head",
        "patch the input then apply a small MLP-mixer",
    ],
    "plan": [
        "sketch_architecture for a depthwise-sep conv baseline",
        "size the hidden dim with size_to_flops to land mid-bucket",
        "validate_code, then submit with a hypothesis note",
    ],
}


# ── Researcher prompts ────────────────────────────────────────────────

def build_researcher_system_prompt(
    challenge: dict, bucket: str | None = None,
) -> str:
    """Researcher system prompt.

    Goal: a JSON brief the designer can act on. The researcher does
    NOT write code, sketch architectures, or call validate_code /
    submit (those tools aren't even in its surface). Its only job is
    to surface what's been tried, what's missing, and a 3-5 step
    plan.
    """
    task = challenge.get("task", {}) or {}
    task_name = task.get("name", "unknown")
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = bucket or identify_bucket(flops_min, flops_max)
    target = int(flops_max * 0.6) if flops_max else 0

    parts: list[str] = []

    parts.append(
        "You are the **researcher subagent** in a multi-agent miner "
        "competing on the Radar subnet. The orchestrator runs you "
        "first, then hands your output to a designer subagent that "
        "writes the actual code.\n\n"
        f"Task: `{task_name}`. Bucket: `{bucket}`. "
        f"FLOPs range: [{flops_min:,}, {flops_max:,}] (target ~{target:,})."
    )

    parts.append(
        "## Your output (the brief)\n\n"
        "Produce a single JSON object with these keys. The designer "
        "reads this verbatim — be specific, not generic.\n\n"
        "- `relevant_prior_work`: list of strings — papers / methods "
        "you found that bear on this task. Brief citations + the "
        "key idea, not abstracts.\n"
        "- `frontier_gaps`: list of strings — what the current "
        "frontier is *missing* (architectures absent, ideas un-tried, "
        "objectives no member optimizes for).\n"
        "- `ideas_to_try`: list of strings — concrete architectural "
        "ideas the designer could implement. Each one should fit the "
        "FLOPs bucket.\n"
        "- `plan`: list of 3-5 short strings — the exact sequence of "
        "tool calls / decisions you'd run if you were the designer.\n\n"
        "Example shape:\n```json\n"
        + json.dumps(BRIEF_SCHEMA_EXAMPLE, indent=2)
        + "\n```"
    )

    parts.append(
        "## Tools available to you\n\n"
        "- `analyze_task` — task spec, params, constraints, "
        "objectives, FLOPs budget as JSON. Call this first.\n"
        "- `list_frontier` — current best models on this bucket. "
        "Read for ideas, not to copy.\n"
        "- `cognition_wiki_index` / `cognition_wiki_read` — curated "
        "task-specific design recipes. Try the wiki before "
        "`search_papers` — it's cheaper and pre-filtered for this task.\n"
        "- `query_db` — experiment DB. Useful paths: "
        "`/frontier?task=`, `/experiments/recent?n=`, "
        "`/experiments/pareto?task=`, `/experiments/failures?n=`, "
        "`/experiments/families?task=`, `/experiments/stats?task=`, "
        "`/experiments/{idx}`, `/experiments/{idx}/diff`, "
        "`/experiments/lineage/{idx}`, "
        "`/provenance/component_stats`, `/provenance/dead_ends`, "
        "`/provenance/{idx}/similar?top_k=` (novelty check). "
        "POST `/experiments/search` body `{\"query\": \"...\"}`. "
        "~60 calls/min; calls are logged on the public dashboard.\n"
        "- `search_papers` — arxiv search. Use sparingly — papers "
        "are expensive and the LLM context is bounded.\n"
        "- `time_remaining` — seconds left in your slice of the "
        "round. The orchestrator caps you at 90s; budget accordingly."
    )

    parts.append(
        "## Principles\n\n"
        "- **Beat the frontier, don't match it.** A tie loses the "
        "Pareto dominance bonus. Your `frontier_gaps` should make "
        "this concrete.\n"
        "- **Concrete > abstract.** \"Try a transformer\" is "
        "useless. \"Try PatchTST with patch_len=16, FLOPs target "
        f"{target:,}\" is actionable.\n"
        "- **Don't write code.** That's the designer's job. If you "
        "find yourself sketching architectures, stop and rephrase as "
        "a plan step.\n"
        "- **Stop early.** A two-tool brief is fine if you already "
        "know the task. Spend tools when you don't."
    )

    parts.append(
        "## Final turn\n\n"
        "When you're ready, return ONLY a single fenced ```json block "
        "with the brief. No prose around it, no other commentary in "
        "that final message. The orchestrator parses your last "
        "assistant message — anything else costs context for nothing."
    )

    return "\n\n".join(parts)


def build_researcher_user_prompt(challenge: dict) -> str:
    """Kickoff message for the researcher."""
    task = challenge.get("task", {}) or {}
    frontier = (
        challenge.get("feasible_frontier")
        or challenge.get("pareto_frontier")
        or []
    )
    return (
        f"Begin researching the `{task.get('name', 'unknown')}` task. "
        f"Frontier currently has {len(frontier)} model(s).\n\n"
        "Use the research tools as needed, then return your brief as "
        "a single fenced ```json block."
    )


# ── Designer prompts ──────────────────────────────────────────────────

def build_designer_system_prompt(
    challenge: dict, bucket: str | None = None,
) -> str:
    """Designer system prompt.

    Carries the FLOPs compliance protocol, code-shape rules, and
    optional-hooks list verbatim from openai_sdk/prompts.py — those
    are well-tuned and a re-write would just regress them. The
    designer-specific additions are: critic feedback contract,
    submit hook contract, and the smaller tool surface.
    """
    task = challenge.get("task", {}) or {}
    tp = task.get("task_params", {}) or {}
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = bucket or identify_bucket(flops_min, flops_max)
    target = int(flops_max * 0.6) if flops_max else 0
    gate_min = int(flops_min * 0.9) if flops_min else 0
    gate_max = int(flops_max * 1.1) if flops_max else 0

    parts: list[str] = []

    parts.append(
        "You are the **designer subagent** in a multi-agent miner. "
        "The researcher already ran and handed you a brief. Your "
        "job: implement one of the brief's ideas, validate it lands "
        "in the FLOPs gate, and submit. You own the code — the "
        "researcher does not."
    )

    parts.append(
        "## Tools available to you\n\n"
        "- `sketch_architecture` — probe a `build_model`: FLOPs, "
        "per-layer trace, output-shape check. Cheaper than "
        "`validate_code`. Returns a `candidate_id` (cand_<hex>) "
        "that `validate_code` and `submit` accept in place of "
        "source.\n"
        "- `estimate_layer_flops` — forward-pass FLOPs for one "
        "layer. Useful for budgeting before you commit to a full "
        "model.\n"
        "- `validate_code` — final pre-submission check. **Required "
        "before submit** (see hook below). Accepts a `candidate_id`.\n"
        "- `submit` — ship in one call: "
        "`submit(candidate_id=..., name=..., motivation=..., "
        "note=...)`. The harness blocks this call until "
        "`validate_code` has returned `ok` within the last 3 turns.\n"
        "- `time_remaining` — seconds left in your slice of the round."
    )

    parts.append(
        "## How the round flows\n\n"
        "1. Read the researcher brief (in the user message).\n"
        "2. Pick one idea. Sketch + size if you need to land FLOPs.\n"
        "3. `validate_code`. If it fails, revise — the FLOPs counter "
        "tells you exactly which way to move.\n"
        "4. After every `validate_code` call you'll receive a critic "
        "message in the next user turn formatted as "
        "`KEEP / CHANGE / DROP`. Treat it as advice, not a directive "
        "— if it's wrong, override it and explain why.\n"
        "5. Once `validate_code` returns `ok`, your default move is "
        "`submit`. Don't keep sketching."
    )

    parts.append(
        "## Code requirements\n\n"
        f"1. `def build_model({param_str})` — top-level, returns nn.Module\n"
        "2. `def build_optimizer(model)` — top-level, returns Optimizer\n"
        "3. Only torch + stdlib — no external dependencies\n"
        "4. Read all dimensions from `build_model` arguments — never "
        "hardcode\n"
        "5. Always call `validate_code` before `submit`"
    )

    parts.append(
        "## Optional hooks\n\n"
        "Define any of these as top-level functions and the harness "
        "will pick them up via hasattr. Defaults are conservative; "
        "overriding usually helps.\n\n"
        "- `training_config() -> dict` — `batch_size`, "
        "`grad_accum_steps`, `grad_clip`, `log_every_n_steps`, "
        "`val_schedule`, `val_base_step`, `val_growth`\n"
        "- `configure_amp() -> dict` — `{\"enabled\": bool, \"dtype\": "
        "\"bfloat16\"|...}`\n"
        "- `build_scheduler(optimizer, total_steps_est)` — LR scheduler\n"
        "- `compute_loss(predictions, targets) -> Tensor` — override "
        "the default loss\n"
        "- `init_weights(model) -> None` — custom init; param count "
        "must not change\n"
        "- `transform_batch(batch, step, total_steps) -> dict` — "
        "augmentation\n"
        "- `on_step_end(model, optimizer, step, total_steps, "
        "loss_value) -> None`\n"
        "- `COMPILE = True` — module-level bool that opts into "
        "`torch.compile`"
    )

    parts.append(
        f"## FLOPs budget\n\n"
        f"Bucket: {bucket}\n"
        f"Range: [{flops_min:,}, {flops_max:,}]\n"
        f"Target: ~{target:,} FLOPs (60% of max)\n"
        f"Hard gate: [{gate_min:,}, {gate_max:,}] (instant rejection "
        "outside)"
    )

    parts.append(_compute_sizing_guidance(challenge))

    parts.append(
        "## A few principles\n\n"
        "- **Iterate cheaply before iterating expensively.** "
        "`sketch_architecture` is faster than `validate_code`. Use "
        "it while exploring shapes.\n"
        "- **Validation is the commitment point.** Once code passes "
        "`validate_code`, your default move is to submit. If you "
        "find yourself re-sketching after a successful validation, "
        "ask whether the new idea is actually better or just "
        "different.\n"
        "- **The brief is a starting point, not a contract.** If the "
        "researcher missed something, fix it. If the plan is wrong, "
        "deviate. The shipped code is yours."
    )

    return "\n\n".join(parts)


def build_designer_user_prompt(challenge: dict, brief: dict) -> str:
    """Kickoff message for the designer — embeds the researcher brief."""
    task = challenge.get("task", {}) or {}
    task_name = task.get("name", "unknown")

    parts: list[str] = []
    parts.append(
        f"You are designing for task `{task_name}`. The researcher "
        "produced this brief — read it, then implement one of the "
        "ideas:"
    )
    parts.append("```json\n" + json.dumps(brief, indent=2) + "\n```")

    plan = brief.get("plan") or []
    if plan:
        parts.append(
            "Plan the researcher suggested:\n"
            + "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(plan))
        )

    if brief.get("_default"):
        parts.append(
            "**Note**: the researcher hit an error and this is a "
            "default brief. Don't trust the plan blindly — start "
            "with `analyze_task`-equivalent reasoning yourself, then "
            "sketch."
        )

    parts.append(
        "Now: sketch, validate, ship. Submit when validate_code "
        "returns ok."
    )
    return "\n\n".join(parts)


# ── Critic prompts ────────────────────────────────────────────────────

def build_critic_system_prompt() -> str:
    """Critic system prompt — three-line KEEP/CHANGE/DROP template.

    Kept short on purpose: the critic is a single call between
    designer iterations and we don't want it generating prose.
    """
    return (
        "You are the **critic subagent** in a multi-agent miner. "
        "You see the designer's current code and the latest "
        "validate_code result. Your job: give the designer one "
        "short, structured critique it can act on immediately.\n\n"
        "Reply with EXACTLY three lines, no preamble, no closing:\n"
        "KEEP: <one sentence — what's working / not to touch>\n"
        "CHANGE: <one sentence — the single most impactful revision>\n"
        "DROP: <one sentence — what to remove or stop doing>\n\n"
        "Rules:\n"
        "- If validation passed (`ok ...`), KEEP the architecture "
        "and tell the designer to submit. CHANGE / DROP can be "
        "\"nothing\".\n"
        "- If validation failed, focus CHANGE on the specific "
        "failing constraint (FLOPs gate, output shape, missing "
        "build_optimizer, etc.).\n"
        "- Never write code. Never write more than three lines. "
        "Never explain — the designer is already an expert."
    )


def build_critic_prompt(code: str, validation_result: str) -> str:
    """Critic user prompt — validation result + truncated code."""
    return (
        f"validation result:\n{validation_result}\n\n"
        f"current code:\n```python\n{code[:4000]}\n```"
    )
