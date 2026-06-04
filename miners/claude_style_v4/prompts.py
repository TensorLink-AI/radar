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
from core.primitives import format_primitives_for_prompt
from core.prompt_builder import _compute_sizing_guidance, _format_task_params


# ── Brief schema (researcher output contract) ─────────────────────────

BRIEF_SCHEMA_EXAMPLE = {
    "relevant_prior_work": [
        "<paper or method> (<author year>) — <one-line core idea, "
        "described by mechanism, not by brand name>",
    ],
    "frontier_gaps": [
        "<inductive bias / op family / objective / regularizer / "
        "tokenization choice that no frontier member currently uses>",
    ],
    "ideas_to_try": [
        "<concrete architectural idea described by its operations, "
        "shapes, and information flow — no need to attach a paper "
        "name; if a novel combination fits the task, prefer that "
        "over a famous one>",
        "<a structurally different second idea that explores a "
        "different point on the design space (different op family, "
        "different bridging strategy, different objective, etc.)>",
    ],
    "plan": [
        "<3-5 short steps the designer should run, written as "
        "tool-call intents rather than recipes — the designer owns "
        "the implementation choices>",
    ],
}


# ── Analyst prompts (technique #1 — axes-of-variation digest) ─────────

DIGEST_SCHEMA_EXAMPLE = {
    "axes_explored": [
        "depth (3-12 layers)", "hidden_dim (64-512)", "patch_size (8-32)",
    ],
    "axes_fixed_across_frontier": [
        "sequence operator type — every frontier member uses full attention",
        "tokenization — every member patches the input",
        "weight sharing — none uses tied weights across depth",
    ],
    "saturated_families": ["patch-transformer"],
    "absent_families": [
        "state-space (S4/Mamba)", "depthwise-conv stacks", "MLP-mixer",
    ],
    "failure_patterns": [
        "FLOPs-gate rejections cluster on the upper edge for transformer "
        "families — too many heads",
        "build_optimizer omitted in 4 of last 20 failures",
    ],
    "injected_primitives_eval": [
        {
            "primitive": "state-space (S4/S5/Mamba-style selective scan)",
            "status": "absent",
            "recommendation": "explore",
            "note": "no scan-based model on the frontier at this bucket",
        },
    ],
    "recommend_explore": [
        "state-space backbone at ~0.6x max FLOPs",
        "depthwise-conv stack with linear head",
    ],
    "recommend_avoid": [
        "another transformer hyperparameter tweak — family is saturated",
    ],
    # v4: what's identical across frontier members' *training recipes*
    # (build_optimizer / build_scheduler / training_config / compute_loss /
    # configure_amp). The in-round recipe-tuner uses this to pick a
    # divergence direction on the training side rather than the
    # architecture side. Empty strings = unknown / didn't read.
    "training_recipe_norm": {
        "optimizer": "Adam (every frontier member, no AdamW or Lion)",
        "learning_rate": "1e-3 (no schedule, no warmup)",
        "lr_schedule": "none — every member uses a constant LR",
        "weight_decay": "0 (omitted)",
        "loss": "task default (no member overrides compute_loss)",
        "amp_dtype": "bfloat16 default",
        "batch_size": "task default",
        "grad_clip": "none",
        "notes": "training-recipe surface is unexplored — divergence here is cheap",
    },
}


def build_analyst_system_prompt(
    challenge: dict, bucket: str | None = None,
    primitives: list[str] | None = None,
) -> str:
    """Analyst system prompt.

    The analyst is read-only — it inspects the global experiment DB
    plus the current frontier and emits a structured digest the
    researcher embeds verbatim. It does NOT propose code, does NOT
    submit, does NOT call validate_code.
    """
    task = challenge.get("task", {}) or {}
    task_name = task.get("name", "unknown")
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = bucket or identify_bucket(flops_min, flops_max)
    primitives = primitives or []

    parts: list[str] = []

    parts.append(
        "You are the **analyst subagent** in a multi-agent miner. "
        "You run BEFORE the researcher. Your job is to read the global "
        "experiment DB and the current frontier, then emit a structured "
        "design-landscape digest the researcher consumes as its "
        "starting context.\n\n"
        f"Task: `{task_name}`. Bucket: `{bucket}`. "
        f"FLOPs range: [{flops_min:,}, {flops_max:,}]."
    )

    parts.append(
        "## Your output (the digest)\n\n"
        "Produce a single JSON object with these keys. The researcher "
        "reads it verbatim.\n\n"
        "- `axes_explored`: list of strings — the architectural axes "
        "(depth, hidden_dim, patch_size, head_count, ...) that "
        "frontier members are known to vary on.\n"
        "- `axes_fixed_across_frontier`: list of strings — axes the "
        "frontier does NOT vary on (e.g. \"every member uses attention\"). "
        "**This is the highest-signal field — the researcher uses it to "
        "pick a divergence direction.**\n"
        "- `saturated_families`: list of strings — architectural families "
        "with 2+ members on the frontier; another candidate from these "
        "is likely to tie not dominate.\n"
        "- `absent_families`: list of strings — families with zero "
        "frontier presence in this bucket.\n"
        "- `failure_patterns`: list of strings — recurring failure modes "
        "from `/experiments/failures` (FLOPs-gate direction, missing "
        "build_optimizer, output-shape mismatches, ...).\n"
        "- `injected_primitives_eval`: list of objects — one per "
        "injected primitive, with `status` ∈ {absent, present, "
        "saturated, unknown} and `recommendation` ∈ {explore, skip, "
        "hybridize}. The injected primitives are listed in your "
        "user message.\n"
        "- `recommend_explore`: list of strings — 2-3 concrete "
        "directions the researcher should consider, given the axes / "
        "absent families / primitive evaluations.\n"
        "- `recommend_avoid`: list of strings — directions that would "
        "land in saturated regions.\n"
        "- `training_recipe_norm`: object — what's identical across "
        "the frontier's *training recipes* (the hooks the harness "
        "picks up by name: `build_optimizer`, `build_scheduler`, "
        "`training_config`, `compute_loss`, `configure_amp`). One "
        "field each: `optimizer`, `learning_rate`, `lr_schedule`, "
        "`weight_decay`, `loss`, `amp_dtype`, `batch_size`, "
        "`grad_clip`, `notes`. Read 1-2 frontier members' code via "
        "`get_frontier_member` (if available) or `query_db` "
        "`/experiments/{idx}` to populate it — leave a field "
        "blank when you didn't get a read. **This is the second-"
        "highest-signal field** (after `axes_fixed_across_frontier`): "
        "the in-round recipe-tuner uses it to choose a recipe "
        "divergence direction once the designer has a validated "
        "architecture.\n\n"
        "Example shape:\n```json\n"
        + json.dumps(DIGEST_SCHEMA_EXAMPLE, indent=2)
        + "\n```"
    )

    parts.append(
        "## Tools available to you\n\n"
        "- `analyze_task` — task spec as JSON.\n"
        "- `list_frontier` — current best models on this bucket. "
        "Read every entry — these are the saturated points.\n"
        "- `get_frontier_member` — one frontier member's full source "
        "by index. Use it to read 1-2 members' "
        "`build_optimizer` / `build_scheduler` / `training_config` / "
        "`compute_loss` / `configure_amp` hooks so you can populate "
        "`training_recipe_norm`. Don't read more than 2 members — "
        "the signal is what they have in common, not the long tail.\n"
        "- `query_db` — experiment DB. Useful paths for landscape "
        "reflection: `/experiments/pareto?task=`, "
        "`/experiments/families?task=`, `/experiments/recent?n=`, "
        "`/experiments/failures?n=`, `/experiments/stats?task=`, "
        "POST `/experiments/search` body `{\"query\": \"...\"}`. "
        "**Use `families` first** — it's the cheapest signal on "
        "what's been clustered together.\n"
        "- `time_remaining` — seconds left. You're capped at ~60s; "
        "don't browse, scan."
    )

    if primitives:
        parts.append(
            "## Injected primitives this round\n\n"
            "The orchestrator sampled these architectural primitives "
            "for this round (technique: forced cross-family ideation). "
            "You MUST evaluate each one against the DB and include it "
            "in `injected_primitives_eval`:\n\n"
            + format_primitives_for_prompt(primitives)
        )

    parts.append(
        "## Principles\n\n"
        "- **Compute axes, not lists.** The researcher needs to know "
        "which architectural dimensions the frontier varies on (and "
        "which it doesn't). Listing 50 model names is noise; naming "
        "the 3 axes nobody varies is signal.\n"
        "- **Saturated ≠ bad.** A saturated family means the next "
        "candidate from that family probably ties an existing member "
        "— and a tie loses the Pareto bonus. Mark saturated families "
        "so the researcher routes around them.\n"
        "- **Classify every injected primitive.** Check the DB before "
        "guessing — `/experiments/search` with the primitive's name "
        "is the fast path.\n"
        "- **Read 1-2 frontier members' source.** Just enough to "
        "populate `training_recipe_norm` — what optimizer, LR, "
        "schedule, AMP dtype, batch size, loss override does each "
        "actually use? Leave fields blank when uncertain; the "
        "recipe-tuner reads blanks as \"no signal here, default.\"\n"
        "- **No code, no architecture sketches, no submissions.** "
        "Your only output is the digest JSON."
    )

    parts.append(
        "## Final turn\n\n"
        "Return ONLY a single fenced ```json block with the digest. "
        "No prose around it. The orchestrator parses your last "
        "assistant message."
    )

    return "\n\n".join(parts)


def build_analyst_user_prompt(
    challenge: dict, primitives: list[str] | None = None,
) -> str:
    """Kickoff message for the analyst."""
    task = challenge.get("task", {}) or {}
    frontier = (
        challenge.get("feasible_frontier")
        or challenge.get("pareto_frontier")
        or []
    )
    primitives = primitives or []
    body = (
        f"Survey the design landscape for the `{task.get('name', 'unknown')}` "
        f"task. Frontier currently has {len(frontier)} model(s).\n\n"
        "Call list_frontier and query_db (families + failures first), "
        "evaluate the injected primitives, then return the digest "
        "as a single fenced ```json block."
    )
    if primitives:
        body += (
            "\n\nInjected primitives to evaluate:\n"
            + format_primitives_for_prompt(primitives)
        )
    return body


# ── Researcher Phase A (high-temperature brainstorm) ──────────────────

_PHASE_A_INSTRUCTIONS = (
    "You are the **brainstorm pass** of the researcher. This is a "
    "text-only pass: no tools, no JSON, no commitment. Your job is "
    "to flood the candidate space with 10 ideas. Phase B will prune "
    "down to 2.\n\n"
    "Hard quotas (the brainstorm fails if these aren't met):\n"
    "- At least 3 of the 10 ideas must be ones you suspect are bad "
    "or unconventional. Mark them `[risky]`. These widen the "
    "distribution and stop you from converging on the obvious.\n"
    "- At least 2 of the 10 must import a primitive from outside "
    "time-series — e.g. from vision, audio, classical signal "
    "processing, RL, or graph learning. Mark them `[cross-domain]`.\n"
    "- Every injected primitive listed below must appear in at "
    "least one idea. Mark those `[injected: <primitive>]`.\n"
    "- No two ideas may share their primary sequence operator AND "
    "tokenization scheme — same-family duplicates collapse the "
    "exploration value."
)


def build_phase_a_system_prompt(
    challenge: dict, bucket: str | None = None,
    primitives: list[str] | None = None,
    digest_block: str = "",
) -> str:
    """Phase A system prompt — high-temperature brainstorm only.

    Phase A is a single chat() call with no tools. It generates a
    numbered list of 10 ideas under hard quotas. Phase B (the
    existing subagent with tools) consumes that list and produces
    the final brief.
    """
    task = challenge.get("task", {}) or {}
    task_name = task.get("name", "unknown")
    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = bucket or identify_bucket(flops_min, flops_max)
    target = int(flops_max * 0.6) if flops_max else 0
    primitives = primitives or []

    parts: list[str] = []
    parts.append(
        "You are the researcher subagent in a multi-agent miner.\n\n"
        f"Task: `{task_name}`. Bucket: `{bucket}`. "
        f"FLOPs target ~{target:,}."
    )
    parts.append(_PHASE_A_INSTRUCTIONS)

    if digest_block:
        parts.append(
            "## Analyst digest (for context — read before brainstorming)\n\n"
            + digest_block
            + "\n\nUse `axes_fixed_across_frontier` as your divergence "
            "compass: ideas that move on those axes are higher-value "
            "than ideas that don't."
        )

    if primitives:
        parts.append(
            "## Injected primitives (each must appear in at least one idea)\n\n"
            + format_primitives_for_prompt(primitives)
        )

    parts.append(
        "## Output format\n\n"
        "Return ONLY a numbered markdown list, 10 entries. Each entry: "
        "one sentence + tags. Example:\n\n"
        "```\n"
        "1. depthwise-sep conv1d backbone with linear head, "
        "patch=8, hidden_dim≈128 [injected: depthwise-separable]\n"
        "2. FFT-mix front-end + small MLP backbone [cross-domain]\n"
        "3. tied-weight 4-layer transformer at half head_dim [risky]\n"
        "...\n"
        "```\n\n"
        "No prose around the list. No JSON. No tools."
    )
    return "\n\n".join(parts)


def build_phase_a_user_prompt(challenge: dict) -> str:
    """Kickoff for Phase A."""
    task = challenge.get("task", {}) or {}
    return (
        f"Brainstorm 10 candidate architectures for the "
        f"`{task.get('name', 'unknown')}` task. Meet the quotas. "
        "Return ONLY the numbered list."
    )


# ── Researcher prompts ────────────────────────────────────────────────

def build_researcher_system_prompt(
    challenge: dict, bucket: str | None = None,
    primitives: list[str] | None = None,
) -> str:
    """Researcher system prompt.

    Goal: a JSON brief the designer can act on. The researcher does
    NOT write code, sketch architectures, or call validate_code /
    submit (those tools aren't even in its surface). Its only job is
    to surface what's been tried, what's missing, and a 3-5 step
    plan.

    In v3 the researcher receives:
      * a landscape digest from the analyst (in its user message),
      * a Phase A brainstorm output (also in its user message),
      * a list of injected primitives it must address.
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
        "POST `/experiments/search` body `{\"query\": \"...\"}`. "
        "Artifacts: `/artifacts?miner_id=&task=&kind=` "
        "(kinds: result, log, checkpoint, …), `/artifacts/{id}`, "
        "`/artifacts/{id}/download`, `/experiments/{idx}/artifacts` — "
        "use these for full training logs / checkpoints. Failure traces "
        "are already in the `analysis` field of `/experiments/failures`. "
        "~60 calls/min; calls are logged on the public dashboard.\n"
        "- `search_papers` — arxiv search. Use sparingly — papers "
        "are expensive and the LLM context is bounded.\n"
        "- `time_remaining` — seconds left in your slice of the "
        "round. The orchestrator caps you at 90s; budget accordingly."
    )

    primitives = primitives or []
    if primitives:
        parts.append(
            "## Injected primitives this round\n\n"
            "Each MUST be addressed in your brief (used in "
            "`ideas_to_try` or rejected with reason in "
            "`primitive_rejections`):\n\n"
            + format_primitives_for_prompt(primitives)
        )

    parts.append(
        "## Principles\n\n"
        "- **The analyst already mapped the landscape.** Treat its "
        "`axes_fixed_across_frontier` as a generative signal: an "
        "idea is interesting if it moves on an axis nobody varies.\n"
        "- **Beat the frontier, don't match it.** A tie loses the "
        "Pareto dominance bonus. Your `frontier_gaps` should make "
        "this concrete.\n"
        "- **Concrete > abstract.** Name the operations, shapes, "
        "and information flow. \"Try a transformer\" is useless. "
        "\"Stack of <op family A> over patches of size P with a "
        f"<bridge type> head, target ~{target:,} FLOPs\" is "
        "actionable — describe the mechanism, not the brand.\n"
        "- **Use injected primitives or justify skipping them.** "
        "Each primitive must appear in at least one Phase A idea AND "
        "in your final brief, either in `ideas_to_try` or in "
        "`primitive_rejections` with a one-line reason.\n"
        "- **Reach past the familiar shortlist.** Don't default to "
        "the same handful of named architectures every round. Novel "
        "combinations of standard PyTorch ops (gated convs, "
        "spectral mixing, state-space recurrences, learned routing, "
        "non-standard tokenizers, alternative loss formulations, "
        "etc.) are fair game and often unexplored on the frontier.\n"
        "- **Don't write code.** That's the designer's job. If you "
        "find yourself sketching architectures, stop and rephrase as "
        "a plan step.\n"
        "- **Stop early.** With the analyst's digest in hand you "
        "rarely need more than 2 tool calls. Spend them only when "
        "the digest leaves a real gap."
    )

    parts.append(
        "## Final turn\n\n"
        "When you're ready, return ONLY a single fenced ```json block "
        "with the brief. No prose around it, no other commentary in "
        "that final message. The orchestrator parses your last "
        "assistant message — anything else costs context for nothing."
    )

    return "\n\n".join(parts)


def build_researcher_user_prompt(
    challenge: dict,
    digest_block: str = "",
    phase_a_output: str = "",
    primitives: list[str] | None = None,
) -> str:
    """Kickoff message for the Phase B researcher.

    ``digest_block`` is the analyst digest rendered as a fenced JSON
    block (analyst.format_digest_for_researcher). ``phase_a_output``
    is the brainstorm text from Phase A. Both are embedded verbatim;
    the researcher reads them, prunes, and emits the final brief.
    """
    task = challenge.get("task", {}) or {}
    frontier = (
        challenge.get("feasible_frontier")
        or challenge.get("pareto_frontier")
        or []
    )
    primitives = primitives or []

    parts: list[str] = []
    parts.append(
        f"You are designing for task `{task.get('name', 'unknown')}`. "
        f"Frontier currently has {len(frontier)} model(s)."
    )

    if digest_block:
        parts.append("## Analyst digest\n\n" + digest_block)
    else:
        parts.append(
            "## Analyst digest\n\n(analyst skipped — fall back to "
            "calling the tools yourself)"
        )

    if phase_a_output:
        parts.append(
            "## Phase A brainstorm (high-temperature, with quotas)\n\n"
            "These 10 ideas came out of a brainstorm pass. The bad / "
            "cross-domain quotas are intentional — they widen the "
            "distribution. Pick the 2 strongest, drop the rest, and "
            "carry them into the brief.\n\n"
            + phase_a_output.strip()
        )

    if primitives:
        parts.append(
            "## Injected primitives (you MUST address each one)\n\n"
            + format_primitives_for_prompt(primitives)
            + "\n\nIn the brief: either include an `ideas_to_try` "
            "entry that uses the primitive, or add it to a "
            "`primitive_rejections` list with a one-line reason."
        )

    parts.append(
        "Now: prune the Phase A list, run any tools you need to fill "
        "gaps the analyst missed, and return the brief as a single "
        "fenced ```json block."
    )
    return "\n\n".join(parts)


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
        "- `submit` — stash or ship a candidate: "
        "`submit(candidate_id=..., name=..., motivation=..., note=...)`. "
        "Outside the last 5 minutes this stashes your candidate as "
        "best-so-far and tells you to keep iterating. Inside the last "
        "5 minutes it actually ships. The harness blocks this call until "
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
        "5. Once `validate_code` returns `ok`, you have a candidate "
        "worth keeping. Stash it via `submit` (early submits stash as "
        "best-so-far, not ship), then try a different architectural "
        "angle. The strongest run validates 2-3 candidates and submits "
        "the best inside the last 5 minutes."
    )

    parts.append(
        "## Code requirements\n\n"
        f"1. `def build_model({param_str})` — top-level, returns nn.Module\n"
        "2. `def build_optimizer(model)` — top-level, returns Optimizer\n"
        "3. Only torch + stdlib — no external dependencies\n"
        "4. Read all dimensions from `build_model` arguments — never "
        "hardcode\n"
        "5. Treat each length-like task_param as INDEPENDENT — an input "
        "length (e.g. `context_len`) and an output length (e.g. "
        "`prediction_len`) are not related and must not be conflated. "
        "Any layer that bridges them must project explicitly along the "
        "length axis (any op whose output length depends on "
        "`prediction_len` is fine — linear, attention pooling, learned "
        "queries, transposed conv, interpolation + refine, etc.), never "
        "via implicit reshape or residual add\n"
        "6. Always call `validate_code` before `submit`"
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
        "- `build_scheduler(optimizer, total_steps_est)` — LR scheduler. "
        "If using `LambdaLR`, the lambda must return a Python float — "
        "`math.cos(math.pi * x)`, NOT `torch.cos(torch.tensor(x))`.\n"
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
        "- **Validation is a checkpoint, not the finish line.** A "
        "validated candidate that ties the frontier loses the Pareto "
        "bonus. After validation, ask: is there a structurally "
        "different candidate that might dominate this one?\n"
        "- **Use your full budget.** Calling submit before the "
        "late-round window stashes your candidate as best-so-far and "
        "prompts you to keep iterating. The strongest submission "
        "usually comes from comparing 2-3 validated candidates, not "
        "the first one that runs.\n"
        "- **The brief is a starting point, not a contract.** If the "
        "researcher missed something, fix it. If the plan is wrong, "
        "deviate. The shipped code is yours."
    )

    return "\n\n".join(parts)


def _continuation_preamble(context: dict) -> str:
    """Hard architecture-freeze instruction for a continuation round.

    The parent checkpoint is loaded strictly, so the submission must be
    architecturally identical. We hand the designer the parent's own
    architecture source to copy verbatim and confine it to the training
    recipe — which it writes every round anyway.
    """
    parent = context.get("parent", {})
    return (
        f"## CONTINUATION ROUND — architecture is FROZEN\n\n"
        f"You are continuing parent experiment #{context['parent_id']} "
        f"(metric={parent.get('metric')!r}, "
        f"n_rounds={parent.get('n_rounds')!r}). Its checkpoint is loaded "
        "into your model with `strict=True`, so your module MUST be "
        "architecturally identical. Copy the entire block below into your "
        "submission **verbatim** — it is the parent's complete architecture "
        "(imports, helper classes, `build_model`, `init_weights`, "
        "`COMPILE`). Do not rename, reshape, reorder, or otherwise alter "
        "any of it:\n\n"
        "```python\n" + context["frozen_arch"] + "\n```\n\n"
        "Your ONLY job is to improve the **training recipe** so the "
        "warm-started weights train further/better. Add or rewrite ONLY "
        "these hooks: `build_optimizer` (required), `build_scheduler`, "
        "`training_config`, `compute_loss`, `configure_amp`, "
        "`transform_batch`, `on_step_begin`/`on_step_end`. Because the "
        "weights are already trained, favour a lower / decaying learning "
        "rate that extends the parent's curve rather than restarting it. "
        "Do NOT change anything in the frozen block or any tensor shape — "
        "if you do, the warm-start load fails and the run is scored as a "
        "fresh design instead.\n\n"
        "`validate_code`, then `submit`."
    )


def _format_recipe_norm(norm: dict | None) -> str:
    """Render the analyst's ``training_recipe_norm`` as a short bullet
    list, omitting empty fields. Returns an empty string when there's no
    signal to show — caller should skip the section entirely."""
    if not isinstance(norm, dict):
        return ""
    rows: list[str] = []
    labels = [
        ("optimizer", "Optimizer"),
        ("learning_rate", "Learning rate"),
        ("lr_schedule", "LR schedule"),
        ("weight_decay", "Weight decay"),
        ("loss", "Loss"),
        ("amp_dtype", "AMP dtype"),
        ("batch_size", "Batch size"),
        ("grad_clip", "Grad clip"),
        ("notes", "Notes"),
    ]
    for key, label in labels:
        v = norm.get(key)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        rows.append(f"- **{label}**: {v}")
    return "\n".join(rows)


def _inround_recipe_preamble(context: dict) -> str:
    """v4 in-round recipe-pass preamble.

    The designer already shipped a validated architecture in pass 1. Pass
    2 hands the same architecture back as a frozen block and asks the
    designer to focus its remaining budget on the training recipe — the
    same surface the continuation flow uses, minus the parent-checkpoint
    warm-start framing (this is a *fresh* submission, no curve to extend).

    When the analyst's ``training_recipe_norm`` is non-empty it's
    inlined here so the designer has a concrete "everyone does X, try Y"
    target instead of inventing a recipe from defaults.
    """
    norm_block = _format_recipe_norm(context.get("training_recipe_norm"))
    norm_section = (
        "\n\nWhat the frontier currently does (analyst read of "
        f"`training_recipe_norm`):\n\n{norm_block}\n\n"
        "Use this as a divergence compass: a recipe that just matches "
        "the frontier ties; a recipe that moves on a field everyone "
        "leaves at default can dominate."
        if norm_block else ""
    )
    return (
        "## RECIPE-TUNING PASS — architecture is FROZEN\n\n"
        "You already shipped a validated architecture in the first "
        "pass. This is the second pass: same architecture, better "
        "training recipe. Copy the entire block below into your "
        "submission **verbatim** — it is your own previously validated "
        "architecture (imports, helper classes, `build_model`, "
        "`init_weights`, `COMPILE`). Do not rename, reshape, or alter "
        "any of it; if you do, your pass-1 candidate ships unchanged "
        "and this pass is wasted:\n\n"
        "```python\n" + context["frozen_arch"] + "\n```\n\n"
        "Your ONLY job is to write a strong **training recipe** for "
        "this architecture. Add or rewrite ONLY these hooks:\n"
        "- `build_optimizer` (required) — AdamW with weight_decay and "
        "betas tuned to the depth, not the default `Adam(lr=1e-3)`.\n"
        "- `build_scheduler` — warmup + cosine / linear decay is "
        "almost always better than a flat LR. **Lambda must return a "
        "Python float** — `import math; math.cos(math.pi * x)`, NOT "
        "`torch.cos(torch.tensor(x))` (the latter returns a 0-dim tensor, "
        "LambdaLR sets `param_group['lr']` to it, training crashes).\n"
        "- `training_config` — `batch_size`, `grad_clip`, "
        "`val_schedule`, `val_base_step`, `val_growth`. Tighter "
        "val cadence early catches divergence cheaply.\n"
        "- `compute_loss` — override if the task default loss is "
        "weak (e.g. Huber vs MSE for heavy-tailed targets).\n"
        "- `configure_amp` — `{enabled: True, dtype: \"bfloat16\"}` "
        "is usually a free speedup on GPU.\n"
        "- `transform_batch`, `on_step_begin`, `on_step_end` — "
        "augmentation, EMA, custom logging.\n"
        + norm_section + "\n\n"
        "This is a fresh submission (no parent checkpoint, no "
        "warm-start), so the LR scale is the *fresh-training* scale — "
        "you are not continuing a curve. `validate_code`, then "
        "`submit`. Your pass-1 candidate is already stashed as "
        "best-so-far; you only need to beat it."
    )


def build_designer_user_prompt(challenge: dict, brief: dict) -> str:
    """Kickoff message for the designer — embeds the researcher brief."""
    task = challenge.get("task", {}) or {}
    task_name = task.get("name", "unknown")

    parts: list[str] = []
    inround_ctx = challenge.get("_inround_recipe_context")
    cont_ctx = challenge.get("_continuation_context")
    if inround_ctx:
        # v4 second-pass: recipe-only, architecture frozen to pass-1's
        # validated code. Takes precedence over a continuation context
        # (they're mutually exclusive in practice — continuation rounds
        # never run a second pass).
        parts.append(_inround_recipe_preamble(inround_ctx))
    elif cont_ctx:
        parts.append(_continuation_preamble(cont_ctx))
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


# ── ts_data_pipeline (pipeline designer) ──────────────────────────────

def build_pipeline_designer_system_prompt(
    challenge: dict, bucket: str | None = None,
) -> str:
    """System prompt for the ts_data_pipeline single-designer pass.

    The miner designs a *data generator / augmentor*, not an architecture.
    The architecture is supplied frozen by the validator (refreshed every
    50 successful rounds) so this prompt frames the design space, the
    scoring formula, and the tool surface — explicitly suppressing the
    standard build_model / FLOPs framing the v4 designer otherwise uses.
    """
    task = challenge.get("task", {}) or {}
    tp = task.get("task_params", {}) or {}
    param_str = ", ".join(tp.keys()) if tp else "**task_params"
    fa = challenge.get("frozen_arch") or {}
    fa_version = fa.get("version")

    parts: list[str] = []
    parts.append(
        "You are the **pipeline designer** for the `ts_data_pipeline` "
        "task. The validator pairs every proposal with a *frozen* "
        "time-series architecture and trains it from fresh weights on "
        "the data your `build_pipeline` yields. Your job is to design "
        "the data generator / augmentor that makes that fixed model "
        "train faster AND generalize better.\n\n"
        f"Frozen architecture is v{fa_version if fa_version is not None else '?'}; "
        "it refreshes every 50 successful rounds. Pipelines that "
        "over-fit to a single arch version silently degrade at the "
        "cutover — design for the I/O contract, not for the source."
    )

    parts.append(
        "## Scoring\n\n"
        "`metric = sqrt(AULC × sqrt(crps · mase))` — lower is better.\n"
        "- **AULC** = trapezoidal area under the in-training val-loss "
        "curve, normalized per-step. Rewards *both* fast convergence "
        "AND a low end-state.\n"
        "- **GIFT-Eval** (crps + mase) is the held-out gate. Pipelines "
        "that memorize leaderboard distributions lose here.\n"
        "- AULC and gift_metric are stored separately in `objectives` "
        "and segmented by `frozen_arch_version` on the dashboard, so "
        "the trajectory and the generalization signal are visible "
        "independently — optimize both."
    )

    parts.append(
        "## Code contract\n\n"
        f"1. `def build_pipeline({param_str})` — top-level, returns an "
        "iterator.\n"
        "2. Each `next()` yields `{\"input\": tensor, \"target\": "
        "tensor}` or `(input, target)`. `input` shape `(B, "
        "context_len, num_variates)`, `target` shape `(B, "
        "prediction_len, num_variates)`. Float32 unless you have a "
        "specific reason.\n"
        "3. Must yield indefinitely (the harness stops on its own "
        "compute budget). StopIteration mid-run kills the round.\n"
        "4. Only torch + numpy + stdlib. No external deps.\n"
        "5. Always run `pipeline_smoke_test` before `validate_code`, "
        "and `validate_code` before `submit` (the submit hook enforces "
        "the latter)."
    )

    parts.append(
        "## Design space (pick a family, then commit)\n\n"
        "1. **Procedural generators.** Sums of sinusoids, AR(p), random "
        "walks, change-points, regime mixtures, level shifts, "
        "heteroscedastic noise. Cheap, infinite, and the only family "
        "truly disjoint from the GIFT-Eval gate.\n"
        "2. **Augmentors over real pretrain shards** (read via the "
        "harness's standard loader if `RADAR_PRETRAIN_VAL_LOCAL_PATHS` "
        "is set in this environment). Jitter, scaling, magnitude / time "
        "warping, window cropping, channel permutation, mixup / CutMix "
        "in the time axis. See Wen 2021's TS-augmentation survey for "
        "the canonical list — but the harness can't pull real shards "
        "here; if you augment, augment the procedural stream.\n"
        "3. **Hybrid curricula.** Start with easy procedural (clean "
        "sinusoid) and ramp toward noisier mixtures as steps advance. "
        "AULC directly rewards easy-early curricula — they drop val "
        "loss faster, which dominates the trapezoidal area.\n"
        "4. **Distribution mixers.** Sample a regime per batch from a "
        "small library so the frozen arch sees broader coverage than "
        "any single shard could give it (domain randomization)."
    )

    parts.append(
        "## Tools\n\n"
        "- `read_frozen_arch` — compact card (default) or full source "
        "(opt-in, capped at 1 source-read per round). Read the source "
        "only if your design decision actually depends on the model's "
        "inductive bias (e.g. patch-aligned augmentations need the "
        "patch size).\n"
        "- `pipeline_smoke_test(code)` — exec your code, pull 2 "
        "batches, verify shapes. Sub-second; use it freely.\n"
        "- `validate_code` — final structural check (top-level "
        "build_pipeline, no forbidden imports). Required before submit.\n"
        "- `submit(code|candidate_id, name, motivation, note=...)` — "
        "stash early, ship in the last 5 minutes.\n"
        "- `list_frontier` / `get_frontier_member` — what already scored "
        "well on this task. Look at `objectives.aulc` and "
        "`objectives.gift_metric` to see WHY they're on the frontier.\n"
        "- `read_scratchpad` / `read_my_submissions` — cross-round "
        "memory. Read these FIRST so you don't re-run a dead end.\n"
        "- `write_scratchpad` — record what you tried and which axis "
        "moved the score. One line per attempt."
    )

    parts.append(
        "## Loop\n\n"
        "1. `read_scratchpad` → what worked / failed in prior rounds "
        "against this or earlier frozen-arch versions.\n"
        "2. `read_frozen_arch` (card only; source only if necessary) + "
        "`list_frontier` → what's currently strong on the "
        "(AULC, GIFT) plane.\n"
        "3. Pick ONE design-space family. Write `build_pipeline`.\n"
        "4. `pipeline_smoke_test` until shapes are clean.\n"
        "5. `validate_code` → fix any structural error.\n"
        "6. `submit` (stash). Iterate to a structurally different "
        "second candidate; submit again to compare. The strongest "
        "rounds ship 2 candidates from different families.\n"
        "7. Inside the last 5 minutes, submit the better of the two."
    )

    parts.append(
        "## Anti-patterns\n\n"
        "- Hardcoding constants from a specific `frozen_arch_version` "
        "into your generator. Refresh wipes that out.\n"
        "- Generators that emit perfectly-clean signals. The arch "
        "memorizes a single mode; GIFT-Eval gate then collapses.\n"
        "- Building an iterator that runs `for _ in range(N): yield` "
        "and exhausts. Yield indefinitely.\n"
        "- Pasting numpy arrays straight into the batch dict. Convert "
        "to torch (the smoke test will tell you if you forgot)."
    )

    return "\n\n".join(parts)


def build_pipeline_designer_user_prompt(
    challenge: dict, brief: dict | None = None,
) -> str:
    """Kickoff user message — embeds the frozen arch card + the brief."""
    fa = challenge.get("frozen_arch") or {}
    task = challenge.get("task", {}) or {}
    tp = task.get("task_params", {}) or {}
    parts: list[str] = []
    parts.append(
        f"Round task: `ts_data_pipeline`. Frozen arch is "
        f"v{fa.get('version')} (source experiment "
        f"#{fa.get('source_experiment_id')}, name "
        f"{fa.get('source_name')!r}, metric={fa.get('source_metric')!r}, "
        f"flops={fa.get('source_flops')!r}, "
        f"code_bytes={len(fa.get('code') or '')})."
    )
    parts.append(
        "I/O contract — input `(B, "
        f"{tp.get('context_len')}, {tp.get('num_variates')})`, target "
        f"`(B, {tp.get('prediction_len')}, {tp.get('num_variates')})`, "
        f"quantiles={tp.get('quantiles')!r}."
    )
    if brief:
        parts.append(
            "Orchestrator brief (a starting point — deviate freely):\n"
            "```json\n" + json.dumps(brief, indent=2) + "\n```"
        )
    parts.append(
        "Start by reading the scratchpad and the frontier. Then design "
        "a pipeline, smoke-test, validate, submit. Use the late-round "
        "window for the actual ship."
    )
    return "\n\n".join(parts)
