"""Designer subagent.

The designer takes the researcher's brief, generates code, validates
it, and submits. Tools: ``sketch_architecture``, ``estimate_layer_flops``,
``validate_code``, ``submit`` (plus ``time_remaining`` so the LLM can
budget its own clock).

The submit-blocking hook (``submit_requires_recent_validate``) is
applied as a pre-tool-call hook — the LLM must successfully run
``validate_code`` within the last 3 turns before ``submit`` is
allowed through. The critic fires as a post-tool-call callback after
each ``validate_code`` and its critique is injected as a user-role
message into the next designer turn.

v5: the same post-``validate_code`` callback also runs the static
recipe-sanity inspector (``core.recipe_sanity.inspect_recipe``). Any
warnings are appended to the critic's message so the LLM gets a single
combined feedback turn — runtime-found bugs from validate, qualitative
critique from the critic LLM, and recipe-anti-pattern warnings from the
static inspector. This catches the failure mode where the LLM ships a
recipe that runs fine but produces a weak model (lr=1e-2 on a small
transformer, no grad clip on AdamW, etc.).
"""
from __future__ import annotations

import sys
from typing import Optional

try:
    from ..hooks import default_designer_hooks
    from ..prompts import (
        build_designer_system_prompt, build_designer_user_prompt,
    )
    from ..tools import SubmitSignal, build_tools
    from .base import Subagent
    from .critic import run_critic
except ImportError:
    from hooks import default_designer_hooks
    from prompts import (
        build_designer_system_prompt, build_designer_user_prompt,
    )
    from tools import SubmitSignal, build_tools
    from subagents.base import Subagent
    from subagents.critic import run_critic

from core.recipe_sanity import inspect_recipe


# Was 12 — too tight to support multi-candidate exploration on a
# 30-minute budget. Each round is one LLM call; 30 supports ~3-4
# full validate-then-iterate cycles.
DESIGNER_MAX_ROUNDS = 30


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _resolve_code_from_args(
    args: dict, handlers: dict,
) -> str:
    """Pull the source the designer just validated.

    ``validate_code`` accepts either ``code`` (inline source) or
    ``candidate_id`` (lookup against state). When the LLM passes the
    id we read the code out of ``state.candidates`` via the submit
    handler's exposed state holder.
    """
    code = args.get("code") or ""
    if code:
        return code
    cid = args.get("candidate_id") or ""
    if not cid:
        return ""
    submit = handlers.get("submit")
    state_holder = getattr(submit, "_state_holder", None)
    if state_holder is None:
        return ""
    cands = (state_holder.get("state") or {}).get("candidates") or {}
    record = cands.get(cid) or {}
    return record.get("code") or ""


def _make_critic_callback(
    *, handlers: dict, deadline: float, llm_kwargs: dict, challenge: dict,
):
    """Build the on_tool_result callback that fires the critic after
    each ``validate_code``. The callback returns the combined critique
    string so the Subagent loop appends it as a user message.

    v5: the callback runs the static recipe-sanity inspector in addition
    to the LLM critic. The inspector is cheap (AST-only, no network), so
    we run it on every validate; its output is appended below the critic
    block so the LLM sees both at once. Either side may be empty — we
    only return a non-empty string when at least one produced feedback.
    """

    def _on_tool_result(
        name: str, args: dict, result: str, state: dict,
    ) -> Optional[str]:
        if name != "validate_code":
            return None
        code = _resolve_code_from_args(args, handlers)
        if not code:
            return None
        critique = run_critic(
            code=code,
            validation_result=result,
            deadline=deadline,
            llm_kwargs=llm_kwargs,
        )
        # Recipe-sanity inspection — pure static, never raises.
        report = inspect_recipe(code, challenge)
        sanity_block = report.format_for_prompt()
        if sanity_block:
            _log(
                f"[recipe_sanity] {len(report.warnings)} finding(s): "
                + ", ".join(
                    f"{w.code}({w.severity})" for w in report.warnings
                )
            )

        if not critique and not sanity_block:
            return None
        parts: list[str] = []
        if critique:
            parts.append(f"Critic feedback:\n{critique}")
        if sanity_block:
            parts.append(sanity_block)
        return "\n\n".join(parts)

    return _on_tool_result


def run_designer(
    *,
    challenge: dict,
    handlers: dict,
    deadline: float,
    llm_kwargs: dict,
    brief: dict,
    state: dict,
    bucket: str,
) -> Optional[SubmitSignal]:
    """Run the designer subagent. Returns the SubmitSignal when the
    LLM successfully ships, or ``None`` when the designer never
    submitted (deadline / chat error / hook-blocked submits all fall
    through).
    """
    tools = build_tools(challenge, role="designer")

    on_tool_result = _make_critic_callback(
        handlers=handlers,
        deadline=deadline,
        llm_kwargs=llm_kwargs,
        challenge=challenge,
    )

    designer_sys = build_designer_system_prompt(challenge, bucket)
    # Operator prompt directive (GEPA / random_mutate variant for this
    # round) is stashed by the orchestrator. Appending it lets the
    # subnet's prompt optimizer steer the designer's instructions
    # without us forking ``build_designer_system_prompt`` for every
    # variant.
    op_directive = challenge.get("_operator_prompt") or ""
    op_id = challenge.get("_operator_prompt_id") or ""
    if op_directive:
        designer_sys = (
            f"{designer_sys}\n\n"
            f"## Operator Directive (prompt variant {op_id[:8]})\n"
            f"{op_directive}"
        )

    sub = Subagent(
        name="designer",
        system_prompt=designer_sys,
        user_prompt=build_designer_user_prompt(challenge, brief),
        tools=tools,
        handlers=handlers,
        deadline=deadline,
        hooks=default_designer_hooks(),
        state=state,
        max_rounds=DESIGNER_MAX_ROUNDS,
        llm_kwargs=llm_kwargs,
        on_tool_result=on_tool_result,
    )
    result = sub.run()
    if result.submit_sig is not None:
        _log(
            f"[designer] shipped via submit "
            f"(rounds={result.rounds}, name={result.submit_sig.name})"
        )
        return result.submit_sig
    _log(
        f"[designer] no submit "
        f"(rounds={result.rounds}, failure={result.failure})"
    )
    return None
