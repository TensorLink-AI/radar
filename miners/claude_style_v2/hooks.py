"""Pre-tool-call hook framework.

A hook is a callable ``hook(name, args, state) -> Optional[str]``.
When a hook returns a string, that string is surfaced to the LLM as
the tool result and the actual handler is NOT called. This lets the
orchestrator block specific tool invocations without modifying the
underlying handler dict.

Hooks are registered as a list of ``(name_pattern, hook_fn)`` tuples;
the dispatcher in ``subagents.base.Subagent._dispatch_tool`` walks
the list in order. ``name_pattern`` is either an exact tool name or
the literal ``"*"`` to match any tool.

For v1 there's exactly one rule:

  * ``submit_requires_recent_validate`` — block ``submit`` unless the
    last 3 designer turns include a successful ``validate_code``.

The state dict is the per-subagent ``state`` dict on
``Subagent.state`` — the base loop already populates
``state["validate_history"]`` (a list of ``{"round": int, "ok": bool}``
entries) every time ``validate_code`` runs, so the rule just reads
that list.
"""
from __future__ import annotations

from typing import Callable, Optional

HookFn = Callable[[str, dict, dict], Optional[str]]
HookRule = tuple[str, HookFn]


SUBMIT_BLOCKED_MSG = (
    "submit blocked: run validate_code and confirm ok=true first"
)

# How many recent designer turns are inspected for the validate ok flag.
SUBMIT_VALIDATE_LOOKBACK = 3


def submit_requires_recent_validate(
    name: str, args: dict, state: dict,
) -> Optional[str]:
    """Block ``submit`` unless ``validate_code`` returned ok=true in
    one of the last ``SUBMIT_VALIDATE_LOOKBACK`` rounds.

    Reads ``state["validate_history"]`` (populated by the subagent
    base loop). The rule is anchored to the round number on the most
    recent history entry — counting back from "now" using designer
    rounds rather than wall-clock time keeps the rule deterministic
    in tests.
    """
    if name != "submit":
        return None
    history = state.get("validate_history") or []
    if not history:
        return SUBMIT_BLOCKED_MSG
    # Most-recent round number in the loop. We're called *during*
    # that round, so anything newer than (current_round -
    # lookback) counts.
    latest = history[-1].get("round", 0)
    cutoff = latest - SUBMIT_VALIDATE_LOOKBACK
    for entry in reversed(history):
        if entry.get("round", 0) <= cutoff:
            break
        if entry.get("ok"):
            return None
    return SUBMIT_BLOCKED_MSG


def default_designer_hooks() -> list[HookRule]:
    """Return the default hook list applied to the designer subagent."""
    return [("submit", submit_requires_recent_validate)]
