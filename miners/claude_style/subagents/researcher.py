"""Researcher subagent.

The researcher is the first subagent in the orchestrator's sequence.
It gets the research / analysis tools (``analyze_task``,
``list_frontier``, ``query_db``, ``search_papers``) and produces a
JSON brief the designer consumes:

    {
      "relevant_prior_work": [...],
      "frontier_gaps": [...],
      "ideas_to_try": [...],
      "plan": ["step 1", "step 2", "step 3"]
    }

Flow: build a Subagent, run it, pull the JSON object out of the
final assistant text. If the parse fails, retry once with a short
"return ONLY the JSON object" nudge appended; if that also fails,
fall back to a default brief derived from
``core.prompt_builder._compute_sizing_guidance``.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Optional

from core.prompt_builder import _compute_sizing_guidance

try:
    from ..prompts import (
        build_researcher_system_prompt,
        build_researcher_user_prompt,
    )
    from ..tools import build_tools
    from .base import Subagent, SubagentResult
except ImportError:
    from prompts import (
        build_researcher_system_prompt,
        build_researcher_user_prompt,
    )
    from tools import build_tools
    from subagents.base import Subagent, SubagentResult


BRIEF_KEYS = (
    "relevant_prior_work",
    "frontier_gaps",
    "ideas_to_try",
    "plan",
)

# Cap on researcher LLM turns. The brief should land in 3-5 calls; the
# extra headroom lets one tool retry slip in without blowing the budget.
RESEARCHER_MAX_ROUNDS = 6

JSON_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _extract_json_object(text: str) -> Optional[dict]:
    """Pull the first JSON object out of ``text``.

    Tries (in order): fenced ```json``` block, fenced ``` block,
    bare brace-balanced object scan. Returns the parsed dict on
    success, ``None`` on failure.
    """
    if not text:
        return None
    m = JSON_FENCE_RE.search(text)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1))
    # Fallback: scan for the first '{' and walk until braces balance.
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _normalize_brief(obj: dict) -> dict:
    """Coerce a parsed object into the brief's shape: ensure all
    expected keys are present, lists stay lists, and strings inside
    the plan stay strings. Unknown keys are kept (might be useful
    context for the designer)."""
    out = dict(obj)
    for key in BRIEF_KEYS[:3]:  # list-shaped fields
        v = out.get(key)
        if not isinstance(v, list):
            out[key] = [] if v is None else [str(v)]
    plan = out.get("plan")
    if not isinstance(plan, list):
        out["plan"] = [] if plan is None else [str(plan)]
    else:
        out["plan"] = [str(s) for s in plan]
    return out


def default_brief(challenge: dict, bucket: str) -> dict:
    """Construct a default brief when the LLM fails to produce one.

    Built from ``_compute_sizing_guidance`` so the designer at least
    sees the FLOPs target and the self-sizing hint pulled straight
    out of the challenge.
    """
    sizing = _compute_sizing_guidance(challenge)
    task = challenge.get("task", {}) or {}
    return {
        "relevant_prior_work": [],
        "frontier_gaps": [
            "no researcher signal — designer must rely on sizing "
            "guidance and frontier data fetched directly"
        ],
        "ideas_to_try": [
            f"size a candidate to mid-bucket for the {bucket} bucket",
            "embed the sizing target inside build_model so the model "
            "auto-resizes to the budget",
        ],
        "plan": [
            "sketch a small candidate that reads dimensions from "
            "task_params",
            "use size_to_flops if the sketch lands outside the gate",
            "validate_code, then submit",
        ],
        "sizing_guidance": sizing,
        "task_name": task.get("name") or "unknown",
        "_default": True,
    }


def _build_subagent(
    *,
    challenge: dict,
    handlers: dict,
    deadline: float,
    llm_kwargs: dict,
    bucket: str,
    extra_user_msg: str = "",
) -> Subagent:
    tools = build_tools(challenge, role="researcher")
    # Keep only the handlers the researcher actually has tools for —
    # the dict still contains everything else by reference, but a
    # well-behaved LLM only sees the role-filtered surface.
    user_prompt = build_researcher_user_prompt(challenge)
    if extra_user_msg:
        user_prompt = user_prompt + "\n\n" + extra_user_msg
    return Subagent(
        name="researcher",
        system_prompt=build_researcher_system_prompt(challenge, bucket),
        user_prompt=user_prompt,
        tools=tools,
        handlers=handlers,
        deadline=deadline,
        hooks=[],
        state={},
        max_rounds=RESEARCHER_MAX_ROUNDS,
        llm_kwargs=llm_kwargs,
    )


def run_researcher(
    *,
    challenge: dict,
    handlers: dict,
    deadline: float,
    llm_kwargs: dict,
    state: dict,
    bucket: str,
) -> dict:
    """Run the researcher subagent and return a brief.

    ``state`` is the orchestrator-shared state dict (read but not
    written here — research data lives in scratchpad via the tool
    handlers, not in the brief itself).
    """
    _ = state  # explicitly unused at this layer; preserved in signature

    sub = _build_subagent(
        challenge=challenge,
        handlers=handlers,
        deadline=deadline,
        llm_kwargs=llm_kwargs,
        bucket=bucket,
    )
    result = sub.run()
    parsed = _extract_brief(result)
    if parsed is not None:
        _log(
            f"[researcher] brief parsed on first attempt "
            f"(rounds={result.rounds})"
        )
        return _normalize_brief(parsed)

    # Retry once with a short corrective nudge. We don't replay the
    # whole conversation — a fresh subagent with the prior turn count
    # subtracted from its budget is enough.
    _log("[researcher] first attempt did not yield JSON — retrying once")
    sub2 = _build_subagent(
        challenge=challenge,
        handlers=handlers,
        deadline=deadline,
        llm_kwargs=llm_kwargs,
        bucket=bucket,
        extra_user_msg=(
            "Return the JSON brief now. ONLY a single fenced "
            "```json block with relevant_prior_work, frontier_gaps, "
            "ideas_to_try, plan. No prose."
        ),
    )
    result2 = sub2.run()
    parsed2 = _extract_brief(result2)
    if parsed2 is not None:
        _log(
            f"[researcher] brief parsed on retry "
            f"(rounds={result2.rounds})"
        )
        return _normalize_brief(parsed2)

    _log(
        "[researcher] both attempts failed to yield JSON brief — "
        "falling back to default brief"
    )
    return default_brief(challenge, bucket)


def _extract_brief(result: SubagentResult) -> Optional[dict]:
    """Look for the brief in the subagent's final assistant text first,
    then walk back through the message list — the LLM sometimes drops
    JSON in an earlier assistant turn before continuing with prose."""
    for source in (result.content, *_assistant_contents(result.messages)):
        obj = _extract_json_object(source or "")
        if obj is not None and any(k in obj for k in BRIEF_KEYS):
            return obj
    return None


def _assistant_contents(messages: list[dict]):
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            if content:
                yield content
