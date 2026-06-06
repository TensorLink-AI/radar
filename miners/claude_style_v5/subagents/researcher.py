"""Researcher subagent (v3 — two-phase: brainstorm → brief).

The v3 researcher runs in two phases:

* **Phase A** (high-temperature, no tools) — generates 10 candidate
  architectures under hard quotas (3 risky, 2 cross-domain, every
  injected primitive used at least once). Pure text output, no JSON.
* **Phase B** (Subagent loop with tools) — consumes the Phase A list
  and the analyst digest, prunes to 2 strong ideas, runs any tools
  needed to fill gaps the analyst missed, and emits the JSON brief
  the designer consumes:

      {
        "relevant_prior_work": [...],
        "ideas_to_try": [...],
        "divergence_axes": [...],
        "plan": [...],
        "primitive_rejections": [{"primitive": "...", "reason": "..."}]
      }

The split is the cheap way to widen the candidate distribution
without rebuilding the brief schema. Phase A is one extra LLM call
per round; on failure the researcher proceeds with the digest only.
"""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Optional

from core.prompt_builder import _compute_sizing_guidance

try:
    from ..llm_client import chat
    from ..prompts import (
        build_phase_a_system_prompt,
        build_phase_a_user_prompt,
        build_researcher_system_prompt,
        build_researcher_user_prompt,
    )
    from ..tools import build_tools
    from .base import Subagent, SubagentResult
except ImportError:
    from llm_client import chat
    from prompts import (
        build_phase_a_system_prompt,
        build_phase_a_user_prompt,
        build_researcher_system_prompt,
        build_researcher_user_prompt,
    )
    from tools import build_tools
    from subagents.base import Subagent, SubagentResult


# Phase A is a single high-temperature completion. Tighter token cap
# than the main researcher — 10 one-line ideas fit comfortably in
# 1500 tokens. Temperature is intentionally well above the default
# 0.7 to widen the distribution.
PHASE_A_TEMPERATURE = 0.95
PHASE_A_MAX_TOKENS = 1500
PHASE_A_MIN_SECONDS = 20


BRIEF_KEYS = (
    "relevant_prior_work",
    "ideas_to_try",
    "divergence_axes",
    "plan",
)

# Cap on researcher LLM turns. The brief should land in 3-5 calls; the
# extra headroom lets one tool retry slip in without blowing the budget.
RESEARCHER_MAX_ROUNDS = 10

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
        "ideas_to_try": [
            f"size a candidate to mid-bucket for the {bucket} bucket",
            "embed the sizing target inside build_model so the model "
            "auto-resizes to the budget",
        ],
        "divergence_axes": [
            "no researcher signal — designer must pick a divergence "
            "axis directly from the frontier listing before submit"
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


def _run_phase_a(
    *,
    challenge: dict,
    deadline: float,
    llm_kwargs: dict,
    bucket: str,
    primitives: list[str],
    digest_block: str,
) -> str:
    """Run the Phase A brainstorm pass — one chat() call, no tools.

    Returns the raw text (a numbered list) on success, "" on failure.
    Failure is non-fatal: Phase B runs with the digest only.
    """
    remaining = deadline - time.monotonic()
    if remaining < PHASE_A_MIN_SECONDS:
        _log(
            f"[researcher.phase_a] only {remaining:.0f}s left — "
            "skipping brainstorm"
        )
        return ""

    system_prompt = build_phase_a_system_prompt(
        challenge, bucket, primitives, digest_block,
    )
    user_prompt = build_phase_a_user_prompt(challenge)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    phase_a_kwargs = dict(llm_kwargs)
    phase_a_kwargs["temperature"] = PHASE_A_TEMPERATURE
    phase_a_kwargs["max_tokens"] = PHASE_A_MAX_TOKENS

    try:
        resp = chat(
            messages=messages,
            tools=None,
            deadline=deadline,
            **phase_a_kwargs,
        )
    except Exception as exc:
        _log(
            f"[researcher.phase_a] chat failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return ""

    try:
        content = resp.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""

    text = content.strip()
    if not text:
        _log("[researcher.phase_a] empty completion")
        return ""
    _log(f"[researcher.phase_a] {len(text)} chars produced")
    return text


def _build_subagent(
    *,
    challenge: dict,
    handlers: dict,
    deadline: float,
    llm_kwargs: dict,
    bucket: str,
    digest_block: str = "",
    phase_a_output: str = "",
    primitives: list[str] | None = None,
    extra_user_msg: str = "",
) -> Subagent:
    tools = build_tools(challenge, role="researcher")
    primitives = primitives or []
    user_prompt = build_researcher_user_prompt(
        challenge,
        digest_block=digest_block,
        phase_a_output=phase_a_output,
        primitives=primitives,
    )
    if extra_user_msg:
        user_prompt = user_prompt + "\n\n" + extra_user_msg
    return Subagent(
        name="researcher",
        system_prompt=build_researcher_system_prompt(
            challenge, bucket, primitives=primitives,
        ),
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
    digest_block: str = "",
    primitives: list[str] | None = None,
) -> dict:
    """Run the v3 two-phase researcher and return a brief.

    ``digest_block`` is the analyst digest rendered as a fenced JSON
    block (analyst.format_digest_for_researcher); ``primitives`` is
    the per-round injection list. Both are passed through to Phase A
    and Phase B verbatim.
    """
    _ = state
    primitives = primitives or []

    # Phase A: high-temperature brainstorm. Non-blocking — if it
    # fails, Phase B runs with only the digest.
    phase_a_output = _run_phase_a(
        challenge=challenge,
        deadline=deadline,
        llm_kwargs=llm_kwargs,
        bucket=bucket,
        primitives=primitives,
        digest_block=digest_block,
    )

    sub = _build_subagent(
        challenge=challenge,
        handlers=handlers,
        deadline=deadline,
        llm_kwargs=llm_kwargs,
        bucket=bucket,
        digest_block=digest_block,
        phase_a_output=phase_a_output,
        primitives=primitives,
    )
    result = sub.run()
    parsed = _extract_brief(result)
    if parsed is not None:
        _log(
            f"[researcher] brief parsed on first attempt "
            f"(rounds={result.rounds})"
        )
        return _normalize_brief(parsed)

    _log("[researcher] first attempt did not yield JSON — retrying once")
    sub2 = _build_subagent(
        challenge=challenge,
        handlers=handlers,
        deadline=deadline,
        llm_kwargs=llm_kwargs,
        bucket=bucket,
        digest_block=digest_block,
        phase_a_output=phase_a_output,
        primitives=primitives,
        extra_user_msg=(
            "Return the JSON brief now. ONLY a single fenced "
            "```json block with relevant_prior_work, ideas_to_try, "
            "divergence_axes, plan. No prose."
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
