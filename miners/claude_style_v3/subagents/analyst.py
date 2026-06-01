"""Analyst subagent — landscape reflection over the global DB.

The analyst is v3's first subagent. It runs once per round, BEFORE
the researcher, with a read-only tool surface limited to ``query_db``,
``list_frontier``, ``analyze_task``, ``time_remaining``. Its output is
a structured **design-landscape digest** the researcher embeds
verbatim:

    {
      "axes_explored":            [...],
      "axes_fixed_across_frontier": [...],
      "saturated_families":       [...],
      "absent_families":          [...],
      "failure_patterns":         [...],
      "injected_primitives_eval": [
        {"primitive": "...", "status": "absent|present|saturated",
         "recommendation": "explore|skip|hybridize"}
      ],
      "recommend_explore":        [...],
      "recommend_avoid":          [...]
    }

The point isn't to enumerate every model on the DB — it's to compute
**axes of variation** the existing frontier doesn't cover, so the
researcher has a generative signal pointing toward novel directions
(technique #1).

The injected primitives (technique #5) flow through the analyst so
it can pre-classify each one against the DB before the researcher
sees them — saves the researcher from re-querying.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Optional

try:
    from ..prompts import (
        build_analyst_system_prompt, build_analyst_user_prompt,
    )
    from ..tools import build_tools
    from .base import Subagent, SubagentResult
except ImportError:
    from prompts import (
        build_analyst_system_prompt, build_analyst_user_prompt,
    )
    from tools import build_tools
    from subagents.base import Subagent, SubagentResult


DIGEST_KEYS = (
    "axes_explored",
    "axes_fixed_across_frontier",
    "saturated_families",
    "absent_families",
    "failure_patterns",
    "injected_primitives_eval",
    "recommend_explore",
    "recommend_avoid",
)

ANALYST_MAX_ROUNDS = 6

JSON_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    m = JSON_FENCE_RE.search(text)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1))
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


def _normalize_digest(obj: dict, primitives: list[str]) -> dict:
    """Coerce the parsed digest into the expected shape.

    All list-shaped fields default to ``[]``. The
    ``injected_primitives_eval`` field is back-filled from the
    injection list so the researcher always sees one entry per
    primitive even when the analyst forgot it.
    """
    out = dict(obj)
    for key in DIGEST_KEYS:
        if key == "injected_primitives_eval":
            continue
        v = out.get(key)
        if not isinstance(v, list):
            out[key] = [] if v is None else [str(v)]

    raw_eval = out.get("injected_primitives_eval") or []
    indexed: dict[str, dict] = {}
    if isinstance(raw_eval, list):
        for row in raw_eval:
            if isinstance(row, dict) and isinstance(row.get("primitive"), str):
                indexed[row["primitive"]] = row
    rebuilt: list[dict] = []
    for p in primitives:
        existing = indexed.get(p)
        if existing:
            rebuilt.append({
                "primitive": p,
                "status": str(existing.get("status") or "unknown"),
                "recommendation": str(
                    existing.get("recommendation") or "explore"
                ),
                "note": str(existing.get("note") or ""),
            })
        else:
            rebuilt.append({
                "primitive": p, "status": "unknown",
                "recommendation": "explore", "note": "analyst skipped",
            })
    out["injected_primitives_eval"] = rebuilt
    return out


def default_digest(
    challenge: dict, bucket: str, primitives: list[str],
) -> dict:
    """Digest emitted when the analyst LLM fails. Conservative — the
    researcher should treat ``_default: True`` as "do the work yourself".
    """
    return {
        "axes_explored": [],
        "axes_fixed_across_frontier": [],
        "saturated_families": [],
        "absent_families": [],
        "failure_patterns": [],
        "injected_primitives_eval": [
            {"primitive": p, "status": "unknown",
             "recommendation": "explore", "note": ""}
            for p in primitives
        ],
        "recommend_explore": [],
        "recommend_avoid": [],
        "_default": True,
    }


def run_analyst(
    *,
    challenge: dict,
    handlers: dict,
    deadline: float,
    llm_kwargs: dict,
    bucket: str,
    primitives: list[str],
) -> dict:
    """Run the analyst subagent and return a digest."""
    tools = build_tools(challenge, role="analyst")
    system_prompt = build_analyst_system_prompt(
        challenge, bucket, primitives,
    )
    user_prompt = build_analyst_user_prompt(challenge, primitives)

    sub = Subagent(
        name="analyst",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tools,
        handlers=handlers,
        deadline=deadline,
        hooks=[],
        state={},
        max_rounds=ANALYST_MAX_ROUNDS,
        llm_kwargs=llm_kwargs,
        # Lower than Subagent default (30s) — analyst's slice is
        # 10% of total, so a 30s floor would prematurely bail on
        # short budgets. 10s is enough for one short tool round.
        min_round_seconds=10,
    )
    result = sub.run()
    parsed = _extract_digest(result)
    if parsed is not None:
        _log(
            f"[analyst] digest parsed "
            f"(rounds={result.rounds})"
        )
        return _normalize_digest(parsed, primitives)

    _log(
        "[analyst] failed to yield digest JSON — using default digest"
    )
    return default_digest(challenge, bucket, primitives)


def _extract_digest(result: SubagentResult) -> Optional[dict]:
    for source in (result.content, *_assistant_contents(result.messages)):
        obj = _extract_json_object(source or "")
        if obj is not None and any(k in obj for k in DIGEST_KEYS):
            return obj
    return None


def _assistant_contents(messages: list[dict]):
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            if content:
                yield content


def format_digest_for_researcher(digest: dict) -> str:
    """Render the digest as a fenced JSON block for the researcher's
    user prompt."""
    return "```json\n" + json.dumps(digest, indent=2) + "\n```"
