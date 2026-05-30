"""Critic subagent.

A single text-completion call (no tools) between designer iterations.
Input: the most recent ``validate_code`` result + the current code.
Output: a short structured ``keep / change / drop`` critique the
orchestrator injects into the designer's next user-role turn.

The critic uses ``llm_client.chat`` directly — no Subagent loop,
because the contract is a single completion with no tool surface.
"""
from __future__ import annotations

import sys

try:
    from ..llm_client import chat
    from ..prompts import build_critic_prompt, build_critic_system_prompt
except ImportError:
    from llm_client import chat
    from prompts import build_critic_prompt, build_critic_system_prompt


# Critic calls are short. Smaller max_tokens than the designer keeps
# the round budget in check, and a lower temperature keeps the
# critique consistent across iterations.
CRITIC_MAX_TOKENS = 512
CRITIC_TEMPERATURE = 0.4


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_critic(
    *,
    code: str,
    validation_result: str,
    deadline: float,
    llm_kwargs: dict,
    operator_directive: str = "",
    operator_directive_id: str = "",
) -> str:
    """Return a short critique string. Empty string on failure or
    when there's no code to critique.

    ``operator_directive`` is the GEPA-evolved critic-slot prompt;
    when set, it's appended to the hardcoded critic system prompt so
    the active variant steers the KEEP/CHANGE/DROP critique.
    """
    if not code or not code.strip():
        return ""
    if not validation_result:
        return ""

    system_prompt = build_critic_system_prompt()
    if operator_directive:
        system_prompt = (
            f"{system_prompt}\n\n"
            f"## Operator Directive — critic slot "
            f"(variant {operator_directive_id[:8]})\n"
            f"{operator_directive}"
        )
    user_prompt = build_critic_prompt(code, validation_result)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Override max_tokens / temperature for the critic — the rest of
    # llm_kwargs (llm_url, agent_token, miner_uid, model) flow
    # through unchanged so the same proxy / cache is reused.
    critic_kwargs = dict(llm_kwargs)
    critic_kwargs["max_tokens"] = CRITIC_MAX_TOKENS
    critic_kwargs["temperature"] = CRITIC_TEMPERATURE

    try:
        resp = chat(
            messages=messages,
            tools=None,
            deadline=deadline,
            **critic_kwargs,
        )
    except Exception as exc:
        _log(f"[critic] chat failed: {type(exc).__name__}: {exc}")
        return ""

    try:
        content = resp.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""
    return content.strip()
