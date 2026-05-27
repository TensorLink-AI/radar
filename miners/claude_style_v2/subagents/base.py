"""Subagent dataclass and shared tool-calling loop.

A ``Subagent`` is a thin shell around a private message list, a tool
subset, and a tool-handler dispatch dict. Each subagent runs against
the same OpenAI-compatible chat endpoint via ``llm_client.chat`` —
the harness deliberately reuses one transport across roles so the
proxy / retry / model-pool path is the same for every call.

The run loop is a stripped-down version of ``openai_sdk.agent._run_tool_loop``:
no per-turn header replacement, no submit-error reprompts, no
post-validation grace counter — those are orchestrator-level concerns
that don't apply to every subagent. The loop's only job is to drive
the tool-calling rounds, dispatch each call through the handler dict
(after consulting the pre-tool-call hooks), and stop on submit /
deadline / no-tool-calls / max-rounds.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    from ..llm_client import chat
    from ..tools import SubmitSignal
except ImportError:
    from llm_client import chat
    from tools import SubmitSignal


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# A pre-tool-call hook: called with (name, args, state) right before
# the handler runs. If it returns a string, that string is surfaced to
# the LLM as the tool result and the actual handler is NOT called. If
# it returns None, the handler runs normally.
HookFn = Callable[[str, dict, dict], Optional[str]]
HookRule = tuple[str, HookFn]


@dataclass
class SubagentResult:
    """What a subagent returns to the orchestrator.

    ``submit_sig`` is set only when the subagent's tools include
    ``submit`` and the LLM actually called it. ``content`` is the
    final assistant text (used by the critic, which has no tools).
    ``rounds`` is the number of LLM turns spent.
    """

    content: str = ""
    submit_sig: Optional[SubmitSignal] = None
    rounds: int = 0
    failure: Optional[str] = None
    messages: list[dict] = field(default_factory=list)


@dataclass
class Subagent:
    """One specialist subagent.

    ``name`` is informational (logged + used in failure strings).
    ``system_prompt`` / ``user_prompt`` seed the message list.
    ``tools`` is the OpenAI-format tool list this subagent sees.
    ``handlers`` is the ``{name: callable(**kwargs) -> str}`` dispatch
    dict. ``deadline`` is the monotonic-clock deadline (subagent stops
    early if the remaining budget is too small to be useful).
    ``hooks`` is a list of ``(name_pattern, hook_fn)`` rules consulted
    before each tool call. ``state`` is the orchestrator-shared dict
    each hook can read/write — used by the submit-blocking rule to
    track recent ``validate_code`` results.
    ``max_rounds`` caps the number of LLM turns this subagent will
    take. ``llm_kwargs`` forwards llm_url / agent_token / miner_uid /
    model / temperature into ``chat()``.
    ``on_tool_result``, when set, is called after each tool dispatch
    with ``(name, args, result, state)``. If it returns a non-empty
    string the string is appended to the message list as a user-role
    message — the hook the designer uses to inject critic feedback.
    """

    name: str
    system_prompt: str
    user_prompt: str
    tools: list[dict]
    handlers: dict[str, Callable[..., str]]
    deadline: float
    hooks: list[HookRule] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    max_rounds: int = 8
    llm_kwargs: dict = field(default_factory=dict)
    on_tool_result: Optional[
        Callable[[str, dict, str, dict], Optional[str]]
    ] = None
    # Min seconds remaining required to start a new chat round. Below
    # this, the loop bails to leave room for the orchestrator's
    # packaging window.
    min_round_seconds: int = 30

    def run(self) -> SubagentResult:
        """Drive the tool-calling loop. Stops on submit signal,
        deadline, or no-tool-calls assistant turn."""
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]
        result = SubagentResult(messages=messages)
        for round_num in range(1, self.max_rounds + 1):
            remaining = self.deadline - time.monotonic()
            if remaining < self.min_round_seconds:
                _log(
                    f"[{self.name}] round {round_num}: only "
                    f"{remaining:.0f}s left, stopping"
                )
                if result.failure is None:
                    result.failure = "deadline"
                break

            try:
                resp = chat(
                    messages=messages,
                    tools=self.tools or None,
                    deadline=self.deadline,
                    **self.llm_kwargs,
                )
            except Exception as exc:
                _log(
                    f"[{self.name}] round {round_num} chat failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                result.failure = f"chat: {type(exc).__name__}: {exc}"
                break

            try:
                msg = resp.choices[0].message
            except (AttributeError, IndexError) as exc:
                result.failure = f"chat: malformed response ({exc})"
                break

            messages.append(_serialize_assistant_message(msg))
            result.rounds = round_num
            result.content = getattr(msg, "content", "") or ""

            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                # Pure-text turn — done.
                break

            stopped = False
            for tc in tool_calls:
                try:
                    name = tc.function.name
                    call_id = getattr(tc, "id", "") or ""
                    raw_args = tc.function.arguments or "{}"
                except AttributeError as exc:
                    _log(f"[{self.name}] malformed tool call: {exc}")
                    continue

                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                tool_result = self._dispatch_tool(name, args)
                # Submit shipped — the dispatch helper raises this back
                # to us via the special sentinel in ``self.state``.
                sig = self.state.get("__submit_sig__")
                if sig is not None:
                    self.state.pop("__submit_sig__", None)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": "submitted",
                    })
                    result.submit_sig = sig
                    stopped = True
                    break

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(tool_result)[:4000],
                })

                # Hook-side bookkeeping: track the most recent
                # validate_code outcome so the submit-blocking rule
                # can reach it.
                if name == "validate_code":
                    ok = isinstance(tool_result, str) and (
                        tool_result.startswith("ok")
                    )
                    history = self.state.setdefault(
                        "validate_history", [],
                    )
                    history.append({"round": round_num, "ok": ok})

                # Per-tool post-result callback. Used by the designer
                # to fire the critic between iterations. Returning a
                # non-empty string appends a user-role message; None
                # is a no-op.
                if self.on_tool_result is not None:
                    try:
                        injected = self.on_tool_result(
                            name, args, str(tool_result), self.state,
                        )
                    except Exception as exc:
                        _log(
                            f"[{self.name}] on_tool_result raised: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        injected = None
                    if injected:
                        messages.append({
                            "role": "user",
                            "content": str(injected)[:4000],
                        })

            if stopped:
                break
        result.messages = messages
        return result

    def _dispatch_tool(self, name: str, args: dict) -> str:
        """Run the pre-tool-call hooks, then the actual handler.

        If a hook returns a string it short-circuits the handler.
        SubmitSignal is converted into a state-side flag so the
        caller can break out of the round cleanly without leaking
        an exception across the subagent boundary.
        """
        for pattern, hook in self.hooks:
            if pattern == name or pattern == "*":
                short = hook(name, args, self.state)
                if isinstance(short, str):
                    return short

        handler = self.handlers.get(name)
        if handler is None:
            return f"unknown tool: {name}"
        try:
            return handler(**args)
        except SubmitSignal as sig:
            self.state["__submit_sig__"] = sig
            return "submitted"
        except Exception as exc:
            return f"tool error: {exc}"


def _serialize_assistant_message(msg) -> dict:
    """Convert an SDK ChatCompletionMessage into a plain dict suitable
    for feeding back into the next chat call. Mirrors the helper in
    ``openai_sdk.agent`` — kept here so the subagent module is
    self-contained.
    """
    if hasattr(msg, "model_dump"):
        try:
            return msg.model_dump(exclude_none=True)
        except Exception:
            pass
    out: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(msg, "content", "") or "",
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        serialised = []
        for tc in tool_calls:
            serialised.append({
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
        out["tool_calls"] = serialised
    return out
