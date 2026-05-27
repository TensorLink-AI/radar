"""Orchestrator for the claude_style multi-subagent miner.

Entry point: ``design_architecture(challenge, gated_client) -> dict``

The orchestrator owns the deadline and runs three specialist subagents
in sequence:

  1. researcher (≤ 20% of the budget, cap 90s) — produces a JSON brief
  2. designer (≤ 60% of the budget) — generates code, validates, ships
  3. critic — single text call between designer iterations (managed
     inside the designer loop, not at orchestrator level)

The last 30s of the wall-clock budget are reserved for packaging /
fallback. If the designer fails to ship, the orchestrator falls
through to ``core.fallback_templates.generate_fallback`` so the
round still produces a valid submission.

This file is a SKELETON — most of the wiring is stubbed pending the
stop-and-show checkpoints. The orchestrator end-to-end already
returns a packaged dict; the subagent calls are placeholders.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from typing import Optional

from core import history
from core.fallback_templates import (
    fallback_name_for, generate_fallback,
)
from core.history import extract_flops_budget, identify_bucket
from core.validation import validate_code

try:
    from .hooks import default_designer_hooks
    from .llm_client import chat, get_client
    from .prompts import (
        build_critic_prompt,
        build_designer_system_prompt, build_designer_user_prompt,
        build_researcher_system_prompt, build_researcher_user_prompt,
    )
    from .subagents.critic import run_critic
    from .subagents.designer import run_designer
    from .subagents.researcher import default_brief, run_researcher
    from .tools import SubmitSignal, build_handlers, build_tools
except ImportError:
    from hooks import default_designer_hooks
    from llm_client import chat, get_client
    from prompts import (
        build_critic_prompt,
        build_designer_system_prompt, build_designer_user_prompt,
        build_researcher_system_prompt, build_researcher_user_prompt,
    )
    from subagents.critic import run_critic
    from subagents.designer import run_designer
    from subagents.researcher import default_brief, run_researcher
    from tools import SubmitSignal, build_handlers, build_tools


FALLBACK_RESERVE_SECONDS = 30
RESEARCHER_BUDGET_FRACTION = 0.20
RESEARCHER_BUDGET_CAP = 90
DESIGNER_BUDGET_FRACTION = 0.60

DEFAULT_MODEL = "moonshotai/Kimi-K2.5-TEE"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _agent_budget(challenge: dict) -> int:
    """Resolve the seconds available to this agent.

    Same precedence as the openai_sdk agent — challenge override,
    env var, then trainer's task.time_budget as a last resort.
    """
    b = int(challenge.get("agent_seconds") or 0)
    if b <= 0:
        try:
            b = int(os.environ.get("AGENT_BUDGET_SECONDS", "0") or 0)
        except ValueError:
            b = 0
    if b <= 0:
        task = challenge.get("task", {}) or {}
        b = int(task.get("time_budget", 300) or 300)
        _log(
            f"[orchestrator] WARN: falling back to "
            f"task.time_budget={b}s — set challenge.agent_seconds or "
            "AGENT_BUDGET_SECONDS env."
        )
    return b


def _package(
    code: str, name: str, motivation: str, prompt_id: str = "",
) -> dict:
    out = {"code": code, "name": name, "motivation": motivation}
    if prompt_id:
        out["prompt_id"] = prompt_id
    return out


def _load_active_prompt(round_id: int) -> dict:
    """Return ``{id, template}`` for the prompt variant this round.

    Reads ``prompts/active.json`` (override via ``MINER_PROMPTS_DIR``)
    and round-robins the population by ``round_id``. Empty when the
    miner hasn't run ``miner/neuron.py optimize`` — designer then
    falls back to its hardcoded system prompt. ``id`` round-trips back
    via ``experiments.prompt_id`` so Phase C scores attribute to the
    variant that produced them, closing the GEPA loop.
    """
    prompts_dir = os.getenv("MINER_PROMPTS_DIR", "prompts")
    path = os.path.join(prompts_dir, "active.json")
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"id": "", "template": ""}
    rows = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        return {"id": "", "template": ""}
    pick = rows[round_id % len(rows)]
    if not isinstance(pick, dict):
        return {"id": "", "template": ""}
    return {
        "id": str(pick.get("id", "")),
        "template": str(pick.get("template", "")),
    }


def _llm_kwargs(challenge: dict) -> dict:
    """Common kwargs forwarded into every chat() call. Resolved once
    per round so the cached client is reused across subagents."""
    return {
        "llm_url": challenge.get("llm_url", "") or "",
        "agent_token": challenge.get("agent_token", "") or "",
        "miner_uid": str(challenge.get("miner_uid", "") or ""),
        "model": DEFAULT_MODEL,
        "temperature": 0.7,
        "max_tokens": 16384,
    }


def design_architecture(challenge: dict, gated_client=None) -> dict:
    """Entry point required by the harness.

    Drives researcher → designer → fallback in sequence under one
    monotonic deadline. Persists state to scratchpad so candidate
    history, hypotheses, and submissions survive across rounds.
    """
    t_start = time.monotonic()
    budget = _agent_budget(challenge)
    deadline = t_start + budget - FALLBACK_RESERVE_SECONDS

    _log(
        f"[orchestrator] start budget={budget}s "
        f"deadline_in={budget - FALLBACK_RESERVE_SECONDS}s"
    )

    flops_min, flops_max = extract_flops_budget(challenge)
    bucket = identify_bucket(flops_min, flops_max)

    # ── Active prompt variant (GEPA / random_mutate coevolution) ─
    # Stashed under "_operator_prompt" so the designer subagent can
    # append it to its system prompt without changing its builder
    # signature. ``id`` round-trips back via the returned dict so
    # Phase C scores attribute to this variant.
    round_id = int(challenge.get("round_id", 0) or 0)
    active_prompt = _load_active_prompt(round_id)
    if active_prompt["id"]:
        _log(
            f"[orchestrator] prompt variant {active_prompt['id'][:8]}… "
            f"(round_id={round_id})"
        )
        challenge["_operator_prompt"] = active_prompt["template"]
        challenge["_operator_prompt_id"] = active_prompt["id"]

    # ── Scratchpad load ─────────────────────────────────────────
    scratch_dir: Optional[str] = None
    try:
        scratch_dir = load_scratchpad(challenge)  # noqa: F821 — injected
    except NameError:
        _log(
            "[orchestrator] load_scratchpad not injected — "
            "running without scratchpad"
        )
    except Exception as exc:
        _log(f"[orchestrator] scratchpad load failed: {exc}")

    state = history.load_state(scratch_dir) if scratch_dir else {}
    prev_results = challenge.get("previous_results") or []
    if prev_results:
        history.merge_results_into_state(state, prev_results)

    # Single shared handler dict — each subagent gets the same
    # handlers but a different tool subset, so the dispatch path
    # behaves identically across roles.
    handlers = build_handlers(
        challenge,
        client=gated_client,
        scratch_dir=scratch_dir,
        deadline=deadline,
        state=state,
    )
    llm_kwargs = _llm_kwargs(challenge)

    # ── Startup config check ────────────────────────────────────
    config_broken = False
    config_error: Optional[str] = None
    try:
        get_client(
            llm_kwargs["llm_url"],
            llm_kwargs["agent_token"],
            llm_kwargs["miner_uid"],
        )
    except RuntimeError as exc:
        _log(f"[orchestrator] startup config check failed: {exc}")
        config_broken = True
        config_error = f"config error: {exc}"

    submit_sig: Optional[SubmitSignal] = None
    last_validated_code: Optional[str] = None

    if not config_broken:
        # ── Phase 1: researcher ─────────────────────────────────
        researcher_deadline = min(
            deadline,
            t_start + min(
                RESEARCHER_BUDGET_CAP,
                int(budget * RESEARCHER_BUDGET_FRACTION),
            ),
        )
        try:
            brief = run_researcher(
                challenge=challenge,
                handlers=handlers,
                deadline=researcher_deadline,
                llm_kwargs=llm_kwargs,
                state=state,
                bucket=bucket,
            )
        except Exception as exc:
            _log(f"[orchestrator] researcher crashed: {exc}")
            brief = default_brief(challenge, bucket)

        # ── Phase 2: designer ───────────────────────────────────
        designer_deadline = min(
            deadline,
            t_start + int(budget * (
                RESEARCHER_BUDGET_FRACTION + DESIGNER_BUDGET_FRACTION
            )),
        )
        # Stop early if too little time is left after research.
        if designer_deadline - time.monotonic() < FALLBACK_RESERVE_SECONDS:
            _log(
                "[orchestrator] not enough time after research for "
                "designer — skipping to fallback"
            )
        else:
            try:
                submit_sig = run_designer(
                    challenge=challenge,
                    handlers=handlers,
                    deadline=designer_deadline,
                    llm_kwargs=llm_kwargs,
                    brief=brief,
                    state=state,
                    bucket=bucket,
                )
            except Exception as exc:
                _log(f"[orchestrator] designer crashed: {exc}")

        # If the designer produced validated code without explicitly
        # submitting, recover it from the submit handler's stash —
        # same recovery path the openai_sdk agent uses.
        last_validated_code = getattr(
            handlers.get("submit", None), "_last_validated_code", "",
        ) or None

    # ── Scratchpad save ─────────────────────────────────────────
    try:
        state_holder = getattr(
            handlers.get("submit", None), "_state_holder", None,
        )
        if state_holder is not None:
            scratch_dir = scratch_dir or tempfile.mkdtemp()
            history.save_state(scratch_dir, state_holder["state"])
            try:
                save_scratchpad(challenge, scratch_dir)  # noqa: F821 — injected
            except NameError:
                pass
            except Exception as exc:
                _log(
                    f"[orchestrator] scratchpad save failed: {exc}"
                )
    except Exception as exc:
        _log(f"[orchestrator] scratchpad finalize crashed: {exc}")

    # ── Phase 3: package ────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    _log(f"[orchestrator] phase=reserve elapsed={elapsed:.0f}s")

    if submit_sig is not None:
        return _package(
            submit_sig.code, submit_sig.name, submit_sig.motivation,
            prompt_id=active_prompt["id"],
        )
    if last_validated_code:
        return _package(
            last_validated_code,
            f"auto_submit_{bucket}",
            "Auto-submitted validated code — designer did not call "
            "submit explicitly.",
            prompt_id=active_prompt["id"],
        )

    # Designer failed → fallback template path.
    fb_code = generate_fallback(challenge)
    fb_name = fallback_name_for(challenge)
    motivation = (
        f"FALLBACK: {config_error}"
        if config_error
        else "FALLBACK: designer failed to produce validated code"
    )
    return _package(
        fb_code, fb_name, motivation, prompt_id=active_prompt["id"],
    )
