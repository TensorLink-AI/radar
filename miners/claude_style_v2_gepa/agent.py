"""Orchestrator for the claude_style_gepa multi-subagent miner.

GEPA-tuned variant of ``claude_style_v2``. Identical orchestration
(researcher → designer → critic) but the prompt-population surface
is **multi-slot**: ``prompts/active.json`` carries rows tagged via
``metadata.slot`` ∈ {``researcher``, ``designer``, ``critic``}, and
each subagent appends its own operator directive to its hand-written
system prompt. This lets the GEPA loop co-evolve three independent
instruction strings instead of a single tail-appended blob.

Entry point: ``design_architecture(challenge, gated_client) -> dict``

The shipped result's ``prompt_id`` is a composite
``r:<rid>|d:<did>|c:<cid>`` so each slot's variant ID round-trips
back to the optimizer; ``local/optimize.py --slot <name>`` filters
experiments to one slot's variants for per-slot GEPA passes.
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
# Was 0.20 — researcher capped too tight on long budgets.
RESEARCHER_BUDGET_FRACTION = 0.15
# Was 90s — barely enough on a 30-min budget. Allows up to 5 min of
# research when the budget is large; small budgets stay bounded by the
# fraction.
RESEARCHER_BUDGET_CAP = 300
# Was 0.60 — designer was structurally capped at 60% of the budget,
# leaving 20%+ idle even when the LLM wanted to keep iterating.
DESIGNER_BUDGET_FRACTION = 0.80

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


SLOTS = ("researcher", "designer", "critic")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PROMPTS_DIR = os.path.join(_HERE, "prompts")


def _load_active_prompts(round_id: int) -> dict:
    """Return ``{slot: {id, template}}`` for each subagent slot.

    Reads ``prompts/active.json`` (override via ``MINER_PROMPTS_DIR``,
    defaults to ``<agent_dir>/prompts``), groups rows by
    ``metadata.slot``, and round-robins within each slot by
    ``round_id``. Rows without a slot tag (or with an unknown one)
    are ignored — the corresponding subagent then runs on its
    hardcoded system prompt alone. Empty / unreadable file → all
    slots return ``{"id": "", "template": ""}``.

    Each slot's ``id`` round-trips back via the composite
    ``prompt_id`` on the submission so the optimizer can attribute
    Phase C scores to specific slot variants.
    """
    blank = {slot: {"id": "", "template": ""} for slot in SLOTS}
    prompts_dir = os.getenv("MINER_PROMPTS_DIR", _DEFAULT_PROMPTS_DIR)
    path = os.path.join(prompts_dir, "active.json")
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return blank
    rows = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        return blank

    by_slot: dict[str, list[dict]] = {slot: [] for slot in SLOTS}
    for row in rows:
        if not isinstance(row, dict):
            continue
        slot = (row.get("metadata") or {}).get("slot")
        if slot in by_slot:
            by_slot[slot].append(row)

    picks = dict(blank)
    for slot in SLOTS:
        bucket = by_slot[slot]
        if not bucket:
            continue
        pick = bucket[round_id % len(bucket)]
        picks[slot] = {
            "id": str(pick.get("id", "")),
            "template": str(pick.get("template", "")),
        }
    return picks


def _composite_prompt_id(picks: dict) -> str:
    """Compose ``r:<rid>|d:<did>|c:<cid>``. Empty pieces are dropped so
    a missing slot doesn't pollute the audit string."""
    parts = []
    for prefix, slot in (("r", "researcher"), ("d", "designer"), ("c", "critic")):
        pid = (picks.get(slot) or {}).get("id") or ""
        if pid:
            parts.append(f"{prefix}:{pid}")
    return "|".join(parts)


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

    # ── Active prompt variants (GEPA co-evolution, one per slot) ─
    # Each subagent reads its own ``_operator_prompt_<slot>`` and
    # appends it to its hand-written system prompt. The composite
    # ``prompt_id`` round-trips all three variant IDs so the
    # optimizer can attribute Phase C scores to per-slot variants.
    round_id = int(challenge.get("round_id", 0) or 0)
    active_prompts = _load_active_prompts(round_id)
    prompt_id_composite = _composite_prompt_id(active_prompts)
    for slot in SLOTS:
        slot_data = active_prompts[slot]
        if not slot_data["id"]:
            continue
        challenge[f"_operator_prompt_{slot}"] = slot_data["template"]
        challenge[f"_operator_prompt_{slot}_id"] = slot_data["id"]
        _log(
            f"[orchestrator] {slot} variant {slot_data['id'][:8]}… "
            f"(round_id={round_id})"
        )

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
            prompt_id=prompt_id_composite,
        )

    # Recovery: deadline hit and no SubmitSignal raised, but the LLM
    # stashed a best-so-far via the time-gated submit handler. Ship it.
    if submit_sig is None:
        state_holder = getattr(handlers.get("submit", None), "_state_holder", None)
        best = (state_holder or {}).get("state", {}).get("best_so_far") if state_holder else None
        if best and best.get("code"):
            _log(
                f"[agent] shipping stashed best-so-far "
                f"(name={best.get('name')!r}, no late-window submit)"
            )
            return _package(
                best["code"],
                best.get("name") or f"best_so_far_{bucket}",
                best.get("motivation") or "Auto-shipped best-so-far candidate.",
                prompt_id=prompt_id_composite,
            )

    if last_validated_code:
        return _package(
            last_validated_code,
            f"auto_submit_{bucket}",
            "Auto-submitted validated code — designer did not call "
            "submit explicitly.",
            prompt_id=prompt_id_composite,
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
        fb_code, fb_name, motivation, prompt_id=prompt_id_composite,
    )
