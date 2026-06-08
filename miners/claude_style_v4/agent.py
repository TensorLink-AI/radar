"""Orchestrator for the claude_style_v4 multi-subagent miner.

v4 extends ``claude_style_v3``'s analyst → researcher → designer
pipeline with a focused **in-round recipe-tuning second pass** so the
designer doesn't have to juggle architecture and training recipe in a
single decision. Three concrete additions over v3:

1. **Analyst now reports a `training_recipe_norm`** (optimizer / LR /
   schedule / loss / AMP / batch identical across the frontier).
   The analyst gains ``get_frontier_member`` to read 1-2 members'
   recipe hooks. The new field rides in the digest alongside the
   architectural axes and is consumed by the recipe-tuner, NOT the
   researcher — the researcher's brief stays architecture-only so
   it doesn't have to balance two surfaces.

2. **Recipe-tuner pass** runs after the designer ships a validated
   architecture (only on fresh rounds — continuation rounds already
   ARE a recipe pass). It reuses the existing continuation
   machinery: the just-validated architecture is extracted via
   ``continuation.extract_arch_source`` and pinned into the next
   designer call as a frozen block. The designer's user prompt
   switches to a recipe-only preamble via
   ``_inround_recipe_context`` — same focused-single-task framing
   the continuation flow uses, minus the warm-start. The agent never
   sees architecture + recipe at the same time.

3. **Fallback templates ship a strong default recipe** (AdamW +
   warmup-cosine + bfloat16 AMP + grad clip) instead of the
   bare ``Adam(lr=1e-3)`` v3 emitted. This is the prior the agent
   inherits when generation fails or when it copies a fallback
   template; a stronger prior raises the floor without the agent
   having to do anything.

Pipeline (fresh round):
  ``analyst → (phase A) → researcher → designer (arch) → recipe-tuner``

Pipeline (continuation round):
  ``designer (recipe-only, parent arch frozen)`` — unchanged from v3.

Entry point: ``design_architecture(challenge, gated_client) -> dict``

Budget split: analyst 10% (cap 120s), researcher 12% (cap 300s),
designer ~60% (cap = full budget minus the recipe reserve), recipe
tuner ~18% (cap 330s, the EARLY_SUBMIT_GATE + buffer so pass-1's
submit always stashes and pass-2's submit always ships). Last 30s
reserved for packaging. Continuation rounds skip the recipe-tuner —
the designer already handles the recipe via the continuation preamble.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from typing import Optional

from core import continuation
from core import history
from core.fallback_templates import (
    fallback_name_for, generate_fallback,
)
from core.history import extract_flops_budget, identify_bucket
from core.primitives import sample_primitives
from core.validation import validate_code

try:
    from .hooks import default_designer_hooks
    from .llm_client import chat, get_client
    from .prompts import (
        build_critic_prompt,
        build_designer_system_prompt, build_designer_user_prompt,
        build_pipeline_designer_system_prompt,
        build_pipeline_designer_user_prompt,
        build_researcher_system_prompt, build_researcher_user_prompt,
    )
    from .subagents.analyst import (
        default_digest, format_digest_for_researcher, run_analyst,
    )
    from .subagents.base import Subagent
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
        build_pipeline_designer_system_prompt,
        build_pipeline_designer_user_prompt,
        build_researcher_system_prompt, build_researcher_user_prompt,
    )
    from subagents.analyst import (
        default_digest, format_digest_for_researcher, run_analyst,
    )
    from subagents.base import Subagent
    from subagents.critic import run_critic
    from subagents.designer import run_designer
    from subagents.researcher import default_brief, run_researcher
    from tools import SubmitSignal, build_handlers, build_tools


FALLBACK_RESERVE_SECONDS = 30
# v3 carves the analyst out of the researcher's slice. Analyst is
# cheap (read-only, ≤6 tool calls) so 10% / 120s cap is plenty.
ANALYST_BUDGET_FRACTION = 0.10
ANALYST_BUDGET_CAP = 120
# Was 0.15 — dropped slightly to fund the analyst. Phase A
# brainstorm draws from this slice too.
RESEARCHER_BUDGET_FRACTION = 0.12
RESEARCHER_BUDGET_CAP = 300
DESIGNER_BUDGET_FRACTION = 0.78
# v4: reserve a fixed tail for the in-round recipe-tuning second
# pass. Chosen ≥ EARLY_SUBMIT_GATE_SECONDS (300) so the designer's
# pass-1 submit at its deadline always *stashes* (never ships),
# leaving pass-2 free to ship the recipe-tuned candidate as the
# round's actual submission. Extra 30s is buffer for pass-2's own
# packaging window. Continuation rounds bypass this — they're
# already a recipe pass.
RECIPE_TUNER_BUDGET_CAP = 330
# Below this much wall-clock at pass-2 start, skip the recipe pass
# and ship pass-1's stashed best-so-far. The recipe tuner needs at
# least ~one validate cycle to be worth running.
RECIPE_TUNER_MIN_SECONDS = 90
# Skip the recipe pass for tiny round budgets — splitting a 5-min
# round buys us nothing. Threshold = 2x RECIPE_TUNER_BUDGET_CAP so
# pass-1 still gets at least the same window pass-2 does.
RECIPE_TUNER_MIN_TOTAL_BUDGET = 660
# Number of architectural primitives injected per round (technique #5).
PRIMITIVES_PER_ROUND = 2

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
    mode: Optional[str] = None, parent_index: Optional[int] = None,
) -> dict:
    out = {"code": code, "name": name, "motivation": motivation}
    if prompt_id:
        out["prompt_id"] = prompt_id
    # Continuation: the validator reads payload["mode"]/["parent_index"]
    # and warm-starts from the parent checkpoint. Only set when the
    # submitted code actually reproduced the parent's frozen architecture.
    if mode:
        out["mode"] = mode
    if parent_index is not None:
        out["parent_index"] = parent_index
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


def _run_recipe_tuner_pass(
    *,
    challenge: dict,
    handlers: dict,
    deadline: float,
    llm_kwargs: dict,
    state: dict,
    bucket: str,
    digest: Optional[dict],
    pass1_submit: Optional[SubmitSignal],
    pass1_validated: Optional[str],
) -> Optional[SubmitSignal]:
    """v4 in-round recipe-tuning second pass.

    Reuses the designer subagent with a recipe-only preamble. The
    architecture from pass 1 is pinned via ``_inround_recipe_context``
    so the designer sees the same focused single-task framing as a
    continuation round (frozen arch + recipe surface only) — never a
    mix of both. Returns the SubmitSignal from pass 2 when it ships a
    candidate whose architecture still matches pass-1; ``None``
    otherwise (caller should ship pass-1's candidate instead).

    Failure modes (all degrade silently — pass-1's candidate ships):
      * remaining time below ``RECIPE_TUNER_MIN_SECONDS``
      * no validated architecture from pass-1 to freeze
      * ``extract_arch_source`` couldn't parse the pass-1 code
      * pass-2's submission diverged from the frozen architecture
    """
    remaining = deadline - time.monotonic()
    if remaining < RECIPE_TUNER_MIN_SECONDS:
        _log(
            f"[orchestrator] recipe-tuner: only {remaining:.0f}s left "
            "— shipping pass-1 candidate"
        )
        return None

    # Pick the architecture to freeze. Preference order: an explicit
    # SubmitSignal beats best_so_far beats the last validated code.
    state_holder = getattr(handlers.get("submit", None), "_state_holder", None)
    pass1_best: dict = {}
    if state_holder:
        sb = state_holder.get("state", {}).get("best_so_far")
        if isinstance(sb, dict):
            pass1_best = sb
    arch_code = (
        (pass1_submit.code if pass1_submit else "")
        or pass1_best.get("code", "")
        or (pass1_validated or "")
    )
    if not arch_code:
        _log(
            "[orchestrator] recipe-tuner: pass-1 produced no validated "
            "architecture — skipping"
        )
        return None

    recipe_norm = {}
    if isinstance(digest, dict):
        rn = digest.get("training_recipe_norm")
        if isinstance(rn, dict):
            recipe_norm = rn

    recipe_ctx = continuation.make_inround_recipe_context(
        arch_code, recipe_norm,
    )
    if recipe_ctx is None:
        _log(
            "[orchestrator] recipe-tuner: could not extract pass-1 "
            "architecture — skipping"
        )
        return None

    # Snapshot pass-1's best_so_far so we can restore it if pass-2
    # diverges from the frozen architecture (and the recovery path
    # would otherwise ship pass-2's diverged code).
    saved_best = dict(pass1_best) if pass1_best else None

    pass2_brief = {
        "_inround_recipe": True,
        "summary": (
            "Recipe-tuning pass. The architecture from pass 1 is pinned "
            "in the preamble above as a frozen block — copy it verbatim. "
            "Spend this slice on the training recipe only."
        ),
        "plan": [
            "Pick a real optimizer: AdamW with weight_decay and betas "
            "tuned to depth, not Adam(lr=1e-3).",
            "Add build_scheduler: warmup + cosine/linear decay.",
            "Set training_config(batch_size, grad_clip, val cadence).",
            "Enable bfloat16 AMP via configure_amp on GPU.",
            "If the analyst's training_recipe_norm shows everyone "
            "uses the same loss / lacks weight decay / lacks "
            "warmup, that's a divergence target — break with it.",
            "Copy the frozen architecture verbatim, validate_code, submit.",
        ],
    }

    challenge["_inround_recipe_context"] = recipe_ctx
    pass2_sig: Optional[SubmitSignal] = None
    try:
        pass2_sig = run_designer(
            challenge=challenge,
            handlers=handlers,
            deadline=deadline,
            llm_kwargs=llm_kwargs,
            brief=pass2_brief,
            state=state,
            bucket=bucket,
        )
    except Exception as exc:
        _log(f"[orchestrator] recipe-tuner crashed: {exc}")
    finally:
        challenge.pop("_inround_recipe_context", None)

    # Architecture-divergence guard. If pass 2 changed the architecture,
    # discard its submission and restore pass-1's stashed best so the
    # recovery path picks it up instead.
    pass2_code = ""
    if pass2_sig is not None:
        pass2_code = pass2_sig.code
    elif state_holder:
        sb2 = state_holder.get("state", {}).get("best_so_far") or {}
        if isinstance(sb2, dict):
            pass2_code = sb2.get("code") or ""

    if pass2_code and not continuation.arch_matches(
        pass2_code, recipe_ctx["frozen_arch"],
    ):
        _log(
            "[orchestrator] recipe-tuner: pass-2 diverged from frozen "
            "architecture — keeping pass-1 candidate"
        )
        if state_holder and saved_best:
            state_holder["state"]["best_so_far"] = saved_best
        return None

    if pass2_sig is not None:
        _log("[orchestrator] recipe-tuner shipped a recipe-tuned candidate")
    else:
        _log(
            "[orchestrator] recipe-tuner did not ship a SubmitSignal "
            "— recovery path will pick up the stashed best-so-far"
        )
    return pass2_sig


PIPELINE_DESIGNER_MAX_ROUNDS = 30


def _design_data_pipeline(challenge: dict, gated_client=None) -> dict:
    """ts_data_pipeline flow: single designer pass against the frozen arch.

    No analyst, no researcher, no recipe-tuner. The design space is
    well-scoped (procedural / augmentor / hybrid / mixer) and described
    in the pipeline-designer system prompt; the analyst's "axes of
    variation" framing doesn't apply because the architecture is fixed
    externally. The scratchpad still threads cross-round memory through
    the same handlers the other v4 subagents use.
    """
    from core.fallback_templates import fallback_name_for, generate_fallback
    from core import history

    t_start = time.monotonic()
    budget = _agent_budget(challenge)
    deadline = t_start + budget - FALLBACK_RESERVE_SECONDS
    _log(
        f"[orchestrator] ts_data_pipeline start budget={budget}s "
        f"deadline_in={budget - FALLBACK_RESERVE_SECONDS}s"
    )

    round_id = int(challenge.get("round_id", 0) or 0)
    active_prompt = _load_active_prompt(round_id)
    if active_prompt["id"]:
        challenge["_operator_prompt"] = active_prompt["template"]
        challenge["_operator_prompt_id"] = active_prompt["id"]

    fa = challenge.get("frozen_arch") or {}
    _log(
        f"[orchestrator] frozen_arch v{fa.get('version')} "
        f"(source exp #{fa.get('source_experiment_id')}, "
        f"metric={fa.get('source_metric')!r})"
    )

    scratch_dir: Optional[str] = None
    try:
        scratch_dir = load_scratchpad(challenge)  # noqa: F821 — injected
    except NameError:
        _log("[orchestrator] load_scratchpad not injected — no scratchpad")
    except Exception as exc:
        _log(f"[orchestrator] scratchpad load failed: {exc}")

    state = history.load_state(scratch_dir) if scratch_dir else {}
    prev_results = challenge.get("previous_results") or []
    if prev_results:
        history.merge_results_into_state(state, prev_results)

    handlers = build_handlers(
        challenge,
        client=gated_client,
        scratch_dir=scratch_dir,
        deadline=deadline,
        state=state,
    )
    llm_kwargs = _llm_kwargs(challenge)

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
        tools = build_tools(challenge, role="pipeline_designer")
        designer_sys = build_pipeline_designer_system_prompt(challenge)
        op_directive = challenge.get("_operator_prompt") or ""
        op_id = challenge.get("_operator_prompt_id") or ""
        if op_directive:
            designer_sys = (
                f"{designer_sys}\n\n"
                f"## Operator Directive (prompt variant {op_id[:8]})\n"
                f"{op_directive}"
            )
        # No researcher brief on this task — the design space is fully
        # described by the system prompt. Pass an empty dict so the user
        # prompt skips the brief section.
        sub = Subagent(
            name="pipeline_designer",
            system_prompt=designer_sys,
            user_prompt=build_pipeline_designer_user_prompt(challenge, None),
            tools=tools,
            handlers=handlers,
            deadline=deadline,
            hooks=default_designer_hooks(),
            state=state,
            max_rounds=PIPELINE_DESIGNER_MAX_ROUNDS,
            llm_kwargs=llm_kwargs,
        )
        try:
            result = sub.run()
            submit_sig = result.submit_sig
            if submit_sig is not None:
                _log(
                    f"[pipeline_designer] shipped via submit "
                    f"(rounds={result.rounds}, name={submit_sig.name})"
                )
            else:
                _log(
                    f"[pipeline_designer] no submit "
                    f"(rounds={result.rounds}, failure={result.failure})"
                )
        except Exception as exc:
            _log(f"[orchestrator] pipeline_designer crashed: {exc}")
        last_validated_code = getattr(
            handlers.get("submit", None), "_last_validated_code", "",
        ) or None

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
                _log(f"[orchestrator] scratchpad save failed: {exc}")
    except Exception as exc:
        _log(f"[orchestrator] scratchpad finalize crashed: {exc}")

    elapsed = time.monotonic() - t_start
    _log(f"[orchestrator] ts_data_pipeline elapsed={elapsed:.0f}s")

    if submit_sig is not None:
        return _package(
            submit_sig.code, submit_sig.name, submit_sig.motivation,
            prompt_id=active_prompt["id"],
        )

    state_holder = getattr(handlers.get("submit", None), "_state_holder", None)
    best = (
        (state_holder or {}).get("state", {}).get("best_so_far")
        if state_holder else None
    )
    if best and best.get("code"):
        _log(
            f"[orchestrator] shipping stashed best-so-far "
            f"(name={best.get('name')!r})"
        )
        return _package(
            best["code"],
            best.get("name") or f"pipeline_best_v{fa.get('version')}",
            best.get("motivation") or "Auto-shipped best-so-far candidate.",
            prompt_id=active_prompt["id"],
        )

    if last_validated_code:
        return _package(
            last_validated_code,
            f"pipeline_auto_v{fa.get('version')}",
            "Auto-submitted validated code — designer did not call submit.",
            prompt_id=active_prompt["id"],
        )

    fb_code = generate_fallback(challenge)
    fb_name = fallback_name_for(challenge)
    motivation = (
        f"FALLBACK: {config_error}"
        if config_error
        else "FALLBACK: pipeline designer did not produce validated code"
    )
    return _package(fb_code, fb_name, motivation, prompt_id=active_prompt["id"])


def design_architecture(challenge: dict, gated_client=None) -> dict:
    """Entry point required by the harness.

    Drives researcher → designer → fallback in sequence under one
    monotonic deadline. Persists state to scratchpad so candidate
    history, hypotheses, and submissions survive across rounds.

    ``ts_data_pipeline`` rounds short-circuit to a focused single-designer
    flow (``_design_data_pipeline``) — the analyst/researcher/recipe
    machinery doesn't apply to a data-generator task and the prompt
    surface is entirely different.
    """
    task_name = (challenge.get("task") or {}).get("name") or ""
    # synthetic_data_generator shares ts_data_pipeline's build_pipeline surface
    # (only the validator-side arch + scoring differ), so it uses the same flow.
    if task_name in ("ts_data_pipeline", "synthetic_data_generator"):
        return _design_data_pipeline(challenge, gated_client)

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

    # ── Continuation context ────────────────────────────────────
    # On a scheduled continuation round, lift the chosen parent's frozen
    # architecture so the designer reproduces it verbatim and tunes only
    # the training recipe. Any failure → cont_ctx is None → design fresh.
    db_url = (challenge.get("db_url") or "").rstrip("/")
    try:
        cont_ctx = continuation.build_context(challenge, gated_client, db_url)
    except Exception as exc:  # noqa: BLE001
        _log(f"[orchestrator] continuation context failed: {exc}")
        cont_ctx = None
    if cont_ctx:
        challenge["_continuation_context"] = cont_ctx
        _log(
            f"[orchestrator] CONTINUATION round — parent "
            f"#{cont_ctx['parent_id']}, architecture frozen, skipping "
            "analyst + researcher"
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
    # v4: digest is created in the analyst branch below; we hold a ref
    # here so the recipe-tuner pass can read `training_recipe_norm` off
    # it without re-running the analyst.
    digest: Optional[dict] = None

    # ── Primitive injection (technique #5) ──────────────────────
    # Deterministic on (round_id, task_name) so parallel miners on the
    # same round see the same injection — clean attribution if you
    # ever want to A/B specific primitives across miners.
    task_name = (challenge.get("task") or {}).get("name") or ""
    primitives = sample_primitives(
        round_id=round_id, task_name=task_name, n=PRIMITIVES_PER_ROUND,
    )
    _log(
        f"[orchestrator] injected primitives "
        f"(round_id={round_id}, task={task_name!r}): {primitives}"
    )

    if not config_broken:
        if cont_ctx:
            # Continuation: the architecture is frozen, so the analyst
            # (axes of variation) and the brainstorm researcher have
            # nothing to explore — hand the designer a recipe-focused
            # brief and let it use the whole budget on the recipe.
            brief = continuation.recipe_brief(cont_ctx)
        else:
            # ── Phase 1: analyst (technique #1) ─────────────────
            analyst_deadline = min(
                deadline,
                t_start + min(
                    ANALYST_BUDGET_CAP,
                    int(budget * ANALYST_BUDGET_FRACTION),
                ),
            )
            try:
                digest = run_analyst(
                    challenge=challenge,
                    handlers=handlers,
                    deadline=analyst_deadline,
                    llm_kwargs=llm_kwargs,
                    bucket=bucket,
                    primitives=primitives,
                )
            except Exception as exc:
                _log(f"[orchestrator] analyst crashed: {exc}")
                digest = default_digest(challenge, bucket, primitives)
            # Researcher reads architectural fields only — the
            # ``training_recipe_norm`` slot rides through to the
            # recipe-tuner pass without leaking into the brief.
            digest_block = format_digest_for_researcher(digest)

            # ── Phase 2: researcher (two-phase, technique #3) ───
            # Cumulative-from-t_start cap; if the analyst finished
            # early the researcher reclaims the slack naturally.
            researcher_deadline = min(
                deadline,
                t_start + min(
                    ANALYST_BUDGET_CAP + RESEARCHER_BUDGET_CAP,
                    int(budget * (
                        ANALYST_BUDGET_FRACTION + RESEARCHER_BUDGET_FRACTION
                    )),
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
                    digest_block=digest_block,
                    primitives=primitives,
                )
            except Exception as exc:
                _log(f"[orchestrator] researcher crashed: {exc}")
                brief = default_brief(challenge, bucket)

        # ── Phase 3: designer (architecture pass) ───────────────
        # v4: reserve a tail for the recipe-tuner pass on fresh rounds.
        # Continuation rounds keep v3's full-slice designer because the
        # continuation preamble already converts the designer into a
        # recipe-only pass — there's no architecture to invent. Tiny
        # budgets fall back to the v3 behaviour: splitting buys nothing.
        do_recipe_pass = (
            cont_ctx is None
            and budget >= RECIPE_TUNER_MIN_TOTAL_BUDGET
        )
        designer_deadline = min(
            deadline,
            t_start + int(budget * (
                ANALYST_BUDGET_FRACTION
                + RESEARCHER_BUDGET_FRACTION
                + DESIGNER_BUDGET_FRACTION
            )),
        )
        if do_recipe_pass:
            # Cap the architecture pass so it finishes ≥ RECIPE_TUNER_BUDGET_CAP
            # before the deadline. Pass-1 submits then stash (remaining >
            # EARLY_SUBMIT_GATE_SECONDS = 300) rather than ship.
            designer_deadline = min(
                designer_deadline, deadline - RECIPE_TUNER_BUDGET_CAP,
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

        # ── Phase 3b: recipe-tuner (in-round second pass) ───────
        # v4 — a separate, single-focus prompt that freezes the
        # validated architecture and asks the designer to write the
        # training recipe only. Same designer subagent, same tools;
        # only the user-prompt preamble switches (via
        # ``_inround_recipe_context``). Skipped on continuation rounds
        # and when there's no validated architecture to pin.
        if do_recipe_pass and not config_broken:
            submit_sig = _run_recipe_tuner_pass(
                challenge=challenge,
                handlers=handlers,
                deadline=deadline,
                llm_kwargs=llm_kwargs,
                state=state,
                bucket=bucket,
                digest=digest,
                pass1_submit=submit_sig,
                pass1_validated=last_validated_code,
            ) or submit_sig
            # Re-read after pass 2 — recipe-tuner may have stashed a
            # new validated candidate without explicit submit.
            last_validated_code = getattr(
                handlers.get("submit", None), "_last_validated_code", "",
            ) or last_validated_code

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

    def _cont_fields(code: str):
        """(mode, parent_index) for a candidate on a continuation round —
        only ``continue`` when the code reproduced the frozen architecture,
        so the validator's strict warm-start load can't fail."""
        if not cont_ctx:
            return None, None
        if code and continuation.arch_matches(code, cont_ctx["frozen_arch"]):
            return "continue", cont_ctx["parent_id"]
        _log(
            "[orchestrator] candidate diverged from frozen parent "
            "architecture — shipping as a new design"
        )
        return None, None

    if submit_sig is not None:
        mode, parent_index = _cont_fields(submit_sig.code)
        return _package(
            submit_sig.code, submit_sig.name, submit_sig.motivation,
            prompt_id=active_prompt["id"], mode=mode, parent_index=parent_index,
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
            mode, parent_index = _cont_fields(best["code"])
            return _package(
                best["code"],
                best.get("name") or f"best_so_far_{bucket}",
                best.get("motivation") or "Auto-shipped best-so-far candidate.",
                prompt_id=active_prompt["id"], mode=mode,
                parent_index=parent_index,
            )

    if last_validated_code:
        mode, parent_index = _cont_fields(last_validated_code)
        return _package(
            last_validated_code,
            f"auto_submit_{bucket}",
            "Auto-submitted validated code — designer did not call "
            "submit explicitly.",
            prompt_id=active_prompt["id"], mode=mode,
            parent_index=parent_index,
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
