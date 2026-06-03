"""Continuation-round support for the designer pipeline.

On a continuation round the validator stamps ``challenge["round_type"] ==
"continuation"`` and supplies ``challenge["eligible_parents"]`` (compact,
weights-free parent summaries). To actually warm-start, a proposal must

  1. carry ``mode="continue"`` and an integer ``parent_index`` (the parent
     experiment id), and
  2. ship ``build_model`` code whose tensor shapes match the parent
     checkpoint exactly — the validator loads it strictly.

We make (2) trivial and safe instead of asking the LLM to regenerate a
shape-identical architecture from a signature. The harness splits the
submission into a *training-recipe* surface — the hooks it picks up by
name (``build_optimizer`` / ``build_scheduler`` / ``training_config`` /
``compute_loss`` / ``configure_amp`` / ``transform_batch`` /
``on_step_*``) — and everything else (imports, helper classes,
``build_model``, ``init_weights``, ``COMPILE``, module constants), which
is the *architecture*. On a continuation round we lift the parent's whole
architecture (everything but the recipe hooks) verbatim, pin it into the
prompt, and let the designer rewrite only the recipe. Shapes are then
identical by construction, so the strict load can't fail — and
recipe-tuning is something the designer already does every round.

This module is duplicated byte-for-byte into each standalone agent's
``core/`` (the miners don't share an importable package).
"""

from __future__ import annotations

import ast
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# The training-recipe hooks the harness picks up by name. These — and ONLY
# these — are what a continuation is allowed to change. Everything else at
# module level (imports, helper classes/functions, build_model,
# init_weights, COMPILE, module constants) is the "architecture" and is
# frozen verbatim so the parent checkpoint loads strictly. We blacklist the
# recipe rather than whitelist build_model, because build_model routinely
# depends on top-level helper classes / imports that must come along too.
RECIPE_FUNCS = frozenset({
    "build_optimizer", "build_scheduler", "training_config",
    "compute_loss", "configure_amp", "transform_batch",
    "on_step_begin", "on_step_end",
})


def is_continuation_round(challenge: dict) -> bool:
    """True when the validator scheduled this round as a continuation and
    handed us at least one eligible parent."""
    return (
        challenge.get("round_type") == "continuation"
        and bool(challenge.get("eligible_parents"))
    )


def choose_parent(parents: list[dict]) -> Optional[dict]:
    """Pick the parent worth continuing.

    Only parents whose checkpoint is available can be warm-started. Among
    those we take the best (lowest) GIFT-eval ``metric``, breaking ties
    toward the one that has accumulated less compute (more headroom on the
    continuation frontier, which scores Δmetric per unit compute).
    """
    usable = [p for p in parents if p.get("checkpoint_available")]
    if not usable:
        return None

    def _key(p: dict) -> tuple[float, float]:
        metric = p.get("metric")
        metric = float(metric) if isinstance(metric, (int, float)) else float("inf")
        compute = p.get("cumulative_compute") or 0.0
        return (metric, float(compute))

    return min(usable, key=_key)


def _get_json(client, db_url: str, path: str) -> Optional[dict]:
    """GET ``{db_url}{path}`` via the agent's gated client. Returns the
    decoded body or ``None`` on any failure (so callers degrade to a
    fresh design rather than crashing the round)."""
    if client is None or not db_url:
        return None
    url = f"{db_url}{path}"
    try:
        body = client.get_json(url)
    except Exception as exc:  # noqa: BLE001 — never let this kill the round
        logger.debug("continuation: GET %s failed: %s", url, exc)
        return None
    return body if isinstance(body, dict) else None


def fetch_parent_code(client, db_url: str, parent_id: int) -> Optional[str]:
    """Full submission source of the parent experiment."""
    body = _get_json(client, db_url, f"/experiments/{parent_id}")
    if not body:
        return None
    code = body.get("code")
    return code if isinstance(code, str) and code.strip() else None


def _node_name(node: ast.AST) -> Optional[str]:
    """Name of a top-level def/class, else ``None``."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    return None


def _is_recipe(node: ast.AST) -> bool:
    return _node_name(node) in RECIPE_FUNCS


def _parse_body(code: str) -> Optional[list[ast.stmt]]:
    """Top-level statements of ``code``, or ``None`` if it doesn't parse."""
    try:
        return ast.parse(code).body
    except SyntaxError as exc:
        logger.debug("continuation: code did not parse: %s", exc)
        return None


def _has_build_model(body: list[ast.stmt]) -> bool:
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "build_model"
        for n in body
    )


def extract_arch_source(code: str) -> Optional[str]:
    """Verbatim source of the parent's architecture — everything at module
    level *except* the recipe hooks.

    This deliberately keeps imports, helper classes/functions, module
    constants, ``build_model``, ``init_weights`` and ``COMPILE`` so the
    designer can paste a self-contained, loadable architecture. Returns
    ``None`` when the code doesn't parse or has no ``build_model``.
    """
    body = _parse_body(code)
    # build_model is never a recipe hook, so its presence in the full body
    # guarantees it survives the filter below.
    if body is None or not _has_build_model(body):
        return None
    segments: list[str] = []
    for node in body:
        if _is_recipe(node):
            continue
        seg = ast.get_source_segment(code, node)
        if seg:
            segments.append(seg)
    return "\n\n".join(segments) if segments else None


def arch_matches(submitted_code: str, frozen_arch: str) -> bool:
    """True iff the submission reproduces the frozen architecture exactly.

    Every frozen top-level node (imports, helper classes, ``build_model``,
    …) must appear structurally identical (AST-equal, ignoring formatting /
    comments) among the submission's non-recipe top-level nodes. The
    submission may add extra nodes and freely write the recipe hooks —
    those don't affect tensor shapes. Any change to a frozen node fails the
    check, so the proposal ships as a fresh design rather than risking a
    strict warm-start load failure.
    """
    if not submitted_code or not frozen_arch:
        return False
    frozen_body = _parse_body(frozen_arch)
    sub_body = _parse_body(submitted_code)
    if not frozen_body or sub_body is None:
        return False
    if not _has_build_model(frozen_body):
        return False
    # Multiset of the submission's non-recipe nodes; each frozen node must
    # be matched (and consumed) exactly once.
    pool = [ast.dump(n) for n in sub_body if not _is_recipe(n)]
    for node in frozen_body:
        dump = ast.dump(node)
        if dump in pool:
            pool.remove(dump)
        else:
            return False
    return True


def build_context(challenge: dict, client, db_url: str) -> Optional[dict]:
    """Resolve the continuation context for this round, or ``None``.

    ``None`` means "design fresh" — either it isn't a continuation round,
    no parent is usable, or we couldn't fetch/parse the parent's code. The
    orchestrator treats any ``None`` as a normal (new) round.
    """
    if not is_continuation_round(challenge):
        return None
    parent = choose_parent(challenge.get("eligible_parents") or [])
    if parent is None:
        logger.info("continuation: no checkpoint-bearing parent — designing fresh")
        return None
    parent_id = parent.get("id")
    if not isinstance(parent_id, int):
        return None
    code = fetch_parent_code(client, db_url, parent_id)
    frozen = extract_arch_source(code) if code else None
    if not frozen:
        logger.info(
            "continuation: parent #%s code/arch unavailable — designing fresh",
            parent_id,
        )
        return None
    return {
        "parent_id": parent_id,
        "parent": parent,
        "frozen_arch": frozen,
    }


def recipe_brief(context: dict) -> dict:
    """A designer brief for a continuation round.

    The researcher is skipped on continuations (there's no new architecture
    to invent), so we hand the designer a recipe-focused brief that points
    it at the parent's trajectory.
    """
    parent = context.get("parent", {})
    return {
        "_continuation": True,
        "summary": (
            f"Continuation of parent #{context['parent_id']} "
            f"(metric={parent.get('metric')!r}). Architecture is frozen; "
            "improve only the training recipe so the warm-started weights "
            "train further."
        ),
        "plan": [
            "Read the parent's loss_curve_tail to see where its curve flattened.",
            "Pick an optimizer + LR + schedule that extends rather than "
            "restarts that curve (warm-start, so favour lower/decaying LR).",
            "Optionally revisit compute_loss / amp / batch transform.",
            "Copy the frozen architecture verbatim, validate_code, submit.",
        ],
    }
