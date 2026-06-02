"""Continuation-round support for the designer pipeline.

On a continuation round the validator stamps ``challenge["round_type"] ==
"continuation"`` and supplies ``challenge["eligible_parents"]`` (compact,
weights-free parent summaries). To actually warm-start, a proposal must

  1. carry ``mode="continue"`` and an integer ``parent_index`` (the parent
     experiment id), and
  2. ship ``build_model`` code whose tensor shapes match the parent
     checkpoint exactly — the validator loads it strictly.

We make (2) trivial and safe instead of asking the LLM to regenerate a
shape-identical architecture from a signature. The harness already splits
the submission into an *architecture* surface (``build_model`` /
``init_weights`` / ``COMPILE``) and a *training-recipe* surface
(``build_optimizer`` / ``build_scheduler`` / ``training_config`` /
``compute_loss`` / ``configure_amp`` / ``transform_batch`` /
``on_step_*``). On a continuation round we lift the parent's architecture
definitions verbatim, pin them into the prompt, and let the designer
rewrite only the recipe. Shapes are then identical by construction, so
the strict load can't fail — and recipe-tuning is something the designer
already does every round.

This module is duplicated byte-for-byte into each standalone agent's
``core/`` (the miners don't share an importable package).
"""

from __future__ import annotations

import ast
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Top-level symbols that define the architecture. These must be reproduced
# verbatim on a continuation so the parent checkpoint loads strictly. The
# recipe hooks (build_optimizer, build_scheduler, training_config,
# compute_loss, configure_amp, transform_batch, on_step_*) are free to
# change — that's the whole point of a continuation.
ARCH_FUNCS = ("build_model", "init_weights")
ARCH_GLOBALS = ("COMPILE",)

_HTTP_TIMEOUT = 15.0


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


def fetch_parent_signature(client, db_url: str, parent_id: int) -> dict:
    """``{tensor_name: [shape]}`` of the parent checkpoint (no weights)."""
    body = _get_json(client, db_url, f"/experiments/{parent_id}/signature")
    if not body:
        return {}
    sig = body.get("signature")
    return sig if isinstance(sig, dict) else {}


def _arch_nodes(code: str) -> dict[str, ast.AST]:
    """Map architecture symbol → its top-level AST node.

    Captures ``def build_model`` / ``def init_weights`` and any module-level
    assignment to ``COMPILE``. Returns ``{}`` if the code doesn't parse.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        logger.debug("continuation: parent code did not parse: %s", exc)
        return {}
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ARCH_FUNCS:
                out[node.name] = node
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in ARCH_GLOBALS:
                    out[tgt.id] = node
        elif isinstance(node, ast.AnnAssign):
            tgt = node.target
            if isinstance(tgt, ast.Name) and tgt.id in ARCH_GLOBALS:
                out[tgt.id] = node
    return out


def extract_arch_source(code: str) -> Optional[str]:
    """Verbatim source of the parent's architecture definitions.

    Returns the concatenated ``build_model`` (required), ``init_weights``,
    and ``COMPILE`` segments — exactly what the designer must copy. ``None``
    when the code doesn't parse or has no ``build_model``.
    """
    nodes = _arch_nodes(code)
    if "build_model" not in nodes:
        return None
    segments: list[str] = []
    # Stable, readable order: COMPILE flag, then build_model, then init.
    for name in ("COMPILE", "build_model", "init_weights"):
        node = nodes.get(name)
        if node is None:
            continue
        seg = ast.get_source_segment(code, node)
        if seg:
            segments.append(seg)
    return "\n\n".join(segments) if segments else None


def arch_matches(submitted_code: str, frozen_arch: str) -> bool:
    """True iff the submission reproduces the frozen architecture exactly.

    Compares the structural AST of each architecture symbol (ignoring
    formatting/comments). A continuation only stays ``mode="continue"``
    when this passes; otherwise the proposal is shipped as a fresh design
    so the validator scores it cleanly instead of failing the strict load.
    """
    if not submitted_code or not frozen_arch:
        return False
    want = _arch_nodes(frozen_arch)
    got = _arch_nodes(submitted_code)
    if not want or set(want) != set(got):
        return False
    for name, node in want.items():
        if ast.dump(node) != ast.dump(got[name]):
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
        "signature": fetch_parent_signature(client, db_url, parent_id),
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
