"""Agent-side handling of validator-owned special rounds + screening.

The validator schedules special round types beyond new/continuation
(see radar's ``docs/experiment_engine.md``) and advertises them on the
challenge payload:

* ``round_type == "ablate"``      + ``ablation_target`` {id, name,
  metric, code} — submit a *minimal one-component diff* of the target.
* ``round_type == "recipe_only"`` + ``recipe_base`` — architecture is
  frozen (validator AST-checks build_model + classes); only the recipe
  hooks may change. Reuses the in-round recipe machinery.
* ``round_type == "transfer"``    + ``transfer_source`` (incl.
  ``source_bucket``) — scale a smaller-bucket winner into this bucket.

An agent that ignores these wastes the round (recipe_only downgrades to
a normal round on violation; ablate/transfer lose attribution). This
module is dependency-free (no torch, no sibling imports) so the
orchestrator can use it on any round and tests can load it standalone.

``screening_candidates`` packages extra validated designs from the
round's state when the challenge advertises the screening tier — the
validator gives each a short-budget run and promotes only the winner.
"""

from __future__ import annotations

from typing import Optional

_CARDS = {
    "ablate": "ablation_target",
    "recipe_only": "recipe_base",
    "transfer": "transfer_source",
}


def build(challenge: dict) -> Optional[dict]:
    """Resolve the special-round context, or None on a normal round."""
    kind = str(challenge.get("round_type") or "")
    key = _CARDS.get(kind)
    if key is None:
        return None
    card = challenge.get(key) or {}
    if not isinstance(card, dict) or not card.get("code"):
        return None
    return {"kind": kind, "target": card}


def brief(ctx: dict, bucket: str = "") -> dict:
    """Designer brief replacing the researcher's on a special round."""
    t = ctx["target"]
    label = f"#{t.get('id')} ('{t.get('name')}', metric={t.get('metric')!r})"
    if ctx["kind"] == "ablate":
        return {
            "_special": "ablate",
            "summary": (
                f"ABLATION round. The validator picked frontier member "
                f"{label}. Submit its code with EXACTLY ONE component "
                "changed — one layer type, one connectivity choice, one "
                "normalization, or one hyperparameter family. The point "
                "is attribution: the diff vs the target must isolate a "
                "single design decision."
            ),
            "plan": [
                "Read the target architecture in the preamble above.",
                "Pick ONE component whose contribution is untested — "
                "state the hypothesis in your motivation.",
                "Change only that component; keep everything else "
                "byte-identical where possible.",
                "validate_code, then submit. Do NOT also tweak the "
                "recipe — a second diff destroys attribution.",
            ],
        }
    if ctx["kind"] == "recipe_only":
        return {
            "_special": "recipe_only",
            "summary": (
                f"RECIPE-ONLY round. The architecture of {label} is "
                "frozen in the preamble — the validator AST-compares "
                "build_model and every class; any architectural change "
                "downgrades the round and loses the comparison. Spend "
                "the whole budget on optimizer / scheduler / "
                "training_config / AMP."
            ),
            "plan": [
                "Copy the frozen architecture verbatim.",
                "Redesign the training recipe: optimizer family + LR + "
                "weight decay, warmup/decay schedule, batch/grad-clip, "
                "AMP. One coherent recipe hypothesis, not ten tweaks.",
                "validate_code, then submit.",
            ],
        }
    src_bucket = t.get("source_bucket") or "a smaller bucket"
    return {
        "_special": "transfer",
        "summary": (
            f"TRANSFER round. {label} won in {src_bucket}; scale the "
            f"same design into this round's bucket ({bucket or 'larger'}) "
            "— keep the architectural idea, re-balance depth/width/heads "
            "to land in the FLOPs window. The experiment tests whether "
            "the design's advantage survives scale."
        ),
        "plan": [
            "Read the source architecture in the preamble above and "
            "identify its core idea (what made it win small).",
            "Scale it: more width/depth/heads, longer patching — keep "
            "the idea recognizable, hit ~60-80% of max FLOPs.",
            "Scale the recipe with it (LR vs width, warmup vs steps).",
            "estimate FLOPs, validate_code, submit.",
        ],
    }


def preamble(ctx: dict) -> str:
    """User-prompt preamble carrying the target's code (ablate/transfer;
    recipe_only rides the existing in-round recipe context instead)."""
    t = ctx["target"]
    head = {
        "ablate": (
            "## ABLATION ROUND — minimal one-diff of the target below\n\n"
            "Validator-declared comparison anchor: experiment "
            f"#{t.get('id')} (metric={t.get('metric')!r}). Your score and "
            "the round's lab report compare you against it directly."
        ),
        "transfer": (
            "## TRANSFER ROUND — scale the source design into this bucket\n\n"
            f"Source: experiment #{t.get('id')} from bucket "
            f"{t.get('source_bucket')!r} (metric={t.get('metric')!r})."
        ),
    }.get(ctx["kind"], "")
    return f"{head}\n\n```python\n{t.get('code', '')}\n```\n"


def screening_candidates(challenge: dict, state: dict, *,
                         primary_name: str,
                         primary_code: str) -> Optional[list[dict]]:
    """Extra validated candidates for the validator's screening tier.

    Returns ``[{name, code}, ...]`` (primary first) when the challenge
    advertises screening and the round produced ≥1 distinct validated
    alternate; None otherwise (caller omits the field).
    """
    scr = challenge.get("screening") or {}
    if not scr.get("enabled") or not primary_code:
        return None
    cap = int(scr.get("max_candidates", 0) or 0)
    if cap < 2:
        return None
    out = [{"name": primary_name, "code": primary_code}]
    seen = {primary_code}
    cands = state.get("candidates") or {}
    rows = sorted(
        cands.items(),
        key=lambda kv: -(kv[1].get("created_at") or 0.0),
    )
    for cid, rec in rows:
        if len(out) >= cap:
            break
        code = rec.get("code") or ""
        if not code or code in seen or not rec.get("validated"):
            continue
        suffix = cid.replace("cand_", "")[:8]
        out.append({"name": f"{primary_name}-alt-{suffix}", "code": code})
        seen.add(code)
    return out if len(out) >= 2 else None
