"""Execution + challenge wiring for the special round types.

``local/round_types.py`` owns scheduling and target selection; this
module owns what the validator *does* on those rounds:

* ``run_replicate_round`` — the whole round, validator-only. Re-trains a
  seeded frontier member's exact code with this round's seed, records
  the result as ``mode='replicate'`` (score 0, excluded from frontiers)
  so ``local/noise.py`` gains a (original, replicate) pair.
* ``annotate_special`` — mutates a freshly built challenge payload for
  ablate / recipe_only / transfer rounds (target card + round_type),
  downgrading to a normal round when no target exists.
* ``stamp_special_objectives`` — pins the validator-declared comparison
  anchor onto the trained result's objectives (and enforces the
  recipe_only architecture freeze via AST comparison).
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Callable, Optional

from local.round_types import (
    enforce_recipe_only, pick_ablation_target, pick_replicate_source,
    pick_transfer_source, source_card,
)
from local.scoring import passes_size_gate
from local.task import (
    SyntheticDataGeneratorSpec, TSDataPipelineSpec, buckets_for,
)

logger = logging.getLogger(__name__)

REPLICATE_MINER_ID = "validator:replicate"


def _bucket_for_source(task, src: dict) -> tuple[str, int, int]:
    """Bucket bounds the source experiment actually lives in (so the
    replicate passes the same size gate). Pipeline tasks run gate-off."""
    if isinstance(task, (TSDataPipelineSpec, SyntheticDataGeneratorSpec)):
        return "frozen", 0, 0
    for name, (lo, hi) in buckets_for(task).items():
        if passes_size_gate(src.get("objectives", {}) or {}, lo, hi):
            return name, lo, hi
    return "frozen", 0, 0


def run_replicate_round(store, task, round_id: int, *,
                        epoch: dict, train_fn: Callable[..., dict],
                        sink=None) -> bool:
    """Run a full validator-only replicate round. Returns False when no
    eligible source exists (caller falls through to a normal round)."""
    exps = store.recent_experiments(n=10_000)
    gate_off = isinstance(
        task, (TSDataPipelineSpec, SyntheticDataGeneratorSpec),
    )
    src = pick_replicate_source(
        exps, task=task.name, round_id=round_id, epoch=epoch,
        gate_off=gate_off,
    )
    if src is None:
        logger.info(
            "round=%d scheduled replicate downgraded: no frontier source",
            round_id,
        )
        return False

    bucket, lo, hi = _bucket_for_source(task, src)
    challenge_id = f"r{round_id:06d}-{uuid.uuid4().hex[:8]}"
    payload = {
        "challenge_id": challenge_id,
        "round_id": round_id,
        "seed": round_id * 31 + 7,
        "round_type": "replicate",
        "replicate_of": src["id"],
        "bucket": bucket,
        "min_flops_equivalent": lo,
        "max_flops_equivalent": hi,
        "task": {"name": task.name},
    }
    # 'collected' status: bookkeeping row for round numbering / dashboards
    # without any miner ever seeing it as open.
    store.post_challenge(challenge_id, round_id, payload, status="collected")
    if sink is not None:
        sink.record_challenge(task.name, round_id, payload)
    logger.info(
        "round=%d type=replicate source=#%d ('%s' metric=%s) bucket=%s",
        round_id, src["id"], src.get("name"),
        f"{src['metric']:.6f}" if src.get("metric") is not None else "?",
        bucket,
    )

    result = train_fn(
        src["code"], seed=round_id, task=task, min_flops=lo, max_flops=hi,
    )
    objectives = result.get("objectives") or {}
    objectives["replicate_of"] = int(src["id"])
    if result.get("success") and result.get("metric") is not None:
        delta = float(src["metric"]) - float(result["metric"])
        objectives["replicate_delta"] = delta
        logger.info(
            "  replicate done: metric=%.6f vs original=%.6f (Δ=%.6f)",
            result["metric"], src["metric"], delta,
        )
    else:
        logger.warning(
            "  replicate failed: %s", result.get("error", "?"),
        )

    analysis = (result.get("analysis") or "") + (
        f" [replicate of #{src['id']}]"
    )
    exp_id = store.add_experiment(
        round_id=round_id,
        miner_id=REPLICATE_MINER_ID,
        name=f"replicate-of-{src['id']}",
        code=src["code"],
        motivation=f"noise probe: re-run of experiment #{src['id']} "
                   f"with seed {round_id}",
        reasoning="",
        tool_calls=[],
        metric=result.get("metric"),
        success=bool(result.get("success")),
        objectives=objectives,
        score=0.0,
        loss_curve=result.get("loss_curve") or [],
        val_curve=result.get("val_curve") or [],
        analysis=analysis,
        parent_index=None,
        task=task.name,
        mode="replicate",
    )
    workdir_str = result.get("workdir") or ""
    workdir = Path(workdir_str) if workdir_str else None
    try:
        if sink is not None:
            sink.record_result(
                task.name, round_id, REPLICATE_MINER_ID, result, workdir,
            )
    finally:
        from local.artifacts import cleanup_workdir
        cleanup_workdir(workdir)

    store.mark_challenge(challenge_id, "done")
    try:
        from local.lab_reports import generate_round_reports
        generate_round_reports(store, round_id, task.name)
    except Exception as e:  # noqa: BLE001
        logger.warning("replicate lab report failed: %s", e)
    logger.debug("replicate experiment recorded as id=%d", exp_id)
    return True


def annotate_special(payload: dict, special: str, store, task, *,
                     round_id: int, epoch: dict) -> None:
    """Mutate a built challenge payload into an ablate / recipe_only /
    transfer round, or record a downgrade when no target exists.

    Special rounds replace the continuation flow for the round: the
    payload's round_type is overwritten and continuation disabled.
    """
    bucketed = not isinstance(
        task, (TSDataPipelineSpec, SyntheticDataGeneratorSpec),
    )
    if special in ("recipe_only", "transfer") and not bucketed:
        payload["downgrade_reason"] = f"{special}_not_applicable"
        return
    exps = store.recent_experiments(n=10_000)
    lo = int(payload.get("min_flops_equivalent", 0) or 0)
    hi = int(payload.get("max_flops_equivalent", 0) or 0)

    card: Optional[dict] = None
    key = ""
    if special == "ablate":
        target = pick_ablation_target(
            exps, task=task.name, round_id=round_id,
            min_flops=lo, max_flops=hi, epoch=epoch,
            gate_off=not bucketed,
        )
        if target is not None:
            card, key = source_card(target), "ablation_target"
    elif special == "recipe_only":
        target = pick_ablation_target(
            exps, task=task.name, round_id=round_id,
            min_flops=lo, max_flops=hi, epoch=epoch,
        )
        if target is not None:
            card, key = source_card(target), "recipe_base"
    elif special == "transfer":
        target = pick_transfer_source(
            exps, task=task.name, round_id=round_id, min_flops=lo,
            buckets=buckets_for(task), epoch=epoch,
        )
        if target is not None:
            card, key = source_card(target, buckets_for(task)), "transfer_source"

    payload["scheduled_round_type"] = special
    if card is None:
        payload["downgrade_reason"] = f"no_{special}_target"
        return
    payload["round_type"] = special
    payload[key] = card
    # Special rounds are fresh-training rounds; continuation is off.
    payload["continuation_allowed"] = False
    payload["eligible_parents"] = []
    payload["continuation_kind"] = ""


def stamp_special_objectives(result: dict, payload: dict,
                             challenge: dict) -> None:
    """Pin the validator-declared comparison anchor into objectives so
    scoring / lab reports / lineage queries can attribute the round."""
    objectives = result.get("objectives")
    if not isinstance(objectives, dict):
        return
    rt = challenge.get("round_type", "")
    if rt == "ablate" and challenge.get("ablation_target"):
        objectives["ablation_of"] = int(challenge["ablation_target"]["id"])
    elif rt == "recipe_only" and challenge.get("recipe_base"):
        base = challenge["recipe_base"]
        ok, note = enforce_recipe_only(
            payload.get("code", ""), base.get("code") or "",
        )
        if ok:
            objectives["recipe_only_of"] = int(base["id"])
        else:
            result["continuation_note"] = (
                (result.get("continuation_note") or "")
                + ("; " if result.get("continuation_note") else "") + note
            )
    elif rt == "transfer" and challenge.get("transfer_source"):
        objectives["transfer_of"] = int(challenge["transfer_source"]["id"])
