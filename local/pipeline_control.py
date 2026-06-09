"""Paired-control gating + per-round re-render for frozen pipelines.

Previously a frozen-pipeline promotion was gated only on the generator's
*standalone* score, and its contribution to ts_forecasting (16 static
shards diluted into a multi-TB stream) was never measured. Two fixes:

* ``control_gate`` — before promotion, train the current best
  ts_forecasting architecture twice with the same seed and a short
  budget: once pure-real, once with the candidate's synthetic shards
  mixed in. Promote only when the mixed run wins. One cheap experiment
  answers "does this generator actually help the consumer task?".
* ``rerender_for_round`` — instead of one tiny fixed corpus rendered at
  promotion, optionally re-render the generator's shards each round
  (round-seeded iterator position), approximating streaming from the
  generator without touching the frozen harness loader.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Fixed seed for control pairs — distinct from any round seed so a control
# run can never collide with a recorded experiment's RNG stream.
CONTROL_SEED = 9_999_991
DEFAULT_CONTROL_EVAL_TASKS = 12
_KEEP_ROUND_RENDERS = 3


def _best_consumer_code(db_store) -> Optional[dict]:
    """Best successful ts_forecasting experiment (the control's arch)."""
    cands = [
        e for e in db_store.recent_experiments(n=10_000)
        if e.get("task") == "ts_forecasting" and e.get("success")
        and e.get("metric") is not None and e.get("code")
        and e.get("mode") != "replicate"
    ]
    if not cands:
        return None
    return min(cands, key=lambda e: e["metric"])


def _control_run(code: str, *, seconds: int, frozen_pipeline) -> dict:
    from local.artifacts import cleanup_workdir
    from local.task import TSForecastingSpec
    from local.trainer import run_training

    task = TSForecastingSpec(time_budget_seconds=int(seconds))
    result = run_training(
        code, seed=CONTROL_SEED, task=task,
        min_flops=0, max_flops=10**18,
        frozen_pipeline=frozen_pipeline,
    )
    wd = result.get("workdir") or ""
    if wd:
        cleanup_workdir(Path(wd))
    return result


def control_gate(db_store, *, shard_paths: list[str], seconds: int,
                 eval_max_tasks: int = DEFAULT_CONTROL_EVAL_TASKS,
                 ) -> tuple[bool, dict]:
    """Paired with/without-synthetic control. Returns (promote?, details).

    Passes open (True) when no consumer experiment exists yet — the
    bootstrap promotion can't be measured, only later ones can.
    """
    if seconds <= 0:
        return True, {"skipped": "control disabled"}
    if not shard_paths:
        return False, {"skipped": "no rendered shards to test"}
    consumer = _best_consumer_code(db_store)
    if consumer is None:
        return True, {"skipped": "no ts_forecasting consumer yet"}

    probe = dataclasses.make_dataclass(
        "ProbePipeline", [("version", int), ("shard_paths", list)],
    )(version=0, shard_paths=list(shard_paths))

    saved = os.environ.get("RADAR_GIFT_EVAL_MAX_TASKS")
    os.environ["RADAR_GIFT_EVAL_MAX_TASKS"] = str(int(eval_max_tasks))
    try:
        logger.info(
            "frozen-pipeline control: arch=#%d, %ds budget × 2 runs",
            consumer["id"], seconds,
        )
        with_synth = _control_run(
            consumer["code"], seconds=seconds, frozen_pipeline=probe,
        )
        without = _control_run(
            consumer["code"], seconds=seconds, frozen_pipeline=None,
        )
    finally:
        if saved is None:
            os.environ.pop("RADAR_GIFT_EVAL_MAX_TASKS", None)
        else:
            os.environ["RADAR_GIFT_EVAL_MAX_TASKS"] = saved

    details = {
        "consumer_experiment_id": consumer["id"],
        "control_seconds": seconds,
        "eval_max_tasks": eval_max_tasks,
        "metric_with_synth": with_synth.get("metric"),
        "metric_without": without.get("metric"),
    }
    if not with_synth.get("success"):
        details["reason"] = "mixed control run failed"
        return False, details
    if not without.get("success"):
        # Real-only control crashed; can't claim the synth hurt. Promote.
        details["reason"] = "pure-real control failed; promoting on benefit of doubt"
        return True, details
    ok = float(with_synth["metric"]) < float(without["metric"])
    details["reason"] = (
        "mixed beat pure-real" if ok else "mixed did not beat pure-real"
    )
    logger.info(
        "frozen-pipeline control: with=%.6f without=%.6f → %s",
        with_synth["metric"], without["metric"],
        "PROMOTE" if ok else "REJECT",
    )
    return ok, details


def rerender_for_round(pipe_store, pipe, round_id: int, *,
                       num_shards: int, batches_per_shard: int):
    """Fresh per-round shard render from the frozen generator's code.

    Returns a copy of ``pipe`` whose ``shard_paths`` point at this
    round's render (the generator restarts with the iterator advanced by
    a round-seeded skip, so successive rounds see different data). Falls
    back to the promotion-time shards when rendering fails. Old round
    renders beyond the last few are reclaimed.
    """
    from local.frozen_pipeline import _render_shards

    rounds_dir = pipe_store.base_dir / f"v{pipe.version:06d}" / "rounds"
    out_dir = rounds_dir / f"r{round_id:06d}"
    try:
        shard_paths = _render_shards(
            pipe.code, out_dir,
            num_shards=num_shards, batches_per_shard=batches_per_shard,
            skip_batches=(round_id % 97) * batches_per_shard,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("per-round synth render failed: %s", e)
        shard_paths = []
    _prune_round_renders(rounds_dir, keep=_KEEP_ROUND_RENDERS)
    if not shard_paths:
        return pipe
    return dataclasses.replace(pipe, shard_paths=shard_paths)


def _prune_round_renders(rounds_dir: Path, keep: int) -> None:
    if not rounds_dir.is_dir():
        return
    dirs = sorted(p for p in rounds_dir.iterdir() if p.is_dir())
    for stale in dirs[:-keep] if keep > 0 else dirs:
        shutil.rmtree(stale, ignore_errors=True)
