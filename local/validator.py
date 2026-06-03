"""Local validator process.

One Python process that drives Phase A → B → C against a SQLite-backed
store. No HMAC, no HTTP, no chain — see ``local/README.md`` for how this
maps to the real radar architecture.

Loop (per round):
  1. Pick a size bucket deterministically from ``round_id``.
  2. Read the current Pareto frontier from past experiments.
  3. Post a Challenge to SQLite (Phase A starts).
  4. Wait for proposals up to ``--phase_a_seconds``.
  5. For each proposal, run training + held-out eval in-process.
  6. Score the round, write one experiment row per miner.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path

# Allow ``python local/validator.py`` from repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local.artifacts import ArtifactSink, cleanup_workdir, sweep_orphan_workdirs
from local.backup import from_env as backup_from_env
from local.checkpoints import CheckpointStore
from local.continuation import (
    continuation_frontier,
    continuation_rate,
    is_continuation_round,
    prepare_continuation,
)
from local.experiments_api import _parent_summary
from local.frozen_arch import FrozenArchStore, maybe_refresh as maybe_refresh_frozen
from local.scoring import compute_pareto, passes_size_gate, score_round
from local.services import ServicesServer
from local.store import LocalStore
from local.task import (
    SIZE_BUCKETS, TaskSpec, TSDataPipelineSpec, TSForecastingSpec,
    buckets_for, make_spec,
)
from local.trainer import run_training


logger = logging.getLogger("local.validator")


def _pick_bucket(round_id: int, task=None) -> tuple[str, int, int]:
    buckets = buckets_for(task) if task is not None else SIZE_BUCKETS
    names = list(buckets.keys())
    name = names[round_id % len(names)]
    lo, hi = buckets[name]
    return name, lo, hi


def _task_dict(task) -> dict:
    """Serialise a TaskSpec / TSForecastingSpec for the challenge payload.

    Miners read this verbatim — see ``miners/*/agent.py`` and the
    ``challenge['task']`` description in radar-miner-examples/README.md.
    """
    if isinstance(task, TSDataPipelineSpec):
        return {
            "name": task.name,
            "task_params": {
                "context_len": task.context_len,
                "prediction_len": task.prediction_len,
                "num_variates": task.num_variates,
                "quantiles": list(task.quantiles),
            },
            "constraints": [
                "torch + stdlib only",
                "build_pipeline(context_len, prediction_len, num_variates, "
                "quantiles) returns an iterator of {'input', 'target'} batches",
            ],
            "objectives": [
                {"name": "aulc", "primary": False, "minimize": True},
                {"name": "gift_metric", "primary": False, "minimize": True},
            ],
            "time_budget": task.time_budget_seconds,
            "runner_dir": "ts_data_pipeline",
        }
    if isinstance(task, TSForecastingSpec):
        return {
            "name": task.name,
            "task_params": {
                "context_len": task.context_len,
                "prediction_len": task.prediction_len,
                "num_variates": task.num_variates,
                "quantiles": list(task.quantiles),
            },
            "constraints": ["torch + stdlib only", "build_model(context_len, "
                            "prediction_len, num_variates, quantiles) returns nn.Module"],
            "objectives": [{"name": "val_loss", "primary": True, "minimize": True}],
            "time_budget": task.time_budget_seconds,
            "runner_dir": "ts_forecasting",
        }
    return {
        "name": task.name,
        "input_dim": task.input_dim,
        "output_dim": task.output_dim,
    }


def _current_epoch(task, frozen_arch) -> dict:
    """Continuation-pinning context for this round.

    A parent's saved checkpoint is only a valid warm-start when it was
    trained in the same context as this round. For ts_data_pipeline that
    means the frozen architecture version — the metric isn't comparable
    across arch refreshes. ts_forecasting has no epoch axis yet (returns
    empty dict — gate is a no-op).
    """
    epoch: dict = {}
    if isinstance(task, TSDataPipelineSpec) and frozen_arch is not None:
        epoch["frozen_arch_version"] = int(frozen_arch.version)
    return epoch


def _parent_in_epoch(parent: dict, epoch: dict) -> bool:
    if not epoch:
        return True
    objs = parent.get("objectives", {}) or {}
    return all(objs.get(k) == v for k, v in epoch.items())


def _build_challenge(round_id: int, store: LocalStore, task,
                     services_url: str, agent_seconds: int = 180,
                     continuation_enabled: bool = False,
                     scheduled_continuation: bool = False,
                     frozen_arch=None) -> dict:
    name, lo, hi = _pick_bucket(round_id, task=task)
    # The data-pipeline task uses a fixed frozen architecture, so its
    # FLOPs are identical across miners in a round — size buckets are
    # meaningless here. Zero the bounds so the harness skips the gate.
    if isinstance(task, TSDataPipelineSpec):
        name, lo, hi = "frozen", 0, 0
    all_exps = store.recent_experiments(n=10_000)
    # Mixed-task DBs (e.g. ts_forecasting + ts_data_pipeline against the
    # same SQLite file) mean we can't compare metrics across tasks — filter
    # to this task before computing the per-round feasible frontier.
    same_task = [e for e in all_exps if e.get("task") == task.name]
    pareto = compute_pareto(same_task)
    feasible = [
        {
            "code": e["code"],
            "metric": e["metric"],
            "objectives": e["objectives"],
            "name": e["name"],
        }
        for e in pareto
        if passes_size_gate(e["objectives"], lo, hi)
    ]
    # The validator owns the round type: a scheduled continuation round only
    # becomes one if eligible (fully-eval'd, checkpoint-bearing, in-bucket)
    # parents actually exist — otherwise it degrades to a fresh round. This
    # is why continuation pressure ramps in slowly: early rounds have no
    # parents to continue from regardless of the schedule.
    eligible_parents = []
    epoch = _current_epoch(task, frozen_arch)
    if continuation_enabled:
        eligible_parents = [
            _parent_summary(e)
            for e in store.eligible_parents(
                task=task.name, min_flops=lo, max_flops=hi,
            )
            if _parent_in_epoch(e, epoch)
        ]
    round_type = (
        "continuation"
        if (scheduled_continuation and eligible_parents) else "new"
    )
    continuation_allowed = round_type == "continuation"
    # ``allowed_urls`` is what the miner-side GatedClient enforces. Every
    # endpoint the agent can reach lives under ``services_url`` so a
    # single prefix is enough.
    payload = {
        "challenge_id": f"r{round_id:06d}-{uuid.uuid4().hex[:8]}",
        "round_id": round_id,
        "seed": round_id * 31 + 7,
        "min_flops_equivalent": lo,
        "max_flops_equivalent": hi,
        "bucket": name,
        "task": _task_dict(task),
        "feasible_frontier": feasible,
        "round_type": round_type,
        "continuation_allowed": bool(continuation_allowed),
        "eligible_parents": eligible_parents if continuation_allowed else [],
        "agent_seconds": int(agent_seconds),
        # Dummy agent_token — local services.py doesn't enforce auth, but
        # the real agents' startup checks require a non-empty value.
        "agent_token": "local-dev",
        # URL surface the agent uses via GatedClient.
        "db_url": services_url,
        "llm_url": f"{services_url}/llm",
        "desearch_url": f"{services_url}/desearch",
        "cognition_wiki_url": f"{services_url}/wiki",
        "allowed_urls": services_url,
    }
    if frozen_arch is not None:
        payload["frozen_arch"] = {
            "version": frozen_arch.version,
            "source_experiment_id": frozen_arch.source_experiment_id,
            "source_metric": frozen_arch.source_metric,
            "source_crps": frozen_arch.source_crps,
            "source_mase": frozen_arch.source_mase,
            "source_flops": frozen_arch.source_flops,
            "source_name": frozen_arch.source_name,
            "code": frozen_arch.code,
        }
    return payload


def _ensure_ts_caches() -> bool:
    """Ensure the GIFT-Eval benchmark cache and the held-out pretrain val
    shard are populated before the validator starts a ts_forecasting round.

    Returns True on success (caches ready), False on hard failure (no creds /
    download failed) — caller exits early so we don't blow ``--training_seconds``
    on every round only to fail Phase C.
    """
    gift_cache = os.environ.get("RADAR_GIFT_EVAL_CACHE", "/tmp/radar_gift_eval")
    val_cache = os.environ.get("RADAR_PRETRAIN_VAL_CACHE", "/tmp/radar_pretrain_val")

    # 1. GIFT-Eval benchmark — every leaderboard dataset must be on disk so
    # Phase C can iterate the full 97-task list. Skip the import + R2 dance
    # when everything is already cached.
    try:
        from shared.gift_eval import (
            MED_LONG_DATASETS, SHORT_DATASETS, ensure_datasets_cached,
        )
    except ImportError as e:
        logger.error("ts_forecasting requires shared.gift_eval: %s", e)
        return False

    leaderboard = sorted({*SHORT_DATASETS, *MED_LONG_DATASETS})
    Path(gift_cache).mkdir(parents=True, exist_ok=True)
    logger.info(
        "checking GIFT-Eval cache (%d datasets) at %s", len(leaderboard), gift_cache,
    )
    status = ensure_datasets_cached(leaderboard, cache_dir=gift_cache)
    missing = [k for k, ok in status.items() if not ok]
    if missing or not status:
        logger.error(
            "GIFT-Eval cache incomplete (%d missing). First few: %s. "
            "Check HIPPIUS_*/R2_* creds and re-run.",
            len(missing) or len(leaderboard), missing[:5],
        )
        return False
    logger.info("GIFT-Eval cache OK (%d datasets ready)", len(status))

    # 2. Pretrain val shard — small fixed held-out split. If the dir already
    # has parquet files we trust them; otherwise pull from the pretrain bucket.
    val_dir = Path(val_cache)
    existing_val = sorted(val_dir.glob("*.parquet")) if val_dir.is_dir() else []
    if existing_val:
        logger.info(
            "pretrain val cache OK (%d shard(s) at %s)", len(existing_val), val_cache,
        )
        return True

    logger.info("pretrain val cache empty at %s — fetching", val_cache)
    try:
        from local.fetch_pretrain import make_pretrain_client
        from shared.pretrain_data import PretrainBenchmark
    except ImportError as e:
        logger.error("cannot fetch val shard (missing deps): %s", e)
        return False

    r2 = make_pretrain_client()
    if r2 is None:
        logger.error(
            "no pretrain S3 client (install boto3 + set HIPPIUS_*/R2_* env)",
        )
        return False
    bench = PretrainBenchmark(r2=r2)
    val_keys = bench.get_val_shard_keys()
    if not val_keys:
        logger.warning(
            "pretrain manifest declares no val shards — in-training val "
            "will be disabled by trainer fallback",
        )
        return True

    val_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for key in val_keys:
        dest = val_dir / Path(key).name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        if r2.download_file_to_disk(key, str(dest)):
            ok += 1
            logger.info("  fetched %s (%d bytes)", dest.name, dest.stat().st_size)
        else:
            fail += 1
            logger.error("  failed to download %s", key)
    if fail:
        logger.error("pretrain val fetch failed (%d/%d shards)", fail, ok + fail)
        return False
    logger.info("pretrain val cache ready (%d shard(s))", ok)
    return True


def _next_round_id(store: LocalStore) -> int:
    """Continue from the highest round_id seen so far so consecutive
    runs of the validator don't repeat round 0."""
    rows = store._conn.execute(
        "SELECT COALESCE(MAX(round_id), -1) AS r FROM challenges"
    ).fetchone()
    return int(rows["r"]) + 1


def _pretrain_pool() -> list[str]:
    """Sorted parquet shard paths in the pretrain cache (empty if absent)."""
    import os
    pdir = Path(os.environ.get("RADAR_PRETRAIN_CACHE", "/tmp/radar_pretrain"))
    if not pdir.is_dir():
        return []
    return sorted(str(p) for p in pdir.glob("*.parquet"))


def _train_proposal(payload: dict, task, round_id: int,
                    challenge: dict, prep: dict, frozen_arch=None) -> dict:
    """Run one proposal, retrying as a fresh run if a continuation
    warm-start turns out to be architecture-incompatible (strict load)."""
    def _run() -> dict:
        return run_training(
            payload.get("code", ""),
            seed=round_id,
            task=task,
            min_flops=challenge["min_flops_equivalent"],
            max_flops=challenge["max_flops_equivalent"],
            parent_checkpoint_path=prep["parent_checkpoint_path"],
            compute_offset=prep["compute_offset"],
            step_offset=prep["step_offset"],
            shard_paths=prep["shard_paths"],
            shard_reuse=prep["shard_reuse"],
            frozen_arch=frozen_arch,
        )

    result = _run()
    if (not result.get("success")
            and result.get("harness_status") == "checkpoint_incompatible"
            and prep["mode"] == "continue"):
        logger.warning("    warm-start incompatible; retrying as a fresh run")
        prefix = (prep["note"] + "; ") if prep["note"] else ""
        prep["note"] = prefix + "continuation incompatible: retried fresh"
        prep.update(
            mode="new", parent_index=None, parent_metric=None,
            parent_checkpoint_path=None, compute_offset=0.0,
            step_offset=0, n_rounds=1,
        )
        result = _run()
    return result


def _gc_checkpoints(store: LocalStore, ckpt_store: CheckpointStore,
                    round_id: int, keep_rounds: int = 5) -> None:
    """Drop checkpoints that aren't on a frontier or a recent parent."""
    all_exps = store.recent_experiments(n=10_000)
    keep = {e["id"] for e in compute_pareto(all_exps)}
    keep |= {e["id"] for e in continuation_frontier(all_exps)}
    keep |= {e["id"] for e in all_exps if e["round_id"] > round_id - keep_rounds}
    removed = ckpt_store.gc(keep)
    if removed:
        logger.info("  gc: removed %d stale checkpoint(s)", removed)


def run_round(store: LocalStore, task, round_id: int,
              phase_a_seconds: float, services_url: str,
              agent_seconds: int = 180,
              sink: ArtifactSink | None = None,
              ckpt_store: CheckpointStore | None = None,
              continuation_enabled: bool = False,
              continuation_equilibrium: float = 0.7,
              continuation_warmup_rounds: int = 50,
              continuation_step_pct: float = 1.0,
              continuation_step_every: int = 5,
              shards_per_round: int = 0,
              frozen_arch_store: FrozenArchStore | None = None,
              frozen_arch_refresh_every: int = 50) -> None:
    # The validator owns the cadence: a scheduled coin flip decides whether
    # this is a continuation round. The rate stays at 0 until
    # ``warmup_rounds`` successful rounds, then climbs as a staircase to the
    # equilibrium; _build_challenge further downgrades to "new" when no
    # eligible parents exist yet.
    successful_rounds = store.successful_round_count(task=task.name)
    rate = continuation_rate(
        successful_rounds,
        equilibrium=continuation_equilibrium,
        warmup_rounds=continuation_warmup_rounds,
        step_pct=continuation_step_pct,
        step_every=continuation_step_every,
    ) if continuation_enabled else 0.0
    scheduled = continuation_enabled and is_continuation_round(
        round_id, successful_rounds,
        equilibrium=continuation_equilibrium,
        warmup_rounds=continuation_warmup_rounds,
        step_pct=continuation_step_pct,
        step_every=continuation_step_every,
    )
    # Frozen-arch refresh (ts_data_pipeline only). Snapshots at every_n
    # successful data-pipeline rounds; the first one bootstraps from the
    # ts_forecasting frontier.
    frozen_arch = None
    if isinstance(task, TSDataPipelineSpec) and frozen_arch_store is not None:
        maybe_refresh_frozen(
            frozen_arch_store, store, every_n=frozen_arch_refresh_every,
        )
        frozen_arch = frozen_arch_store.current()
        if frozen_arch is None:
            logger.warning(
                "  no frozen architecture available yet — round drops. "
                "Run ts_forecasting first so there's a frontier to snapshot.",
            )
            return

    challenge = _build_challenge(
        round_id, store, task, services_url, agent_seconds=agent_seconds,
        continuation_enabled=continuation_enabled,
        scheduled_continuation=scheduled,
        frozen_arch=frozen_arch,
    )
    challenge_id = challenge["challenge_id"]
    bucket = challenge["bucket"]
    continuation_allowed = challenge["continuation_allowed"]
    logger.info(
        "round=%d bucket=%s flops=[%d, %d] frontier=%d type=%s "
        "(cont_rate=%.2f ok_rounds=%d parents=%d)",
        round_id, bucket, challenge["min_flops_equivalent"],
        challenge["max_flops_equivalent"], len(challenge["feasible_frontier"]),
        challenge["round_type"], rate, successful_rounds,
        len(challenge["eligible_parents"]),
    )

    if sink is not None:
        sink.record_challenge(task.name, round_id, challenge)

    # ── Phase A: publish challenge, wait for proposals ──────
    store.post_challenge(challenge_id, round_id, challenge)
    deadline = time.time() + phase_a_seconds
    last_count = -1
    while time.time() < deadline:
        proposals = store.proposals_for(challenge_id)
        if len(proposals) != last_count:
            logger.info("  phase A: %d proposal(s) so far", len(proposals))
            last_count = len(proposals)
        # Early exit if at least one proposal in and 3 seconds have passed
        # without new arrivals — keeps single-miner laptop runs snappy.
        if proposals and time.time() > deadline - phase_a_seconds + 3:
            break
        time.sleep(0.5)
    store.mark_challenge(challenge_id, "collected")

    proposals = store.proposals_for(challenge_id)
    if not proposals:
        logger.warning("  no proposals received; round drops")
        store.mark_challenge(challenge_id, "done")
        return

    # Pretrain shard pool (ts_forecasting only) — assigned per-run so
    # continuations can avoid lineage-seen shards. ts_data_pipeline's
    # data source is the miner's pipeline; shard assignment is N/A.
    pool = (
        _pretrain_pool()
        if continuation_allowed and isinstance(task, TSForecastingSpec)
        else []
    )

    # ── Phase B + Phase C: train and evaluate every proposal ──
    results: list[dict] = []
    for p in proposals:
        payload = p["payload"]
        miner_id = p["miner_id"]
        name = payload.get("name", "unnamed")
        if sink is not None:
            sink.record_proposal(task.name, round_id, miner_id, payload)

        prep = prepare_continuation(
            store, ckpt_store,
            payload=payload, task_name=task.name,
            min_flops=challenge["min_flops_equivalent"],
            max_flops=challenge["max_flops_equivalent"],
            pool=pool, shards_per_round=shards_per_round, seed=round_id,
            current_epoch=_current_epoch(task, frozen_arch),
        ) if continuation_allowed else {
            # Continuation disabled — still preserve any payload parent_index
            # so the existing lineage/diff tracking keeps working.
            "mode": "new",
            "parent_index": (
                payload.get("parent_index")
                if isinstance(payload.get("parent_index"), int) else None
            ),
            "parent_metric": None,
            "parent_checkpoint_path": None, "compute_offset": 0.0,
            "step_offset": 0, "n_rounds": 1, "shard_paths": None,
            "shard_reuse": False, "note": "",
        }
        if prep["note"]:
            logger.info("    %s", prep["note"])
        logger.info(
            "  phase B/C: training '%s' from miner=%s mode=%s",
            name, miner_id, prep["mode"],
        )
        result = _train_proposal(
            payload, task, round_id, challenge, prep, frozen_arch=frozen_arch,
        )

        result["miner_id"] = miner_id
        result["name"] = name
        result["code"] = payload.get("code", "")
        result["motivation"] = payload.get("motivation", "")
        result["reasoning"] = payload.get("reasoning", "")
        result["tool_calls"] = payload.get("tool_calls", [])
        result["prompt_id"] = payload.get("prompt_id", "")
        result["mode"] = prep["mode"]
        result["parent_index"] = prep["parent_index"]
        result["parent_metric"] = prep["parent_metric"]
        result["n_rounds"] = prep["n_rounds"]
        result["continuation_note"] = prep["note"]
        if result["success"]:
            objs = result["objectives"]
            extra = ""
            if "crps" in objs and "mase" in objs:
                extra = f" crps={objs['crps']:.4f} mase={objs['mase']:.4f}"
            logger.info(
                "    metric=%.6f params=%d flops_eq=%d%s",
                result["metric"], objs["num_params"],
                objs["flops_equivalent_size"], extra,
            )
        else:
            logger.warning("    failed: %s", result.get("error", "?"))
        results.append(result)

    # ── Scoring ─────────────────────────────────────────────
    cont_frontier = continuation_frontier(store.recent_experiments(n=10_000))
    score_round(
        results,
        min_flops=challenge["min_flops_equivalent"],
        max_flops=challenge["max_flops_equivalent"],
        frontier=challenge["feasible_frontier"],
        continuation_frontier=cont_frontier,
    )

    # Write experiments + persist checkpoints + mirror artifacts
    for p, r in zip(proposals, results):
        # On failure, fold the trainer's error trace into analysis so
        # /experiments/failures returns something actionable to other
        # miners (the analysis label alone is just "training crashed").
        analysis = r["analysis"]
        if not r["success"]:
            err = (r.get("error") or "").strip()
            if err and err not in analysis:
                analysis = f"{analysis}\n{err}" if analysis else err
        note = r.get("continuation_note") or ""
        if note:
            analysis = f"{analysis} [{note}]"
        exp_id = store.add_experiment(
            round_id=round_id,
            miner_id=r["miner_id"],
            name=r["name"],
            code=r["code"],
            motivation=r["motivation"],
            reasoning=r["reasoning"],
            tool_calls=r["tool_calls"],
            metric=r["metric"],
            success=r["success"],
            objectives=r["objectives"],
            score=r["score"],
            loss_curve=r["loss_curve"],
            val_curve=r.get("val_curve") or [],
            analysis=analysis,
            parent_index=r.get("parent_index"),
            prompt_id=r["prompt_id"],
            task=task.name,
            n_rounds=int(r.get("n_rounds", 1) or 1),
            cumulative_compute=float(
                r["objectives"].get("cumulative_compute", 0.0)
            ),
            mode=r.get("mode", "new"),
        )
        workdir_str = r.get("workdir") or ""
        workdir = Path(workdir_str) if workdir_str else None
        # Persist the trained checkpoint so this experiment can later be a
        # warm-start parent, BEFORE the workdir is cleaned up.
        if r["success"] and ckpt_store is not None and workdir is not None:
            ckpt_src = workdir / "checkpoints" / "model.safetensors"
            ref = ckpt_store.save(exp_id, ckpt_src)
            if ref is not None:
                store.set_checkpoint_ref(exp_id, ref)
        try:
            if sink is not None:
                sink.record_result(task.name, round_id, r["miner_id"], r, workdir)
        finally:
            # Always reclaim the trainer workdir, even if mirroring raised —
            # otherwise the /tmp/radar_ts_* dir leaks for the run's lifetime.
            cleanup_workdir(workdir)

    # Keep only checkpoints worth warm-starting from (frontier + recent
    # lineage parents); drop the rest to bound disk use.
    if ckpt_store is not None:
        _gc_checkpoints(store, ckpt_store, round_id)

    store.mark_challenge(challenge_id, "done")

    # ── Round summary ───────────────────────────────────────
    winner = max(results, key=lambda r: r.get("score", 0.0))
    w_objs = winner.get("objectives", {}) or {}
    if "crps" in w_objs and "mase" in w_objs:
        win_extra = f" crps={w_objs['crps']:.4f} mase={w_objs['mase']:.4f}"
    else:
        win_extra = ""
    logger.info(
        "  round done: winner=%s metric=%s score=%.3f%s",
        winner["name"],
        f"{winner['metric']:.6f}" if winner.get("metric") is not None else "FAIL",
        winner.get("score", 0.0),
        win_extra,
    )
    stats = store.stats()
    logger.info(
        "  store: total=%d successful=%d best=%s",
        stats["total"], stats["successful"],
        f"{stats['best_metric']:.6f}" if stats["best_metric"] is not None else "—",
    )

    # Flush this round's agent events to R2 if configured. On success the
    # local rows are dropped so the SQLite file stays bounded on long runs;
    # on failure they're kept and the next round (or a manual export) can
    # retry. Wrapped in try/except because the round is already complete
    # by this point and the flush must not break the loop.
    bucket = os.getenv("RADAR_EVENT_LOG_R2_BUCKET", "").strip()
    if bucket:
        try:
            from local.export_events import flush_round_to_r2
            explicit_prefix = os.getenv("RADAR_EVENT_LOG_R2_PREFIX", "").strip()
            if explicit_prefix:
                prefix = explicit_prefix
            else:
                instance = os.getenv("RADAR_INSTANCE_ID", "").strip()
                prefix = f"agent-events/{instance}" if instance else "agent-events"
            flush_round_to_r2(store, round_id, bucket, prefix=prefix)
        except Exception as e:  # noqa: BLE001
            logger.warning("agent-events flush failed for round=%d: %s",
                           round_id, e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local radar validator")
    parser.add_argument("--db", default="local/radar_local.db",
                        help="SQLite path (default: local/radar_local.db)")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Number of rounds to run; 0 = forever")
    parser.add_argument("--phase_a_seconds", type=float, default=1900.0,
                        help="Max wait for miner proposals each round. "
                             "Should be ≥ --agent_seconds. Default 1900s "
                             "leaves 100s of slack on top of a 30-min agent.")
    parser.add_argument("--agent_seconds", type=int, default=1800,
                        help="Time budget published to the agent in "
                             "challenge['agent_seconds'] — LLM design loop. "
                             "Agents reserve the last 30s for a fallback, "
                             "so anything < 60 leaves no room for the LLM. "
                             "Default 1800s (30 min).")
    parser.add_argument("--training_seconds", type=int, default=3600,
                        help="Phase B training budget for the ts_forecasting "
                             "task (passed to runner.harness as time_budget). "
                             "Default 3600s (1 hour). Ignored for "
                             "synth_regression.")
    parser.add_argument("--gap_seconds", type=float, default=2.0,
                        help="Sleep between rounds")
    parser.add_argument("--services_port", type=int, default=0,
                        help="HTTP port for the agent-facing services "
                             "server (db/llm/desearch/wiki). 0 = pick free.")
    parser.add_argument("--services_url_file", default="",
                        help="If set, write the chosen services URL here "
                             "after binding (used by run.py to forward it "
                             "to the miner process).")
    parser.add_argument("--wiki_dir", default="",
                        help="Local directory of markdown files exposed to "
                             "the agent at GET /wiki. Empty = no wiki.")
    parser.add_argument(
        "--task", default="synth_regression",
        choices=["synth_regression", "ts_forecasting", "ts_data_pipeline"],
        help="Which task this validator drives.",
    )
    parser.add_argument(
        "--frozen_arch_refresh_every", type=int, default=50,
        help="ts_data_pipeline only: refresh the frozen architecture every "
             "N successful data-pipeline rounds (default 50).",
    )
    parser.add_argument(
        "--frozen_arch_dir", default="",
        help="ts_data_pipeline only: where versioned frozen archs live. "
             "Empty = $RADAR_FROZEN_ARCH_DIR or local/frozen_archs.",
    )
    parser.add_argument("--continuation", default="auto",
                        choices=["auto", "on", "off"],
                        help="Allow continuation (warm-start) proposals. "
                             "'auto' = on for ts_forecasting, off for "
                             "synth_regression (no checkpoints there).")
    parser.add_argument("--shards_per_round", type=int, default=0,
                        help="Pretrain shards assigned per run. 0 = all "
                             "(legacy; continuations then reuse shards). "
                             "Set >0 to leave disjoint headroom so "
                             "continuations train on unseen shards.")
    parser.add_argument("--checkpoint_dir", default="",
                        help="Durable checkpoint store dir for continuation "
                             "warm-starts. Empty = $RADAR_CHECKPOINT_DIR or "
                             "local/checkpoints.")
    parser.add_argument("--continuation_equilibrium", type=float, default=0.7,
                        help="Steady-state fraction of rounds the validator "
                             "schedules as continuations (default 0.70).")
    parser.add_argument("--continuation_warmup_rounds", type=int, default=50,
                        help="Successful rounds with 0%% continuation before "
                             "the ramp starts (default 50).")
    parser.add_argument("--continuation_step_pct", type=float, default=1.0,
                        help="Percentage points the continuation rate climbs "
                             "per step after warmup (default 1.0).")
    parser.add_argument("--continuation_step_every", type=int, default=5,
                        help="Successful rounds per ramp step (default 5). "
                             "Defaults give 0→70%% over ~350 rounds post-warmup.")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [validator] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Reclaim trainer workdirs orphaned by a prior crashed/killed run before
    # we start minting new ones.
    sweep_orphan_workdirs()

    # Optional R2/Hippius backup: restore latest snapshot if the local DB
    # is missing, then run a periodic upload daemon for the lifetime of
    # the validator. No-op unless RADAR_BACKUP_BUCKET is set.
    backup = backup_from_env(args.db)
    if backup is not None:
        backup.restore_if_missing()

    store = LocalStore(args.db)
    if backup is not None:
        backup.start()
    task = make_spec(args.task)
    if isinstance(task, (TSForecastingSpec, TSDataPipelineSpec)):
        task.time_budget_seconds = args.training_seconds
        # GIFT-Eval is non-negotiable: Phase C always runs the full 97-task
        # leaderboard, so there's no point starting rounds without it. No
        # skip flag — fix creds / disk and rerun.
        if not _ensure_ts_caches():
            logger.error("%s caches not ready — aborting.", task.name)
            return 2

    frozen_arch_store: FrozenArchStore | None = None
    if isinstance(task, TSDataPipelineSpec):
        frozen_arch_store = FrozenArchStore(
            base_dir=args.frozen_arch_dir or None,
        )
        # Best-effort bootstrap so the very first round has a snapshot.
        maybe_refresh_frozen(
            frozen_arch_store, store, every_n=args.frozen_arch_refresh_every,
        )
        current = frozen_arch_store.current()
        if current is None:
            logger.warning(
                "ts_data_pipeline: no frozen arch available yet — "
                "early rounds will drop until a ts_forecasting experiment "
                "with both crps + mase lands in the same db.",
            )
        else:
            logger.info(
                "ts_data_pipeline: frozen arch v%d (exp=%d metric=%.6f)",
                current.version, current.source_experiment_id,
                current.source_metric,
            )

    if args.continuation == "on":
        continuation_enabled = True
    elif args.continuation == "off":
        continuation_enabled = False
    else:  # auto
        # Continuation is on by default for both ts_forecasting and
        # ts_data_pipeline (the latter gates parents on
        # frozen_arch_version so Δ is honest across epochs). Synthetic
        # regression has no checkpoints, so it stays off.
        continuation_enabled = isinstance(
            task, (TSForecastingSpec, TSDataPipelineSpec),
        )

    logger.info(
        "starting; db=%s task=%s agent_seconds=%d training_seconds=%s "
        "continuation=%s (eq=%.2f warmup=%d +%.1f%%/%d) shards_per_round=%d",
        args.db, task.name, args.agent_seconds,
        getattr(task, "time_budget_seconds", "n/a"),
        continuation_enabled, args.continuation_equilibrium,
        args.continuation_warmup_rounds, args.continuation_step_pct,
        args.continuation_step_every, args.shards_per_round,
    )

    sink = ArtifactSink.from_env(store)
    ckpt_store = CheckpointStore(
        base_dir=args.checkpoint_dir or None, sink=sink,
    ) if continuation_enabled else None

    wiki_dir = args.wiki_dir or None
    if not wiki_dir:
        try:
            from shared.cognition_wiki import ensure_wiki_cached
            cached = ensure_wiki_cached(task.name)
        except Exception as e:
            logger.warning("cognition-wiki fetch raised %s — continuing without", e)
            cached = None
        if cached is not None:
            wiki_dir = str(cached)
            logger.info("cognition-wiki: serving %s at /wiki", wiki_dir)

    services = ServicesServer(
        store=store,
        wiki_dir=wiki_dir,
        port=args.services_port,
        sink=sink,
    )
    services_url = services.start()
    logger.info(
        "services: db=%s llm=%s/llm desearch=%s/desearch wiki=%s/wiki",
        services_url, services_url, services_url, services_url,
    )
    if args.services_url_file:
        Path(args.services_url_file).write_text(services_url)

    round_id = _next_round_id(store)
    completed = 0
    try:
        while args.rounds == 0 or completed < args.rounds:
            run_round(store, task, round_id,
                      phase_a_seconds=args.phase_a_seconds,
                      services_url=services_url,
                      agent_seconds=args.agent_seconds,
                      sink=sink,
                      ckpt_store=ckpt_store,
                      continuation_enabled=continuation_enabled,
                      continuation_equilibrium=args.continuation_equilibrium,
                      continuation_warmup_rounds=args.continuation_warmup_rounds,
                      continuation_step_pct=args.continuation_step_pct,
                      continuation_step_every=args.continuation_step_every,
                      shards_per_round=args.shards_per_round,
                      frozen_arch_store=frozen_arch_store,
                      frozen_arch_refresh_every=args.frozen_arch_refresh_every)
            round_id += 1
            completed += 1
            if args.rounds == 0 or completed < args.rounds:
                time.sleep(args.gap_seconds)
    except KeyboardInterrupt:
        logger.info("interrupted; bye")
    finally:
        services.stop()
        store.close()
        if backup is not None:
            backup.stop(final=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
