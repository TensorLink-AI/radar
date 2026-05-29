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
import sys
import time
import uuid
from pathlib import Path

# Allow ``python local/validator.py`` from repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local.artifacts import ArtifactSink, cleanup_workdir
from local.checkpoints import CheckpointStore
from local.continuation import continuation_frontier, prepare_continuation
from local.experiments_api import _parent_summary
from local.scoring import compute_pareto, passes_size_gate, score_round
from local.services import ServicesServer
from local.store import LocalStore
from local.task import SIZE_BUCKETS, TaskSpec, TSForecastingSpec, buckets_for, make_spec
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


def _build_challenge(round_id: int, store: LocalStore, task,
                     services_url: str, agent_seconds: int = 180,
                     continuation_allowed: bool = False) -> dict:
    name, lo, hi = _pick_bucket(round_id, task=task)
    all_exps = store.recent_experiments(n=10_000)
    pareto = compute_pareto(all_exps)
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
    # Continuation-eligible parents (compact, weights-free) so a
    # zero-network agent can pick a warm-start candidate from the
    # challenge alone; full trajectories are at /experiments/{id}/trajectory.
    eligible_parents = []
    if continuation_allowed:
        eligible_parents = [
            _parent_summary(e)
            for e in store.eligible_parents(
                task=task.name, min_flops=lo, max_flops=hi,
            )
        ]
    # ``allowed_urls`` is what the miner-side GatedClient enforces. Every
    # endpoint the agent can reach lives under ``services_url`` so a
    # single prefix is enough.
    return {
        "challenge_id": f"r{round_id:06d}-{uuid.uuid4().hex[:8]}",
        "round_id": round_id,
        "seed": round_id * 31 + 7,
        "min_flops_equivalent": lo,
        "max_flops_equivalent": hi,
        "bucket": name,
        "task": _task_dict(task),
        "feasible_frontier": feasible,
        "continuation_allowed": bool(continuation_allowed),
        "eligible_parents": eligible_parents,
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
                    challenge: dict, prep: dict) -> dict:
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
              continuation_allowed: bool = False,
              shards_per_round: int = 0) -> None:
    challenge = _build_challenge(
        round_id, store, task, services_url, agent_seconds=agent_seconds,
        continuation_allowed=continuation_allowed,
    )
    challenge_id = challenge["challenge_id"]
    bucket = challenge["bucket"]
    logger.info(
        "round=%d bucket=%s flops=[%d, %d] frontier=%d",
        round_id, bucket, challenge["min_flops_equivalent"],
        challenge["max_flops_equivalent"], len(challenge["feasible_frontier"]),
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
    # continuations can avoid lineage-seen shards.
    pool = _pretrain_pool() if continuation_allowed else []

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
        result = _train_proposal(payload, task, round_id, challenge, prep)

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
        note = r.get("continuation_note") or ""
        analysis = r["analysis"] + (f" [{note}]" if note else "")
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
        if sink is not None:
            sink.record_result(task.name, round_id, r["miner_id"], r, workdir)
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
    parser.add_argument("--task", default="synth_regression",
                        choices=["synth_regression", "ts_forecasting"],
                        help="Which task this validator drives.")
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
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [validator] %(message)s",
        datefmt="%H:%M:%S",
    )

    store = LocalStore(args.db)
    task = make_spec(args.task)
    if isinstance(task, TSForecastingSpec):
        task.time_budget_seconds = args.training_seconds

    if args.continuation == "on":
        continuation_allowed = True
    elif args.continuation == "off":
        continuation_allowed = False
    else:  # auto
        continuation_allowed = isinstance(task, TSForecastingSpec)

    logger.info(
        "starting; db=%s task=%s agent_seconds=%d training_seconds=%s "
        "continuation=%s shards_per_round=%d",
        args.db, task.name, args.agent_seconds,
        getattr(task, "time_budget_seconds", "n/a"),
        continuation_allowed, args.shards_per_round,
    )

    sink = ArtifactSink.from_env(store)
    ckpt_store = CheckpointStore(
        base_dir=args.checkpoint_dir or None, sink=sink,
    ) if continuation_allowed else None

    services = ServicesServer(
        store=store,
        wiki_dir=args.wiki_dir or None,
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
                      continuation_allowed=continuation_allowed,
                      shards_per_round=args.shards_per_round)
            round_id += 1
            completed += 1
            if args.rounds == 0 or completed < args.rounds:
                time.sleep(args.gap_seconds)
    except KeyboardInterrupt:
        logger.info("interrupted; bye")
    finally:
        services.stop()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
