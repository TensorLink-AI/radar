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

from local.scoring import compute_pareto, passes_size_gate, score_round
from local.store import LocalStore
from local.task import SIZE_BUCKETS, TaskSpec
from local.trainer import run_training


logger = logging.getLogger("local.validator")


def _pick_bucket(round_id: int) -> tuple[str, int, int]:
    names = list(SIZE_BUCKETS.keys())
    name = names[round_id % len(names)]
    lo, hi = SIZE_BUCKETS[name]
    return name, lo, hi


def _build_challenge(round_id: int, store: LocalStore, task: TaskSpec) -> dict:
    name, lo, hi = _pick_bucket(round_id)
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
    return {
        "challenge_id": f"r{round_id:06d}-{uuid.uuid4().hex[:8]}",
        "round_id": round_id,
        "seed": round_id * 31 + 7,
        "min_flops_equivalent": lo,
        "max_flops_equivalent": hi,
        "bucket": name,
        "task": {
            "name": task.name,
            "input_dim": task.input_dim,
            "output_dim": task.output_dim,
        },
        "feasible_frontier": feasible,
        "agent_seconds": 30,  # local agent runs in milliseconds
    }


def _next_round_id(store: LocalStore) -> int:
    """Continue from the highest round_id seen so far so consecutive
    runs of the validator don't repeat round 0."""
    rows = store._conn.execute(
        "SELECT COALESCE(MAX(round_id), -1) AS r FROM challenges"
    ).fetchone()
    return int(rows["r"]) + 1


def run_round(store: LocalStore, task: TaskSpec, round_id: int,
              phase_a_seconds: float) -> None:
    challenge = _build_challenge(round_id, store, task)
    challenge_id = challenge["challenge_id"]
    bucket = challenge["bucket"]
    logger.info(
        "round=%d bucket=%s flops=[%d, %d] frontier=%d",
        round_id, bucket, challenge["min_flops_equivalent"],
        challenge["max_flops_equivalent"], len(challenge["feasible_frontier"]),
    )

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

    # ── Phase B + Phase C: train and evaluate every proposal ──
    results: list[dict] = []
    for p in proposals:
        payload = p["payload"]
        miner_id = p["miner_id"]
        name = payload.get("name", "unnamed")
        logger.info("  phase B/C: training '%s' from miner=%s", name, miner_id)
        result = run_training(payload.get("code", ""), seed=round_id)
        result["miner_id"] = miner_id
        result["name"] = name
        result["code"] = payload.get("code", "")
        result["motivation"] = payload.get("motivation", "")
        result["reasoning"] = payload.get("reasoning", "")
        result["tool_calls"] = payload.get("tool_calls", [])
        result["prompt_id"] = payload.get("prompt_id", "")
        if result["success"]:
            logger.info(
                "    metric=%.6f params=%d flops_eq=%d",
                result["metric"], result["objectives"]["num_params"],
                result["objectives"]["flops_equivalent_size"],
            )
        else:
            logger.warning("    failed: %s", result.get("error", "?"))
        results.append(result)

    # ── Scoring ─────────────────────────────────────────────
    score_round(
        results,
        min_flops=challenge["min_flops_equivalent"],
        max_flops=challenge["max_flops_equivalent"],
        frontier=challenge["feasible_frontier"],
    )

    # Write experiments
    for r in results:
        store.add_experiment(
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
            analysis=r["analysis"],
            prompt_id=r["prompt_id"],
            task=task.name,
        )

    store.mark_challenge(challenge_id, "done")

    # ── Round summary ───────────────────────────────────────
    winner = max(results, key=lambda r: r.get("score", 0.0))
    logger.info(
        "  round done: winner=%s metric=%s score=%.3f",
        winner["name"],
        f"{winner['metric']:.6f}" if winner.get("metric") is not None else "FAIL",
        winner.get("score", 0.0),
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
    parser.add_argument("--phase_a_seconds", type=float, default=10.0,
                        help="Max wait for miner proposals each round")
    parser.add_argument("--gap_seconds", type=float, default=2.0,
                        help="Sleep between rounds")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [validator] %(message)s",
        datefmt="%H:%M:%S",
    )

    store = LocalStore(args.db)
    task = TaskSpec()
    logger.info("starting; db=%s task=%s", args.db, task.name)

    round_id = _next_round_id(store)
    completed = 0
    try:
        while args.rounds == 0 or completed < args.rounds:
            run_round(store, task, round_id,
                      phase_a_seconds=args.phase_a_seconds)
            round_id += 1
            completed += 1
            if args.rounds == 0 or completed < args.rounds:
                time.sleep(args.gap_seconds)
    except KeyboardInterrupt:
        logger.info("interrupted; bye")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
