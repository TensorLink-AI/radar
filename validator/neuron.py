"""Radar Validator Neuron — 3-phase validation pipeline.

Phase A: Collect architecture submissions from miner agents
Phase B: Dispatch training to miner trainers (cross-eval)
Phase C: Every validator independently evaluates every checkpoint (trust anchor)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import random
import threading
import traceback
from typing import Optional

import bittensor as bt
import uvicorn

from config import Config
from shared.access_logger import AccessLogger
from shared.challenge import (
    generate_challenge, round_start_block, current_phase,
)
from shared.commitment import read_miner_commitments
from shared.database import DataElement
from shared.sqlite_store import SQLiteExperimentStore
from shared.pareto import ParetoFront
from shared.protocol import Challenge, Proposal
from shared.provenance import detect_components
from shared.scoring import score_round, scores_to_weights, ema_update, compute_penalties
from shared.task import TaskSpec, load_task
from validator.analyzer import analyze
from validator.collection import run_and_collect_agents
from validator.coordinator import (
    TrainingCoordinator, compute_assignments, compute_fallback,
)
from validator.db_server import (
    app as db_app, set_db, set_auth, set_challenge, set_frontier,
    set_access_logger, set_hotkey_map,
)
from validator.evaluator import evaluate_all_checkpoints
from validator.pod_manager import pre_validate_code

logger = logging.getLogger(__name__)


def get_my_assignments(
    all_uids: list[int],
    validator_uids: list[int],
    my_uid: int,
    seed: int,
) -> list[int]:
    """Deterministic assignment of UIDs to validators.

    Shuffle all_uids with seed, round-robin across sorted validator_uids.
    Returns the UIDs assigned to my_uid.
    """
    if not validator_uids or my_uid not in validator_uids:
        return list(all_uids)

    rng = random.Random(seed)
    shuffled = list(all_uids)
    rng.shuffle(shuffled)

    sorted_validators = sorted(validator_uids)
    my_idx = sorted_validators.index(my_uid)

    return [uid for i, uid in enumerate(shuffled) if i % len(sorted_validators) == my_idx]


class Validator:
    """Radar subnet validator — 3-phase pipeline."""

    def __init__(self, config: bt.Config):
        self.config = config
        self.netuid = config.netuid

        # Bittensor components
        self.wallet = bt.Wallet(config=config)
        self.subtensor = bt.Subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(self.netuid)

        # Task
        task_path = getattr(config, "task", "ml_training")
        self.task = load_task(task_path)

        # Experiment DB
        db_dir = getattr(config, "db_dir", "./experiments")
        self.db = SQLiteExperimentStore(
            db_path=os.path.join(db_dir, "experiments.db"),
        )
        set_db(self.db)

        # Per-task Pareto fronts
        self._task_cache: dict[str, TaskSpec] = {self.task.name: self.task}
        self.pareto_fronts: dict[str, ParetoFront] = {}
        self._rebuild_pareto()

        # EMA scores per UID
        self.ema_scores: dict[int, float] = {}

        # Trainer reliability tracking (keyed by trainer UID)
        self.trainer_reliability: dict[int, float] = {}

        # DB server config
        self.db_port = int(getattr(config, "db_port", 8080))

        # R2 audit log
        self.r2 = None
        if Config.R2_BUCKET:
            try:
                from shared.r2_audit import R2AuditLog
                self.r2 = R2AuditLog(bucket=Config.R2_BUCKET)
            except Exception as e:
                logger.warning("R2 audit log unavailable: %s", e)

        # Training coordinator
        my_uid = self._my_uid()
        if self.r2:
            self.coordinator = TrainingCoordinator(
                wallet=self.wallet, metagraph=self.metagraph,
                r2=self.r2, my_uid=my_uid,
            )
        else:
            self.coordinator = None

        # Desearch proxy
        self.desearch_proxy = None
        if Config.DESEARCH_ENABLED:
            from validator.desearch_proxy import DesearchProxy, set_proxy, register_routes
            self.desearch_proxy = DesearchProxy(
                sn22_url=Config.DESEARCH_SN22_URL,
                max_queries=Config.DESEARCH_MAX_QUERIES,
            )
            set_proxy(self.desearch_proxy)
            register_routes(db_app)

        # Access logger — shares the same SQLite connection
        self.access_logger = AccessLogger(conn=self.db.conn)
        set_access_logger(self.access_logger)

        # Auth on DB server
        set_auth(self.metagraph)

        logger.info(
            "Validator initialized. Task: %s, DB: %d experiments, R2: %s",
            self.task.name, self.db.size, "enabled" if self.r2 else "disabled",
        )

    def _load_task_spec(self, task_name: str) -> TaskSpec:
        """Load a TaskSpec by name, with caching."""
        if task_name not in self._task_cache:
            self._task_cache[task_name] = load_task(task_name)
        return self._task_cache[task_name]

    def _get_pareto(self, task_name: str) -> ParetoFront:
        """Get or create a ParetoFront for a task."""
        if task_name not in self.pareto_fronts:
            ts = self._load_task_spec(task_name)
            objective_fn = lambda elem, _ts=ts: _ts.objective_vector(elem.objectives)
            self.pareto_fronts[task_name] = ParetoFront(
                max_size=50, objective_fn=objective_fn,
            )
        return self.pareto_fronts[task_name]

    def _rebuild_pareto(self):
        """Rebuild Pareto fronts from all DB elements, grouped by task."""
        self.pareto_fronts = {}
        skipped = 0
        all_elements = self.db.get_pareto_elements()
        for elem in all_elements:
            if not elem.task:
                skipped += 1
                continue
            try:
                self._get_pareto(elem.task).update(elem)
            except Exception:
                logger.warning(
                    "Skipping experiment %d: unknown task %r (no YAML definition)",
                    elem.index, elem.task,
                )
        if skipped:
            logger.warning(
                "Pareto rebuild: skipped %d/%d experiments with empty task "
                "(pre-migration data excluded from Pareto fronts)",
                skipped, len(all_elements),
            )

    def _my_uid(self) -> int:
        """Get this validator's UID."""
        hotkey = self.wallet.hotkey.ss58_address
        if hotkey in self.metagraph.hotkeys:
            return self.metagraph.hotkeys.index(hotkey)
        return -1

    def _get_validator_uids(self) -> list[int]:
        """Get UIDs of all validators with permits."""
        return [
            uid for uid in range(self.metagraph.n)
            if self.metagraph.validator_permit[uid]
        ]

    def start_db_server(self):
        """Start the experiment DB server in a background thread."""
        def _run():
            uvicorn.run(db_app, host="0.0.0.0", port=self.db_port, log_level="warning")
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        logger.info("DB server started on port %d", self.db_port)

    async def _wait_until_block(self, target_block: int):
        """Wait until the chain reaches target_block."""
        while True:
            current = self.subtensor.block
            if current >= target_block:
                return
            remaining = (target_block - current) * 12  # ~12s per block
            await asyncio.sleep(min(remaining, 30))

    async def run_round(self):
        """Execute one full 3-phase round."""
        # ── SETUP ──────────────────────────────────────────────
        self.metagraph.sync()
        current_block = self.subtensor.block
        round_start = round_start_block(current_block, Config.ROUND_INTERVAL_BLOCKS)
        block_hash = hashlib.sha256(str(round_start).encode()).hexdigest()

        challenge = generate_challenge(block_hash, self.task.to_dict())
        task_name = self.task.name

        # Include the feasible frontier for this size bucket (per-task)
        pareto = self._get_pareto(task_name)
        feasible = pareto.get_feasible(
            challenge.min_flops_equivalent, challenge.max_flops_equivalent,
        )
        challenge.feasible_frontier = [
            {
                "code": c.element.code,
                "metric": c.element.metric,
                "objectives": c.element.objectives,
                "parent_diff": self.db.get_diff(c.element.index),
                "motivation": c.element.motivation,
                "task": task_name,
            }
            for c in feasible
        ]

        # Set up access logging for this round
        self.access_logger.set_round(challenge.round_id)
        hotkey_map = {
            self.metagraph.hotkeys[uid]: uid
            for uid in range(self.metagraph.n)
            if uid < len(self.metagraph.hotkeys)
        }
        set_hotkey_map(hotkey_map)

        challenge.db_url = f"http://localhost:{self.db_port}"
        challenge.desearch_url = (
            f"http://localhost:{self.db_port}/desearch" if self.desearch_proxy else ""
        )

        # Set challenge on DB server for miners to query
        set_challenge(challenge.to_json())

        logger.info(
            "Round %d: size bucket [%d, %d], frontier points in range: %d",
            challenge.round_id, challenge.min_flops_equivalent,
            challenge.max_flops_equivalent, len(feasible),
        )

        # ── PHASE A: RUN AGENTS + COLLECT ────────────────────
        # Run agents DURING Phase A (not after), then wait for the window
        # to close so all validators finish before Phase B starts.
        validator_uids = self._get_validator_uids()
        all_commitments = read_miner_commitments(self.subtensor, self.netuid, self.metagraph)

        # Filter out validator neurons — only miners should be in the pipeline
        commitments = {
            uid: c for uid, c in all_commitments.items()
            if uid not in validator_uids
        }
        if len(commitments) < len(all_commitments):
            logger.info(
                "Filtered %d validator neurons from %d commitments → %d miners",
                len(all_commitments) - len(commitments),
                len(all_commitments), len(commitments),
            )

        if not self.r2:
            logger.warning("No R2 configured — skipping Phase A/B/C")
            return

        submissions, agent_logs = await run_and_collect_agents(
            wallet=self.wallet,
            metagraph=self.metagraph,
            challenge_json=challenge.to_json(),
            round_id=challenge.round_id,
            seed=challenge.seed,
            r2=self.r2,
            my_uid=self._my_uid(),
            validator_uids=validator_uids,
            commitments=commitments,
            get_my_assignments_fn=get_my_assignments,
        )

        # Pre-filter: syntax check
        filtered: dict[int, Proposal] = {}
        for uid, proposal in submissions.items():
            ok, reason = pre_validate_code(proposal.code)
            if not ok:
                continue
            filtered[uid] = proposal

        logger.info("Phase A: %d proposals, %d after filter", len(submissions), len(filtered))

        if not filtered:
            logger.info("No valid submissions this round")
            return

        # Wait for Phase A window to close before starting Phase B
        await self._wait_until_block(round_start + Config.SUBMISSION_WINDOW_BLOCKS)

        # ── PHASE B: DISPATCH TRAINING ────────────────────────
        if not self.coordinator:
            logger.warning("No coordinator configured — skipping Phase B/C")
            return

        my_uid = self._my_uid()

        # Ensure this validator is in the dispatch rotation even if
        # validator_permit hasn't propagated yet (common on localnet).
        dispatch_validators = list(validator_uids)
        if my_uid >= 0 and my_uid not in dispatch_validators:
            logger.warning(
                "My UID %d not in validator_uids %s — adding self for dispatch",
                my_uid, dispatch_validators,
            )
            dispatch_validators.append(my_uid)

        # Filter commitments: exclude validators and neurons without
        # trainer endpoints so they are never assigned as trainers.
        trainer_endpoints = {uid: c.pod_url for uid, c in commitments.items() if c.pod_url}
        miner_uids = [
            uid for uid in commitments
            if uid not in validator_uids and uid in trainer_endpoints
        ]
        excluded = [uid for uid in commitments if uid not in miner_uids]
        if excluded:
            logger.info(
                "Phase B: excluded %d neurons from trainer pool "
                "(validators or missing trainer endpoint): %s",
                len(excluded), excluded,
            )

        all_jobs = compute_assignments(
            block_hash, filtered,
            miner_uids=miner_uids,
            validator_uids=dispatch_validators,
            round_id=challenge.round_id,
        )
        my_jobs = [j for j in all_jobs if j.dispatcher == my_uid]

        # Safety net: if round-robin gave us 0 jobs (assigned to offline
        # validators like the subnet owner), dispatch ALL jobs so training
        # isn't lost.  R2 uploads are keyed by miner hotkey, so duplicate
        # dispatches from multiple validators are harmless.
        if not my_jobs and all_jobs:
            logger.warning(
                "Round-robin assigned 0 of %d jobs to my_uid=%d — "
                "dispatching all jobs (other dispatchers may be offline)",
                len(all_jobs), my_uid,
            )
            my_jobs = list(all_jobs)

        logger.info(
            "Phase B: %d total jobs, %d mine (my_uid=%d, validators=%s), %d trainer endpoints",
            len(all_jobs), len(my_jobs), my_uid, dispatch_validators, len(trainer_endpoints),
        )
        if not trainer_endpoints:
            logger.warning(
                "Phase B: no trainer endpoints! Miners must set --trainer_url "
                "or trainer pods must be deployed. Training will fail for all jobs."
            )

        my_results = await self.coordinator.dispatch_jobs(
            my_jobs, challenge, filtered, trainer_endpoints,
            commitments=commitments,
        )
        succeeded = sum(1 for r in my_results if r.status == "success")
        failed = sum(1 for r in my_results if r.status != "success")
        logger.info("Phase B dispatch: %d succeeded, %d failed", succeeded, failed)
        for r in my_results:
            if r.status != "success":
                logger.warning(
                    "  Job arch=%d trainer=%d: %s — %s",
                    r.arch_owner, r.trainer_uid, r.status, r.error,
                )

        await self.coordinator.write_dispatch_record(challenge.round_id, my_results)

        # Wait for training window to close (keeps validators in sync on mainnet).
        # In test environments, skip the wait since local dispatch already finished.
        if Config.SKIP_TRAINING_WAIT:
            logger.info("SKIP_TRAINING_WAIT enabled — skipping block wait after dispatch")
        else:
            await self._wait_until_block(
                round_start + Config.SUBMISSION_WINDOW_BLOCKS + Config.TRAINING_WINDOW_BLOCKS
            )

        # ── Checkpoint collection ─────────────────────────────
        training_metas = await self.coordinator.wait_for_checkpoints(
            challenge.round_id,
            expected_miners=list(filtered.keys()),
            timeout=Config.EVAL_WINDOW_BLOCKS * 12,
        )

        logger.info(
            "Checkpoint collection: %d/%d miners have training metas",
            len(training_metas), len(filtered),
        )
        for uid, meta in training_metas.items():
            logger.info(
                "  UID %d: status=%s flops=%d error=%s",
                uid, meta.get("status", "?"),
                meta.get("flops_equivalent_size", 0),
                meta.get("error", ""),
            )

        # Fallback for missing validators (off by default)
        missing_trainers = [uid for uid in filtered if uid not in training_metas]
        if missing_trainers:
            logger.info("Missing checkpoints from %d miners: %s", len(missing_trainers), missing_trainers)

        if Config.FALLBACK_ENABLED and missing_trainers:
            logger.info("Fallback enabled — reassigning %d missing jobs", len(missing_trainers))
            missing_dispatchers = list({
                j.dispatcher for j in all_jobs
                if j.arch_owner in missing_trainers
            })
            remaining_valis = [v for v in dispatch_validators if v not in missing_dispatchers]
            if remaining_valis:
                fallback_jobs = compute_fallback(
                    block_hash, missing_dispatchers, all_jobs, remaining_valis,
                )
                my_fallback = [j for j in fallback_jobs if j.dispatcher == my_uid]
                if my_fallback:
                    fb_results = await self.coordinator.dispatch_jobs(
                        my_fallback, challenge, filtered, trainer_endpoints,
                        commitments=commitments,
                    )
                    await self.coordinator.write_dispatch_record(
                        challenge.round_id, fb_results,
                    )

                # Wait for fallback window
                await self._wait_until_block(
                    round_start + Config.SUBMISSION_WINDOW_BLOCKS
                    + Config.TRAINING_WINDOW_BLOCKS
                    + Config.FALLBACK_WINDOW_BLOCKS
                )

                # Re-read checkpoints
                fb_metas = await self.coordinator.wait_for_checkpoints(
                    challenge.round_id,
                    expected_miners=missing_trainers,
                    timeout=Config.FALLBACK_WINDOW_BLOCKS * 12,
                )
                training_metas.update(fb_metas)

        # ── Pre-cache GIFT-Eval data for Phase C ────────────────
        if self.r2:
            try:
                from shared.gift_eval import GiftEvalBenchmark
                gift = GiftEvalBenchmark(
                    r2=self.r2,
                    cache_dir=Config.GIFT_EVAL_CACHE_DIR,
                    r2_prefix=Config.GIFT_EVAL_R2_PREFIX,
                )
                selected = gift.select_datasets(
                    eval_split_seed=challenge.eval_split_seed,
                    n=Config.GIFT_EVAL_DATASETS_PER_ROUND,
                )
                for ds_name in selected:
                    try:
                        gift.download_dataset(ds_name)
                    except Exception as e:
                        logger.warning("GIFT-Eval download failed for %s: %s", ds_name, e)
                logger.info(
                    "GIFT-Eval: cached %d datasets for seed=%d: %s",
                    len(selected), challenge.eval_split_seed, selected,
                )
            except Exception as e:
                logger.warning("GIFT-Eval pre-cache failed: %s", e)

        # ── PHASE C: EVALUATION (TRUST ANCHOR) ────────────────
        eval_results = await evaluate_all_checkpoints(
            r2=self.r2,
            round_id=challenge.round_id,
            training_metas=training_metas,
            challenge=challenge,
            task=self.task,
        )

        logger.info("Phase C: evaluated %d checkpoints", len(eval_results))
        for uid, metrics in eval_results.items():
            logger.info(
                "  UID %d: crps=%.6f flops=%d gate=%s error=%s",
                uid,
                metrics.get("crps", float("inf")),
                metrics.get("flops_equivalent_size", 0),
                metrics.get("passed_size_gate", False),
                metrics.get("error", ""),
            )

        # ── SCORING ───────────────────────────────────────────
        penalties = compute_penalties(training_metas, eval_results)

        # Track trainer reliability for future job assignment
        for uid, meta in training_metas.items():
            trainer_uid = meta.get("trainer_uid", uid)
            status = meta.get("status", "")
            score = 1.0 if status == "success" else 0.0
            alpha = Config.EMA_ALPHA
            prev = self.trainer_reliability.get(trainer_uid, 1.0)
            self.trainer_reliability[trainer_uid] = alpha * score + (1 - alpha) * prev

        # Record which frontier experiments were shown this round
        for entry in challenge.feasible_frontier:
            entry_code = entry.get("code", "")
            if entry_code:
                # Find matching experiment by code (use DB lock)
                with self.db._lock:
                    match = self.db.conn.execute(
                        "SELECT id FROM experiments WHERE code = ? LIMIT 1",
                        (entry_code,),
                    ).fetchone()
                if match:
                    self.db.provenance.record_round_context(
                        challenge.round_id, match[0], "frontier",
                    )

        # Update frontier with eval-verified metrics
        for uid, metrics in eval_results.items():
            if metrics.get("passed_size_gate"):
                proposal = filtered.get(uid, Proposal())

                miner_hk = self.metagraph.hotkeys[uid] if uid < len(self.metagraph.hotkeys) else ""
                element = DataElement(
                    name=f"round_{challenge.round_id}_miner_{uid}",
                    code=proposal.code,
                    metric=metrics.get("crps"),
                    success=True,
                    objectives=metrics,
                    miner_uid=uid,
                    miner_hotkey=miner_hk,
                    motivation=proposal.motivation,
                    trace=agent_logs.get(uid, ""),
                    task=task_name,
                    round_id=challenge.round_id,
                )
                self.db.add(element)
                pareto.update(element)

                # Detect and store architectural components
                components = detect_components(proposal.code)
                if components:
                    self.db.provenance.record_components(element.index, components)

        round_scores = score_round(
            eval_results, challenge, pareto, self.task.objectives, penalties,
            training_metas=training_metas,
        )

        if round_scores:
            nonzero = {uid: s for uid, s in round_scores.items() if s > 0}
            logger.info(
                "Round scores: %d total, %d nonzero — %s",
                len(round_scores), len(nonzero),
                {uid: f"{s:.4f}" for uid, s in sorted(nonzero.items())} if nonzero else "all zero",
            )
        else:
            logger.warning("Round scores empty — no eval results passed size gate")

        # EMA on raw scores, softmax applied once in _set_weights()
        all_uids = list(range(self.metagraph.n))
        self.ema_scores = ema_update(self.ema_scores, round_scores, all_uids, Config.EMA_ALPHA)
        self._set_weights()

        # Write frontier to R2
        if self.coordinator:
            await self.coordinator.write_frontier(
                [c.element.to_dict() for c in pareto.candidates],
                task_name=task_name,
            )

        nonzero_scored = sum(1 for s in round_scores.values() if s > 0)
        logger.info(
            "Round %d complete. DB: %d, Pareto: %d, scored: %d (%d nonzero)",
            challenge.round_id, self.db.size, pareto.size,
            len(round_scores), nonzero_scored,
        )

    def _set_weights(self):
        """Set weights on chain — softmax on EMA scores (single normalization)."""
        if not self.ema_scores:
            logger.warning("No EMA scores — skipping weight setting")
            return
        try:
            uids, weights = scores_to_weights(self.ema_scores, Config.SOFTMAX_TEMPERATURE)
            nonzero_weights = {
                uid: f"{w:.4f}" for uid, w in zip(uids, weights) if w > 0
            }
            if nonzero_weights:
                logger.info(
                    "Setting weights for %d UIDs (%d nonzero): %s",
                    len(uids), len(nonzero_weights), nonzero_weights,
                )
            else:
                logger.warning(
                    "Setting all-zero weights for %d UIDs (no miner scored this round)",
                    len(uids),
                )
            self.subtensor.set_weights(
                netuid=self.netuid, wallet=self.wallet,
                uids=uids, weights=weights,
            )
            logger.info("Weights set successfully on chain")
        except Exception as e:
            logger.error("Failed to set weights: %s", e)

    async def run(self):
        """Main validator loop."""
        self.start_db_server()
        while True:
            try:
                if self.desearch_proxy:
                    self.desearch_proxy.reset_limits()
                await self.run_round()
            except Exception:
                logger.error("Round error:\n%s", traceback.format_exc())
            await asyncio.sleep(60)


def get_config() -> bt.Config:
    parser = argparse.ArgumentParser(description="Radar Validator")
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--task", type=str, default="ml_training")
    parser.add_argument("--db_dir", type=str, default="./experiments")
    parser.add_argument("--db_port", type=int, default=8080)
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    return bt.Config(parser)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = get_config()
    validator = Validator(config)
    asyncio.run(validator.run())


if __name__ == "__main__":
    main()
