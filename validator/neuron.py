"""Radar Validator Neuron — 3-phase validation pipeline.

Phase A: Collect architecture submissions from miner agents
Phase B: Dispatch training to miner trainers (cross-eval)
Phase C: Every validator independently evaluates every checkpoint (trust anchor)

Validators no longer run a local database. They proxy reads to the
centralized DB server and write results via DatabaseClient.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import logging
import os
import random
import threading
import traceback
from typing import Optional

import bittensor as bt
import uvicorn

from config import Config
from shared.challenge import (
    generate_challenge, round_start_block, current_phase, select_task,
)
from shared.commitment import read_miner_commitments
from shared.database import DataElement
from shared.db_client import DatabaseClient
from shared.protocol import Challenge, Proposal
from shared.provenance import detect_components
from shared.scoring import score_round, scores_to_weights, ema_update, compute_penalties
from shared.task import TaskSpec, load_task, load_enabled_tasks
from validator.collection import run_and_collect_agents
from validator.coordinator import (
    TrainingCoordinator, compute_assignments, compute_fallback,
)
from validator.db_proxy import (
    app as proxy_app, set_config as set_proxy_config, set_metagraph,
    set_hotkey_map, rotate_agent_token,
)
from validator.evaluator import evaluate_all_checkpoints
from validator.pod_manager import pre_validate_code
from validator.substrate_publisher import run_substrate_publish_step

logger = logging.getLogger(__name__)


def compute_live_validator_uids(
    metagraph,
    miner_uids: Optional[set[int]] = None,
    current_block: Optional[int] = None,
    stale_blocks: int = 600,
) -> list[int]:
    """Filter ``metagraph.validator_permit`` to live, non-miner validators.

    A UID counts as a live validator iff:
      * it has ``validator_permit=True`` on chain, AND
      * it is not running as a miner this round (no commitment), AND
      * its on-chain ``last_update`` is within ``stale_blocks`` of
        ``current_block``.

    ``miner_uids`` / ``current_block`` are optional; when absent, the
    corresponding check is skipped. A ``last_update`` of 0 is treated as
    "never updated" and is not filtered (bootstrap case).
    """
    permits = getattr(metagraph, "validator_permit", None)
    if permits is None:
        return []

    last_update = getattr(metagraph, "last_update", None)
    miners = miner_uids or set()
    result: list[int] = []
    for uid in range(metagraph.n):
        if uid >= len(permits) or not permits[uid]:
            continue
        if uid in miners:
            continue
        if (
            current_block is not None
            and last_update is not None
            and uid < len(last_update)
        ):
            try:
                last = int(last_update[uid])
            except (TypeError, ValueError):
                last = 0
            if last > 0 and current_block - last > stale_blocks:
                continue
        result.append(uid)
    return result


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

        # Tasks — load all enabled tasks for multi-task rounds
        self.tasks = load_enabled_tasks(Config.ENABLED_TASKS)
        task_path = getattr(config, "task", "")
        if task_path and task_path in self.tasks:
            self.task = self.tasks[task_path]
        else:
            self.task = next(iter(self.tasks.values()))

        # Database client — talks to centralized DB server
        if not Config.DB_API_URL:
            raise RuntimeError(
                "RADAR_DB_API_URL is not set. Validators must point to the "
                "centralized DB server (e.g. http://<db-host>:8090)."
            )
        self.db_client = DatabaseClient(
            db_url=Config.DB_API_URL, wallet=self.wallet,
            api_key=Config.DB_API_KEY,
        )

        # EMA scores per UID
        self.ema_scores: dict[int, float] = {}

        # Track last completed round to avoid re-running on restart
        self._last_completed_round: Optional[int] = None

        # Trainer reliability tracking (keyed by trainer UID)
        self.trainer_reliability: dict[int, float] = {}

        # Proxy port
        self.proxy_port = int(getattr(config, "db_port", Config.PROXY_PORT))

        # R2 audit log
        self.r2 = None
        if Config.R2_BUCKET:
            try:
                from shared.r2_audit import R2AuditLog
                self.r2 = R2AuditLog(bucket=Config.R2_BUCKET)
            except Exception as e:
                logger.warning("R2 audit log unavailable: %s", e)

        # GIFT-Eval R2 client (separate bucket for benchmark data)
        self.gift_r2 = None
        if self.r2 and Config.GIFT_EVAL_R2_BUCKET:
            try:
                from shared.r2_audit import R2AuditLog
                self.gift_r2 = R2AuditLog(bucket=Config.GIFT_EVAL_R2_BUCKET)
            except Exception as e:
                logger.warning("GIFT-Eval R2 unavailable: %s", e)

        # Pretrain R2 client (separate bucket for training data)
        self.pretrain_r2 = None
        if self.r2 and Config.PRETRAIN_R2_BUCKET:
            try:
                from shared.r2_audit import R2AuditLog
                self.pretrain_r2 = R2AuditLog(bucket=Config.PRETRAIN_R2_BUCKET)
                logger.info("Pretrain R2 client initialized for bucket=%s prefix=%s",
                            Config.PRETRAIN_R2_BUCKET, Config.PRETRAIN_R2_PREFIX)
            except Exception as e:
                logger.warning("Pretrain R2 unavailable: %s", e)

        # Cognition-wiki R2 client (separate bucket for per-task markdown corpus)
        self.cognition_wiki_r2 = None
        if Config.COGNITION_WIKI_R2_BUCKET:
            try:
                from shared.cognition_wiki import build_wiki_r2
                self.cognition_wiki_r2 = build_wiki_r2(
                    bucket=Config.COGNITION_WIKI_R2_BUCKET,
                )
                if self.cognition_wiki_r2:
                    logger.info(
                        "Cognition-wiki R2 client initialized for bucket=%s prefix=%s",
                        Config.COGNITION_WIKI_R2_BUCKET, Config.COGNITION_WIKI_R2_PREFIX,
                    )
            except Exception as e:
                logger.warning("Cognition-wiki R2 unavailable: %s", e)

        # Substrate publishing (best-effort, opt-in via Config.HIPPIUS_ENABLED).
        # The Hippius client wrapper from TEN-242 isn't shipped yet, so the
        # import is lazy and tolerant: with HIPPIUS_ENABLED=false (the default)
        # nothing is imported and `self.hippius` stays None. Operators who
        # opt in before the wrapper lands get a warning and the validator
        # continues to run normally — substrate publishing is never a
        # precondition for weight setting.
        self.hippius = self._init_hippius()

        # Dual-write artifact store (TEN-240 Phase 7). Constructed only
        # when an operator has explicitly opted in via
        # RADAR_DUAL_WRITE_ARTIFACTS — default deploy keeps the historical
        # R2-only path. When None, coordinator/etc. fall back to direct
        # R2 calls and behave exactly as before.
        self.artifact_store = None
        if Config.DUAL_WRITE_ARTIFACTS and (self.r2 or self.hippius):
            from shared.artifact_store import ArtifactStore
            self.artifact_store = ArtifactStore(
                r2=self.r2, hippius=self.hippius,
                dual_write=True,
                allow_fallback=Config.HIPPIUS_ARTIFACT_FALLBACK,
            )
            logger.info(
                "Artifact dual-write enabled (r2=%s, hippius=%s, fallback=%s)",
                bool(self.r2), bool(self.hippius),
                Config.HIPPIUS_ARTIFACT_FALLBACK,
            )

        # Training coordinator
        my_uid = self._my_uid()
        if self.r2:
            self.coordinator = TrainingCoordinator(
                wallet=self.wallet, metagraph=self.metagraph,
                r2=self.r2, my_uid=my_uid,
                artifact_store=self.artifact_store,
            )
        else:
            self.coordinator = None

        # Desearch and LLM proxies run on the DB server (subnet owner).
        # Validators just forward /desearch/* and /llm/* via db_proxy.py.
        # These flags control whether URLs are injected into the challenge.
        self.desearch_enabled = Config.DESEARCH_ENABLED
        self.llm_enabled = Config.LLM_ENABLED

        # Configure proxy
        set_proxy_config(
            db_api_url=Config.DB_API_URL,
            wallet=self.wallet,
            metagraph=self.metagraph,
            api_key=Config.DB_API_KEY,
        )

        logger.info(
            "Validator initialized. Task: %s, DB API: %s, R2: %s",
            list(self.tasks.keys()), Config.DB_API_URL,
            "enabled" if self.r2 else "disabled",
        )

    def _my_uid(self) -> int:
        """Get this validator's UID."""
        hotkey = self.wallet.hotkey.ss58_address
        hotkeys = self.metagraph.hotkeys
        if hotkeys is not None and hotkey in hotkeys:
            return hotkeys.index(hotkey)
        return -1

    def _init_hippius(self):
        """Construct the Hippius client when opted in; return None otherwise."""
        if not Config.HIPPIUS_ENABLED:
            return None
        try:
            from shared.hippius_client import HippiusClient  # type: ignore
        except ImportError:
            logger.warning(
                "HIPPIUS_ENABLED=true but shared.hippius_client is not "
                "available yet (TEN-242). Substrate publishing disabled."
            )
            return None
        try:
            return HippiusClient(
                ipfs_api_url=Config.HIPPIUS_IPFS_API_URL,
                hippius_key=Config.HIPPIUS_KEY,
                substrate_rpc=Config.HIPPIUS_SUBSTRATE_RPC,
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning("Hippius client init failed: %s", e)
            return None

    def _get_validator_uids(
        self,
        miner_uids: Optional[set[int]] = None,
        current_block: Optional[int] = None,
    ) -> list[int]:
        """Instance wrapper for :func:`compute_live_validator_uids`."""
        return compute_live_validator_uids(
            self.metagraph,
            miner_uids=miner_uids,
            current_block=current_block,
            stale_blocks=Config.VALIDATOR_STALE_BLOCKS,
        )

    def _proxy_base_url(self) -> str:
        """Return the externally-reachable base URL for the validator proxy.

        Uses VALIDATOR_EXTERNAL_URL if set, but replaces the port with the
        actual proxy port when they differ — the proxy hosts routes like
        ``/trainer/ready`` that don't exist on the database server.
        """
        from urllib.parse import urlparse, urlunparse

        ext = Config.VALIDATOR_EXTERNAL_URL.rstrip("/") if Config.VALIDATOR_EXTERNAL_URL else ""
        if not ext:
            return f"http://localhost:{self.proxy_port}"

        parsed = urlparse(ext)
        ext_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if ext_port != self.proxy_port:
            logger.warning(
                "VALIDATOR_EXTERNAL_URL port %d differs from proxy port %d — "
                "rewriting to use proxy port (the proxy hosts /trainer/ready)",
                ext_port, self.proxy_port,
            )
            # Replace port in netloc
            host = parsed.hostname or "localhost"
            new_netloc = f"{host}:{self.proxy_port}"
            parsed = parsed._replace(netloc=new_netloc)
            return urlunparse(parsed).rstrip("/")
        return ext

    def _build_dispatch_extras(self, challenge, task) -> dict:
        """Call the task's `dispatch.build_dispatch_extras` hook if present.

        Loads `{runner_dir}/dispatch.py` by file path and invokes
        `build_dispatch_extras` to produce any task-specific keys that
        get merged into the trainer payload. Missing module returns {}.
        File-path loading avoids depending on `runner` being on sys.path
        (it isn't pip-installed).
        """
        runner_dir = ""
        if isinstance(challenge.task, dict):
            runner_dir = challenge.task.get("runner_dir", "")
        if not runner_dir:
            return {}
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dispatch_path = os.path.join(project_root, runner_dir, "dispatch.py")
        if not os.path.isfile(dispatch_path):
            return {}
        mod_name = f"_radar_dispatch_{runner_dir.replace('/', '_').replace('.', '_')}"
        spec = importlib.util.spec_from_file_location(mod_name, dispatch_path)
        if spec is None or spec.loader is None:
            return {}
        dispatch_mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(dispatch_mod)
        except Exception as e:
            logger.warning(
                "Failed to load dispatch module for runner_dir=%s: %s",
                runner_dir, e,
            )
            return {}
        if not hasattr(dispatch_mod, "build_dispatch_extras"):
            return {}
        try:
            return dispatch_mod.build_dispatch_extras(
                task,
                gift_r2=self.gift_r2,
                pretrain_r2=self.pretrain_r2,
                seed=challenge.seed,
                shards_per_round=Config.PRETRAIN_SHARDS_PER_ROUND,
                r2_prefixes={
                    "gift": Config.GIFT_EVAL_R2_PREFIX,
                    "pretrain": Config.PRETRAIN_R2_PREFIX,
                },
            )
        except Exception as e:
            logger.warning(
                "build_dispatch_extras failed for runner_dir=%s: %s",
                runner_dir, e,
            )
            return {}

    def start_proxy_server(self):
        """Start the reverse proxy server in a background thread."""
        def _run():
            uvicorn.run(proxy_app, host="0.0.0.0", port=self.proxy_port, log_level="warning")
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        logger.info("Proxy server started on port %d", self.proxy_port)

    async def _wait_until_block(self, target_block: int):
        """Wait until the chain reaches target_block."""
        while True:
            current = self.subtensor.block
            if current >= target_block:
                return
            remaining = (target_block - current) * 12
            await asyncio.sleep(min(remaining, 30))

    async def _cache_training_metas(
        self, round_id: int, metas: dict[int, dict],
    ) -> None:
        """Push training_meta blobs to the centralized DB so the public
        dashboard can render loss curves without R2 access. Best-effort —
        errors are logged but never raised, since scoring must proceed even
        if the cache write fails."""
        if not metas:
            return
        hotkeys = self.metagraph.hotkeys if hasattr(self.metagraph, "hotkeys") else []
        cached = 0
        for uid, meta in metas.items():
            if not isinstance(meta, dict):
                continue
            hk = (
                meta.get("miner_hotkey")
                or (hotkeys[uid] if uid < len(hotkeys) else "")
            )
            if not hk:
                continue
            try:
                ok = await self.db_client.submit_training_meta(round_id, hk, meta)
                if ok:
                    cached += 1
            except Exception as e:
                logger.warning(
                    "training_meta cache failed: round=%d uid=%d err=%s",
                    round_id, uid, e,
                )
        if cached:
            logger.info(
                "Cached %d training_meta blobs for round %d", cached, round_id,
            )

    async def run_round(self):
        """Execute one full 3-phase round."""
        # ── SETUP ──────────────────────────────────────────────
        self.metagraph.sync()
        set_metagraph(self.metagraph)
        hotkeys = self.metagraph.hotkeys or []
        hotkey_map = {
            hotkeys[uid]: uid
            for uid in range(self.metagraph.n)
            if uid < len(hotkeys)
        }
        set_hotkey_map(hotkey_map)
        current_block = self.subtensor.block
        round_start = round_start_block(current_block, Config.ROUND_INTERVAL_BLOCKS)
        block_hash = hashlib.sha256(str(round_start).encode()).hexdigest()

        # Select task for this round (deterministic from block hash)
        task_name = select_task(block_hash, list(self.tasks.keys()))
        round_task = self.tasks[task_name]
        challenge = generate_challenge(
            block_hash,
            round_task.to_dict(),
            default_agent_seconds=Config.AGENT_TIMEOUT,
        )

        # Fetch feasible frontier from centralized DB
        pareto_elements = await self.db_client.get_pareto_elements(task=task_name)
        feasible_frontier = []
        for elem_dict in pareto_elements:
            objectives = elem_dict.get("results", {})
            flops = objectives.get("flops_equivalent_size", 0)
            if challenge.min_flops_equivalent <= flops <= challenge.max_flops_equivalent:
                diff_data = await self.db_client.get_diff(elem_dict.get("index", -1))
                parent_diff = diff_data.get("diff", "") if diff_data else ""
                feasible_frontier.append({
                    "code": elem_dict.get("code", ""),
                    "metric": elem_dict.get("results", {}).get("metric"),
                    "objectives": objectives,
                    "parent_diff": parent_diff,
                    "motivation": elem_dict.get("motivation", ""),
                    "task": task_name,
                })

        challenge.feasible_frontier = feasible_frontier

        proxy_base = self._proxy_base_url()
        challenge.db_url = proxy_base
        challenge.desearch_url = (
            f"{proxy_base}/desearch" if self.desearch_enabled else ""
        )
        challenge.llm_url = (
            f"{proxy_base}/llm" if self.llm_enabled else ""
        )

        # Rotate agent token for this round — agents use it to auth proxy requests
        challenge.agent_token = rotate_agent_token()

        # Per-task cognition wiki tarball (skipped silently if not configured
        # or if the task has no wiki uploaded).
        if self.cognition_wiki_r2 is not None:
            try:
                from shared.cognition_wiki import presign_wiki_url
                challenge.cognition_wiki_url = presign_wiki_url(
                    self.cognition_wiki_r2,
                    task_name,
                    prefix=Config.COGNITION_WIKI_R2_PREFIX,
                    ttl=Config.COGNITION_WIKI_TTL,
                )
            except Exception as e:
                logger.warning("Cognition-wiki URL generation failed: %s", e)
                challenge.cognition_wiki_url = ""

        logger.info(
            "Round %d: size bucket [%d, %d], frontier points in range: %d",
            challenge.round_id, challenge.min_flops_equivalent,
            challenge.max_flops_equivalent, len(feasible_frontier),
        )

        # ── PHASE A: RUN AGENTS + COLLECT ────────────────────
        commitments = read_miner_commitments(self.subtensor, self.netuid, self.metagraph)
        try:
            current_block = self.subtensor.get_current_block()
        except Exception as e:
            logger.warning("Failed to read current block for liveness filter: %s", e)
            current_block = None
        validator_uids = self._get_validator_uids(
            miner_uids=set(commitments.keys()),
            current_block=current_block,
        )
        logger.info(
            "Live validators this round: %s (filtered miners + stale last_update)",
            validator_uids,
        )

        if not self.r2:
            logger.warning("No R2 configured — skipping Phase A/B/C")
            return

        # ── PREPARE TRAINERS (warm-standby) ──────────────────
        if self.coordinator:
            prepare_coro = self.coordinator.prepare_trainers(
                challenge, commitments,
                db_url=proxy_base,
            )
        else:
            async def _no_endpoints():
                return {}
            prepare_coro = _no_endpoints()

        collect_coro = run_and_collect_agents(
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

        dynamic_endpoints, (submissions, agent_meta) = await asyncio.gather(
            prepare_coro, collect_coro,
        )

        # Pre-filter: syntax check
        filtered: dict[int, Proposal] = {}
        for uid, proposal in submissions.items():
            ok, reason = pre_validate_code(proposal.code)
            if not ok:
                logger.warning(
                    "UID %d proposal rejected (pre_validate_code): %s | "
                    "name=%r code_len=%d",
                    uid, reason, proposal.name[:80], len(proposal.code),
                )
                continue
            filtered[uid] = proposal

        rejected_count = len(submissions) - len(filtered)
        logger.info(
            "Phase A: %d proposals, %d after filter (%d rejected)",
            len(submissions), len(filtered), rejected_count,
        )

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
        dispatch_validators = list(validator_uids)
        if my_uid >= 0 and my_uid not in dispatch_validators:
            logger.info(
                "My UID %d not in validator_uids %s — adding self as safety-net dispatcher",
                my_uid, dispatch_validators,
            )
            dispatch_validators.append(my_uid)

        all_jobs = compute_assignments(
            block_hash, filtered,
            miner_uids=list(commitments.keys()),
            validator_uids=dispatch_validators,
            round_id=challenge.round_id,
        )
        my_jobs = [j for j in all_jobs if j.dispatcher == my_uid]

        if not my_jobs and all_jobs:
            logger.warning(
                "Round-robin assigned 0 of %d jobs to my_uid=%d — "
                "dispatching all jobs (other dispatchers may be offline)",
                len(all_jobs), my_uid,
            )
            my_jobs = list(all_jobs)

        trainer_endpoints = dynamic_endpoints
        logger.info(
            "Phase B: %d total jobs, %d mine (my_uid=%d, validators=%s), %d trainer endpoints",
            len(all_jobs), len(my_jobs), my_uid, dispatch_validators, len(trainer_endpoints),
        )
        if not trainer_endpoints:
            logger.warning(
                "Phase B: no trainer endpoints! Miners must set --listener_port "
                "or trainer pods failed to deploy. Training will fail for all jobs."
            )

        # Task-specific dispatch extras (e.g. GIFT-Eval + pretrain shard URLs
        # for ts_forecasting). Each task's runner_dir may ship a `dispatch`
        # module exposing `build_dispatch_extras(task, ...)`. Missing module
        # is fine — the extras dict stays empty.
        extras = self._build_dispatch_extras(challenge, round_task)

        my_results = await self.coordinator.dispatch_jobs(
            my_jobs, challenge, filtered, trainer_endpoints,
            commitments=commitments,
            extras=extras,
        )
        _OK_STATUSES = ("success", "accepted", "already_running")
        succeeded = sum(1 for r in my_results if r.status == "success")
        accepted = sum(1 for r in my_results if r.status in ("accepted", "already_running"))
        failed = sum(1 for r in my_results if r.status not in _OK_STATUSES)
        logger.info(
            "Phase B dispatch: %d succeeded, %d accepted, %d failed",
            succeeded, accepted, failed,
        )
        for r in my_results:
            if r.status == "success":
                logger.info(
                    "  Job arch=%d trainer=%d: success (%.1fs, %d flops)",
                    r.arch_owner, r.trainer_uid,
                    r.training_time_seconds, r.flops_equivalent_size,
                )
            elif r.status in ("accepted", "already_running"):
                logger.info(
                    "  Job arch=%d trainer=%d: %s (awaiting R2 checkpoint)",
                    r.arch_owner, r.trainer_uid, r.status,
                )
            else:
                logger.warning(
                    "  Job arch=%d trainer=%d: %s — %s",
                    r.arch_owner, r.trainer_uid, r.status, r.error,
                )

        await self.coordinator.write_dispatch_record(challenge.round_id, my_results)

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

        # ── RELEASE TRAINERS ─────────────────────────────────
        if self.coordinator:
            collected_uids = set(training_metas.keys())
            all_prepared = set(dynamic_endpoints.keys())
            timed_out = all_prepared - collected_uids
            if timed_out:
                logger.info(
                    "Timed-out trainers (no checkpoint): %s", sorted(timed_out),
                )
            all_release_uids = all_prepared | collected_uids
            await self.coordinator.release_trainers(
                challenge.round_id, commitments, all_release_uids,
            )

        # Fallback for missing validators
        missing_trainers = [uid for uid in filtered if uid not in training_metas]
        if missing_trainers:
            logger.info("Missing checkpoints from %d miners: %s", len(missing_trainers), missing_trainers)

        # Cache training metas in Postgres so the public dashboard can render
        # loss curves without R2 access. Best-effort: a failed cache write
        # must not block scoring.
        await self._cache_training_metas(challenge.round_id, training_metas)

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
                    fb_extras = self._build_dispatch_extras(challenge, round_task)
                    fb_results = await self.coordinator.dispatch_jobs(
                        my_fallback, challenge, filtered, trainer_endpoints,
                        commitments=commitments,
                        extras=fb_extras,
                    )
                    await self.coordinator.write_dispatch_record(
                        challenge.round_id, fb_results,
                    )

                await self._wait_until_block(
                    round_start + Config.SUBMISSION_WINDOW_BLOCKS
                    + Config.TRAINING_WINDOW_BLOCKS
                    + Config.FALLBACK_WINDOW_BLOCKS
                )

                fb_metas = await self.coordinator.wait_for_checkpoints(
                    challenge.round_id,
                    expected_miners=missing_trainers,
                    timeout=Config.FALLBACK_WINDOW_BLOCKS * 12,
                )
                training_metas.update(fb_metas)
                await self._cache_training_metas(challenge.round_id, fb_metas)

        # ── Pre-cache GIFT-Eval data for Phase C ────────────────
        if self.gift_r2:
            try:
                from shared.gift_eval import GiftEvalBenchmark
                gift = GiftEvalBenchmark(
                    r2=self.gift_r2,
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
            task=round_task,
        )

        logger.info("Phase C: evaluated %d checkpoints", len(eval_results))
        primary_obj = round_task.primary_objective
        primary_name = primary_obj.name if primary_obj else "metric"
        primary_default = primary_obj.default if primary_obj else float("inf")
        for uid, metrics in eval_results.items():
            logger.info(
                "  UID %d: %s=%.6f flops=%d gate=%s error=%s",
                uid, primary_name,
                metrics.get(primary_name, primary_default),
                metrics.get("flops_equivalent_size", 0),
                metrics.get("passed_size_gate", False),
                metrics.get("error", ""),
            )

        # ── SCORING ───────────────────────────────────────────
        logger.info(
            "Phase D (scoring): round %d, %d eval results, frontier size %d, bucket=[%d, %d]",
            challenge.round_id, len(eval_results), len(feasible_frontier),
            challenge.min_flops_equivalent, challenge.max_flops_equivalent,
        )
        penalties = compute_penalties(training_metas, eval_results)
        nonzero_penalties = {uid: p for uid, p in penalties.items() if p > 0}
        if nonzero_penalties:
            logger.info(
                "Penalties applied to %d UIDs: %s",
                len(nonzero_penalties),
                {uid: f"{p:.2f}" for uid, p in sorted(nonzero_penalties.items())},
            )
        else:
            logger.info("Penalties: none")

        # Track trainer reliability
        fallback_uids = (
            self.coordinator._fallback_uids.get(challenge.round_id, set())
            if self.coordinator else set()
        )
        for uid, meta in training_metas.items():
            trainer_uid = meta.get("trainer_uid", uid)
            if trainer_uid in fallback_uids:
                score = 0.0
            else:
                status = meta.get("status", "")
                score = 1.0 if status == "success" else 0.0
            alpha = Config.EMA_ALPHA
            prev = self.trainer_reliability.get(trainer_uid, 1.0)
            self.trainer_reliability[trainer_uid] = alpha * score + (1 - alpha) * prev

        # Record frontier context via DB client
        for entry in challenge.feasible_frontier:
            entry_code = entry.get("code", "")
            if entry_code:
                # Look up experiment by querying DB
                pareto_data = await self.db_client.get_pareto_elements(task=task_name)
                for pe in pareto_data:
                    if pe.get("code", "") == entry_code:
                        await self.db_client.record_round_context(
                            challenge.round_id, pe.get("index", -1), "frontier",
                        )
                        break

        # Write experiments to centralized DB
        # Build a local ParetoFront for scoring purposes
        from shared.pareto import ParetoFront
        ts = round_task
        objective_fn = lambda elem, _ts=ts: _ts.objective_vector(elem.objectives)
        pareto = ParetoFront(max_size=50, objective_fn=objective_fn)

        # Populate pareto from fetched elements for scoring
        for elem_dict in pareto_elements:
            try:
                de = DataElement.from_dict({
                    "index": elem_dict.get("index", -1),
                    "code": elem_dict.get("code", ""),
                    "metric": elem_dict.get("results", {}).get("metric"),
                    "success": elem_dict.get("results", {}).get("success", True),
                    "objectives": {k: v for k, v in elem_dict.get("results", {}).items()
                                   if k not in ("success", "metric", "loss_curve")},
                    "task": elem_dict.get("task", task_name),
                })
                pareto.update(de)
            except Exception:
                pass

        gate_passed = 0
        gate_failed = 0
        for uid, metrics in eval_results.items():
            if metrics.get("passed_size_gate"):
                gate_passed += 1
                proposal = filtered.get(uid, Proposal())

                miner_hk = self.metagraph.hotkeys[uid] if uid < len(self.metagraph.hotkeys) else ""
                meta = agent_meta.get(uid, {})
                element = DataElement(
                    name=f"round_{challenge.round_id}_miner_{uid}",
                    code=proposal.code,
                    metric=metrics.get(primary_name),
                    success=True,
                    objectives=metrics,
                    miner_uid=uid,
                    miner_hotkey=miner_hk,
                    motivation=proposal.motivation,
                    trace=meta.get("agent_log", ""),
                    reasoning=meta.get("reasoning", "") or proposal.reasoning,
                    tool_calls=meta.get("tool_calls", []) or proposal.tool_calls,
                    agent_behavior=meta.get("agent_behavior", {}),
                    task=task_name,
                    round_id=challenge.round_id,
                )

                # Write to centralized DB. When this validator published
                # a Phase C bundle this round, attach the CID so the row
                # carries an audit-trail entry pointing at the bundle.
                new_idx = await self.db_client.add_experiment(
                    element.to_dict(),
                    substrate_cid=substrate_cids.get(uid, ""),
                    validator_hotkey=self.wallet.hotkey.ss58_address,
                )
                if new_idx is not None:
                    element.index = new_idx
                pareto.update(element)

                # Detect and store architectural components
                components = detect_components(proposal.code)
                if components and element.index >= 0:
                    await self.db_client.record_components(element.index, components)
            else:
                gate_failed += 1
                logger.warning(
                    "UID %d failed size gate: flops=%d range=[%d, %d] error=%s",
                    uid, metrics.get("flops_equivalent_size", 0),
                    challenge.min_flops_equivalent, challenge.max_flops_equivalent,
                    metrics.get("error", ""),
                )

        logger.info(
            "Experiment recording: %d passed size gate, %d failed, Pareto size: %d",
            gate_passed, gate_failed, pareto.size,
        )

        round_scores = score_round(
            eval_results, challenge, pareto, round_task.objectives, penalties,
            training_metas=training_metas,
        )

        # Best-effort: publish signed Phase C records to Hippius/substrate.
        # Returns {miner_uid: bundle_cid} so Phase 5 can thread CIDs into
        # the per-miner DB writes. Empty dict when disabled, no client, or
        # publish failed — never raises.
        substrate_cids = await run_substrate_publish_step(
            hippius=self.hippius, wallet=self.wallet,
            challenge=challenge, eval_results=eval_results,
            training_metas=training_metas, commitments=commitments,
            metagraph=self.metagraph, my_uid=self._my_uid(),
            current_block=current_block, task_name=task_name,
            block_hash=block_hash, netuid=int(self.netuid),
        )
        if substrate_cids:
            logger.info(
                "Substrate published: round_id=%d cid=%s miners=%d",
                challenge.round_id, next(iter(substrate_cids.values())),
                len(substrate_cids),
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
        nonzero_ema = {uid: s for uid, s in self.ema_scores.items() if s > 0}
        logger.info(
            "EMA updated (alpha=%.2f): %d UIDs nonzero — %s",
            Config.EMA_ALPHA, len(nonzero_ema),
            {uid: f"{s:.4f}" for uid, s in sorted(nonzero_ema.items())} if nonzero_ema else "all zero",
        )
        self._set_weights()

        # Write frontier to centralized DB + R2
        await self.db_client.update_frontier(
            [c.element.to_dict() for c in pareto.candidates],
            task=task_name,
        )
        if self.coordinator:
            await self.coordinator.write_frontier(
                [c.element.to_dict() for c in pareto.candidates],
                task_name=task_name,
            )

        nonzero_scored = sum(1 for s in round_scores.values() if s > 0)
        logger.info(
            "Round %d complete. Pareto: %d, scored: %d (%d nonzero)",
            challenge.round_id, pareto.size,
            len(round_scores), nonzero_scored,
        )

    def _set_weights(self):
        """Set weights on chain — softmax on EMA scores."""
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
        self.start_proxy_server()
        while True:
            try:
                current_block = self.subtensor.block
                round_start = round_start_block(current_block, Config.ROUND_INTERVAL_BLOCKS)
                phase = current_phase(
                    current_block, round_start,
                    submission_window=Config.SUBMISSION_WINDOW_BLOCKS,
                    training_window=Config.TRAINING_WINDOW_BLOCKS,
                    eval_window=Config.EVAL_WINDOW_BLOCKS,
                    scoring_window=Config.FALLBACK_WINDOW_BLOCKS,
                )
                if phase != "submission":
                    next_round = round_start + Config.ROUND_INTERVAL_BLOCKS
                    wait_blocks = next_round - current_block
                    wait_secs = wait_blocks * 12
                    logger.info(
                        "Current phase is '%s' (block %d, round started at %d) "
                        "— waiting %d blocks (~%ds) for next round at block %d",
                        phase, current_block, round_start,
                        wait_blocks, wait_secs, next_round,
                    )
                    await self._wait_until_block(next_round)
                    continue
                if self._last_completed_round == round_start:
                    next_round = round_start + Config.ROUND_INTERVAL_BLOCKS
                    wait_blocks = next_round - current_block
                    logger.info(
                        "Round %d already processed — waiting %d blocks for next round at %d",
                        round_start, wait_blocks, next_round,
                    )
                    await self._wait_until_block(next_round)
                    continue
                await self.run_round()
                self._last_completed_round = round_start
            except Exception:
                logger.error("Round error:\n%s", traceback.format_exc())
            await asyncio.sleep(60)


def get_config() -> bt.Config:
    parser = argparse.ArgumentParser(description="Radar Validator")
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--task", type=str, default="ml_training")
    parser.add_argument("--db_dir", type=str, default="./experiments")
    parser.add_argument("--db_port", type=int, default=Config.PROXY_PORT)
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    return bt.Config(parser)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    config = get_config()
    validator = Validator(config)
    asyncio.run(validator.run())


if __name__ == "__main__":
    main()
