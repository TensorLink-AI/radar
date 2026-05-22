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
import threading
import time
import traceback
from typing import Optional

import uvicorn

from config import Config
from shared.challenge import (
    generate_challenge, round_start_block, current_phase, select_task,
)
from shared.commitment import ImageCommitment
from shared.database import DataElement
from shared.db_client import DatabaseClient
from shared.protocol import Proposal
from shared.provenance import detect_components
from shared.scoring import (
    apply_round_metadata, compute_penalties, ema_update, score_round,
)
from shared.task import load_enabled_tasks
from validator.collection import run_and_collect_agents
from validator.coordinator import (
    TrainingCoordinator, compute_assignments, compute_fallback,
)
from validator.db_proxy import (
    app as proxy_app, set_config as set_proxy_config, rotate_agent_token,
)
from validator.evaluator import evaluate_all_checkpoints
from validator.pod_manager import pre_validate_code, reap_orphan_agent_pods

logger = logging.getLogger(__name__)


def get_my_assignments(
    all_uids: list[int],
    validator_uids: list[int],
    my_uid: int,
    seed: int,
) -> list[int]:
    """Single-validator deployment: this validator owns every UID.

    Kept as a stub so collection.py's pass-through still works.
    """
    return list(all_uids)


class Validator:
    """Radar validator — 3-phase pipeline (off-chain, single-validator)."""

    def __init__(self, config):
        self.config = config

        # Fail-fast on Targon misconfig — operator gets a clear error at
        # startup instead of a stack trace mid-round when verify fires.
        if Config.HOSTING_BACKEND == "targon" and not os.environ.get("TARGON_API_KEY"):
            raise RuntimeError(
                "RADAR_HOSTING_BACKEND=targon but TARGON_API_KEY is not set. "
                "Validators need a Targon API key to verify image digests + attestations. "
                "See https://docs.targon.com."
            )
        if Config.HOSTING_BACKEND == "targon" and not Config.OFFICIAL_TRAINING_IMAGE_DIGEST:
            raise RuntimeError(
                "RADAR_HOSTING_BACKEND=targon but OFFICIAL_TRAINING_IMAGE_DIGEST is empty. "
                "Without a pinned digest the verify chain becomes a no-op — refusing to start."
            )

        if not Config.SERVICE_KEY:
            raise RuntimeError(
                "RADAR_SERVICE_KEY must be set (shared HMAC secret for "
                "service-to-service auth)."
            )

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
            db_url=Config.DB_API_URL,
            service_secret=Config.SERVICE_KEY.encode(),
            key_id=Config.SERVICE_KEY_ID,
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

        # Hippius client (best-effort, opt-in via Config.HIPPIUS_ENABLED).
        # Used only for the optional artifact dual-write path.
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
        if self.r2:
            self.coordinator = TrainingCoordinator(
                r2=self.r2, my_uid=-1,
                artifact_store=self.artifact_store,
            )
        else:
            self.coordinator = None

        # Desearch and LLM proxies run on the DB server. Validators just
        # forward /desearch/* and /llm/* via db_proxy.py.
        self.desearch_enabled = Config.DESEARCH_ENABLED
        self.llm_enabled = Config.LLM_ENABLED

        # Configure proxy
        set_proxy_config(
            db_api_url=Config.DB_API_URL,
            api_key=Config.DB_API_KEY,
        )

        logger.info(
            "Validator initialized. Task: %s, DB API: %s, R2: %s",
            list(self.tasks.keys()), Config.DB_API_URL,
            "enabled" if self.r2 else "disabled",
        )

    def _init_hippius(self):
        """Construct the Hippius client when opted in; return None otherwise."""
        if not Config.HIPPIUS_ENABLED:
            return None
        try:
            from shared.hippius_client import HippiusClient  # type: ignore
        except ImportError:
            logger.warning(
                "HIPPIUS_ENABLED=true but shared.hippius_client is not "
                "available — artifact dual-write disabled."
            )
            return None
        try:
            return HippiusClient(
                ipfs_api_url=Config.HIPPIUS_IPFS_API_URL,
                hippius_key=Config.HIPPIUS_KEY,
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning("Hippius client init failed: %s", e)
            return None

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
        """Wait until the wall-clock-derived block counter reaches target_block.

        Off-chain deploys use ``time.time() // 12`` as a stand-in for block
        height so timing stays compatible with the per-phase block-window
        constants in ``config.py``.
        """
        while True:
            current = int(time.time() // 12)
            if current >= target_block:
                return
            remaining = (target_block - current) * 12
            await asyncio.sleep(min(remaining, 30))

    async def _fetch_active_commitments(self) -> dict[int, ImageCommitment]:
        """Build {uid: ImageCommitment} from the DB's active miner list.

        The off-chain deploy has no on-chain ImageCommitment; miners
        self-register their listener URL via POST /miners/me/listener
        after submit_agent_code. We synthesise an ImageCommitment per
        active miner so the rest of Phase A/B/C (which is still typed
        around commitments) keeps working.

        UIDs:
          * Prefer agent_submissions.miner_uid when the miner posted
            one via X-Miner-UID (lets operators keep stable, small ids).
          * Otherwise derive a stable synthetic int from the hotkey.
            Collisions are vanishingly unlikely in practice but we
            re-roll on conflict so two miners never share a uid.
        """
        if self.db_client is None:
            logger.warning("No db_client configured — cannot fetch active miners")
            return {}

        try:
            rows = await self.db_client.get_active_miners()
        except Exception as e:
            logger.warning("get_active_miners failed: %s", e)
            return {}

        db_url = getattr(self.db_client, "db_url", "?")
        if not rows:
            logger.warning(
                "get_active_miners returned 0 rows from db=%s — "
                "either no miner heartbeated within the 6h window, the GET "
                "failed silently (check DatabaseClient GET warnings), or "
                "the validator is pointed at a different DB than the miners.",
                db_url,
            )
        else:
            logger.info(
                "get_active_miners returned %d rows from db=%s (hotkeys=%s)",
                len(rows), db_url,
                [(r.get("hotkey") or "")[:16] for r in rows[:8]],
            )

        out: dict[int, ImageCommitment] = {}
        skipped_no_hotkey = 0
        skipped_no_listener = 0
        for row in rows:
            hotkey = row.get("hotkey") or ""
            listener_url = row.get("listener_url") or ""
            if not hotkey:
                skipped_no_hotkey += 1
                continue
            if not listener_url:
                skipped_no_listener += 1
                continue
            preferred = int(row.get("miner_uid", -1))
            uid = preferred if preferred >= 0 and preferred not in out else -1
            if uid < 0:
                base = int.from_bytes(
                    hashlib.sha256(hotkey.encode()).digest()[:4],
                    "big",
                ) % (2**31 - 1)
                uid = base
                # Re-roll on collision — extremely rare but cheap.
                bump = 0
                while uid in out and bump < 16:
                    bump += 1
                    uid = (base + bump) % (2**31 - 1)
                if uid in out:
                    logger.warning(
                        "Skipping miner %s — could not assign unique synthetic uid",
                        hotkey[:16],
                    )
                    continue
            out[uid] = ImageCommitment(
                code_hash=row.get("code_hash", "") or "",
                listener_url=listener_url,
                miner_uid=uid,
                hotkey=hotkey,
            )
        if rows and not out:
            logger.warning(
                "get_active_miners returned %d rows but 0 survived filter "
                "(skipped_no_hotkey=%d, skipped_no_listener=%d)",
                len(rows), skipped_no_hotkey, skipped_no_listener,
            )
        return out

    async def _cache_training_metas(
        self, round_id: int, metas: dict[int, dict],
    ) -> None:
        """Push training_meta blobs to the centralized DB so the public
        dashboard can render loss curves without R2 access. Best-effort —
        errors are logged but never raised, since scoring must proceed even
        if the cache write fails."""
        if not metas:
            return
        cached = 0
        for uid, meta in metas.items():
            if not isinstance(meta, dict):
                continue
            hk = meta.get("miner_hotkey", "")
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

    async def _publish_submission_reveal(
        self,
        round_id: int,
        uid_to_sid: dict[int, str],
        hotkeys: list[str],
    ) -> None:
        """Push the per-round submission_id → (miner_hotkey, miner_uid) map
        to the centralised DB after Phase C closes.

        Best-effort — if the cache write fails the dashboard simply falls
        back to the historic ``miner_{hotkey}/`` paths it knows for older
        rounds. Scoring never depends on this call. Only the dispatching
        validator owns the truth for its submission_ids; idempotent
        upsert in the DB merges across validators safely.
        """
        if not uid_to_sid:
            return
        entries = []
        for uid, sid in uid_to_sid.items():
            if not sid:
                continue
            hk = hotkeys[uid] if 0 <= uid < len(hotkeys) else ""
            if not hk:
                continue
            entries.append({
                "submission_id": sid,
                "miner_hotkey": hk,
                "miner_uid": int(uid),
            })
        if not entries:
            return
        try:
            ok = await self.db_client.submit_submission_reveal(
                round_id=round_id, entries=entries,
            )
            if ok:
                logger.info(
                    "Published submission reveal: round=%d entries=%d",
                    round_id, len(entries),
                )
            else:
                logger.warning(
                    "Submission reveal returned non-OK: round=%d", round_id,
                )
        except Exception as e:
            logger.warning(
                "Submission reveal failed: round=%d err=%s", round_id, e,
            )

    async def run_round(self):
        """Execute one full 3-phase round."""
        # ── SETUP ──────────────────────────────────────────────
        # No chain — round_start is a wall-clock-derived integer so the
        # deterministic seed stays stable for ~12s like a block.
        current_block = int(time.time() // 12)
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
        # Off-chain deploy: miner identity comes from operator-CLI tokens,
        # not on-chain ImageCommitments. We rebuild the equivalent record
        # by asking the DB for miners that have recently registered a
        # listener URL (POST /miners/me/listener). Without this lookup
        # `commitments` would be empty and the round would dead-end at
        # "0 miners with listener_urls".
        commitments = await self._fetch_active_commitments()
        validator_uids: list[int] = []
        logger.info(
            "Active miner registry: %d miners with live listener URLs",
            len(commitments),
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
            challenge_json=challenge.to_json(),
            round_id=challenge.round_id,
            seed=challenge.seed,
            r2=self.r2,
            my_uid=-1,
            validator_uids=validator_uids,
            commitments=commitments,
            get_my_assignments_fn=get_my_assignments,
            db_client=self.db_client,
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

        # Single-validator deploy: we own every dispatch.
        my_uid = -1
        dispatch_validators = [my_uid]

        all_jobs = compute_assignments(
            block_hash, filtered,
            miner_uids=list(filtered.keys()),
            validator_uids=dispatch_validators,
            round_id=challenge.round_id,
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
        # Targon: schedule mid-round reverify of every accepted CVM at
        # deterministic offsets within the training window. No-op for
        # Basilica deployments. Tasks self-clean; release_trainers
        # cancels stragglers.
        try:
            window_s = float(Config.TRAINING_WINDOW_BLOCKS * 12)
            self.coordinator.schedule_mid_round_reverify(
                challenge.round_id,
                block_hash=block_hash,
                training_window_seconds=window_s,
            )
        except Exception as e:
            logger.warning("schedule_mid_round_reverify failed (round %d): %s", challenge.round_id, e)
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

        # Build the uid → submission_id map. Start with my own jobs (in-memory
        # source of truth for the submission_ids I just minted), then merge in
        # other validators' submission_ids after Phase B closes.
        uid_to_sid: dict[int, str] = {
            r.arch_owner: r.submission_id
            for r in my_results if r.submission_id
        }

        if Config.SKIP_TRAINING_WAIT:
            logger.info("SKIP_TRAINING_WAIT enabled — skipping block wait after dispatch")
        else:
            await self._wait_until_block(
                round_start + Config.SUBMISSION_WINDOW_BLOCKS + Config.TRAINING_WINDOW_BLOCKS
            )

        # Publish the dispatch record AFTER the Phase B training window closes.
        # The record carries the submission_id ↔ arch_owner mapping that lets
        # other validators resolve checkpoints in Phase C. Hippius is public
        # by design (decentralized object store), so writing the record any
        # earlier would let a trainer-host fetch it mid-training and learn
        # whose architecture they're running — defeating the anonymisation.
        await self.coordinator.write_dispatch_record(challenge.round_id, my_results)

        # Pull other validators' dispatch records so we can poll for their
        # checkpoints too (Phase C requires every validator to evaluate every
        # checkpoint, so we need the union of submission_ids).
        try:
            other_sids = await self.coordinator.collect_submission_map(challenge.round_id)
            for uid, sid in other_sids.items():
                uid_to_sid.setdefault(uid, sid)
        except Exception as e:
            logger.warning("Failed to collect cross-validator submission map: %s", e)

        # ── Checkpoint collection ─────────────────────────────
        training_metas = await self.coordinator.wait_for_checkpoints(
            challenge.round_id,
            expected_miners=list(filtered.keys()),
            uid_to_submission_id=uid_to_sid,
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
                fb_results: list = []
                if my_fallback:
                    fb_extras = self._build_dispatch_extras(challenge, round_task)
                    fb_results = await self.coordinator.dispatch_jobs(
                        my_fallback, challenge, filtered, trainer_endpoints,
                        commitments=commitments,
                        extras=fb_extras,
                    )
                    for r in fb_results:
                        if r.submission_id:
                            uid_to_sid[r.arch_owner] = r.submission_id

                await self._wait_until_block(
                    round_start + Config.SUBMISSION_WINDOW_BLOCKS
                    + Config.TRAINING_WINDOW_BLOCKS
                    + Config.FALLBACK_WINDOW_BLOCKS
                )

                # Publish fallback dispatch record AFTER the fallback training
                # window closes — same anonymity reason as the primary record.
                if fb_results:
                    await self.coordinator.write_dispatch_record(
                        challenge.round_id, fb_results,
                    )

                # Re-pull other validators' fallback dispatch records too.
                try:
                    other_sids = await self.coordinator.collect_submission_map(
                        challenge.round_id,
                    )
                    for uid, sid in other_sids.items():
                        uid_to_sid.setdefault(uid, sid)
                except Exception as e:
                    logger.warning(
                        "Failed to collect fallback submission map: %s", e,
                    )

                fb_metas = await self.coordinator.wait_for_checkpoints(
                    challenge.round_id,
                    expected_miners=missing_trainers,
                    uid_to_submission_id=uid_to_sid,
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

        # ── REVEAL: publish submission_id → miner_hotkey mapping ──
        # Phase B is over and Phase C has read every checkpoint, so
        # revealing the mapping no longer lets a trainer-host selectively
        # sandbag a rival. The dashboard uses this map to render
        # per-miner training history for past rounds.
        await self._publish_submission_reveal(
            round_id=challenge.round_id,
            uid_to_sid=uid_to_sid,
            hotkeys=[],
        )
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

                meta = agent_meta.get(uid, {})
                miner_hk = meta.get("miner_hotkey", "") if isinstance(meta, dict) else ""
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
                    prompt_id=(
                        meta.get("prompt_id", "")
                        or getattr(proposal, "prompt_id", "")
                        or ""
                    ),
                )

                new_idx = await self.db_client.add_experiment(
                    element.to_dict(),
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

        # Apply Targon migration's hybrid-fallback policy: targon_unavailable
        # gets a 0.5× multiplier (default), compromised UIDs zeroed entirely.
        # No-op when round_metadata is empty (Basilica deployments, or
        # Targon round with no flags fired).
        round_metadata = (
            self.coordinator.round_metadata(challenge.round_id)
            if hasattr(self.coordinator, "round_metadata") else None
        )
        if round_metadata and (round_metadata.get("targon_unavailable") or round_metadata.get("compromised")):
            logger.info(
                "Applying Targon round metadata (round %d): unavailable=%s compromised=%s",
                challenge.round_id,
                round_metadata.get("targon_unavailable"),
                round_metadata.get("compromised"),
            )
            round_scores = apply_round_metadata(round_scores, round_metadata)

        if round_scores:
            nonzero = {uid: s for uid, s in round_scores.items() if s > 0}
            logger.info(
                "Round scores: %d total, %d nonzero — %s",
                len(round_scores), len(nonzero),
                {uid: f"{s:.4f}" for uid, s in sorted(nonzero.items())} if nonzero else "all zero",
            )
        else:
            logger.warning("Round scores empty — no eval results passed size gate")

        # EMA on raw scores. No on-chain weight set — the EMA dict is the
        # source of truth for the dashboard and operator-CLI reporting.
        all_uids = sorted(set(self.ema_scores) | set(round_scores))
        self.ema_scores = ema_update(self.ema_scores, round_scores, all_uids, Config.EMA_ALPHA)
        nonzero_ema = {uid: s for uid, s in self.ema_scores.items() if s > 0}
        logger.info(
            "EMA updated (alpha=%.2f): %d UIDs nonzero — %s",
            Config.EMA_ALPHA, len(nonzero_ema),
            {uid: f"{s:.4f}" for uid, s in sorted(nonzero_ema.items())} if nonzero_ema else "all zero",
        )

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

    async def run(self):
        """Main validator loop."""
        self.start_proxy_server()
        # Reap any agent pods left over from a prior crash/redeploy
        # before we start spawning new ones. Pure safety-net — never
        # raises, never blocks startup on Basilica latency.
        try:
            await reap_orphan_agent_pods(
                max_age_seconds=Config.AGENT_POD_REAP_MAX_AGE_S,
            )
        except Exception as e:
            logger.warning("Startup orphan reap failed: %s", e)
        while True:
            try:
                current_block = int(time.time() // 12)
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
            # Sweep any agent pods that the round left behind — covers
            # mid-round exceptions, retry-spawned pods cleanup missed,
            # and load-shed scenarios.
            try:
                await reap_orphan_agent_pods(
                    max_age_seconds=Config.AGENT_POD_REAP_MAX_AGE_S,
                )
            except Exception as e:
                logger.debug("Post-round orphan reap failed: %s", e)
            await asyncio.sleep(60)


def get_config():
    parser = argparse.ArgumentParser(description="Radar Validator")
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--task", type=str, default="ml_training")
    parser.add_argument("--db_dir", type=str, default="./experiments")
    parser.add_argument("--db_port", type=int, default=Config.PROXY_PORT)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    config = get_config()
    validator = Validator(config)
    asyncio.run(validator.run())


if __name__ == "__main__":
    main()
