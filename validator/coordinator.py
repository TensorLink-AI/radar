"""Training coordinator — Phase B job assignment and dispatch.

Handles deterministic job assignment, dispatch to miner trainers,
and R2 artifact monitoring.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import httpx

from config import Config
from shared.auth import sign_request
from shared.protocol import Proposal, TrainerRequest, TrainerReady, TrainerRelease

if TYPE_CHECKING:
    from shared.commitment import ImageCommitment
    from shared.r2_audit import R2AuditLog

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """A training job assignment."""
    arch_owner: int      # miner whose architecture is being trained
    trainer_uid: int     # miner whose trainer runs the job (self-training allowed)
    dispatcher: int      # validator UID responsible for dispatching
    round_id: int = 0


@dataclass
class TrainingResult:
    """What the trainer returns — NOT final eval metrics."""
    round_id: int = 0
    arch_owner: int = -1
    trainer_uid: int = -1
    dispatcher: int = -1
    seed: int = 0
    flops_equivalent_size: int = 0
    status: str = ""             # success | failed | timeout | build_failed | size_violation
    error: str = ""
    training_time_seconds: float = 0.0
    checkpoint_key: str = ""
    architecture_key: str = ""


def compute_assignments(
    block_hash: str,
    submissions: dict[int, Proposal],
    miner_uids: list[int],
    validator_uids: list[int],
    round_id: int,
) -> list[Job]:
    """Deterministic job assignment — fully randomised each round.

    Any miner can train any architecture, including their own.
    Trainer pool is shuffled fresh from the block hash each round.
    """
    if not submissions or not miner_uids or not validator_uids:
        return []

    seed = int(block_hash[:16], 16)
    rng = random.Random(seed)

    arch_uids = sorted(submissions.keys())
    trainer_pool = sorted(miner_uids)
    sorted_validators = sorted(validator_uids)

    # Shuffle trainer pool — completely random pairing each round
    rng.shuffle(trainer_pool)

    jobs: list[Job] = []
    for i, arch_uid in enumerate(arch_uids):
        # Round-robin through shuffled trainer pool (self-training allowed)
        trainer_uid = trainer_pool[i % len(trainer_pool)]

        # Round-robin dispatch across validators
        dispatcher = sorted_validators[i % len(sorted_validators)]

        jobs.append(Job(
            arch_owner=arch_uid,
            trainer_uid=trainer_uid,
            dispatcher=dispatcher,
            round_id=round_id,
        ))

    return jobs


def compute_fallback(
    block_hash: str,
    missing_valis: list[int],
    jobs: list[Job],
    remaining_valis: list[int],
) -> list[Job]:
    """Reassign missing validator's jobs deterministically."""
    if not missing_valis or not remaining_valis:
        return []

    seed = int(block_hash[:16], 16) + 999
    rng = random.Random(seed)
    sorted_remaining = sorted(remaining_valis)

    reassigned = []
    orphaned = [j for j in jobs if j.dispatcher in missing_valis]
    for i, job in enumerate(orphaned):
        new_dispatcher = sorted_remaining[i % len(sorted_remaining)]
        reassigned.append(Job(
            arch_owner=job.arch_owner,
            trainer_uid=job.trainer_uid,
            dispatcher=new_dispatcher,
            round_id=job.round_id,
        ))

    return reassigned


class TrainingCoordinator:
    """Orchestrates Phase B dispatch and R2 artifact monitoring."""

    def __init__(
        self, wallet, metagraph, r2: "R2AuditLog", my_uid: int,
        artifact_store=None,
    ):
        self.wallet = wallet
        self.metagraph = metagraph
        self.r2 = r2
        self.my_uid = my_uid
        # Optional dual-write store (TEN-240 Phase 7). When present, JSON
        # artifacts (dispatch records, frontier snapshots) fan out to both
        # R2 and Hippius. Falls back to R2-only when None — historical
        # behaviour and the default.
        self.artifact_store = artifact_store
        self._fallback_uids: dict[int, set[int]] = {}  # round_id → UIDs using proxy
        # round_id → {uid → True} for miners verified during a Targon outage.
        # Scoring multiplies their contribution by TARGON_UNAVAILABLE_SCORE_MULTIPLIER.
        self._targon_unavailable: dict[int, dict[int, bool]] = {}
        # round_id → {uid → True} for miners hosted on a non-attested
        # backend (currently: RunPod). Scoring multiplies their
        # contribution by NON_ATTESTED_SCORE_MULTIPLIER.
        self._non_attested: dict[int, dict[int, bool]] = {}
        # round_id → {uid → True} for miners flagged as compromised during
        # mid-round re-verification. Excluded from scoring.
        self._compromised: dict[int, dict[int, bool]] = {}
        # round_id → {uid → TrainerReady} so mid-run reverify knows where
        # each accepted miner's CVM lives. Populated in prepare_trainers.
        self._ready_msgs: dict[int, dict[int, "TrainerReady"]] = {}
        # round_id → list[asyncio.Task] of in-flight mid-round reverify
        # tasks. Cancelled by release_trainers / cancel_mid_round_reverify.
        self._reverify_tasks: dict[int, list[asyncio.Task]] = {}
        self._targon_client = None
        self._runpod_client = None

    def compute_my_jobs(
        self,
        block_hash: str,
        submissions: dict[int, Proposal],
        miner_uids: list[int],
        validator_uids: list[int],
        round_id: int,
    ) -> list[Job]:
        """Compute all jobs, return only mine."""
        all_jobs = compute_assignments(
            block_hash, submissions, miner_uids, validator_uids, round_id,
        )
        return [j for j in all_jobs if j.dispatcher == self.my_uid]

    async def dispatch_jobs(
        self,
        jobs: list[Job],
        challenge,
        submissions: dict[int, Proposal],
        trainer_endpoints: dict[int, str],
        commitments: dict[int, "ImageCommitment"] | None = None,
        extras: dict | None = None,
    ) -> list[TrainingResult]:
        """POST to trainer endpoints with Epistula-signed payload.

        Dispatches concurrently. Attestation is already verified in
        prepare_trainers() when TrainerReady arrives; fallback proxy
        is trusted subnet-owner infrastructure.

        `extras` is a task-provided dict merged into every dispatch
        payload (e.g. ts_forecasting supplies `gift_eval_urls` and
        `pretrain_shard_urls`).
        """
        commitments = commitments or {}
        extras = extras or {}
        # Each job is sent to exactly one trainer (1:1 arch→trainer).
        # `trainer_endpoints` is the available pool, not the fan-out.
        logger.info(
            "Dispatching %d jobs across %d available trainers (1 trainer per job)",
            len(jobs), len(trainer_endpoints),
        )

        # Trainer returns 202 Accepted immediately; this timeout only
        # covers auth + request parsing, not the full training run.
        time_budget = challenge.task.get("time_budget", 300)
        job_timeout = 60

        # Build tasks for concurrent dispatch
        immediate_results: list[TrainingResult] = []
        dispatch_tasks: list[tuple[Job, str, bytes]] = []

        for job in jobs:
            proposal = submissions.get(job.arch_owner)
            if not proposal:
                continue

            trainer_url = trainer_endpoints.get(job.trainer_uid)
            if not trainer_url:
                logger.warning(
                    "No trainer endpoint for UID %d (arch_owner=%d)",
                    job.trainer_uid, job.arch_owner,
                )
                immediate_results.append(TrainingResult(
                    round_id=job.round_id,
                    arch_owner=job.arch_owner,
                    trainer_uid=job.trainer_uid,
                    dispatcher=self.my_uid,
                    status="failed",
                    error="No trainer endpoint",
                ))
                continue

            miner_hotkey = (
                self.metagraph.hotkeys[job.arch_owner]
                if job.arch_owner < len(self.metagraph.hotkeys)
                else f"uid_{job.arch_owner}"
            )

            # Generate presigned PUT URLs so trainer can upload without R2 creds.
            from shared.artifacts import generate_upload_urls
            presigned_ttl = int(os.getenv("RADAR_PRESIGNED_TTL", "5400"))
            upload_urls = generate_upload_urls(
                self.r2, challenge.round_id, miner_hotkey, ttl=presigned_ttl,
            )

            if not upload_urls.get("architecture"):
                logger.error(
                    "Missing architecture presigned URL for arch_owner=%d miner=%s",
                    job.arch_owner, miner_hotkey[:16],
                )
            else:
                logger.info(
                    "Generated %d presigned URLs for arch_owner=%d: %s",
                    len(upload_urls), job.arch_owner, list(upload_urls.keys()),
                )

            task_name = challenge.task.get("name", "")
            runner_dir = challenge.task.get("runner_dir", "")
            dispatch_payload = {
                "architecture": proposal.code,
                "seed": challenge.seed,
                "round_id": challenge.round_id,
                "min_flops_equivalent": challenge.min_flops_equivalent,
                "max_flops_equivalent": challenge.max_flops_equivalent,
                "miner_hotkey": miner_hotkey,
                "time_budget": time_budget,
                "upload_urls": upload_urls,
                "task_name": task_name,
                "runner_dir": runner_dir,
            }
            dispatch_payload.update(extras)
            payload = json.dumps(dispatch_payload).encode()

            dispatch_tasks.append((job, trainer_url, payload))

        if not dispatch_tasks:
            return immediate_results

        async def _dispatch_one(
            client: httpx.AsyncClient, job: Job, url: str, payload: bytes,
        ) -> TrainingResult:
            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    # Sign fresh every attempt so the Epistula timestamp
                    # stays inside the EPISTULA_TIMESTAMP_TOLERANCE window
                    # (default 120s; tunable via RADAR_EPISTULA_TOLERANCE).
                    headers = sign_request(self.wallet, payload)
                    headers["Content-Type"] = "application/json"
                    resp = await client.post(
                        f"{url.rstrip('/')}/train",
                        content=payload,
                        headers=headers,
                    )

                    # Retry on transient server errors:
                    # - 403: Timestamp stale (clock skew or slow dispatch)
                    # - 500: Internal Server Error (transient Basilica platform error)
                    # - 502: Bad Gateway (trainer pod proxy up, server still starting)
                    # - 503: Service Unavailable (metagraph/auth not ready yet)
                    # 429 is handled separately below — rate-limited 429s are
                    # retried, while already-running 429s go to R2 polling.
                    if resp.status_code in (403, 500, 502, 503) and attempt < max_retries:
                        wait = 5 * (attempt + 1)
                        body_preview = resp.text[:300] if resp.text else "(empty)"
                        logger.warning(
                            "Trainer UID %d returned HTTP %d: %s — retrying in %ds (attempt %d/%d)",
                            job.trainer_uid, resp.status_code, body_preview,
                            wait, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(wait)
                        continue

                    # 429 from trainer — check reason to decide retry vs accept.
                    # "already_running" = semaphore busy, job IS running → poll R2.
                    # "rate_limited"    = per-hotkey cooldown, no job running → retry.
                    if resp.status_code == 429:
                        try:
                            reason_data = resp.json()
                        except (json.JSONDecodeError, ValueError):
                            reason_data = {}
                        reason = reason_data.get("reason", "already_running")

                        if reason == "rate_limited" and attempt < max_retries:
                            retry_after = int(reason_data.get("retry_after", 30))
                            logger.warning(
                                "Trainer UID %d rate-limited (HTTP 429), retrying in %ds (attempt %d/%d)",
                                job.trainer_uid, retry_after, attempt + 1, max_retries,
                            )
                            await asyncio.sleep(retry_after)
                            continue

                        logger.info(
                            "Trainer UID %d already running job (HTTP 429) — "
                            "checkpoint will be collected via R2 polling",
                            job.trainer_uid,
                        )
                        return TrainingResult(
                            round_id=job.round_id,
                            arch_owner=job.arch_owner,
                            trainer_uid=job.trainer_uid,
                            dispatcher=self.my_uid,
                            seed=challenge.seed,
                            status="already_running",
                        )

                    try:
                        data = resp.json()
                    except (json.JSONDecodeError, ValueError):
                        body_preview = resp.text[:200] if resp.text else "(empty)"
                        logger.error(
                            "Trainer UID %d returned non-JSON (HTTP %d): %s",
                            job.trainer_uid, resp.status_code, body_preview,
                        )
                        return TrainingResult(
                            round_id=job.round_id,
                            arch_owner=job.arch_owner,
                            trainer_uid=job.trainer_uid,
                            dispatcher=self.my_uid,
                            status="failed",
                            error=f"Non-JSON response (HTTP {resp.status_code}): {body_preview}",
                        )

                    # 202 Accepted = trainer acknowledged, training in background
                    if resp.status_code == 202:
                        logger.info(
                            "Trainer UID %d accepted job (HTTP 202) — "
                            "checkpoint will be collected via R2 polling",
                            job.trainer_uid,
                        )
                        return TrainingResult(
                            round_id=job.round_id,
                            arch_owner=job.arch_owner,
                            trainer_uid=job.trainer_uid,
                            dispatcher=self.my_uid,
                            seed=challenge.seed,
                            status="accepted",
                        )

                    # Log HTTP errors with the response body for diagnostics
                    if resp.status_code >= 400:
                        logger.warning(
                            "Trainer UID %d returned HTTP %d: %s",
                            job.trainer_uid, resp.status_code, data.get("error", ""),
                        )

                    return TrainingResult(
                        round_id=job.round_id,
                        arch_owner=job.arch_owner,
                        trainer_uid=job.trainer_uid,
                        dispatcher=self.my_uid,
                        seed=challenge.seed,
                        flops_equivalent_size=int(data.get("flops_equivalent_size", 0)),
                        status=data.get("status", "failed"),
                        error=data.get("error", ""),
                        training_time_seconds=float(data.get("training_time_seconds", 0)),
                        checkpoint_key=data.get("checkpoint_key", ""),
                        architecture_key=data.get("architecture_key", ""),
                    )
                except httpx.TimeoutException:
                    logger.error("Dispatch to trainer UID %d timed out", job.trainer_uid)
                    return TrainingResult(
                        round_id=job.round_id,
                        arch_owner=job.arch_owner,
                        trainer_uid=job.trainer_uid,
                        dispatcher=self.my_uid,
                        status="timeout",
                        error="Request timed out",
                    )
                except Exception as e:
                    if attempt < max_retries:
                        wait = 3 * (attempt + 1)
                        logger.warning(
                            "Dispatch to trainer UID %d failed (%s), retrying in %ds",
                            job.trainer_uid, e, wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    logger.error("Dispatch to trainer UID %d failed: %s", job.trainer_uid, e)
                    return TrainingResult(
                        round_id=job.round_id,
                        arch_owner=job.arch_owner,
                        trainer_uid=job.trainer_uid,
                        dispatcher=self.my_uid,
                        status="failed",
                        error=str(e),
                    )
            # Should not reach here, but handle defensively
            return TrainingResult(
                round_id=job.round_id, arch_owner=job.arch_owner,
                trainer_uid=job.trainer_uid, dispatcher=self.my_uid,
                status="failed", error="Retries exhausted",
            )

        async with httpx.AsyncClient(timeout=job_timeout) as client:
            coros = [
                _dispatch_one(client, job, url, payload)
                for job, url, payload in dispatch_tasks
            ]
            dispatch_results = await asyncio.gather(*coros)

        return immediate_results + list(dispatch_results)

    async def wait_for_checkpoints(
        self,
        round_id: int,
        expected_miners: list[int],
        timeout: int = 300,
    ) -> dict[int, dict]:
        """Poll R2 for training_meta.json from each miner.

        Uses direct key lookups (strongly consistent) instead of
        list_objects (eventually consistent) to avoid missing recently
        uploaded artifacts.

        Performs post-upload verification: checks that the meta's round_id
        and miner_hotkey match expectations, rejecting tampered artifacts.
        """
        from shared.artifacts import meta_key as mk_fn, checkpoint_key as ck_fn

        results: dict[int, dict] = {}
        deadline = asyncio.get_event_loop().time() + timeout

        # Build uid→hotkey mapping
        hotkeys = self.metagraph.hotkeys if hasattr(self.metagraph, "hotkeys") else []
        uid_to_hotkey = {}
        for uid in expected_miners:
            hk = hotkeys[uid] if uid < len(hotkeys) else f"uid_{uid}"
            uid_to_hotkey[uid] = hk

        # Log expected R2 keys for debugging
        for uid, hk in uid_to_hotkey.items():
            logger.info(
                "Expecting R2 meta key for UID %d: %s",
                uid, mk_fn(round_id, hk),
            )

        poll_count = 0
        while asyncio.get_event_loop().time() < deadline:
            poll_count += 1
            remaining = deadline - asyncio.get_event_loop().time()
            logger.info(
                "Checkpoint poll #%d: %d/%d collected, %.0fs remaining (round %d)",
                poll_count, len(results), len(expected_miners), remaining, round_id,
            )
            # Check each pending miner directly (strongly consistent)
            for uid, hk in uid_to_hotkey.items():
                if uid in results:
                    continue
                mk = mk_fn(round_id, hk)
                try:
                    meta = self.r2.download_json(mk)
                    if not meta:
                        if poll_count == 1:
                            logger.info(
                                "UID %d: meta not found at %s (will keep polling)",
                                uid, mk,
                            )
                        continue

                    logger.info(
                        "UID %d: meta found at %s — status=%s",
                        uid, mk, meta.get("status", "?"),
                    )

                    # Verify meta fields match expectations
                    if meta.get("round_id") != round_id:
                        logger.warning(
                            "Meta round_id mismatch for UID %d: expected %d, got %s",
                            uid, round_id, meta.get("round_id"),
                        )
                        continue
                    if meta.get("miner_hotkey") != hk:
                        logger.warning(
                            "Meta miner_hotkey mismatch for UID %d: expected %s, got %s",
                            uid, hk[:16], str(meta.get("miner_hotkey"))[:16],
                        )
                        continue

                    # Failure statuses: collect the meta without requiring a checkpoint.
                    # The trainer only uploads training_meta.json for failures
                    # (no checkpoint is produced). Downstream scoring handles these
                    # via penalties / size gate rejection.
                    _FAILURE_STATUSES = ("build_failed", "size_violation", "failed", "timeout")
                    meta_status = meta.get("status", "")
                    if meta_status in _FAILURE_STATUSES:
                        meta["miner_uid"] = uid
                        meta["miner_hotkey"] = hk
                        results[uid] = meta
                        logger.info(
                            "Training failure collected: UID %d status=%s error=%s (round %d)",
                            uid, meta_status, meta.get("error", ""), round_id,
                        )
                        continue

                    # Verify checkpoint file exists (only for success status)
                    ck = ck_fn(round_id, hk)
                    if not self.r2.key_exists(ck):
                        logger.info(
                            "Meta exists but checkpoint missing for UID %d at %s (round %d)",
                            uid, ck, round_id,
                        )
                        continue

                    meta["miner_uid"] = uid
                    meta["miner_hotkey"] = hk
                    results[uid] = meta
                    logger.info(
                        "Checkpoint received: UID %d status=%s flops=%d (round %d)",
                        uid, meta.get("status", "?"),
                        meta.get("flops_equivalent_size", 0), round_id,
                    )
                except Exception as exc:
                    logger.warning("Error reading checkpoint for UID %s round %d: %s", uid, round_id, exc)

            if len(results) >= len(expected_miners):
                logger.info("All %d checkpoints collected (round %d)", len(expected_miners), round_id)
                break
            await asyncio.sleep(30)
        else:
            # On timeout, check what actually exists in R2 for this round
            round_prefix = f"round_{round_id}/"
            try:
                existing_keys = self.r2.list_keys(round_prefix)
                logger.warning(
                    "Checkpoint collection timed out: %d/%d collected after %ds (round %d). "
                    "R2 keys under %s: %s",
                    len(results), len(expected_miners), timeout, round_id,
                    round_prefix, existing_keys[:20] if existing_keys else "(none)",
                )
            except Exception:
                logger.warning(
                    "Checkpoint collection timed out: %d/%d collected after %ds (round %d). "
                    "R2 list_keys also failed — check R2 connectivity",
                    len(results), len(expected_miners), timeout, round_id,
                )

        return results

    async def write_dispatch_record(self, round_id: int, results: list[TrainingResult]):
        """Write dispatch record to R2 (and Hippius if dual-write enabled)."""
        hotkey = self.wallet.hotkey.ss58_address
        key = f"round_{round_id}/dispatch/vali_{hotkey}.json"
        records = []
        for r in results:
            records.append({
                "arch_owner": r.arch_owner,
                "trainer_uid": r.trainer_uid,
                "status": r.status,
                "flops_equivalent_size": r.flops_equivalent_size,
                "training_time_seconds": r.training_time_seconds,
                "checkpoint_key": r.checkpoint_key,
            })
        body = {"dispatcher": hotkey, "round_id": round_id, "jobs": records}
        if self.artifact_store is not None:
            await self.artifact_store.put_json(key, body, {
                "app": "radar", "kind": "dispatch",
                "round_id": str(round_id), "validator_hotkey": hotkey,
            })
        else:
            self.r2.upload_json(key, body)

    async def write_frontier(
        self, frontier_data: list[dict], task_name: str = "",
    ):
        """Write current frontier to R2 (and Hippius if dual-write enabled)."""
        if task_name:
            key = f"frontier/{task_name}/latest.json"
        else:
            key = "frontier/latest.json"
        body = {"frontier": frontier_data}
        if self.artifact_store is not None:
            await self.artifact_store.put_json(key, body, {
                "app": "radar", "kind": "frontier",
                "task": task_name or "default",
            })
        else:
            self.r2.upload_json(key, body)

    async def prepare_trainers(
        self,
        challenge,
        commitments: dict[int, "ImageCommitment"],
        db_url: str,
    ) -> dict[int, str]:
        """Send TrainerRequest to all miners' listener_urls, wait for TrainerReady.

        Called at Phase A start. Sends to all miners because trainer
        assignments depend on proposals which aren't collected yet.
        Non-responsive miners get routed to the fallback proxy if configured.

        Returns {uid: trainer_url} for all miners that responded or fell back.
        """
        from validator.db_proxy import get_ready_trainers
        from validator.pod_manager import verify_miner_pod

        round_id = challenge.round_id
        time_budget = challenge.task.get("time_budget", 300)

        req = TrainerRequest(
            round_id=round_id,
            challenge_id=challenge.challenge_id,
            seed=challenge.seed,
            min_flops_equivalent=challenge.min_flops_equivalent,
            max_flops_equivalent=challenge.max_flops_equivalent,
            time_budget=time_budget,
            validator_db_url=db_url,
            gpu_count=Config.TRAINER_GPU_COUNT,
            min_gpu_memory_gb=Config.TRAINER_MIN_GPU_MEMORY_GB,
            memory=Config.TRAINER_MEMORY,
            submission_window_seconds=Config.SUBMISSION_WINDOW_BLOCKS * 12,
        )
        body = req.to_json().encode()

        # Fire TrainerRequest to all miners with listener_urls.
        # Re-sign per miner so the Epistula timestamp stays fresh —
        # stale timestamps cause miners to reject with 403.
        with_listener = {uid: c for uid, c in commitments.items() if c.listener_url}
        without_listener = {uid: c for uid, c in commitments.items() if not c.listener_url}
        if without_listener:
            logger.warning(
                "Skipping %d miners with no listener_url: %s",
                len(without_listener), sorted(without_listener.keys()),
            )
        logger.info(
            "Sending TrainerRequest to %d miners with listener_urls (round %d)",
            len(with_listener), round_id,
        )
        sent_uids: set[int] = set()
        async with httpx.AsyncClient(timeout=30) as client:
            for uid, commitment in with_listener.items():
                try:
                    url = f"{commitment.listener_url.rstrip('/')}/prepare"
                    fresh_headers = sign_request(self.wallet, body)
                    fresh_headers["Content-Type"] = "application/json"
                    resp = await client.post(url, content=body, headers=fresh_headers)
                    if resp.status_code < 400:
                        sent_uids.add(uid)
                    else:
                        logger.warning(
                            "TrainerRequest to UID %d rejected (HTTP %d): %s",
                            uid, resp.status_code, resp.text[:200],
                        )
                except Exception as e:
                    logger.warning("Failed to send TrainerRequest to UID %d: %s", uid, e)

        logger.info(
            "Sent TrainerRequest to %d miners (round %d), waiting for readiness",
            len(sent_uids), round_id,
        )

        # Poll for TrainerReady responses
        deadline = asyncio.get_event_loop().time() + Config.TRAINER_PREPARE_TIMEOUT
        result: dict[int, str] = {}

        # Track the per-round flags so dispatch / scoring can read them.
        targon_unavailable: dict[int, bool] = self._targon_unavailable.setdefault(round_id, {})
        non_attested: dict[int, bool] = self._non_attested.setdefault(round_id, {})
        # Cache ready messages so mid-round reverify can re-hit each CVM.
        ready_cache: dict[int, "TrainerReady"] = self._ready_msgs.setdefault(round_id, {})

        while asyncio.get_event_loop().time() < deadline:
            ready = get_ready_trainers(round_id)
            for uid, ready_msg in ready.items():
                if uid in result:
                    continue
                ok, reason, soft_fail, attested = await self._verify_ready(ready_msg, uid)
                if not ok:
                    if soft_fail:
                        # Targon API down — let the round proceed with a flag.
                        targon_unavailable[uid] = True
                        result[uid] = ready_msg.trainer_url
                        logger.warning(
                            "UID %d trainer accepted with targon_unavailable=True: %s",
                            uid, reason,
                        )
                    else:
                        logger.warning(
                            "UID %d TrainerReady failed verification: %s", uid, reason,
                        )
                    continue
                result[uid] = ready_msg.trainer_url
                ready_cache[uid] = ready_msg
                if not attested:
                    non_attested[uid] = True
                    logger.info(
                        "UID %d trainer ready at %s (non-attested backend, "
                        "score discounted by NON_ATTESTED_SCORE_MULTIPLIER)",
                        uid, ready_msg.trainer_url,
                    )
                else:
                    logger.info("UID %d trainer ready at %s", uid, ready_msg.trainer_url)

            if len(result) >= len(sent_uids):
                break
            await asyncio.sleep(Config.TRAINER_READY_POLL_INTERVAL)

        # Fallback proxy for non-responsive miners
        fallback_uids: set[int] = set()
        if Config.FALLBACK_PROXY_URL:
            for uid in sent_uids:
                if uid not in result:
                    result[uid] = Config.FALLBACK_PROXY_URL
                    fallback_uids.add(uid)
            if fallback_uids:
                logger.info(
                    "Routed %d non-responsive trainers to fallback proxy: %s",
                    len(fallback_uids), sorted(fallback_uids),
                )

        self._fallback_uids[round_id] = fallback_uids

        total_miners = len(sent_uids)
        real_count = len(result) - len(fallback_uids)
        fallback_count = len(fallback_uids)
        no_response = total_miners - len(result)
        fallback_pct = (fallback_count / total_miners * 100) if total_miners else 0

        logger.info(
            "prepare_trainers complete: %d ready, %d fallback (%.0f%%), "
            "%d no response (round %d)",
            real_count, fallback_count, fallback_pct, no_response, round_id,
        )
        if fallback_pct > 50:
            logger.warning(
                "FALLBACK_RATE_HIGH: %.0f%% of miners (%d/%d) using fallback "
                "template (round %d) — agents may not be learning",
                fallback_pct, fallback_count, total_miners, round_id,
            )
        return result

    async def release_trainers(
        self,
        round_id: int,
        commitments: dict[int, "ImageCommitment"],
        released_uids: set[int],
    ):
        """Send TrainerRelease to miners whose checkpoints have been collected.

        Skips UIDs that used the fallback proxy.
        """
        from validator.db_proxy import clear_ready_trainers

        fallback_uids = self._fallback_uids.get(round_id, set())
        released = 0

        async with httpx.AsyncClient(timeout=10) as client:
            for uid in released_uids:
                if uid in fallback_uids:
                    continue
                commitment = commitments.get(uid)
                if not commitment or not commitment.listener_url:
                    continue
                try:
                    release = TrainerRelease(
                        round_id=round_id,
                        miner_hotkey=(
                            self.metagraph.hotkeys[uid]
                            if uid < len(self.metagraph.hotkeys)
                            else ""
                        ),
                    )
                    body = release.to_json().encode()
                    headers = sign_request(self.wallet, body)
                    headers["Content-Type"] = "application/json"
                    url = f"{commitment.listener_url.rstrip('/')}/release"
                    await client.post(url, content=body, headers=headers)
                    released += 1
                except Exception as e:
                    logger.debug("Failed to send TrainerRelease to UID %d: %s", uid, e)

        logger.info("Released %d trainers (round %d)", released, round_id)
        clear_ready_trainers(round_id)
        self._fallback_uids.pop(round_id, None)
        await self.cancel_mid_round_reverify(round_id)
        self._ready_msgs.pop(round_id, None)

    # ── Trainer verification (backend-aware) ──────────────────────

    def _get_targon_client(self):
        if self._targon_client is None:
            from shared.targon_breaker import CircuitBreaker
            from shared.targon_client import TargonClient
            self._targon_client = TargonClient(
                base_url=Config.TARGON_API_BASE_URL,
                tower_url=Config.TARGON_TOWER_URL,
                timeout=Config.TARGON_VERIFICATION_TIMEOUT,
                breaker=CircuitBreaker(
                    threshold=Config.TARGON_CIRCUIT_BREAKER_THRESHOLD,
                    reset_after=Config.TARGON_CIRCUIT_BREAKER_RESET,
                ),
            )
        return self._targon_client

    def _get_runpod_client(self):
        if self._runpod_client is None:
            from shared.runpod_client import RunPodClient
            self._runpod_client = RunPodClient(
                base_url=Config.RUNPOD_API_BASE_URL,
                timeout=Config.RUNPOD_VERIFICATION_TIMEOUT,
            )
        return self._runpod_client

    async def _verify_ready(self, ready_msg, uid: int) -> tuple[bool, str, bool, bool]:
        """Verify a TrainerReady envelope.

        Returns ``(ok, reason, soft_fail, attested)``.

        ``soft_fail=True`` means the round should proceed with a
        ``targon_unavailable`` flag instead of excluding the miner —
        the only soft-fail signal is the Targon circuit breaker.

        ``attested=False`` flags the miner as having been hosted on a
        non-attested backend (RunPod / Basilica) — scoring will apply
        ``NON_ATTESTED_SCORE_MULTIPLIER`` to their contribution.
        """
        if Config.HOSTING_BACKEND == "targon":
            from validator.trainer_verify import verify_trainer
            hotkey = (
                self.metagraph.hotkeys[uid]
                if uid < len(self.metagraph.hotkeys) else ready_msg.miner_hotkey
            )
            result = await verify_trainer(
                ready=ready_msg,
                miner_hotkey=hotkey,
                expected_image_digest=Config.OFFICIAL_TRAINING_IMAGE_DIGEST,
                trainer_url=ready_msg.trainer_url,
                targon_client=self._get_targon_client(),
                wallet=self.wallet,
                require_boot_proof=Config.REQUIRE_BOOT_PROOF,
            )
            return result.ok, result.reason, result.targon_unavailable, result.attested

        if Config.HOSTING_BACKEND == "runpod":
            from validator.trainer_verify import verify_trainer_runpod
            hotkey = (
                self.metagraph.hotkeys[uid]
                if uid < len(self.metagraph.hotkeys) else ready_msg.miner_hotkey
            )
            result = await verify_trainer_runpod(
                ready=ready_msg,
                miner_hotkey=hotkey,
                expected_image_digest=Config.OFFICIAL_TRAINING_IMAGE_DIGEST,
                trainer_url=ready_msg.trainer_url,
                runpod_client=self._get_runpod_client(),
                require_boot_proof=Config.REQUIRE_BOOT_PROOF,
            )
            # RunPod has no soft-fail equivalent (no circuit breaker
            # because failures are categorical, not transient outages).
            return result.ok, result.reason, False, result.attested

        # Basilica path — preserve original behavior.
        from validator.pod_manager import verify_miner_pod
        if not ready_msg.instance_name:
            return True, "", False, False
        ok, reason = await verify_miner_pod(ready_msg.instance_name)
        return ok, reason, False, False

    async def reverify_running(
        self, round_id: int, miner_uid: int, ready_msg, block_hash: str,
    ) -> bool:
        """Run a single mid-round re-verification. Returns False on compromise.

        Caller schedules this at deterministic offsets within the
        training window via ``trainer_verify.reverify_offsets``. We
        keep that helper out of the hot path because the offsets are
        block-derived (see compute_assignments for the same pattern).
        """
        if Config.HOSTING_BACKEND != "targon":
            return True  # No mid-run check on Basilica or RunPod.
        from validator.trainer_verify import reverify_workload
        miner_hotkey = (
            self.metagraph.hotkeys[miner_uid]
            if miner_uid < len(self.metagraph.hotkeys) else ready_msg.miner_hotkey
        )
        result = await reverify_workload(
            ready=ready_msg,
            expected_image_digest=Config.OFFICIAL_TRAINING_IMAGE_DIGEST,
            trainer_url=ready_msg.trainer_url,
            targon_client=self._get_targon_client(),
            require_boot_proof=Config.REQUIRE_BOOT_PROOF,
            expected_signer_hotkey=miner_hotkey,
        )
        if result.targon_unavailable:
            self._targon_unavailable.setdefault(round_id, {})[miner_uid] = True
            return True  # treat as soft-pass
        if not result.ok:
            self._compromised.setdefault(round_id, {})[miner_uid] = True
            logger.warning(
                "Mid-run reverify failed for UID %d (round %d): %s",
                miner_uid, round_id, result.reason,
            )
            return False
        return True

    def schedule_mid_round_reverify(
        self,
        round_id: int,
        block_hash: str,
        training_window_seconds: float,
        n_checkpoints: int | None = None,
    ) -> int:
        """Fire background reverify tasks at deterministic offsets.

        Returns the number of tasks spawned. Caller invokes this once at
        the start of Phase B; the tasks self-clean and ``release_trainers``
        cancels any stragglers. No-op for the Basilica backend.
        """
        if Config.HOSTING_BACKEND != "targon":
            return 0
        from validator.trainer_verify import reverify_offsets
        ready_msgs = self._ready_msgs.get(round_id) or {}
        n = n_checkpoints if n_checkpoints is not None else Config.TARGON_REVERIFY_CHECKPOINTS
        spawned: list[asyncio.Task] = []
        for uid, ready_msg in ready_msgs.items():
            if not ready_msg.targon_workload_uid:
                continue  # Basilica miner in a Targon-mode validator — skip.
            offsets = reverify_offsets(
                block_hash, round_id, uid,
                n=n, window_seconds=training_window_seconds,
            )
            spawned.append(asyncio.create_task(
                self._run_reverify_schedule(round_id, uid, ready_msg, offsets, block_hash)
            ))
        self._reverify_tasks.setdefault(round_id, []).extend(spawned)
        if spawned:
            logger.info(
                "Scheduled mid-round reverify: round=%d miners=%d checkpoints_each=%d",
                round_id, len(spawned), n,
            )
        return len(spawned)

    async def _run_reverify_schedule(
        self, round_id: int, miner_uid: int, ready_msg, offsets: list[float],
        block_hash: str = "",
    ) -> None:
        """Sleep to each offset, then run reverify_running once. Stops early on first failure."""
        last = 0.0
        for offset in offsets:
            delta = max(0.0, offset - last)
            try:
                await asyncio.sleep(delta)
            except asyncio.CancelledError:
                return
            last = offset
            try:
                ok = await self.reverify_running(round_id, miner_uid, ready_msg, block_hash)
            except Exception as e:
                logger.warning(
                    "Reverify task crashed for round=%d uid=%d: %s",
                    round_id, miner_uid, e,
                )
                return
            if not ok:
                # reverify_running already logged + recorded compromise.
                return

    async def cancel_mid_round_reverify(self, round_id: int) -> None:
        tasks = self._reverify_tasks.pop(round_id, [])
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    def round_metadata(self, round_id: int) -> dict:
        """Snapshot per-round verification flags — written into experiment rows.

        Returns a dict with:
          - ``targon_unavailable``: UIDs that verified during a Targon
            outage (soft-pass) → ``TARGON_UNAVAILABLE_SCORE_MULTIPLIER``.
          - ``non_attested``: UIDs hosted on a non-attested backend
            (RunPod) → ``NON_ATTESTED_SCORE_MULTIPLIER``.
          - ``compromised``: UIDs that failed mid-round re-verify →
            excluded from scoring entirely.
        """
        return {
            "targon_unavailable": sorted((self._targon_unavailable.get(round_id) or {}).keys()),
            "non_attested": sorted((self._non_attested.get(round_id) or {}).keys()),
            "compromised": sorted((self._compromised.get(round_id) or {}).keys()),
        }
