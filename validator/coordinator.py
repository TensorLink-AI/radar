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

    def __init__(self, wallet, metagraph, r2: "R2AuditLog", my_uid: int):
        self.wallet = wallet
        self.metagraph = metagraph
        self.r2 = r2
        self.my_uid = my_uid
        self._fallback_uids: dict[int, set[int]] = {}  # round_id → UIDs using proxy

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
    ) -> list[TrainingResult]:
        """POST to trainer endpoints with Epistula-signed payload.

        Dispatches concurrently. Attestation is already verified in
        prepare_trainers() when TrainerReady arrives; fallback proxy
        is trusted subnet-owner infrastructure.
        """
        commitments = commitments or {}
        logger.info("Dispatching %d jobs to %d trainer endpoints", len(jobs), len(trainer_endpoints))

        # Per-job timeout: time_budget + 120s buffer for startup/upload overhead
        time_budget = challenge.task.get("time_budget", 300)
        job_timeout = time_budget + 120

        # Build tasks for concurrent dispatch
        immediate_results: list[TrainingResult] = []
        dispatch_tasks: list[tuple[Job, str, bytes, dict]] = []

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

            payload = json.dumps({
                "architecture": proposal.code,
                "seed": challenge.seed,
                "round_id": challenge.round_id,
                "min_flops_equivalent": challenge.min_flops_equivalent,
                "max_flops_equivalent": challenge.max_flops_equivalent,
                "miner_hotkey": miner_hotkey,
                "time_budget": time_budget,
                "upload_urls": upload_urls,
            }).encode()

            headers = sign_request(self.wallet, payload)
            headers["Content-Type"] = "application/json"
            dispatch_tasks.append((job, trainer_url, payload, headers))

        if not dispatch_tasks:
            return immediate_results

        async def _dispatch_one(
            client: httpx.AsyncClient, job: Job, url: str, payload: bytes, headers: dict,
        ) -> TrainingResult:
            try:
                resp = await client.post(
                    f"{url.rstrip('/')}/train",
                    content=payload,
                    headers=headers,
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
                logger.error("Dispatch to trainer UID %d failed: %s", job.trainer_uid, e)
                return TrainingResult(
                    round_id=job.round_id,
                    arch_owner=job.arch_owner,
                    trainer_uid=job.trainer_uid,
                    dispatcher=self.my_uid,
                    status="failed",
                    error=str(e),
                )

        async with httpx.AsyncClient(timeout=job_timeout) as client:
            coros = [
                _dispatch_one(client, job, url, payload, headers)
                for job, url, payload, headers in dispatch_tasks
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

        Performs post-upload verification: checks that the meta's round_id
        and miner_hotkey match expectations, rejecting tampered artifacts.
        """
        from shared.artifacts import list_round_artifacts, meta_key as mk_fn, verify_uploaded_artifacts

        results: dict[int, dict] = {}
        deadline = asyncio.get_event_loop().time() + timeout

        # Build uid→hotkey mapping
        hotkeys = self.metagraph.hotkeys if hasattr(self.metagraph, "hotkeys") else []
        uid_to_hotkey = {}
        hotkey_to_uid = {}
        for uid in expected_miners:
            hk = hotkeys[uid] if uid < len(hotkeys) else f"uid_{uid}"
            uid_to_hotkey[uid] = hk
            hotkey_to_uid[hk] = uid

        poll_count = 0
        while asyncio.get_event_loop().time() < deadline:
            poll_count += 1
            remaining = deadline - asyncio.get_event_loop().time()
            logger.info(
                "Checkpoint poll #%d: %d/%d collected, %.0fs remaining (round %d)",
                poll_count, len(results), len(expected_miners), remaining, round_id,
            )
            # Use list_round_artifacts to find which miners have uploaded
            available = list_round_artifacts(self.r2, round_id)
            for hk in available:
                uid = hotkey_to_uid.get(hk)
                if uid is None or uid in results:
                    continue
                try:
                    # Verify artifact integrity before accepting
                    ok, err = verify_uploaded_artifacts(self.r2, round_id, hk)
                    if not ok:
                        logger.warning(
                            "Artifact verification failed for miner %s (UID %s) round %d: %s",
                            hk[:16], uid, round_id, err,
                        )
                        continue

                    meta = self.r2.download_json(mk_fn(round_id, hk))
                    if meta:
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
            logger.warning(
                "Checkpoint collection timed out: %d/%d collected after %ds (round %d)",
                len(results), len(expected_miners), timeout, round_id,
            )

        return results

    async def write_dispatch_record(self, round_id: int, results: list[TrainingResult]):
        """Write dispatch record to R2 for auditability."""
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
        self.r2.upload_json(key, {"dispatcher": hotkey, "round_id": round_id, "jobs": records})

    async def write_frontier(
        self, frontier_data: list[dict], task_name: str = "",
    ):
        """Write current frontier to R2, scoped by task."""
        if task_name:
            key = f"frontier/{task_name}/latest.json"
        else:
            key = "frontier/latest.json"
        self.r2.upload_json(key, {"frontier": frontier_data})

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
        from config import Config
        from validator.db_server import get_ready_trainers
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
            gpu_model=Config.TRAINER_GPU_MODEL,
            memory=Config.TRAINER_MEMORY,
        )
        body = req.to_json().encode()
        headers = sign_request(self.wallet, body)
        headers["Content-Type"] = "application/json"

        # Fire TrainerRequest to all miners with listener_urls
        sent_uids: set[int] = set()
        async with httpx.AsyncClient(timeout=30) as client:
            for uid, commitment in commitments.items():
                if not commitment.listener_url:
                    continue
                try:
                    url = f"{commitment.listener_url.rstrip('/')}/prepare"
                    await client.post(url, content=body, headers=headers)
                    sent_uids.add(uid)
                except Exception as e:
                    logger.warning("Failed to send TrainerRequest to UID %d: %s", uid, e)

        logger.info(
            "Sent TrainerRequest to %d miners (round %d), waiting for readiness",
            len(sent_uids), round_id,
        )

        # Poll for TrainerReady responses
        deadline = asyncio.get_event_loop().time() + Config.TRAINER_PREPARE_TIMEOUT
        result: dict[int, str] = {}

        while asyncio.get_event_loop().time() < deadline:
            ready = get_ready_trainers(round_id)
            for uid, ready_msg in ready.items():
                if uid in result:
                    continue
                # Verify the pod via Basilica public metadata
                if ready_msg.instance_name:
                    ok, reason = await verify_miner_pod(ready_msg.instance_name)
                    if not ok:
                        logger.warning(
                            "UID %d TrainerReady failed verification: %s", uid, reason,
                        )
                        continue
                result[uid] = ready_msg.trainer_url
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

        logger.info(
            "prepare_trainers complete: %d ready, %d fallback, %d no response (round %d)",
            len(result) - len(fallback_uids), len(fallback_uids),
            len(sent_uids) - len(result), round_id,
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
        from validator.db_server import clear_ready_trainers

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
