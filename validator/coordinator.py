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
from shared.protocol import Proposal

if TYPE_CHECKING:
    from shared.commitment import ImageCommitment
    from shared.r2_audit import R2AuditLog

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """A training job assignment."""
    arch_owner: int      # miner whose architecture is being trained
    trainer_uid: int     # DIFFERENT miner whose trainer runs the job
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
    """Deterministic job assignment from block hash.

    Cross-eval: arch_owner != trainer_uid.
    Dispatch split across validators.
    Everyone with the same inputs gets identical output.
    """
    if not submissions or not miner_uids or not validator_uids:
        return []

    seed = int(block_hash[:16], 16)
    rng = random.Random(seed)

    arch_uids = sorted(submissions.keys())
    trainer_pool = sorted(miner_uids)
    sorted_validators = sorted(validator_uids)

    jobs: list[Job] = []
    for i, arch_uid in enumerate(arch_uids):
        # Cross-eval: pick a different miner as trainer
        candidates = [u for u in trainer_pool if u != arch_uid]
        if not candidates:
            candidates = trainer_pool  # self-train if only one miner

        rng_copy = random.Random(seed + arch_uid)
        trainer_uid = rng_copy.choice(candidates)

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

        Dispatches concurrently and verifies Basilica attestation before
        dispatching to any trainer.
        """
        from validator.pod_manager import verify_miner_pod

        commitments = commitments or {}
        logger.info("Dispatching %d jobs to %d trainer endpoints", len(jobs), len(trainer_endpoints))

        # Verify each trainer pod ONCE before dispatching any jobs to it
        verified_trainers: dict[int, bool] = {}
        for job in jobs:
            tid = job.trainer_uid
            if tid in verified_trainers:
                continue
            url = trainer_endpoints.get(tid, "")
            commitment = commitments.get(tid)
            if not url or not commitment:
                verified_trainers[tid] = bool(url)  # has URL but no commitment = pass (localnet compat)
                continue
            instance_name = getattr(commitment, "pod_attestation_id", "") or ""
            if not instance_name:
                # No attestation committed — allow (localnet compatibility)
                verified_trainers[tid] = True
                continue
            ok, reason = await verify_miner_pod(
                instance_name=instance_name,
            )
            verified_trainers[tid] = ok
            if not ok:
                logger.warning("Trainer UID %d failed attestation: %s", tid, reason)

        # Per-job timeout: time_budget + 120s buffer for startup/upload overhead
        time_budget = challenge.task.get("time_budget", 300)
        job_timeout = time_budget + 120

        # Build tasks for concurrent dispatch
        immediate_results: list[TrainingResult] = []
        dispatch_tasks: list[tuple[Job, str, bytes, dict]] = []

        for job in jobs:
            if not verified_trainers.get(job.trainer_uid, False):
                immediate_results.append(TrainingResult(
                    round_id=job.round_id,
                    arch_owner=job.arch_owner,
                    trainer_uid=job.trainer_uid,
                    dispatcher=self.my_uid,
                    status="attestation_failed",
                    error="Trainer pod failed Basilica attestation check",
                ))
                continue
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

            # Generate presigned PUT URLs so trainer can upload without R2 creds
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
        """Poll R2 for training_meta.json from each miner."""
        from shared.artifacts import list_round_artifacts, meta_key as mk_fn

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

        while asyncio.get_event_loop().time() < deadline:
            # Use list_round_artifacts to find which miners have uploaded
            available = list_round_artifacts(self.r2, round_id)
            for hk in available:
                uid = hotkey_to_uid.get(hk)
                if uid is None or uid in results:
                    continue
                try:
                    meta = self.r2.download_json(mk_fn(round_id, hk))
                    if meta:
                        meta["miner_uid"] = uid
                        meta["miner_hotkey"] = hk
                        results[uid] = meta
                except Exception:
                    pass

            if len(results) >= len(expected_miners):
                break
            await asyncio.sleep(30)

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
