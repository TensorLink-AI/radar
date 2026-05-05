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
import secrets
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
    """A training job assignment.

    ``submission_id`` is an opaque per-job ID minted by the dispatching
    validator at dispatch time (left empty for jobs we don't dispatch).
    It hides the architecture owner's hotkey from the trainer-host (which
    is itself another miner's pod under cross-eval).
    """
    arch_owner: int      # miner whose architecture is being trained
    trainer_uid: int     # miner whose trainer runs the job (self-training allowed)
    dispatcher: int      # validator UID responsible for dispatching
    round_id: int = 0
    submission_id: str = ""


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
    submission_id: str = ""


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
                    submission_id=job.submission_id,
                ))
                continue

            # Mint an opaque submission_id per job at dispatch time.
            # The trainer-host receives only this ID — the miner's hotkey
            # is never on the wire to the trainer or in the bucket path.
            # `secrets.token_hex(16)` → 32-char unguessable hex.
            if not job.submission_id:
                job.submission_id = secrets.token_hex(16)
            submission_id = job.submission_id

            # Generate presigned PUT URLs so trainer can upload without R2 creds.
            from shared.artifacts import generate_upload_urls
            presigned_ttl = int(os.getenv("RADAR_PRESIGNED_TTL", "5400"))
            upload_urls = generate_upload_urls(
                self.r2, challenge.round_id, submission_id, ttl=presigned_ttl,
            )

            if not upload_urls.get("architecture"):
                logger.error(
                    "Missing architecture presigned URL for arch_owner=%d submission=%s",
                    job.arch_owner, submission_id[:12],
                )
            else:
                logger.info(
                    "Generated %d presigned URLs for arch_owner=%d submission=%s: %s",
                    len(upload_urls), job.arch_owner, submission_id[:12],
                    list(upload_urls.keys()),
                )

            task_name = challenge.task.get("name", "")
            runner_dir = challenge.task.get("runner_dir", "")
            dispatch_payload = {
                "architecture": proposal.code,
                "seed": challenge.seed,
                "round_id": challenge.round_id,
                "min_flops_equivalent": challenge.min_flops_equivalent,
                "max_flops_equivalent": challenge.max_flops_equivalent,
                "submission_id": submission_id,
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
                    # is always within the 30s tolerance window.
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
                            submission_id=job.submission_id,
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
                            submission_id=job.submission_id,
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
                            submission_id=job.submission_id,
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
                        submission_id=job.submission_id,
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
                        submission_id=job.submission_id,
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
                        submission_id=job.submission_id,
                    )
            # Should not reach here, but handle defensively
            return TrainingResult(
                round_id=job.round_id, arch_owner=job.arch_owner,
                trainer_uid=job.trainer_uid, dispatcher=self.my_uid,
                status="failed", error="Retries exhausted",
                submission_id=job.submission_id,
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
        uid_to_submission_id: dict[int, str],
        timeout: int = 300,
    ) -> dict[int, dict]:
        """Poll R2/Hippius for training_meta.json by submission_id.

        Artifacts live at ``round_{id}/submission_{sid}/...``. Resolution
        from miner UID → submission_id comes from the dispatching
        validator's in-memory job state plus other validators' published
        dispatch records — assembled by the caller and passed in via
        ``uid_to_submission_id``.

        UIDs missing from the mapping are skipped (no submission_id known
        yet — typically jobs dispatched by another validator that hasn't
        written its dispatch record). Successful collections include the
        miner's hotkey/uid in the returned meta so the rest of the
        validator pipeline keeps working unchanged.
        """
        from shared.artifacts import meta_key as mk_fn, checkpoint_key as ck_fn

        results: dict[int, dict] = {}
        deadline = asyncio.get_event_loop().time() + timeout

        # Build uid→hotkey mapping for stamping the result so downstream
        # scoring/eval still has the resolved hotkey.
        hotkeys = self.metagraph.hotkeys if hasattr(self.metagraph, "hotkeys") else []

        # Log expected keys for debugging
        for uid in expected_miners:
            sid = uid_to_submission_id.get(uid, "")
            if not sid:
                logger.info(
                    "UID %d: no submission_id known yet — skipping until reveal",
                    uid,
                )
                continue
            logger.info(
                "Expecting meta key for UID %d: %s", uid, mk_fn(round_id, sid),
            )

        poll_count = 0
        while asyncio.get_event_loop().time() < deadline:
            poll_count += 1
            remaining = deadline - asyncio.get_event_loop().time()
            logger.info(
                "Checkpoint poll #%d: %d/%d collected, %.0fs remaining (round %d)",
                poll_count, len(results), len(expected_miners), remaining, round_id,
            )
            for uid in expected_miners:
                if uid in results:
                    continue
                sid = uid_to_submission_id.get(uid, "")
                if not sid:
                    continue
                hk = hotkeys[uid] if uid < len(hotkeys) else f"uid_{uid}"
                mk = mk_fn(round_id, sid)
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

                    if meta.get("round_id") != round_id:
                        logger.warning(
                            "Meta round_id mismatch for UID %d: expected %d, got %s",
                            uid, round_id, meta.get("round_id"),
                        )
                        continue
                    if meta.get("submission_id") != sid:
                        logger.warning(
                            "Meta submission_id mismatch for UID %d: expected %s, got %s",
                            uid, sid[:12], str(meta.get("submission_id"))[:12],
                        )
                        continue

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

                    ck = ck_fn(round_id, sid)
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
            round_prefix = f"round_{round_id}/"
            try:
                existing_keys = self.r2.list_keys(round_prefix)
                logger.warning(
                    "Checkpoint collection timed out: %d/%d collected after %ds (round %d). "
                    "Keys under %s: %s",
                    len(results), len(expected_miners), timeout, round_id,
                    round_prefix, existing_keys[:20] if existing_keys else "(none)",
                )
            except Exception:
                logger.warning(
                    "Checkpoint collection timed out: %d/%d collected after %ds (round %d). "
                    "list_keys also failed — check connectivity",
                    len(results), len(expected_miners), timeout, round_id,
                )

        return results

    async def write_dispatch_record(self, round_id: int, results: list[TrainingResult]):
        """Write dispatch record (and the submission_id reveal map for our jobs).

        Each record carries (arch_owner, submission_id, miner_hotkey) so any
        validator can resolve every checkpoint they need for Phase C eval
        — the dispatching validator is the only entity that knows the
        opaque submission_id minted at dispatch.
        """
        hotkey = self.wallet.hotkey.ss58_address
        key = f"round_{round_id}/dispatch/vali_{hotkey}.json"
        hotkeys = self.metagraph.hotkeys if hasattr(self.metagraph, "hotkeys") else []
        records = []
        for r in results:
            arch_hk = (
                hotkeys[r.arch_owner]
                if 0 <= r.arch_owner < len(hotkeys) else ""
            )
            records.append({
                "arch_owner": r.arch_owner,
                "arch_owner_hotkey": arch_hk,
                "trainer_uid": r.trainer_uid,
                "submission_id": r.submission_id,
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

    async def collect_submission_map(
        self, round_id: int,
    ) -> dict[int, str]:
        """Read every validator's dispatch record for ``round_id`` and
        assemble the union ``arch_owner_uid → submission_id`` map.

        Used for Phase C cross-eval: every validator must independently
        download every checkpoint, so each needs the submission_ids minted
        by the *other* validators. Best-effort — missing records mean a
        UID falls out of the map and its checkpoint is skipped this round.
        """
        prefix = f"round_{round_id}/dispatch/"
        try:
            keys = self.r2.list_keys(prefix)
        except Exception as e:
            logger.warning("Failed to list dispatch records for round %d: %s", round_id, e)
            return {}
        mapping: dict[int, str] = {}
        for key in keys:
            if not key.endswith(".json"):
                continue
            try:
                body = self.r2.download_json(key)
            except Exception as e:
                logger.warning("Failed to read dispatch record %s: %s", key, e)
                continue
            if not isinstance(body, dict):
                continue
            for job in body.get("jobs", []) or []:
                if not isinstance(job, dict):
                    continue
                sid = job.get("submission_id", "")
                uid = job.get("arch_owner")
                if isinstance(uid, int) and sid and uid not in mapping:
                    mapping[uid] = sid
        return mapping

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
        from config import Config
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
        # stale timestamps (>30s) cause miners to reject with 403.
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
