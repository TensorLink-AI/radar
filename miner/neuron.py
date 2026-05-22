"""Radar Miner — hosts warm-standby trainer + serves agent code.

The miner has two components:
  1. Agent code: .py files submitted to the central DB. Validators
     fetch and run them inside the official sandboxed agent image.
  2. Trainer listener: a lightweight FastAPI server (no GPU). Deploys
     trainer pods on demand when validators send TrainerRequests.
"""

import argparse
import asyncio
import json
import logging
import os
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import Config
from miner.hosting import (
    Deployment, TargonReadinessTimeout, deploy_basilica, deploy_targon,
    get_targon_registry_creds, teardown_basilica_sync,
    teardown_targon_with_retry, wait_for_trainer_ready,
)
from miner.hosting_runpod import (
    RunpodReadinessTimeout, deploy_runpod, submit_dispatch_to_runpod,
)
from miner.runpod_lifecycle import (
    cancel_jobs_for_round, make_runpod_client,
    validate_credentials as validate_runpod_credentials,
)
from miner.targon_lifecycle import (
    HealthMonitorRegistry, install_shutdown_handlers,
    make_targon_client, validate_and_reap_orphans,
)
from shared.agent_code import bundle_from_directory
from shared.auth import hmac_sign_request, hmac_verify_request, static_key_lookup
from shared.protocol import TrainerRequest, TrainerReady, TrainerRelease

logger = logging.getLogger(__name__)

listener_app = FastAPI(title="RADAR Miner Listener")


class Miner:
    """Radar subnet miner — Docker agent + warm-standby trainer listener."""

    def __init__(self, config):
        self.config = config

        # Miner identity is a free-form label (no chain hotkey).  Used
        # only to namespace agent code in the DB and to tag outbound
        # log lines.
        self.miner_id = (
            getattr(config, "miner_id", "")
            or os.getenv("RADAR_MINER_ID", "")
            or "miner"
        )

        # Per-miner bearer token used to authenticate to the DB (issued
        # via the operator CLI).  Required to submit agent code.
        self.api_key = os.getenv("RADAR_MINER_API_KEY", "").strip()

        # Fail-fast on Targon misconfig — operator gets a clear error at
        # startup instead of a cryptic deploy failure on the first round.
        if Config.HOSTING_BACKEND == "targon" and not os.environ.get("TARGON_API_KEY"):
            raise RuntimeError(
                "RADAR_HOSTING_BACKEND=targon but TARGON_API_KEY is not set. "
                "Set the env var to the validator/miner's Targon account API key."
            )
        if Config.HOSTING_BACKEND == "targon" and not Config.OFFICIAL_TRAINING_IMAGE_DIGEST:
            raise RuntimeError(
                "RADAR_HOSTING_BACKEND=targon but OFFICIAL_TRAINING_IMAGE_DIGEST is empty. "
                "Without a pinned digest validators cannot verify the deployed image — "
                "set OFFICIAL_TRAINING_IMAGE_DIGEST=sha256:... to the subnet-owner-blessed value."
            )
        # Same fail-fast for RunPod. The endpoint id is per-miner (RunPod
        # endpoints are account-scoped) and must be pre-provisioned with
        # the official trainer image; without it there's nothing to
        # dispatch jobs into.
        if Config.HOSTING_BACKEND == "runpod" and not os.environ.get("RUNPOD_API_KEY"):
            raise RuntimeError(
                "RADAR_HOSTING_BACKEND=runpod but RUNPOD_API_KEY is not set. "
                "Issue an API key at https://www.runpod.io/console/user/settings."
            )
        if Config.HOSTING_BACKEND == "runpod" and not Config.RUNPOD_ENDPOINT_ID:
            raise RuntimeError(
                "RADAR_HOSTING_BACKEND=runpod but RADAR_RUNPOD_ENDPOINT_ID is empty. "
                "Pre-provision a RunPod Serverless endpoint with the official trainer "
                "image and set RADAR_RUNPOD_ENDPOINT_ID to its id."
            )

        # Shared HMAC service key — used to verify inbound validator
        # /prepare requests and to sign outbound TrainerReady POSTs.
        if not Config.SERVICE_KEY:
            raise RuntimeError(
                "RADAR_SERVICE_KEY must be set — the miner verifies "
                "validator /prepare requests and signs TrainerReady "
                "posts with the shared HMAC service key."
            )
        self._service_secret = Config.SERVICE_KEY.encode()
        self._service_key_id = Config.SERVICE_KEY_ID

        self._targon_client = None
        self._runpod_client = None
        # Per-round RunPod job ids the miner has submitted on behalf of
        # validators. Cancelled at end-of-round / shutdown.
        # round_id → list[job_id]
        self._runpod_jobs: dict[int, list[str]] = {}

        # Agent code directory — validators fetch the code and run it
        # in the official sandboxed image
        self.agent_dir = getattr(config, "agent_dir", "agent/")

        # Trainer image deployed on-demand via Basilica
        self.trainer_image = getattr(
            config, "trainer_image", Config.OFFICIAL_TRAINING_IMAGE,
        )

        # Listener port for warm-standby HTTP server
        self.listener_port = int(getattr(config, "listener_port", Config.MINER_LISTENER_PORT))
        self.external_ip = getattr(config, "external_ip", "") or "0.0.0.0"

        # Active deployments: round_id → (deployment, created_at, ttl)
        self.active_deployments: dict[int, tuple[object, float, int]] = {}
        # Targon-only — defensive logging via HealthMonitor per round.
        self._health_monitors = HealthMonitorRegistry()
        # Set by SIGTERM/SIGINT so the run loop exits cleanly.
        self._shutdown_event: asyncio.Event | None = None

        # Validator db_urls to notify when deployment completes: round_id → [urls]
        self._pending_notify: dict[int, list[str]] = {}

        # Cached code hash — set after first submission
        self._code_hash: str = ""

        logger.info(
            "Miner initialized. Agent dir: %s, Trainer image: %s, Listener port: %d",
            self.agent_dir, self.trainer_image, self.listener_port,
        )

    async def submit_agent_code(self):
        """Submit agent code to the DB server."""
        if not os.path.isdir(self.agent_dir):
            logger.error("Agent directory not found: %s", self.agent_dir)
            return

        bundle = bundle_from_directory(self.agent_dir)
        code_hash = bundle["code_hash"]

        if code_hash == self._code_hash:
            logger.info("Agent code unchanged (hash=%s), skipping", code_hash[:24])
            return

        db_url = Config.DB_API_URL
        if not db_url:
            logger.error("RADAR_DB_API_URL not set — cannot submit agent code")
            return
        if not self.api_key:
            logger.error(
                "RADAR_MINER_API_KEY is not set — cannot submit agent code. "
                "Issue one via `python -m database.operator_cli issue-key`.",
            )
            return
        from shared.db_client import DatabaseClient
        db = DatabaseClient(db_url=db_url, api_key=self.api_key)
        listener_url = self._listener_external_url()
        try:
            result = await db.submit_agent_code(
                files=bundle["files"],
                entry_point=bundle["entry_point"],
                listener_url=listener_url,
            )
            if result:
                logger.info(
                    "Agent code submitted to %s: hash=%s listener=%s files=%s",
                    db_url, result.get("code_hash", "?")[:24],
                    listener_url, sorted(bundle["files"].keys()),
                )
            else:
                logger.error("Agent code submission failed")
                return
        finally:
            await db.close()

        self._code_hash = code_hash

    async def heartbeat_listener(self):
        """Refresh listener_url on the DB so validators keep seeing us.

        The /miners/active endpoint filters on listener_seen_at, so
        without periodic re-registration the miner drops out of the
        active set after the TTL window even if it's still online.
        """
        db_url = Config.DB_API_URL
        if not db_url or not self.api_key:
            return
        listener_url = self._listener_external_url()
        from shared.db_client import DatabaseClient
        db = DatabaseClient(db_url=db_url, api_key=self.api_key)
        try:
            # register_listener returns the parsed body on success and
            # None on any HTTP/transport failure (db_client logs the
            # underlying warning). Log both outcomes explicitly so a
            # silently-failing heartbeat doesn't hide behind the bare
            # "Heartbeat:" tick log.
            result = await db.register_listener(listener_url)
            if result is None:
                logger.warning(
                    "Listener heartbeat FAILED url=%s db=%s "
                    "(see DatabaseClient warning above)",
                    listener_url, db_url,
                )
            else:
                logger.info(
                    "Listener heartbeat OK url=%s db=%s",
                    listener_url, db_url,
                )
        except Exception as e:
            logger.warning(
                "Listener heartbeat raised url=%s db=%s err=%s",
                listener_url, db_url, e,
            )
        finally:
            await db.close()

    async def _post_trainer_ready(
        self, round_id: int, deployment: Deployment, validator_db_url: str,
    ):
        """POST a signed TrainerReady to a validator's proxy.

        Carries backend-specific metadata when present; empty strings
        for backends that don't use a given field keep the wire format
        backwards-compatible.
        """
        ready = TrainerReady(
            round_id=round_id,
            trainer_url=deployment.url,
            instance_name=deployment.name,
            miner_hotkey=self.miner_id,
            targon_workload_uid=deployment.targon_workload_uid,
            cvm_ip=deployment.cvm_ip,
            gpu_class=deployment.gpu_class,
            deployed_image_digest=deployment.deployed_image_digest,
            runpod_endpoint_id=deployment.runpod_endpoint_id,
            runpod_template_id=deployment.runpod_template_id,
        )
        body = ready.to_json().encode()
        headers = hmac_sign_request(
            self._service_secret, body, key_id=self._service_key_id,
        )
        headers["Content-Type"] = "application/json"
        headers["X-Miner-UID"] = str(getattr(self, "miner_uid", -1))
        url = f"{validator_db_url}/trainer/ready"
        logger.info("Posting TrainerReady to %s", url)
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.post(url, content=body, headers=headers)
                logger.info("TrainerReady response: HTTP %d", resp.status_code)
        except Exception as e:
            logger.warning("Failed to post TrainerReady to %s: %s", url, e)

    async def handle_prepare(self, request: TrainerRequest):
        """Deploy a Basilica GPU pod and POST TrainerReady back to validator."""
        logger.info(
            "Received TrainerRequest for round %d (gpu=%d x %dGB, memory=%s, budget=%ds)",
            request.round_id, request.gpu_count, request.min_gpu_memory_gb,
            request.memory, request.time_budget,
        )

        # Tear down all deployments from prior rounds — this round supersedes them
        await self._teardown_prior_rounds(request.round_id)

        # Deduplicate — only one deployment per round, but notify all validators
        entry = self.active_deployments.get(request.round_id)
        if entry and entry != "pending":
            deployment, _ts, _ttl = entry
            logger.info(
                "Already deployed for round %d, notifying validator at %s",
                request.round_id, request.validator_db_url,
            )
            await self._post_trainer_ready(
                request.round_id, deployment, request.validator_db_url,
            )
            return
        if entry == "pending":
            # Deployment in progress — queue this validator for notification when ready
            logger.info(
                "Deployment in progress for round %d, queuing notification for %s",
                request.round_id, request.validator_db_url,
            )
            self._pending_notify.setdefault(request.round_id, []).append(
                request.validator_db_url,
            )
            return

        # Mark as in-progress to prevent duplicate deploys
        self.active_deployments[request.round_id] = "pending"

        # TTL covers Phase A wait + training + upload buffer.
        submission_wait = int(getattr(request, "submission_window_seconds", 0) or 600)
        ttl = submission_wait + int(request.time_budget) + 600
        hotkey = self.miner_id
        network = os.environ.get("RADAR_NETWORK", "") or ""

        deploy_start = time.time()
        try:
            deployment = await self._deploy_one(request, hotkey, network, ttl)
        except Exception as e:
            logger.error(
                "DEPLOY_FAILED round=%d backend=%s elapsed=%.1fs error=%s",
                request.round_id, Config.HOSTING_BACKEND, time.time() - deploy_start, e,
                exc_info=True,
            )
            self.active_deployments.pop(request.round_id, None)
            self._pending_notify.pop(request.round_id, None)
            return

        logger.info(
            "DEPLOY_OK round=%d backend=%s uid=%s url=%s elapsed=%.1fs",
            request.round_id, Config.HOSTING_BACKEND,
            deployment.targon_workload_uid or deployment.name,
            deployment.url, time.time() - deploy_start,
        )

        # On RunPod: deploy_runpod() already polled the endpoint to
        # readiness, no need for an extra wait_for here. The miner
        # listener (deployment.url) was up before the request landed
        # by definition.

        # On Targon: wait for /health and CVM evidence endpoint before
        # advertising the trainer. Tear down on timeout so we don't leak
        # a half-ready workload.
        if Config.HOSTING_BACKEND == "targon":
            try:
                ready_start = time.time()
                await wait_for_trainer_ready(
                    trainer_url=deployment.url,
                    cvm_ip=deployment.cvm_ip,
                    timeout_s=Config.TARGON_READINESS_TIMEOUT_S,
                )
                logger.info(
                    "READINESS_OK round=%d uid=%s elapsed=%.1fs",
                    request.round_id, deployment.targon_workload_uid,
                    time.time() - ready_start,
                )
            except TargonReadinessTimeout as e:
                logger.error(
                    "READINESS_TIMEOUT round=%d uid=%s: %s — tearing down",
                    request.round_id, deployment.targon_workload_uid, e,
                )
                await teardown_targon_with_retry(
                    self._get_targon_client(), deployment.targon_workload_uid,
                )
                self.active_deployments.pop(request.round_id, None)
                self._pending_notify.pop(request.round_id, None)
                return

        # Tear down the previous round's workload synchronously before
        # we mark this one active. Targon bills by uptime; leaks cost
        # real money.
        await self._teardown_previous_round(request.round_id)

        try:
            await self._post_trainer_ready(
                request.round_id, deployment, request.validator_db_url,
            )
            self.active_deployments[request.round_id] = (deployment, time.time(), ttl)

            for db_url in self._pending_notify.pop(request.round_id, []):
                logger.info(
                    "Notifying queued validator at %s for round %d",
                    db_url, request.round_id,
                )
                await self._post_trainer_ready(request.round_id, deployment, db_url)
            logger.info(
                "TRAINER_READY round=%d uid=%s url=%s instance=%s backend=%s",
                request.round_id,
                deployment.targon_workload_uid or "-",
                deployment.url, deployment.name, Config.HOSTING_BACKEND,
            )
            if Config.HOSTING_BACKEND == "targon" and deployment.targon_workload_uid:
                self._health_monitors.start(request.round_id, deployment)
        except Exception as e:
            logger.error(
                "POST_READY_FAILED round=%d uid=%s error=%s",
                request.round_id, deployment.targon_workload_uid, e, exc_info=True,
            )
            self.active_deployments.pop(request.round_id, None)
            self._pending_notify.pop(request.round_id, None)

    async def _teardown_previous_round(self, current_round_id: int) -> None:
        """Tear down all rounds older than the current one. Sync, with retry on Targon."""
        prior_rounds = [
            rid for rid, entry in self.active_deployments.items()
            if rid < current_round_id and entry != "pending"
        ]
        for rid in prior_rounds:
            await self._health_monitors.stop(rid)
            entry = self.active_deployments.pop(rid)
            deployment, _ts, _ttl = entry
            if deployment.targon_workload_uid:
                ok = await teardown_targon_with_retry(
                    self._get_targon_client(), deployment.targon_workload_uid,
                )
                if not ok:
                    logger.error(
                        "PRIOR_TEARDOWN_LEAK round=%d uid=%s — workload may bill until TTL",
                        rid, deployment.targon_workload_uid,
                    )
            else:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, teardown_basilica_sync, deployment.raw)
                except Exception as e:
                    logger.warning(
                        "Prior-round Basilica teardown failed for round %d: %s", rid, e,
                    )

    async def _deploy_one(
        self, request: TrainerRequest, hotkey: str, network: str, ttl: int,
    ) -> Deployment:
        """Pick backend by Config.HOSTING_BACKEND: basilica | targon | runpod."""
        if Config.HOSTING_BACKEND == "targon":
            gpu_class = (Config.TRAINER_GPU_MODELS.split(",")[0].strip()
                         if Config.TRAINER_GPU_MODELS else "H200")
            return await deploy_targon(
                targon_client=self._get_targon_client(),
                request=request,
                image=self.trainer_image,
                deployed_image_digest=Config.OFFICIAL_TRAINING_IMAGE_DIGEST,
                hotkey=hotkey,
                netuid=0,
                subtensor_network=network,
                gpu_class=gpu_class,
                registry=get_targon_registry_creds(),
            )
        if Config.HOSTING_BACKEND == "runpod":
            gpu_class = (Config.TRAINER_GPU_MODELS.split(",")[0].strip()
                         if Config.TRAINER_GPU_MODELS else "")
            # Listener URL is what the validator dispatches /train to;
            # the listener relays into RunPod with the miner's API key.
            listener_url = self._listener_external_url()
            return await deploy_runpod(
                runpod_client=self._get_runpod_client(),
                endpoint_id=Config.RUNPOD_ENDPOINT_ID,
                listener_url=listener_url,
                deployed_image_digest=Config.OFFICIAL_TRAINING_IMAGE_DIGEST,
                gpu_class=gpu_class,
                request=request,
                readiness_timeout_s=Config.RUNPOD_READINESS_TIMEOUT_S,
            )
        return await deploy_basilica(
            request=request,
            image=self.trainer_image,
            hotkey=hotkey,
            netuid=0,
            subtensor_network=network,
            ttl=ttl,
        )

    def _listener_external_url(self) -> str:
        host = self.external_ip or "0.0.0.0"
        return f"http://{host}:{self.listener_port}"

    def _get_targon_client(self):
        if not getattr(self, "_targon_client", None):
            self._targon_client = make_targon_client()
        return self._targon_client

    def _get_runpod_client(self):
        if not getattr(self, "_runpod_client", None):
            self._runpod_client = make_runpod_client()
        return self._runpod_client

    async def _targon_startup_check(self) -> None:
        await validate_and_reap_orphans(self._get_targon_client())

    async def _runpod_startup_check(self) -> None:
        await validate_runpod_credentials(self._get_runpod_client())

    async def _teardown_all_active(self) -> None:
        """Synchronous-feeling teardown of every active deployment."""
        active = list(self.active_deployments.items())
        for rid, entry in active:
            if entry == "pending":
                self.active_deployments.pop(rid, None)
                continue
            deployment, _ts, _ttl = entry
            self.active_deployments.pop(rid, None)
            try:
                await self._teardown(deployment)
                logger.info("SHUTDOWN_TEARDOWN round=%d ok", rid)
            except Exception as e:
                logger.warning("SHUTDOWN_TEARDOWN round=%d failed: %s", rid, e)

    async def handle_release(self, round_id: int):
        """Tear down the trainer pod for a completed round."""
        await self._health_monitors.stop(round_id)
        entry = self.active_deployments.pop(round_id, None)
        if entry and entry != "pending":
            deployment = entry[0]
            try:
                await self._teardown(deployment)
                logger.info("Released trainer for round %d (deployment=%s)", round_id, deployment.name)
            except Exception as e:
                logger.debug("Teardown failed: %s (TTL will clean up)", e)

    async def _teardown(self, deployment: Deployment) -> None:
        """Backend-aware teardown.

        Targon path retries 3 times with exponential backoff — the workload
        bills by uptime, so quietly leaking it costs the operator real money.
        RunPod path cancels any per-round jobs the miner submitted but
        leaves the endpoint alive (endpoints persist across rounds).
        """
        if deployment.targon_workload_uid:
            await teardown_targon_with_retry(
                self._get_targon_client(), deployment.targon_workload_uid,
            )
            return
        if deployment.runpod_endpoint_id:
            # Any in-flight jobs for this round get cancelled; the
            # endpoint itself stays up because RunPod manages worker
            # lifecycle and we'll reuse it next round.
            jobs = self._collect_runpod_jobs_for_deployment(deployment)
            if jobs:
                await cancel_jobs_for_round(
                    self._get_runpod_client(),
                    deployment.runpod_endpoint_id,
                    jobs,
                )
            return
        # Basilica deployment objects expose a sync .delete() — run in executor.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, teardown_basilica_sync, deployment.raw)

    def _collect_runpod_jobs_for_deployment(self, deployment: Deployment) -> list[str]:
        """Pull job ids submitted under any round bound to this deployment.

        Looks up `_runpod_jobs` by the deployment's round (encoded in the
        instance name) — caller knows the round id, but `_teardown`
        doesn't, so we walk and remove any bucket whose endpoint id
        matches and whose name suffix matches.
        """
        jobs: list[str] = []
        for rid in list(self._runpod_jobs.keys()):
            entry = self.active_deployments.get(rid)
            if isinstance(entry, tuple) and entry[0].name == deployment.name:
                jobs.extend(self._runpod_jobs.pop(rid, []))
        return jobs

    async def _teardown_prior_rounds(self, current_round_id: int):
        """Delete prior-round deployments whose TTL has elapsed."""
        now = time.time()
        expired = [
            rid for rid, entry in self.active_deployments.items()
            if rid != current_round_id
            and entry != "pending"
            and now >= entry[1] + entry[2]
        ]
        for rid in expired:
            deployment, created, _ttl = self.active_deployments.pop(rid)
            try:
                await self._teardown(deployment)
                logger.info(
                    "Tore down expired deployment: round %d (%s, %.0f min old)",
                    rid, deployment.name, (now - created) / 60,
                )
            except Exception as e:
                logger.debug("Teardown failed for round %d: %s (TTL will clean up)", rid, e)

    async def _reap_stale_deployments(self):
        """Heartbeat safety-net: delete any deployment past its TTL."""
        now = time.time()
        expired = [
            rid for rid, entry in self.active_deployments.items()
            if entry != "pending" and now >= entry[1] + entry[2]
        ]
        for rid in expired:
            deployment, created, _ttl = self.active_deployments.pop(rid)
            try:
                await self._teardown(deployment)
                logger.info(
                    "Reaped expired deployment for round %d (%s, %.0f min old)",
                    rid, deployment.name, (now - created) / 60,
                )
            except Exception as e:
                logger.debug("Reap failed for round %d: %s (TTL will clean up)", rid, e)

    def _setup_listener_routes(self):
        """Register listener endpoints on the FastAPI app."""
        miner = self

        verify_lookup = static_key_lookup(
            miner._service_key_id, miner._service_secret,
        )

        @listener_app.post("/prepare")
        async def prepare(request: Request):
            body = await request.body()
            ok, err, _kid = hmac_verify_request(
                dict(request.headers), body, verify_lookup,
            )
            if not ok:
                logger.warning("Auth failed on /prepare: %s", err)
                return JSONResponse(status_code=403, content={"error": err})

            try:
                req = TrainerRequest.from_json(body.decode())
            except Exception as e:
                return JSONResponse(
                    status_code=400, content={"error": f"Bad request: {e}"},
                )

            # Deploy async — don't block the response
            asyncio.create_task(miner.handle_prepare(req))
            return {"status": "accepted", "round_id": req.round_id}

        @listener_app.post("/release")
        async def release(request: Request):
            body = await request.body()
            try:
                data = TrainerRelease.from_json(body.decode())
            except Exception as e:
                return JSONResponse(
                    status_code=400, content={"error": f"Bad request: {e}"},
                )

            asyncio.create_task(miner.handle_release(data.round_id))
            return {"status": "ok", "round_id": data.round_id}

        @listener_app.get("/health")
        def health():
            return {"status": "ok"}

        @listener_app.post("/train")
        async def train(request: Request):
            """Relay /train into RunPod when the miner runs that backend.

            Validators dispatch to ``deployment.url`` (the listener URL
            for RunPod). The listener verifies the validator's Epistula
            signature, submits the payload as ``input`` to the miner's
            RunPod endpoint with the miner's API key, and returns 202.
            The actual training runs on a RunPod worker; the worker
            uploads to R2 via the presigned URLs in the payload, and
            the validator polls R2 for completion just like Basilica /
            Targon today.

            This route is a no-op (404 from the validator's POV) for
            Basilica / Targon backends — those dispatch to the trainer
            pod directly, not the miner listener.
            """
            if Config.HOSTING_BACKEND != "runpod":
                return JSONResponse(
                    status_code=404,
                    content={"error": "miner listener does not relay /train for this backend"},
                )
            body = await request.body()
            ok, err, signed_by = hmac_verify_request(
                dict(request.headers), body, verify_lookup,
            )
            if not ok:
                logger.warning("Auth failed on /train: %s", err)
                return JSONResponse(status_code=403, content={"error": err})

            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

            round_id = int(payload.get("round_id", 0))
            entry = miner.active_deployments.get(round_id)
            if not isinstance(entry, tuple):
                return JSONResponse(
                    status_code=409,
                    content={"error": f"no active RunPod deployment for round {round_id}"},
                )
            deployment, _ts, _ttl = entry
            if not deployment.runpod_endpoint_id:
                return JSONResponse(
                    status_code=409,
                    content={"error": f"round {round_id} not running on RunPod"},
                )

            existing = miner._runpod_jobs.get(round_id, [])
            if existing:
                logger.info(
                    "RunPod relay: round=%d already submitted job=%s — returning idempotently",
                    round_id, existing[0],
                )
                return JSONResponse(
                    status_code=202,
                    content={"status": "accepted", "round_id": round_id, "job_id": existing[0]},
                )

            job_id = await submit_dispatch_to_runpod(
                runpod_client=miner._get_runpod_client(),
                endpoint_id=deployment.runpod_endpoint_id,
                payload=payload,
            )
            if not job_id:
                return JSONResponse(
                    status_code=503,
                    content={"error": "RunPod unavailable", "reason": "submit_failed"},
                )
            miner._runpod_jobs.setdefault(round_id, []).append(job_id)
            logger.info(
                "RunPod relay: round=%d validator=%s job=%s",
                round_id, (signed_by or "?")[:16], job_id,
            )
            return JSONResponse(
                status_code=202,
                content={"status": "accepted", "round_id": round_id, "job_id": job_id},
            )

    def start_listener(self):
        """Start the listener HTTP server in a background thread."""
        self._setup_listener_routes()

        import threading

        def _run():
            uvicorn.run(
                listener_app, host="0.0.0.0", port=self.listener_port,
                log_level="warning",
            )
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        logger.info("Listener started on port %d", self.listener_port)

    async def run(self):
        """Start listener, submit agent code, and keep alive."""
        # Targon startup: validate creds + reap any leaked workloads from
        # a prior crashed process before we start subscribing to validators.
        if Config.HOSTING_BACKEND == "targon":
            await self._targon_startup_check()
        elif Config.HOSTING_BACKEND == "runpod":
            await self._runpod_startup_check()

        self._shutdown_event = asyncio.Event()
        install_shutdown_handlers(
            asyncio.get_event_loop(),
            self._shutdown_event,
            self._teardown_all_active,
        )
        self.start_listener()
        await self.submit_agent_code()
        logger.info(
            "Miner running. Agent dir: %s. "
            "Listener: port %d. Trainer image: %s. "
            "Waiting for validator TrainerRequests on /prepare.",
            self.agent_dir, self.listener_port, self.trainer_image,
        )

        while True:
            try:
                # Sleep up to 300s but wake immediately on shutdown signal.
                if self._shutdown_event is not None:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=300)
                else:
                    await asyncio.sleep(300)
                # Reaching here means the event fired — exit the loop.
                logger.info("Shutdown event set — exiting run loop")
                return
            except asyncio.TimeoutError:
                pass  # normal heartbeat tick

            try:
                await self._reap_stale_deployments()
                logger.info(
                    "Heartbeat: active deployments=%d, listening on port %d",
                    len(self.active_deployments), self.listener_port,
                )
                if not self._code_hash:
                    await self.submit_agent_code()
                else:
                    await self.heartbeat_listener()
            except Exception as e:
                logger.warning("Heartbeat tick failed: %s", e)


def get_config():
    parser = argparse.ArgumentParser(description="Radar Miner")
    parser.add_argument("--miner_id", type=str, default="",
                        help="Free-form miner label (defaults to RADAR_MINER_ID env)")
    parser.add_argument("--agent_dir", type=str, default="agent/",
                        help="Directory containing agent .py files")
    parser.add_argument("--listener_port", type=int, default=Config.MINER_LISTENER_PORT,
                        help="Port for warm-standby trainer listener")
    parser.add_argument("--trainer_image", type=str,
                        default=Config.OFFICIAL_TRAINING_IMAGE,
                        help="Trainer Docker image deployed on demand")
    parser.add_argument("--external_ip", type=str, default="",
                        help="External IP for the listener URL")
    return parser.parse_args()


def main():
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Subcommand router: ``results`` / ``optimize`` / ``prompts`` go to
    # ``miner/cli.py``.  Anything else (or no args) falls through to the
    # legacy ``run`` behavior so existing deployments keep working.
    from miner import cli
    if cli.is_subcommand(sys.argv):
        sys.exit(cli.dispatch(sys.argv))
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        del sys.argv[1]

    config = get_config()
    miner = Miner(config)
    asyncio.run(miner.run())


if __name__ == "__main__":
    main()
