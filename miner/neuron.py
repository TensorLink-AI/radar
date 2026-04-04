"""Radar Miner — commits agent code hash to chain, hosts warm-standby trainer.

The miner has three components:
  1. Agent code: .py files served from the listener. Validators fetch and
     run them inside the official sandboxed agent image.
  2. Trainer listener: a lightweight FastAPI server (no GPU). Deploys
     Basilica GPU pods on-demand when validators send TrainerRequests.
  3. Agent code endpoint: GET /agent_code returns a JSON bundle of the
     miner's agent .py files plus a content hash for verification.
"""

import argparse
import asyncio
import json
import logging
import os

import bittensor as bt
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import Config
from shared.agent_code import bundle_from_directory, compute_code_hash
from shared.auth import sign_request, verify_request
from shared.commitment import ImageCommitment
from shared.protocol import TrainerRequest, TrainerReady, TrainerRelease

logger = logging.getLogger(__name__)

listener_app = FastAPI(title="RADAR Miner Listener")


class Miner:
    """Radar subnet miner — Docker agent + warm-standby trainer listener."""

    def __init__(self, config: bt.Config):
        self.config = config
        self.netuid = config.netuid

        self.wallet = bt.Wallet(config=config)
        self.subtensor = bt.Subtensor(config=config)

        # Cache metagraph — synced periodically in keep-alive loop
        self.metagraph = self.subtensor.metagraph(self.netuid)

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

        # Active Basilica deployments: round_id → deployment object
        self.active_deployments: dict[int, object] = {}

        # Validator db_urls to notify when deployment completes: round_id → [urls]
        self._pending_notify: dict[int, list[str]] = {}

        # Cached code hash — set after first submission
        self._code_hash: str = ""

        logger.info(
            "Miner initialized. Agent dir: %s, Trainer image: %s, Listener port: %d",
            self.agent_dir, self.trainer_image, self.listener_port,
        )

    def _get_proxy_url(self) -> str:
        """Resolve a validator proxy URL for API calls.

        Priority: RADAR_VALIDATOR_PROXY_URL env var > metagraph discovery.
        """
        explicit = os.environ.get("RADAR_VALIDATOR_PROXY_URL", "")
        if explicit:
            return explicit.rstrip("/")

        permits = self.metagraph.validator_permit
        axons = self.metagraph.axons
        if permits is None or axons is None:
            return ""

        proxy_port = int(os.environ.get("RADAR_PROXY_PORT", "8080"))
        for uid in range(self.metagraph.n):
            if uid < len(permits) and permits[uid] and uid < len(axons):
                axon = axons[uid]
                ip = getattr(axon, "ip", "") or ""
                if ip and ip != "0.0.0.0":
                    return f"http://{ip}:{proxy_port}"
        return ""

    async def submit_agent_code(self):
        """Submit agent code via DatabaseClient and commit hash on-chain."""
        if not os.path.isdir(self.agent_dir):
            logger.error("Agent directory not found: %s", self.agent_dir)
            return

        bundle = bundle_from_directory(self.agent_dir)
        code_hash = bundle["code_hash"]

        if code_hash == self._code_hash:
            logger.info("Agent code unchanged (hash=%s), skipping", code_hash[:24])
            return

        proxy_url = self._get_proxy_url()
        if not proxy_url:
            logger.warning(
                "No validator proxy found (set RADAR_VALIDATOR_PROXY_URL). "
                "Will retry on next heartbeat.",
            )
            self._code_hash = code_hash
            self._commit_to_chain(code_hash)
            return

        from shared.db_client import DatabaseClient
        db = DatabaseClient(db_url=proxy_url, wallet=self.wallet)
        try:
            result = await db.submit_agent_code(
                files=bundle["files"],
                entry_point=bundle["entry_point"],
            )
            if result:
                logger.info(
                    "Agent code submitted: hash=%s files=%s",
                    result.get("code_hash", "?")[:24],
                    sorted(bundle["files"].keys()),
                )
            else:
                logger.error("Agent code submission failed (no response)")
                return
        finally:
            await db.close()

        self._code_hash = code_hash
        self._commit_to_chain(code_hash)

    def _commit_to_chain(self, code_hash: str):
        """Commit code_hash + listener URL to chain."""
        commitment = ImageCommitment(
            code_hash=code_hash,
            subnet_version="0.3.0",
            listener_url=f"http://{self.external_ip}:{self.listener_port}",
            trainer_image=self.trainer_image,
        )

        chain_json = commitment.to_chain_json()
        logger.info("Committing to chain (%d bytes): %s", len(chain_json), chain_json)
        response = self.subtensor.set_commitment(
            wallet=self.wallet,
            netuid=self.netuid,
            data=chain_json,
        )
        if hasattr(response, "success") and not response.success:
            raise RuntimeError(f"Chain commit failed: {getattr(response, 'message', response)}")
        logger.info("Committed to chain: code_hash=%s", code_hash[:24])

    async def _post_trainer_ready(
        self, round_id: int, trainer_url: str,
        instance_name: str, validator_db_url: str,
    ):
        """POST signed TrainerReady to a validator's DB server."""
        hotkey = self.wallet.hotkey.ss58_address
        ready = TrainerReady(
            round_id=round_id,
            trainer_url=trainer_url,
            instance_name=instance_name,
            miner_hotkey=hotkey,
        )
        body = ready.to_json().encode()
        headers = sign_request(self.wallet, body)
        headers["Content-Type"] = "application/json"
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

        # Deduplicate — only one deployment per round, but notify all validators
        existing = self.active_deployments.get(request.round_id)
        if existing and existing != "pending":
            # Pod already deployed — just POST TrainerReady to this validator
            logger.info(
                "Already deployed for round %d, notifying validator at %s",
                request.round_id, request.validator_db_url,
            )
            await self._post_trainer_ready(
                request.round_id, existing.url, existing.name,
                request.validator_db_url,
            )
            return
        if existing == "pending":
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

        try:
            from basilica import BasilicaClient
            client = BasilicaClient()
        except ImportError:
            logger.error("basilica SDK not installed — cannot deploy trainer pod")
            return

        # TTL = allocation wait + training time + upload buffer
        # Pod auto-deletes after this even if release signal is missed
        ttl = 900 + int(request.time_budget) + 300  # 15 min alloc + training + 5 min upload
        deploy_timeout = 900  # 15 min max wait for GPU allocation
        hotkey = self.wallet.hotkey.ss58_address
        deploy_name = f"radar-trainer-{hotkey[:8]}-{request.round_id}"

        try:
            # Build env vars for the trainer pod
            # No R2 credentials — trainer uses presigned URLs for uploads
            pod_env = {}
            # Propagate network settings: try env vars first, then fall back
            # to the subtensor config (CLI args like --subtensor.network).
            network = os.environ.get("SUBTENSOR_NETWORK", "")
            if not network:
                network = getattr(self.subtensor, "network", "") or ""
            if network:
                pod_env["SUBTENSOR_NETWORK"] = network

            netuid = os.environ.get("NETUID", "")
            if not netuid:
                netuid = str(self.netuid)
            if netuid:
                pod_env["NETUID"] = netuid

            logger.info(
                "Deploying Basilica pod %s (image=%s, ttl=%ds, gpu=%d x %dGB min)",
                deploy_name, self.trainer_image, ttl, request.gpu_count, request.min_gpu_memory_gb,
            )

            # Run blocking Basilica SDK calls in executor to avoid blocking event loop
            import functools
            loop = asyncio.get_event_loop()

            deploy_kwargs = dict(
                name=deploy_name,
                image=self.trainer_image,
                port=8081,
                public=True,
                replicas=1,
                ttl_seconds=ttl,
                gpu_count=request.gpu_count,
                min_gpu_memory_gb=request.min_gpu_memory_gb,
                memory=request.memory,
                env=pod_env,
                timeout=deploy_timeout,
            )
            # Pin to specific GPU models if configured
            gpu_models_str = os.environ.get("RADAR_TRAINER_GPU_MODELS", "")
            if gpu_models_str:
                deploy_kwargs["gpu_models"] = [m.strip() for m in gpu_models_str.split(",") if m.strip()]

            deployment = await loop.run_in_executor(
                None, functools.partial(client.deploy, **deploy_kwargs),
            )

            trainer_url = deployment.url
            logger.info(
                "Basilica deployment ready: name=%s url=%s",
                deployment.name, trainer_url,
            )

            # Enroll for public metadata so validators can verify the pod
            try:
                await loop.run_in_executor(
                    None,
                    functools.partial(
                        client.enroll_metadata, deployment.name, enabled=True,
                    ),
                )
                logger.info("Public metadata enrolled for %s", deployment.name)
            except Exception as e:
                logger.warning("Failed to enroll public metadata for %s: %s", deployment.name, e)

            # POST signed TrainerReady back to requesting validator
            await self._post_trainer_ready(
                request.round_id, trainer_url, deployment.name,
                request.validator_db_url,
            )

            self.active_deployments[request.round_id] = deployment

            # Notify any validators that arrived while deployment was pending
            pending_urls = self._pending_notify.pop(request.round_id, [])
            for db_url in pending_urls:
                logger.info(
                    "Notifying queued validator at %s for round %d",
                    db_url, request.round_id,
                )
                await self._post_trainer_ready(
                    request.round_id, trainer_url, deployment.name, db_url,
                )
            logger.info(
                "Trainer ready for round %d at %s (instance=%s)",
                request.round_id, deployment.url, deployment.name,
            )
        except Exception as e:
            logger.error("Failed to deploy trainer for round %d: %s", request.round_id, e, exc_info=True)
            # Clean up failed deployment so it doesn't consume quota
            self.active_deployments.pop(request.round_id, None)
            try:
                from basilica import BasilicaClient
                BasilicaClient().delete_deployment(deploy_name)
                logger.info("Cleaned up failed deployment %s", deploy_name)
            except Exception:
                pass  # may not exist or already deleted

    async def handle_release(self, round_id: int):
        """Tear down the Basilica pod for a completed round."""
        deployment = self.active_deployments.pop(round_id, None)
        if deployment and deployment != "pending":
            try:
                deployment.delete()
                logger.info("Released trainer for round %d (deployment=%s)", round_id, deployment.name)
            except Exception as e:
                logger.debug("Teardown failed: %s (TTL will clean up)", e)

    def _setup_listener_routes(self):
        """Register listener endpoints on the FastAPI app."""
        miner = self

        @listener_app.post("/prepare")
        async def prepare(request: Request):
            body = await request.body()
            # Verify Epistula signature from validator (use cached metagraph)
            try:
                ok, err, hotkey = verify_request(
                    dict(request.headers), body,
                    miner.metagraph,
                )
                if not ok:
                    logger.warning(
                        "Auth failed on /prepare: %s (signed-by: %s)",
                        err,
                        request.headers.get("x-epistula-signed-by", "?")[:16],
                    )
                    return JSONResponse(status_code=403, content={"error": err})
            except Exception as exc:
                logger.debug("Auth verification exception (allowing): %s", exc)

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
        self.start_listener()
        await self.submit_agent_code()
        logger.info(
            "Miner running. Agent dir: %s. "
            "Listener: port %d. Trainer image: %s. "
            "Waiting for validator TrainerRequests on /prepare.",
            self.agent_dir, self.listener_port, self.trainer_image,
        )

        while True:
            await asyncio.sleep(300)
            try:
                self.metagraph = self.subtensor.metagraph(self.netuid)
                logger.info(
                    "Heartbeat: metagraph synced (%d neurons). "
                    "Active deployments: %d. Listening on port %d.",
                    self.metagraph.n,
                    len(self.active_deployments),
                    self.listener_port,
                )
                # Retry agent code submission if it failed at startup
                # (e.g. metagraph wasn't populated, no proxy found)
                if not self._code_hash:
                    await self.submit_agent_code()
            except Exception as e:
                logger.warning("Metagraph sync failed: %s", e)


def get_config() -> bt.Config:
    parser = argparse.ArgumentParser(description="Radar Miner")
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--agent_dir", type=str, default="agent/",
                        help="Directory containing agent .py files")
    parser.add_argument("--listener_port", type=int, default=Config.MINER_LISTENER_PORT,
                        help="Port for warm-standby trainer listener")
    parser.add_argument("--trainer_image", type=str,
                        default=Config.OFFICIAL_TRAINING_IMAGE,
                        help="Trainer Docker image deployed on Basilica on-demand")
    parser.add_argument("--external_ip", type=str, default="",
                        help="External IP for listener URL committed to chain")
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    return bt.Config(parser)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = get_config()
    miner = Miner(config)
    asyncio.run(miner.run())


if __name__ == "__main__":
    main()
