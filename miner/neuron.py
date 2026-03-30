"""Radar Miner — commits Docker agent image to chain, hosts warm-standby trainer.

The miner has two components:
  1. Agent: a Docker image committed to chain. Validators pull and run it.
  2. Trainer listener: a lightweight FastAPI server (no GPU). Deploys
     Basilica GPU pods on-demand when validators send TrainerRequests.
"""

import argparse
import asyncio
import logging
import os
import subprocess

import bittensor as bt
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import Config
from shared.auth import sign_request, verify_request
from shared.commitment import ImageCommitment, _commit_to_file
from shared.protocol import TrainerRequest, TrainerReady, TrainerRelease

logger = logging.getLogger(__name__)

listener_app = FastAPI(title="RADAR Miner Listener")


def _get_image_digest(image_url: str) -> str:
    """Get the sha256 digest of a local Docker image."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format",
             "{{index .RepoDigests 0}}", image_url],
            capture_output=True, text=True, timeout=30,
        )
        digest = result.stdout.strip()
        if "@" in digest:
            digest = digest.split("@")[1]
        return digest
    except Exception:
        logger.warning("Could not get image digest for %s", image_url)
        return ""


class Miner:
    """Radar subnet miner — Docker agent + warm-standby trainer listener."""

    def __init__(self, config: bt.Config):
        self.config = config
        self.netuid = config.netuid

        self.wallet = bt.Wallet(config=config)
        self.subtensor = bt.Subtensor(config=config)

        # Agent is a Docker image validators pull and run
        self.docker_image = getattr(config, "docker_image", "systematic:latest")

        # Trainer image deployed on-demand via Basilica
        self.trainer_image = getattr(
            config, "trainer_image", Config.OFFICIAL_TRAINING_IMAGE,
        )

        # Listener port for warm-standby HTTP server
        self.listener_port = int(getattr(config, "listener_port", 8090))
        self.external_ip = getattr(config, "external_ip", "") or "0.0.0.0"

        # Active Basilica deployments: round_id → deployment object
        self.active_deployments: dict[int, object] = {}

        logger.info(
            "Miner initialized. Agent: %s, Trainer image: %s, Listener port: %d",
            self.docker_image, self.trainer_image, self.listener_port,
        )

    def commit_image(self):
        """Commit Docker agent image + listener URL to chain."""
        digest = _get_image_digest(self.docker_image)

        commitment = ImageCommitment(
            image_url=self.docker_image,
            image_digest=digest,
            subnet_version="0.3.0",
            listener_url=f"http://{self.external_ip}:{self.listener_port}",
            trainer_image=self.trainer_image,
        )

        # Always write to file — ensures localnet and testnet work even if
        # chain commitment encoding is unreadable by get_commitment().
        _commit_to_file(self.wallet, self.netuid, commitment)

        try:
            self.subtensor.set_commitment(
                wallet=self.wallet,
                netuid=self.netuid,
                data=commitment.to_json(),
            )
            logger.info("Committed image to chain: %s", self.docker_image)
        except Exception as e:
            logger.warning("Chain commit failed (%s), file fallback already written", e)

    async def handle_prepare(self, request: TrainerRequest):
        """Deploy a Basilica GPU pod and POST TrainerReady back to validator."""
        logger.info(
            "Received TrainerRequest for round %d (gpu=%d x %dGB, memory=%s, budget=%ds)",
            request.round_id, request.gpu_count, request.min_gpu_memory_gb,
            request.memory, request.time_budget,
        )

        # Deduplicate — only one deployment per round
        if request.round_id in self.active_deployments:
            logger.info("Already deploying for round %d, skipping duplicate", request.round_id)
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
        deploy_name = f"radar-trainer-{request.round_id}"

        try:
            # Build env vars for the trainer pod
            # No R2 credentials — trainer uses presigned URLs for uploads
            pod_env = {}
            for key in ("SUBTENSOR_NETWORK", "NETUID"):
                val = os.environ.get(key, "")
                if val:
                    pod_env[key] = val

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

            # POST signed TrainerReady back to validator
            ready = TrainerReady(
                round_id=request.round_id,
                trainer_url=trainer_url,
                instance_name=deployment.name,
                miner_hotkey=hotkey,
            )
            body = ready.to_json().encode()
            headers = sign_request(self.wallet, body)
            headers["Content-Type"] = "application/json"
            url = f"{request.validator_db_url}/trainer/ready"
            logger.info("Posting TrainerReady to %s", url)
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.post(url, content=body, headers=headers)
                logger.info("TrainerReady response: HTTP %d", resp.status_code)

            self.active_deployments[request.round_id] = deployment
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
            # Verify Epistula signature from validator
            try:
                ok, err, hotkey = verify_request(
                    dict(request.headers), body,
                    miner.subtensor.metagraph(miner.netuid),
                )
                if not ok:
                    return JSONResponse(status_code=403, content={"error": err})
            except Exception:
                pass  # Allow unsigned requests for localnet compat

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
        """Start listener, commit image, and keep alive."""
        self.start_listener()
        self.commit_image()
        logger.info(
            "Miner running. Agent: %s (validators pull this image). "
            "Listener: port %d. Trainer image: %s",
            self.docker_image, self.listener_port, self.trainer_image,
        )

        while True:
            await asyncio.sleep(300)
            try:
                self.subtensor.metagraph(self.netuid)
            except Exception:
                pass


def get_config() -> bt.Config:
    parser = argparse.ArgumentParser(description="Radar Miner")
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--docker_image", type=str, default="systematic:latest",
                        help="Docker agent image (validators pull and run this)")
    parser.add_argument("--listener_port", type=int, default=8090,
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
