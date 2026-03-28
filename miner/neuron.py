"""Radar Miner — commits Docker agent image to chain, hosts warm-standby trainer.

The miner has two components:
  1. Agent: a Docker image committed to chain. Validators pull and run it.
  2. Trainer listener: a lightweight FastAPI server (no GPU). Deploys
     Basilica GPU pods on-demand when validators send TrainerRequests.
"""

import argparse
import asyncio
import logging
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
            listener_url=f"http://0.0.0.0:{self.listener_port}",
            trainer_image=self.trainer_image,
        )

        try:
            self.subtensor.commit(
                wallet=self.wallet,
                netuid=self.netuid,
                data=commitment.to_json(),
            )
            logger.info("Committed image to chain: %s", self.docker_image)
        except Exception as e:
            logger.warning("Chain commit failed (%s), using file fallback", e)
            _commit_to_file(self.wallet, self.netuid, commitment)

    async def handle_prepare(self, request: TrainerRequest):
        """Deploy a Basilica GPU pod and POST TrainerReady back to validator."""
        try:
            from basilica import BasilicaClient
            client = BasilicaClient()
        except ImportError:
            logger.error("basilica SDK not installed — cannot deploy trainer pod")
            return

        ttl = int(request.time_budget * Config.TRAINER_RELEASE_SAFETY_MARGIN)
        hotkey = self.wallet.hotkey.ss58_address
        instance_name = f"radar-trainer-r{request.round_id}-{hotkey[:8]}"

        try:
            deployment = client.create_deployment(
                instance_name=instance_name,
                image=self.trainer_image,
                port=8001,
                replicas=1,
                public_metadata=True,
                ttl_seconds=ttl,
                gpu_count=request.gpu_count,
                gpu_models=[request.gpu_model],
                memory=request.memory,
            )
            deployment.wait_until_ready()

            # POST signed TrainerReady back to validator
            ready = TrainerReady(
                round_id=request.round_id,
                trainer_url=deployment.url,
                instance_name=deployment.instance_name,
                miner_hotkey=hotkey,
            )
            body = ready.to_json().encode()
            headers = sign_request(self.wallet, body)
            headers["Content-Type"] = "application/json"
            async with httpx.AsyncClient(timeout=30) as http:
                await http.post(
                    f"{request.validator_db_url}/trainer/ready",
                    content=body,
                    headers=headers,
                )

            self.active_deployments[request.round_id] = deployment
            logger.info(
                "Trainer ready for round %d at %s (instance=%s)",
                request.round_id, deployment.url, instance_name,
            )
        except Exception as e:
            logger.error("Failed to deploy trainer for round %d: %s", request.round_id, e)

    async def handle_release(self, round_id: int):
        """Tear down the Basilica pod for a completed round."""
        deployment = self.active_deployments.pop(round_id, None)
        if deployment:
            try:
                from basilica import BasilicaClient
                client = BasilicaClient()
                client.delete_deployment(deployment.instance_name)
                logger.info("Released trainer for round %d", round_id)
            except Exception:
                pass  # TTL will clean it up anyway

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
        """Commit image, start listener, and keep alive."""
        self.commit_image()
        self.start_listener()
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
