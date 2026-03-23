"""EvoLoop Miner — commits Docker agent image to chain, hosts trainer on Basilica.

The miner has two components:
  1. Agent: a Docker image committed to chain. Validators pull and run it.
  2. Trainer: the sanctioned training image hosted on Basilica.
"""

import argparse
import asyncio
import logging
import subprocess

import bittensor as bt

from config import Config
from shared.commitment import commit_image_to_chain, ImageCommitment, _commit_to_file

logger = logging.getLogger(__name__)


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
    """EvoLoop subnet miner — Docker agent + Basilica trainer."""

    def __init__(self, config: bt.Config):
        self.config = config
        self.netuid = config.netuid

        self.wallet = bt.Wallet(config=config)
        self.subtensor = bt.Subtensor(config=config)

        # Agent is a Docker image validators pull and run
        self.docker_image = getattr(config, "docker_image", "systematic:latest")

        # Trainer is the sanctioned image hosted on Basilica
        self.trainer_url = getattr(config, "trainer_url", "")

        logger.info(
            "Miner initialized. Agent image: %s, Trainer URL: %s",
            self.docker_image, self.trainer_url or "(not set)",
        )

    def commit_image(self):
        """Commit Docker agent image + trainer URL to chain."""
        digest = _get_image_digest(self.docker_image)

        commitment = ImageCommitment(
            image_url=self.docker_image,
            image_digest=digest,
            subnet_version="0.2.0",
            pod_url=self.trainer_url,
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

    async def run(self):
        """Commit image and keep alive."""
        self.commit_image()
        logger.info(
            "Miner running. Agent: %s (validators pull this image). "
            "Trainer: %s",
            self.docker_image, self.trainer_url or "(not set)",
        )

        while True:
            await asyncio.sleep(300)
            try:
                self.subtensor.metagraph(self.netuid)
            except Exception:
                pass


def get_config() -> bt.Config:
    parser = argparse.ArgumentParser(description="EvoLoop Miner")
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--docker_image", type=str, default="systematic:latest",
                        help="Docker agent image (validators pull and run this)")
    parser.add_argument("--trainer_url", type=str, default="",
                        help="Basilica URL for sanctioned trainer pod")
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
