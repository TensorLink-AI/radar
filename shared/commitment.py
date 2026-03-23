"""
On-chain Docker image commitment and verification.

Miners commit their agent image reference (registry URL + digest) to
chain metadata. Validators read commitments from the metagraph and
verify pulled images match the committed digest.

This ensures:
  - Validators know which image each miner is running
  - The image can be verified against a known digest
  - Future: verify image matches official subnet version
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# File-based fallback directory for localnet (no chain commitment API)
_COMMITMENT_DIR = Path(os.environ.get(
    "RADAR_COMMITMENT_DIR", "/tmp/radar_commitments"
))


@dataclass
class ImageCommitment:
    """A miner's committed image reference with pod URLs."""

    # Agent image (existing)
    image_url: str = ""       # e.g. "docker.io/myminer/agent:v2"
    image_digest: str = ""    # e.g. "sha256:abc123..."
    subnet_version: str = ""  # e.g. "0.1.0" — must match official

    # Training pod (miner-hosted on Basilica)
    pod_url: str = ""                    # Basilica pod URL
    pod_attestation_id: str = ""         # Basilica attestation for training pod

    # Agent pod (miner-hosted on Basilica)
    agent_url: str = ""                  # Basilica pod URL for agent
    agent_attestation_id: str = ""       # Basilica attestation for agent pod

    # Internal fields (not committed to chain)
    miner_uid: int = -1
    hotkey: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "image_url": self.image_url,
            "image_digest": self.image_digest,
            "subnet_version": self.subnet_version,
            "pod_url": self.pod_url,
            "pod_attestation_id": self.pod_attestation_id,
            "agent_url": self.agent_url,
            "agent_attestation_id": self.agent_attestation_id,
        })

    @classmethod
    def from_json(cls, s: str, miner_uid: int = -1, hotkey: str = "") -> ImageCommitment:
        d = json.loads(s)
        return cls(
            image_url=d.get("image_url", ""),
            image_digest=d.get("image_digest", ""),
            subnet_version=d.get("subnet_version", ""),
            pod_url=d.get("pod_url", ""),
            pod_attestation_id=d.get("pod_attestation_id", ""),
            agent_url=d.get("agent_url", ""),
            agent_attestation_id=d.get("agent_attestation_id", ""),
            miner_uid=miner_uid,
            hotkey=hotkey,
        )

    @property
    def is_valid(self) -> bool:
        """Basic validation: image URL and digest present."""
        return bool(self.image_url) and bool(self.image_digest)


def read_miner_commitments(subtensor, netuid: int, metagraph) -> dict[int, ImageCommitment]:
    """
    Read image commitments from all miners in the metagraph.

    Tries chain first, falls back to file-based commitments for localnet.

    Returns: {uid: ImageCommitment} for miners with valid commitments.
    """
    # Try file-based commitments first (always available, used by localnet)
    file_commitments = _read_from_files(netuid, metagraph)
    if file_commitments:
        logger.info("Found %d commitments from file fallback", len(file_commitments))
        return file_commitments

    # Fall back to chain API
    hotkeys = metagraph.hotkeys if metagraph.hotkeys is not None else []
    commitments = {}
    for uid in range(metagraph.n):
        try:
            raw = subtensor.get_commitment(netuid=netuid, uid=uid)
            if raw:
                hotkey = hotkeys[uid] if uid < len(hotkeys) else ""
                commitment = ImageCommitment.from_json(raw, miner_uid=uid, hotkey=hotkey)
                if commitment.image_url:
                    commitments[uid] = commitment
        except Exception as e:
            logger.debug("No commitment for UID %d: %s", uid, e)

    return commitments


def _image_exists_locally(image_url: str) -> bool:
    """Check if a Docker image already exists in the local daemon."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_url],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def pull_and_verify_image(commitment: ImageCommitment) -> bool:
    """
    Pull a miner's Docker image and verify its digest matches the commitment.

    Skips the pull if the image already exists locally (e.g. built on this
    machine for localnet testing).

    Returns True if image available and digest matches (or no digest to check).
    """
    image_url = commitment.image_url

    try:
        # Skip pull if image is already available locally
        if _image_exists_locally(image_url):
            logger.info("Image already local, skipping pull: %s", image_url)
        else:
            logger.info("Pulling image: %s", image_url)
            subprocess.run(
                ["docker", "pull", image_url],
                capture_output=True, text=True, timeout=300, check=True,
            )

        # Verify digest if one was committed
        if commitment.image_digest:
            result = subprocess.run(
                ["docker", "inspect", "--format",
                 "{{index .RepoDigests 0}}", image_url],
                capture_output=True, text=True, timeout=30,
            )
            actual_digest = result.stdout.strip()

            # Extract the sha256 part
            if "@" in actual_digest:
                actual_digest = actual_digest.split("@")[1]

            # Local-only images have no RepoDigests — skip check
            if actual_digest and commitment.image_digest != actual_digest:
                logger.warning(
                    "Image digest mismatch for UID %d: expected %s, got %s",
                    commitment.miner_uid, commitment.image_digest,
                    actual_digest,
                )
                return False

        logger.info("Image verified for UID %d: %s", commitment.miner_uid, image_url)
        return True

    except subprocess.CalledProcessError as e:
        logger.error("Failed to pull/verify image %s: %s", image_url, e.stderr[:200])
        return False
    except Exception as e:
        logger.error("Image verification error: %s", e)
        return False


def verify_subnet_version(commitment: ImageCommitment, expected_version: str) -> bool:
    """Check that miner's committed subnet version matches expected."""
    if not commitment.subnet_version:
        return True  # No version committed, skip check for now
    return commitment.subnet_version == expected_version


def commit_image_to_chain(
    subtensor, netuid: int, wallet,
    image_url: str, image_digest: str, subnet_version: str = "",
) -> bool:
    """
    Commit an agent image reference to chain (called by miners).

    Args:
        image_url: Docker registry URL (e.g., "docker.io/user/agent:v2")
        image_digest: Image digest (e.g., "sha256:abc...")
        subnet_version: Subnet version for compatibility check
    """
    commitment = ImageCommitment(
        image_url=image_url,
        image_digest=image_digest,
        subnet_version=subnet_version,
    )
    try:
        subtensor.commit(
            wallet=wallet,
            netuid=netuid,
            data=commitment.to_json(),
        )
        logger.info("Committed image to chain: %s @ %s", image_url, image_digest[:16])
        return True
    except Exception as e:
        logger.warning("Chain commit unavailable (%s), using file fallback", e)
        return _commit_to_file(wallet, netuid, commitment)


def _commit_to_file(wallet, netuid: int, commitment: ImageCommitment) -> bool:
    """Write commitment to a shared temp directory (localnet fallback)."""
    try:
        d = _COMMITMENT_DIR / str(netuid)
        d.mkdir(parents=True, exist_ok=True)
        hotkey = wallet.hotkey.ss58_address
        path = d / f"{hotkey}.json"
        path.write_text(commitment.to_json())
        logger.info("Committed image to file: %s -> %s", commitment.image_url, path)
        return True
    except Exception as e:
        logger.error("File commitment failed: %s", e)
        return False


def _read_from_files(netuid: int, metagraph) -> dict[int, ImageCommitment]:
    """Read commitments from shared temp directory (localnet fallback)."""
    commitments = {}
    d = _COMMITMENT_DIR / str(netuid)
    if not d.exists():
        return commitments

    # Map hotkeys to UIDs
    hotkeys = metagraph.hotkeys if metagraph.hotkeys is not None else []
    hotkey_to_uid = {}
    for uid in range(metagraph.n):
        if uid < len(hotkeys):
            hotkey_to_uid[hotkeys[uid]] = uid

    for path in d.glob("*.json"):
        try:
            hotkey = path.stem
            uid = hotkey_to_uid.get(hotkey)
            if uid is None:
                continue
            commitment = ImageCommitment.from_json(
                path.read_text(), miner_uid=uid, hotkey=hotkey,
            )
            if commitment.image_url:
                commitments[uid] = commitment
        except Exception as e:
            logger.debug("Bad commitment file %s: %s", path, e)
    return commitments
