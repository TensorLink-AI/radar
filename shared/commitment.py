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

    # Warm-standby trainer (miner-hosted lightweight listener, no GPU)
    listener_url: str = ""               # always-on HTTP endpoint on miner's neuron process
    trainer_image: str = ""              # Docker image miner will deploy on Basilica when requested

    # Agent pod (miner-hosted on Basilica)
    agent_url: str = ""                  # Basilica pod URL for agent
    agent_attestation_id: str = ""       # Basilica attestation for agent pod

    # Internal fields (not committed to chain)
    miner_uid: int = -1
    hotkey: str = ""

    # Compact key mapping — keeps on-chain commitment under 256 bytes
    # so bittensor SDK can encode it (Raw256 max).
    _KEY_MAP = {
        "i": "image_url",
        "d": "image_digest",
        "v": "subnet_version",
        "l": "listener_url",
        "t": "trainer_image",
        "a": "agent_url",
        "at": "agent_attestation_id",
    }
    _REV_MAP = {v: k for k, v in _KEY_MAP.items()}

    def to_json(self) -> str:
        """Serialize to compact JSON (short keys, no empty values)."""
        data = {}
        for short, field in self._KEY_MAP.items():
            val = getattr(self, field, "")
            if val:
                data[short] = val
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str, miner_uid: int = -1, hotkey: str = "") -> ImageCommitment:
        d = json.loads(s)
        # Support both compact (short) and legacy (full) keys
        def _get(short: str, full: str) -> str:
            return d.get(short, d.get(full, ""))
        return cls(
            image_url=_get("i", "image_url"),
            image_digest=_get("d", "image_digest"),
            subnet_version=_get("v", "subnet_version"),
            listener_url=_get("l", "listener_url"),
            trainer_image=_get("t", "trainer_image"),
            agent_url=_get("a", "agent_url"),
            agent_attestation_id=_get("at", "agent_attestation_id"),
            miner_uid=miner_uid,
            hotkey=hotkey,
        )

    @property
    def is_valid(self) -> bool:
        """Basic validation: image URL and listener URL present."""
        return bool(self.image_url) and bool(self.listener_url)


def read_miner_commitments(subtensor, netuid: int, metagraph) -> dict[int, ImageCommitment]:
    """
    Read image commitments from all miners in the metagraph.

    Priority: file (localnet) → chain raw query → SDK get_commitment.

    Returns: {uid: ImageCommitment} for miners with valid commitments.
    """
    # Try file-based commitments first (localnet / same-machine)
    file_commitments = _read_from_files(netuid, metagraph)
    if file_commitments:
        logger.info("Found %d commitments from file fallback", len(file_commitments))
        return file_commitments

    hotkeys = metagraph.hotkeys if metagraph.hotkeys is not None else []
    commitments: dict[int, ImageCommitment] = {}

    # Try raw substrate query — bypasses SDK type decoder that chokes
    # on Raw241+ variants not in its type_mapping.
    raw_commitments = _read_from_chain_raw(subtensor, netuid, hotkeys, metagraph.n)
    if raw_commitments:
        logger.info("Found %d commitments from chain (raw query)", len(raw_commitments))
        return raw_commitments

    # Final fallback: SDK get_commitment (works for short commitments)
    bt_logger = logging.getLogger("bittensor")
    prev_level = bt_logger.level
    bt_logger.setLevel(logging.CRITICAL)
    try:
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
    finally:
        bt_logger.setLevel(prev_level)

    return commitments


def _read_from_chain_raw(
    subtensor, netuid: int, hotkeys: list, n: int,
) -> dict[int, ImageCommitment]:
    """Read commitments via raw substrate query, bypassing type decoder.

    The bittensor SDK's get_commitment() fails when commitment data is
    encoded as Raw241+ (not in type_mapping). We query the raw storage
    directly and decode the bytes ourselves.
    """
    commitments = {}
    substrate = getattr(subtensor, "substrate", None)
    if substrate is None:
        return commitments

    for uid in range(n):
        hotkey = hotkeys[uid] if uid < len(hotkeys) else ""
        if not hotkey:
            continue
        try:
            result = substrate.query(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[netuid, hotkey],
            )
            if result is None or result.value is None:
                continue
            # result.value is the commitment info map
            # Extract the 'info' field which contains the 'fields' list
            info = result.value
            if isinstance(info, dict):
                fields = info.get("info", {}).get("fields", [])
                if not fields:
                    fields = info.get("fields", [])
                # Each field is a dict like {"Raw241": "0x..."} or a string
                for field in fields:
                    text = _decode_raw_field(field)
                    if text:
                        try:
                            c = ImageCommitment.from_json(
                                text, miner_uid=uid, hotkey=hotkey,
                            )
                            if c.image_url:
                                commitments[uid] = c
                                break
                        except Exception:
                            pass
        except Exception as e:
            logger.debug("Raw chain query failed for UID %d: %s", uid, e)
    return commitments


def _decode_raw_field(field) -> str:
    """Decode a substrate Data enum field to a UTF-8 string.

    Fields come as dicts like {"Raw241": "0x7b2269..."} where the hex
    is the UTF-8 encoded commitment JSON.
    """
    if isinstance(field, str):
        # Already decoded string
        if field.startswith("{"):
            return field
        if field.startswith("0x"):
            try:
                return bytes.fromhex(field[2:]).decode("utf-8", errors="ignore")
            except Exception:
                return ""
        return field
    if isinstance(field, dict):
        for key, val in field.items():
            if key == "None" or val is None:
                continue
            if isinstance(val, str):
                if val.startswith("0x"):
                    try:
                        return bytes.fromhex(val[2:]).decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                if val.startswith("{"):
                    return val
                return val
    return ""


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
        subtensor.set_commitment(
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
