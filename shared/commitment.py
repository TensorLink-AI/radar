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
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ImageCommitment:
    """A miner's committed identity: agent code hash + listener URL.

    Agent code is served from the miner's listener (GET /agent_code).
    The code_hash is committed on-chain so validators can verify
    integrity after fetching.
    """

    # Agent code (new — replaces Docker image)
    code_hash: str = ""       # e.g. "sha256:abc123..." — hash of agent code bundle
    image_url: str = ""       # DEPRECATED: Docker image (kept for backward compat reads)
    image_digest: str = ""    # DEPRECATED

    subnet_version: str = ""  # e.g. "0.3.0" — must match official

    # Warm-standby trainer (miner-hosted lightweight listener, no GPU)
    listener_url: str = ""               # always-on HTTP endpoint on miner's neuron process
    trainer_image: str = ""              # Docker image miner will deploy on Basilica when requested

    # Agent pod (miner-hosted on Basilica)
    agent_url: str = ""                  # Basilica pod URL for agent
    agent_attestation_id: str = ""       # Basilica attestation for agent pod

    # Internal fields (not committed to chain)
    miner_uid: int = -1
    hotkey: str = ""

    # Compact key mapping for serialization.
    _KEY_MAP = {
        "i": "image_url",
        "d": "image_digest",
        "v": "subnet_version",
        "l": "listener_url",
        "t": "trainer_image",
        "a": "agent_url",
        "at": "agent_attestation_id",
        "ch": "code_hash",
    }
    _REV_MAP = {v: k for k, v in _KEY_MAP.items()}

    # On-chain limit is Raw128 (128 bytes). Only essential fields go on-chain.
    _CHAIN_KEYS = ("ch", "l", "v")
    _MAX_CHAIN_BYTES = 128

    def to_json(self) -> str:
        """Serialize to compact JSON (short keys, no empty values)."""
        data = {}
        for short, field in self._KEY_MAP.items():
            val = getattr(self, field, "")
            if val:
                data[short] = val
        return json.dumps(data, separators=(",", ":"))

    def to_chain_json(self) -> str:
        """Serialize for on-chain storage (<=128 bytes, essential fields only).

        The bittensor Commitments pallet Data enum supports Raw0-Raw128.
        Only image_url and listener_url (+ version if space allows) are included.
        Full data should be committed to file or fetched via the listener.
        """
        data: dict[str, str] = {}
        for short in self._CHAIN_KEYS:
            field = self._KEY_MAP[short]
            val = getattr(self, field, "")
            if val:
                data[short] = val
        result = json.dumps(data, separators=(",", ":"))
        # If still over limit, drop version to save space
        if len(result) > self._MAX_CHAIN_BYTES and "v" in data:
            del data["v"]
            result = json.dumps(data, separators=(",", ":"))
        return result

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
            code_hash=_get("ch", "code_hash"),
            miner_uid=miner_uid,
            hotkey=hotkey,
        )

    @property
    def is_valid(self) -> bool:
        """Basic validation: code_hash and listener_url present."""
        return bool(self.code_hash) and bool(self.listener_url)


def read_miner_commitments(subtensor, netuid: int, metagraph) -> dict[int, ImageCommitment]:
    """
    Read image commitments from all miners in the metagraph.

    Priority: get_all_commitments (single RPC) → per-UID fallback.

    Returns: {uid: ImageCommitment} for miners with valid commitments.
    """
    hotkeys = metagraph.hotkeys if metagraph.hotkeys is not None else []

    # Build hotkey→uid map and miner UID set
    hotkey_to_uid: dict[str, int] = {}
    validator_permits = metagraph.validator_permit
    miner_uids: set[int] = set()
    for uid in range(metagraph.n):
        if uid < len(hotkeys) and hotkeys[uid]:
            hotkey_to_uid[hotkeys[uid]] = uid
        is_validator = (
            validator_permits is not None
            and uid < len(validator_permits)
            and validator_permits[uid]
        )
        if not is_validator:
            miner_uids.add(uid)
    logger.info("Checking %d miner UIDs for commitments", len(miner_uids))

    # Primary: get_all_commitments — single RPC call, SDK handles decoding
    commitments = _read_all_commitments(
        subtensor, netuid, hotkey_to_uid, miner_uids,
    )
    if commitments:
        logger.info("Found %d commitments via get_all_commitments", len(commitments))
        return commitments

    # Fallback: per-UID get_commitment (slower, but works if map query fails)
    commitments = _read_per_uid_commitments(
        subtensor, netuid, hotkeys, sorted(miner_uids),
    )
    if commitments:
        logger.info("Found %d commitments via per-UID fallback", len(commitments))
    return commitments


def _read_all_commitments(
    subtensor, netuid: int,
    hotkey_to_uid: dict[str, int],
    miner_uids: set[int],
) -> dict[int, ImageCommitment]:
    """Read all commitments in a single RPC call via SDK get_all_commitments."""
    commitments: dict[int, ImageCommitment] = {}
    try:
        all_raw = subtensor.get_all_commitments(netuid=netuid)
    except Exception as e:
        logger.warning("get_all_commitments RPC failed: %s", e)
        return commitments

    miner_entries = 0
    for hotkey, raw_text in all_raw.items():
        uid = hotkey_to_uid.get(hotkey)
        if uid is None or uid not in miner_uids:
            continue
        miner_entries += 1
        if not raw_text:
            logger.warning("UID %d: commitment on chain is empty", uid)
            continue
        try:
            c = ImageCommitment.from_json(raw_text, miner_uid=uid, hotkey=hotkey)
            if c.code_hash:
                commitments[uid] = c
            else:
                logger.warning(
                    "UID %d: commitment has no code_hash (raw=%s)",
                    uid, raw_text[:120],
                )
        except Exception as e:
            logger.warning("UID %d: bad commitment JSON: %s (raw=%s)", uid, e, raw_text[:120])
    logger.info(
        "get_all_commitments: %d total on chain, %d miner entries, %d valid",
        len(all_raw), miner_entries, len(commitments),
    )
    return commitments


def _read_per_uid_commitments(
    subtensor, netuid: int, hotkeys: list, miner_uids: list[int],
) -> dict[int, ImageCommitment]:
    """Per-UID commitment reads via get_commitment_metadata + decode."""
    commitments: dict[int, ImageCommitment] = {}
    bt_logger = logging.getLogger("bittensor")
    prev_level = bt_logger.level
    bt_logger.setLevel(logging.CRITICAL)
    try:
        for uid in miner_uids:
            hotkey = hotkeys[uid] if uid < len(hotkeys) else ""
            if not hotkey:
                continue
            try:
                metadata = subtensor.get_commitment_metadata(
                    netuid=netuid, hotkey_ss58=hotkey,
                )
                if not metadata:
                    logger.info("UID %d: no commitment on chain", uid)
                    continue
                if not isinstance(metadata, dict):
                    logger.warning(
                        "UID %d: commitment metadata is %s, not dict: %s",
                        uid, type(metadata).__name__, str(metadata)[:120],
                    )
                    continue
                # Use SDK decoding — handles current on-chain byte tuple format
                from bittensor.core.chain_data.utils import decode_metadata
                raw = decode_metadata(metadata)
                if raw:
                    c = ImageCommitment.from_json(raw, miner_uid=uid, hotkey=hotkey)
                    if c.code_hash:
                        commitments[uid] = c
                    else:
                        logger.warning(
                            "UID %d: per-UID commitment has no code_hash (raw=%s)",
                            uid, raw[:120],
                        )
            except Exception as e:
                logger.warning("Per-UID commitment read failed for UID %d: %s", uid, e)
    finally:
        bt_logger.setLevel(prev_level)
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
    chain_json = commitment.to_chain_json()
    subtensor.set_commitment(
        wallet=wallet,
        netuid=netuid,
        data=chain_json,
    )
    logger.info("Committed image to chain: %s @ %s", image_url, image_digest[:16])
    return True
