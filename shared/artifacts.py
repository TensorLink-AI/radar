"""Training artifact storage — upload, download, verify.

Handles the R2 storage protocol for training artifacts:
  round_{round_id}/miner_{hotkey}/checkpoint.safetensors
  round_{round_id}/miner_{hotkey}/architecture.py
  round_{round_id}/miner_{hotkey}/training_meta.json
  round_{round_id}/miner_{hotkey}/stdout.log

Hash verification chain: training_meta.json contains sha256 hashes
binding all artifact files together.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.r2_audit import R2AuditLog

logger = logging.getLogger(__name__)


# ── R2 path helpers ──────────────────────────────────────────────

def checkpoint_key(round_id: int, miner_hotkey: str) -> str:
    return f"round_{round_id}/miner_{miner_hotkey}/checkpoint.safetensors"


def architecture_key(round_id: int, miner_hotkey: str) -> str:
    return f"round_{round_id}/miner_{miner_hotkey}/architecture.py"


def meta_key(round_id: int, miner_hotkey: str) -> str:
    return f"round_{round_id}/miner_{miner_hotkey}/training_meta.json"


def stdout_key(round_id: int, miner_hotkey: str) -> str:
    return f"round_{round_id}/miner_{miner_hotkey}/stdout.log"


def scratchpad_key(miner_hotkey: str) -> str:
    """R2 key for a miner's persistent scratchpad archive."""
    return f"scratchpad/{miner_hotkey}/state.tar.gz"


def generate_scratchpad_urls(
    r2: "R2AuditLog",
    miner_hotkey: str,
    ttl: int = 900,
    max_size_bytes: int = 10 * 1024 * 1024,
) -> tuple[str, str]:
    """Generate presigned GET and PUT URLs for a miner's scratchpad.

    Returns (get_url, put_url). GET URL returns 404 if no scratchpad exists yet.
    PUT URL includes a Content-Length condition to enforce max_size_bytes.
    """
    key = scratchpad_key(miner_hotkey)
    get_url = r2.generate_presigned_get_url(key, ttl=ttl)
    put_url = r2.generate_presigned_put_url(
        key, ttl=ttl, max_content_length=max_size_bytes,
    )
    return get_url, put_url


# ── TrainingMeta ─────────────────────────────────────────────────

@dataclass
class TrainingMeta:
    """Metadata written alongside checkpoint after training."""
    round_id: int = 0
    miner_hotkey: str = ""
    status: str = ""                    # success | failed | timeout | build_failed | size_violation
    error: str = ""
    flops_equivalent_size: int = 0
    training_time_seconds: float = 0.0
    num_steps: int = 0
    num_params_M: float = 0.0
    loss_curve: list[float] = field(default_factory=list)
    peak_vram_mb: float = 0.0
    # Hash verification chain
    checkpoint_sha256: str = ""
    architecture_sha256: str = ""
    stdout_sha256: str = ""

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "miner_hotkey": self.miner_hotkey,
            "status": self.status,
            "error": self.error,
            "flops_equivalent_size": self.flops_equivalent_size,
            "training_time_seconds": self.training_time_seconds,
            "num_steps": self.num_steps,
            "num_params_M": self.num_params_M,
            "loss_curve": self.loss_curve,
            "peak_vram_mb": self.peak_vram_mb,
            "checkpoint_sha256": self.checkpoint_sha256,
            "architecture_sha256": self.architecture_sha256,
            "stdout_sha256": self.stdout_sha256,
        }

    @classmethod
    def from_json(cls, text: str) -> "TrainingMeta":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingMeta":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(text.encode()).hexdigest()


# ── Upload / download ────────────────────────────────────────────

def upload_training_artifacts(
    r2: "R2AuditLog",
    round_id: int,
    miner_hotkey: str,
    checkpoint_path: str,
    architecture_code: str,
    stdout_log: str,
    meta: TrainingMeta,
) -> bool:
    """Upload all training artifacts to R2 with hash verification chain.

    Computes sha256 of each artifact and stores in training_meta.json.
    """
    # Compute hashes
    meta.checkpoint_sha256 = sha256_file(checkpoint_path)
    meta.architecture_sha256 = sha256_text(architecture_code)
    meta.stdout_sha256 = sha256_text(stdout_log)

    ck = checkpoint_key(round_id, miner_hotkey)
    ak = architecture_key(round_id, miner_hotkey)
    mk = meta_key(round_id, miner_hotkey)
    sk = stdout_key(round_id, miner_hotkey)

    ok = True
    ok = r2.upload_file_from_disk(checkpoint_path, ck) and ok
    ok = r2.upload_text(ak, architecture_code) and ok
    ok = r2.upload_text(sk, stdout_log) and ok
    ok = r2.upload_json(mk, meta.to_dict()) and ok

    if ok:
        logger.info("Uploaded artifacts for round %d miner %s", round_id, miner_hotkey)
    else:
        logger.error("Some artifacts failed to upload for round %d miner %s", round_id, miner_hotkey)

    return ok


def generate_upload_urls(
    r2: "R2AuditLog",
    round_id: int,
    miner_hotkey: str,
    ttl: int = 5400,
) -> dict[str, str]:
    """Generate pre-signed PUT URLs for all training artifacts.

    TTL defaults to 5400s (~90 min). URLs are path-locked to the specific
    round/miner key.

    Returns dict mapping artifact name to presigned URL.
    """
    keys = {
        "checkpoint": checkpoint_key(round_id, miner_hotkey),
        "architecture": architecture_key(round_id, miner_hotkey),
        "meta": meta_key(round_id, miner_hotkey),
        "stdout": stdout_key(round_id, miner_hotkey),
    }
    urls = {}
    for name, key in keys.items():
        url = r2.generate_presigned_put_url(key, ttl=ttl)
        if url:
            urls[name] = url
        else:
            logger.error("Failed to generate presigned URL for %s", key)
    return urls


def verify_uploaded_artifacts(
    r2: "R2AuditLog",
    round_id: int,
    miner_hotkey: str,
) -> tuple[bool, str]:
    """Verify that expected artifacts exist in R2 after training upload.

    Checks that checkpoint and meta files exist at the expected keys.
    This catches cases where a presigned URL was used to write to an
    unexpected path (which S3 presigned URLs prevent, but defense-in-depth).

    Returns (ok, error_message).
    """
    ck = checkpoint_key(round_id, miner_hotkey)
    mk = meta_key(round_id, miner_hotkey)

    if not r2.key_exists(mk):
        return False, f"training_meta.json missing at {mk}"
    if not r2.key_exists(ck):
        return False, f"checkpoint missing at {ck}"

    # Verify meta contains correct round_id and miner_hotkey
    meta_dict = r2.download_json(mk)
    if not meta_dict:
        return False, "Failed to download training_meta.json for verification"

    if meta_dict.get("round_id") != round_id:
        return False, (
            f"Meta round_id mismatch: expected {round_id}, "
            f"got {meta_dict.get('round_id')}"
        )
    if meta_dict.get("miner_hotkey") != miner_hotkey:
        return False, (
            f"Meta miner_hotkey mismatch: expected {miner_hotkey}, "
            f"got {meta_dict.get('miner_hotkey')}"
        )

    return True, ""


def upload_training_artifacts_presigned(
    presigned_urls: dict[str, str],
    checkpoint_path: str,
    architecture_code: str,
    stdout_log: str,
    meta: "TrainingMeta",
) -> bool:
    """Upload training artifacts using pre-signed PUT URLs.

    No R2 credentials needed — uses time-limited presigned URLs.
    """
    import httpx

    # Compute hashes before upload
    meta.checkpoint_sha256 = sha256_file(checkpoint_path)
    meta.architecture_sha256 = sha256_text(architecture_code)
    meta.stdout_sha256 = sha256_text(stdout_log)

    ok = True

    # Upload checkpoint (binary file)
    if "checkpoint" in presigned_urls:
        try:
            with open(checkpoint_path, "rb") as f:
                resp = httpx.put(presigned_urls["checkpoint"], content=f.read(), timeout=120)
                resp.raise_for_status()
        except Exception as e:
            logger.error("Presigned upload failed for checkpoint: %s", e)
            ok = False
    else:
        logger.error("No presigned URL for checkpoint")
        ok = False

    # Upload architecture (text)
    if "architecture" in presigned_urls:
        try:
            resp = httpx.put(presigned_urls["architecture"], content=architecture_code.encode(), timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Presigned upload failed for architecture: %s", e)
            ok = False

    # Upload stdout (text)
    if "stdout" in presigned_urls:
        try:
            resp = httpx.put(presigned_urls["stdout"], content=stdout_log.encode(), timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Presigned upload failed for stdout: %s", e)
            ok = False

    # Upload meta (JSON) — must be last since it contains hashes
    if "meta" in presigned_urls:
        try:
            body = json.dumps(meta.to_dict(), indent=2).encode()
            resp = httpx.put(presigned_urls["meta"], content=body, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Presigned upload failed for meta: %s", e)
            ok = False

    if ok:
        logger.info("Uploaded artifacts via presigned URLs for miner %s", meta.miner_hotkey)
    return ok


@dataclass
class DownloadedArtifacts:
    """All artifacts downloaded from R2 for a single miner's training run."""
    meta: TrainingMeta
    checkpoint_path: str = ""    # local path where checkpoint was saved
    architecture_code: str = ""
    stdout_log: str = ""
    verified: bool = False
    verification_error: str = ""


def download_training_artifacts(
    r2: "R2AuditLog",
    round_id: int,
    miner_hotkey: str,
    download_dir: str,
) -> Optional[DownloadedArtifacts]:
    """Download all training artifacts from R2 and verify hashes.

    Returns None if meta can't be downloaded.
    Sets verified=False with verification_error if hash mismatch.
    """
    # Download meta
    mk = meta_key(round_id, miner_hotkey)
    meta_dict = r2.download_json(mk)
    if not meta_dict:
        return None

    meta = TrainingMeta.from_dict(meta_dict)

    # Download checkpoint
    ck = checkpoint_key(round_id, miner_hotkey)
    os.makedirs(download_dir, exist_ok=True)
    local_checkpoint = os.path.join(download_dir, f"checkpoint_{miner_hotkey}.safetensors")
    if not r2.download_file_to_disk(ck, local_checkpoint):
        return DownloadedArtifacts(
            meta=meta, verification_error="Failed to download checkpoint",
        )

    # Download architecture
    ak = architecture_key(round_id, miner_hotkey)
    architecture_code = r2.download_text(ak) or ""

    # Download stdout
    sk = stdout_key(round_id, miner_hotkey)
    stdout_log = r2.download_text(sk) or ""

    # Verify hashes
    result = DownloadedArtifacts(
        meta=meta,
        checkpoint_path=local_checkpoint,
        architecture_code=architecture_code,
        stdout_log=stdout_log,
    )

    errors = []
    if meta.checkpoint_sha256:
        actual = sha256_file(local_checkpoint)
        if actual != meta.checkpoint_sha256:
            errors.append(f"checkpoint hash mismatch: {actual[:16]}... vs {meta.checkpoint_sha256[:16]}...")

    if meta.architecture_sha256 and architecture_code:
        actual = sha256_text(architecture_code)
        if actual != meta.architecture_sha256:
            errors.append(f"architecture hash mismatch")

    if meta.stdout_sha256 and stdout_log:
        actual = sha256_text(stdout_log)
        if actual != meta.stdout_sha256:
            errors.append(f"stdout hash mismatch")

    if errors:
        result.verification_error = "; ".join(errors)
    else:
        result.verified = True

    return result


def list_round_artifacts(
    r2: "R2AuditLog",
    round_id: int,
) -> list[str]:
    """List all miner hotkeys that have training_meta.json for a round."""
    prefix = f"round_{round_id}/"
    keys = r2.list_keys(prefix)

    hotkeys = set()
    for key in keys:
        # round_{id}/miner_{hotkey}/training_meta.json
        if key.endswith("/training_meta.json"):
            parts = key.split("/")
            if len(parts) >= 3 and parts[1].startswith("miner_"):
                hotkeys.add(parts[1][len("miner_"):])

    return sorted(hotkeys)
