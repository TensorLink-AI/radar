"""Training artifact storage — upload, download, verify.

Handles the R2/Hippius storage protocol for training artifacts:
  round_{round_id}/submission_{submission_id}/checkpoint.safetensors
  round_{round_id}/submission_{submission_id}/architecture.py
  round_{round_id}/submission_{submission_id}/training_meta.json
  round_{round_id}/submission_{submission_id}/stdout.log

`submission_id` is an opaque per-round per-job ID minted by the dispatching
validator. It hides the miner's hotkey from the trainer-host (which is itself
a miner's pod under cross-eval). The validator publishes the
``submission_id → miner_hotkey`` reveal map after Phase C closes — see
``database/server.py:/round_submissions/reveal``.

Hash verification chain: training_meta.json contains sha256 hashes
binding all artifact files together.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re as _re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.r2_audit import R2AuditLog

logger = logging.getLogger(__name__)


# ── Path helpers ─────────────────────────────────────────────────

_SAFE_ID = _re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_submission_id(submission_id: str) -> str:
    """Validate submission_id contains no path-traversal characters."""
    if not submission_id or not _SAFE_ID.match(submission_id):
        raise ValueError(f"Invalid submission_id (unsafe characters): {submission_id!r}")
    return submission_id


def _validate_hotkey(miner_hotkey: str) -> str:
    """Validate hotkey contains no path-traversal characters."""
    if not _SAFE_ID.match(miner_hotkey):
        raise ValueError(f"Invalid miner_hotkey (unsafe characters): {miner_hotkey!r}")
    return miner_hotkey


def checkpoint_key(round_id: int, submission_id: str) -> str:
    _validate_submission_id(submission_id)
    return f"round_{round_id}/submission_{submission_id}/checkpoint.safetensors"


def architecture_key(round_id: int, submission_id: str) -> str:
    _validate_submission_id(submission_id)
    return f"round_{round_id}/submission_{submission_id}/architecture.py"


def meta_key(round_id: int, submission_id: str) -> str:
    _validate_submission_id(submission_id)
    return f"round_{round_id}/submission_{submission_id}/training_meta.json"


def stdout_key(round_id: int, submission_id: str) -> str:
    _validate_submission_id(submission_id)
    return f"round_{round_id}/submission_{submission_id}/stdout.log"


def scratchpad_key(miner_hotkey: str) -> str:
    """R2 key for a miner's persistent scratchpad archive.

    Scratchpads stay keyed by hotkey — they're miner-private state the
    miner reads/writes themselves through their own creds; no cross-miner
    anonymity concern applies.
    """
    _validate_hotkey(miner_hotkey)
    return f"scratchpad/{miner_hotkey}/state.tar.gz"


def generate_scratchpad_urls(
    r2: "R2AuditLog",
    miner_hotkey: str,
    ttl: int = 1800,
) -> tuple[str, str]:
    """Generate presigned GET and PUT URLs for a miner's scratchpad.

    Returns (get_url, put_url). GET URL returns 404 if no scratchpad exists yet.
    Size limit is enforced agent-side in save_scratchpad — signing a
    Content-Length into the PUT URL would force the upload to match it
    exactly (S3/R2 signature check), breaking every smaller upload with 403.
    """
    key = scratchpad_key(miner_hotkey)
    get_url = r2.generate_presigned_get_url(key, ttl=ttl)
    put_url = r2.generate_presigned_put_url(key, ttl=ttl)
    return get_url, put_url


# ── TrainingMeta ─────────────────────────────────────────────────

@dataclass
class TrainingMeta:
    """Metadata written alongside checkpoint after training.

    The trainer fills ``submission_id`` (from the dispatch payload). The
    validator only resolves ``submission_id → miner_hotkey`` via its own
    in-memory dispatch state and the post-Phase-C reveal map.
    """
    round_id: int = 0
    submission_id: str = ""
    status: str = ""                    # success | failed | timeout | build_failed | size_violation
    error: str = ""
    flops_equivalent_size: int = 0
    training_time_seconds: float = 0.0
    num_steps: int = 0
    num_params_M: float = 0.0
    peak_vram_mb: float = 0.0
    # Loss tracking (Phase B). Self-reported by the miner pod —
    # NOT a scoring trust anchor (Phase C remains the only one).
    train_loss_history: list[dict] = field(default_factory=list)  # [{step: int, loss: float}, ...]
    val_loss_history: list[dict] = field(default_factory=list)
    best_val_loss: float | None = None
    best_val_step: int = -1
    # Validation schedule policy — persisted so loss trajectories from different
    # model sizes / batch sizes / seq lens can be compared on a shared FLOPs axis.
    val_cadence_unit: str = "step"          # "step" for legacy, "flops" new
    val_base: float = 0.0                   # FLOPs or steps depending on unit
    val_growth: float = 0.0
    val_eval_tokens: int = 0
    flops_per_step_estimate: float = 0.0
    reference_eval_loss_history: list[dict] = field(default_factory=list)
    # Hash verification chain
    checkpoint_sha256: str = ""
    architecture_sha256: str = ""
    stdout_sha256: str = ""

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "submission_id": self.submission_id,
            "status": self.status,
            "error": self.error,
            "flops_equivalent_size": self.flops_equivalent_size,
            "training_time_seconds": self.training_time_seconds,
            "num_steps": self.num_steps,
            "num_params_M": self.num_params_M,
            "peak_vram_mb": self.peak_vram_mb,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "best_val_loss": self.best_val_loss,
            "best_val_step": self.best_val_step,
            "val_cadence_unit": self.val_cadence_unit,
            "val_base": self.val_base,
            "val_growth": self.val_growth,
            "val_eval_tokens": self.val_eval_tokens,
            "flops_per_step_estimate": self.flops_per_step_estimate,
            "reference_eval_loss_history": self.reference_eval_loss_history,
            "checkpoint_sha256": self.checkpoint_sha256,
            "architecture_sha256": self.architecture_sha256,
            "stdout_sha256": self.stdout_sha256,
        }

    @classmethod
    def from_json(cls, text: str) -> "TrainingMeta":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingMeta":
        # Old metas may carry `loss_curve` — silently drop (filtered by field set).
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
    submission_id: str,
    checkpoint_path: str,
    architecture_code: str,
    stdout_log: str,
    meta: TrainingMeta,
) -> bool:
    """Upload all training artifacts with hash verification chain."""
    meta.checkpoint_sha256 = sha256_file(checkpoint_path)
    meta.architecture_sha256 = sha256_text(architecture_code)
    meta.stdout_sha256 = sha256_text(stdout_log)

    ck = checkpoint_key(round_id, submission_id)
    ak = architecture_key(round_id, submission_id)
    mk = meta_key(round_id, submission_id)
    sk = stdout_key(round_id, submission_id)

    ok = True
    ok = r2.upload_file_from_disk(checkpoint_path, ck) and ok
    ok = r2.upload_text(ak, architecture_code) and ok
    ok = r2.upload_text(sk, stdout_log) and ok
    ok = r2.upload_json(mk, meta.to_dict()) and ok

    if ok:
        logger.info("Uploaded artifacts for round %d submission %s", round_id, submission_id[:12])
    else:
        logger.error("Some artifacts failed to upload for round %d submission %s", round_id, submission_id[:12])

    return ok


def generate_upload_urls(
    r2: "R2AuditLog",
    round_id: int,
    submission_id: str,
    ttl: int = 5400,
) -> dict[str, str]:
    """Generate pre-signed PUT URLs for all training artifacts.

    TTL defaults to 5400s (~90 min). URLs are path-locked to the specific
    round/submission key, so the trainer-host can only write into its
    assigned opaque submission slot.
    """
    keys = {
        "checkpoint": checkpoint_key(round_id, submission_id),
        "architecture": architecture_key(round_id, submission_id),
        "meta": meta_key(round_id, submission_id),
        "stdout": stdout_key(round_id, submission_id),
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
    submission_id: str,
) -> tuple[bool, str]:
    """Verify expected artifacts exist after training upload.

    Checks that checkpoint and meta files exist at the expected keys, and
    that the meta self-reports the same round_id and submission_id.
    """
    ck = checkpoint_key(round_id, submission_id)
    mk = meta_key(round_id, submission_id)

    if not r2.key_exists(mk):
        return False, f"training_meta.json missing at {mk}"
    if not r2.key_exists(ck):
        return False, f"checkpoint missing at {ck}"

    meta_dict = r2.download_json(mk)
    if not meta_dict:
        return False, "Failed to download training_meta.json for verification"

    if meta_dict.get("round_id") != round_id:
        return False, (
            f"Meta round_id mismatch: expected {round_id}, "
            f"got {meta_dict.get('round_id')}"
        )
    # The meta lives at a presigned-PUT, path-locked key, so its location
    # already authenticates it. Older trainer images omit ``submission_id``
    # from the JSON body — accept them; reject only on an explicit mismatch.
    meta_sid = meta_dict.get("submission_id")
    if meta_sid and meta_sid != submission_id:
        return False, (
            f"Meta submission_id mismatch: expected {submission_id}, "
            f"got {meta_sid}"
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

    meta.checkpoint_sha256 = sha256_file(checkpoint_path)
    meta.architecture_sha256 = sha256_text(architecture_code)
    meta.stdout_sha256 = sha256_text(stdout_log)

    ok = True

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

    # Architecture: retry once — eval depends on it.
    if "architecture" in presigned_urls:
        arch_bytes = architecture_code.encode()
        arch_uploaded = False
        for attempt in range(2):
            try:
                if attempt == 0:
                    logger.info("Uploading architecture (%d bytes) to presigned URL", len(arch_bytes))
                else:
                    logger.info("Retrying architecture upload (attempt %d)", attempt + 1)
                resp = httpx.put(presigned_urls["architecture"], content=arch_bytes, timeout=30)
                resp.raise_for_status()
                logger.info("Architecture upload succeeded (HTTP %d)", resp.status_code)
                arch_uploaded = True
                break
            except Exception as e:
                logger.error(
                    "Presigned upload failed for architecture (%d bytes, attempt %d): %s",
                    len(arch_bytes), attempt + 1, e,
                )
        if not arch_uploaded:
            ok = False
    else:
        logger.error("No presigned URL for architecture — upload_urls keys: %s", list(presigned_urls.keys()))
        ok = False

    if "stdout" in presigned_urls:
        try:
            resp = httpx.put(presigned_urls["stdout"], content=stdout_log.encode(), timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Presigned upload failed for stdout: %s", e)
            ok = False

    if "meta" in presigned_urls:
        try:
            body = json.dumps(meta.to_dict(), indent=2).encode()
            resp = httpx.put(presigned_urls["meta"], content=body, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Presigned upload failed for meta: %s", e)
            ok = False

    if ok:
        logger.info("Uploaded artifacts via presigned URLs for submission %s", meta.submission_id[:12])
    return ok


@dataclass
class DownloadedArtifacts:
    """All artifacts downloaded for a single miner's training run."""
    meta: TrainingMeta
    checkpoint_path: str = ""    # local path where checkpoint was saved
    architecture_code: str = ""
    stdout_log: str = ""
    verified: bool = False
    verification_error: str = ""


def download_training_artifacts(
    r2: "R2AuditLog",
    round_id: int,
    submission_id: str,
    download_dir: str,
) -> Optional[DownloadedArtifacts]:
    """Download all training artifacts for a submission and verify hashes.

    Returns None if meta can't be downloaded.
    Sets verified=False with verification_error if hash mismatch.
    """
    sid_short = submission_id[:12]
    logger.info("Downloading artifacts for submission %s round %d", sid_short, round_id)

    mk = meta_key(round_id, submission_id)
    meta_dict = r2.download_json(mk)
    if not meta_dict:
        logger.warning("No training_meta.json found for submission %s round %d", sid_short, round_id)
        return None

    meta = TrainingMeta.from_dict(meta_dict)
    logger.info(
        "Training meta for %s: status=%s steps=%d time=%.1fs params=%.2fM",
        sid_short, meta.status, meta.num_steps,
        meta.training_time_seconds, meta.num_params_M,
    )

    ck = checkpoint_key(round_id, submission_id)
    os.makedirs(download_dir, exist_ok=True)
    local_checkpoint = os.path.join(download_dir, f"checkpoint_{submission_id}.safetensors")
    if not r2.download_file_to_disk(ck, local_checkpoint):
        logger.warning("Failed to download checkpoint for submission %s round %d", sid_short, round_id)
        return DownloadedArtifacts(
            meta=meta, verification_error="Failed to download checkpoint",
        )

    ckpt_size_mb = os.path.getsize(local_checkpoint) / (1024 * 1024)
    logger.info("Checkpoint downloaded: %s (%.2f MB)", sid_short, ckpt_size_mb)

    ak = architecture_key(round_id, submission_id)
    architecture_code = r2.download_text(ak) or ""

    sk = stdout_key(round_id, submission_id)
    stdout_log = r2.download_text(sk) or ""
    logger.info(
        "Artifacts downloaded for %s: checkpoint=%.2fMB arch=%d bytes stdout=%d bytes",
        sid_short, ckpt_size_mb, len(architecture_code), len(stdout_log),
    )

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
            errors.append("architecture hash mismatch")

    if meta.stdout_sha256 and stdout_log:
        actual = sha256_text(stdout_log)
        if actual != meta.stdout_sha256:
            errors.append("stdout hash mismatch")

    if errors:
        result.verification_error = "; ".join(errors)
        logger.warning("Hash verification failed for %s: %s", sid_short, result.verification_error)
    else:
        result.verified = True
        logger.info("All hashes verified for submission %s", sid_short)

    return result


def list_round_artifacts(
    r2: "R2AuditLog",
    round_id: int,
) -> list[str]:
    """List all submission_ids that have training_meta.json for a round."""
    prefix = f"round_{round_id}/"
    keys = r2.list_keys(prefix)

    sids: set[str] = set()
    for key in keys:
        # round_{id}/submission_{sid}/training_meta.json
        if key.endswith("/training_meta.json"):
            parts = key.split("/")
            if len(parts) >= 3 and parts[1].startswith("submission_"):
                sids.add(parts[1][len("submission_"):])

    return sorted(sids)
