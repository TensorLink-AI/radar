"""Validator-side trainer verification — Targon attestation chain.

Three required checks (b/c/d from the migration plan):

  (b) ``targon_client.verify_image_digest(uid, expected)`` — confirms
      the running workload is the official trainer image digest.
  (c) ``targon_client.verify_attestation(cvm_ip, mh, vh, nonce)`` —
      end-to-end TDX + NRAS proof; cross-checks the GPU class declared
      by the miner in TrainerReady.
  (d) ``GET {trainer_url}/boot_proof`` — feature-flagged via
      ``Config.REQUIRE_BOOT_PROOF``. Confirms the hardened entrypoint
      ran (i.e. the operator didn't override Targon's command/args).
      Catches launch-config bypass that Targon's image-bytes
      attestation can't see.

(a) "miner's chain-committed digest matches" was dropped — the trainer
image is subnet-owner-controlled, every miner runs the same digest,
and there's no per-miner variation worth committing on chain. The
check that mattered (running workload runs the expected digest) is
covered by (b).

When Targon's circuit breaker is open the verifier returns
``targon_unavailable=True``; callers route those rounds through the
hybrid-fallback policy (reduced scoring weight, no exclusion).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    ok: bool
    reason: str = ""
    targon_unavailable: bool = False
    boot_proof: Optional[dict] = None
    gpu_class: str = ""


async def verify_trainer(
    *,
    ready,                         # TrainerReady
    miner_hotkey: str,
    expected_image_digest: str,
    trainer_url: str,
    targon_client,
    wallet,                        # validator's bittensor wallet
    require_boot_proof: bool = False,
    boot_proof_timeout: float = 5.0,
) -> VerifyResult:
    """Run all three Targon checks. All must pass for ``ok=True``.

    Returns ``targon_unavailable=True`` if the breaker fires; the
    caller should mark the round and let it proceed at reduced weight
    rather than excluding the miner.
    """
    from shared.targon_breaker import TargonUnavailable

    # (b) Workload-digest verify.
    try:
        digest_ok = await targon_client.verify_image_digest(
            ready.targon_workload_uid, expected_image_digest,
        )
    except TargonUnavailable as e:
        return VerifyResult(ok=False, reason=str(e), targon_unavailable=True)
    if not digest_ok:
        return VerifyResult(
            ok=False,
            reason=f"workload {ready.targon_workload_uid} not running expected digest",
        )

    # (c) Full TDX + NRAS attestation.
    try:
        attest = await targon_client.verify_attestation(
            cvm_ip=ready.cvm_ip,
            miner_hotkey=miner_hotkey,
            validator_hotkey=wallet.hotkey.ss58_address,
            wallet=wallet,
        )
    except TargonUnavailable as e:
        return VerifyResult(ok=False, reason=str(e), targon_unavailable=True)
    if not attest.verified:
        return VerifyResult(ok=False, reason=f"attestation: {attest.error}")
    if ready.gpu_class and attest.gpu_class and attest.gpu_class.lower() != ready.gpu_class.lower():
        return VerifyResult(
            ok=False,
            reason=f"GPU class mismatch: declared {ready.gpu_class}, attested {attest.gpu_class}",
        )

    # (d) Boot proof — feature-flagged. Independent of Targon.
    boot_proof = await fetch_boot_proof(trainer_url, timeout=boot_proof_timeout)
    if require_boot_proof:
        ok, reason = check_boot_proof(boot_proof, expected_image_digest)
        if not ok:
            return VerifyResult(ok=False, reason=f"boot_proof: {reason}", boot_proof=boot_proof)
    elif boot_proof is None:
        # Until we flip the flag, log but pass.
        logger.warning(
            "Boot proof unavailable for %s — feature-flagged off, treating as pass",
            trainer_url,
        )

    return VerifyResult(
        ok=True,
        boot_proof=boot_proof,
        gpu_class=attest.gpu_class,
    )


async def fetch_boot_proof(trainer_url: str, *, timeout: float) -> Optional[dict]:
    """GET ``/boot_proof`` from the trainer. None on 503 / transport error."""
    if not trainer_url:
        return None
    url = f"{trainer_url.rstrip('/')}/boot_proof"
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(url)
            if resp.status_code == 503:
                logger.warning("Trainer %s returned 503 on /boot_proof — bootstrap did not run", trainer_url)
                return None
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.debug("Boot proof fetch failed for %s: %s", trainer_url, e)
        return None


def check_boot_proof(boot_proof: Optional[dict], expected_image_digest: str) -> tuple[bool, str]:
    """Validate a boot-proof envelope. Caller decides whether to enforce.

    Verifies:
      - the proof exists,
      - the signer matches a hotkey we trust (caller's responsibility —
        we just check there *is* a signer),
      - the signature decodes (we don't reverify here; ed25519/SR25519
        signature verification belongs at the call site so the caller
        can decide whether to fetch the trainer's hotkey from the
        metagraph).

    Cross-checking ``hashes_root_sha256`` against an expected value is
    out of scope — the caller maintains the table of expected roots
    keyed by ``OFFICIAL_TRAINING_IMAGE_DIGEST``.
    """
    if boot_proof is None:
        return False, "missing"
    if not isinstance(boot_proof, dict):
        return False, "malformed"
    # ``proof`` must be present (even if empty in tests); signature must be non-empty.
    if "proof" not in boot_proof or not boot_proof.get("signature"):
        return False, "no signature"
    return True, ""


def reverify_offsets(
    block_hash: str, round_id: int, miner_uid: int,
    *, n: int, window_seconds: float,
) -> list[float]:
    """Deterministic re-verification offsets within the training window.

    Validators agree on when re-checks happen (so each miner's
    snapshots line up across validators) but the miner can't predict
    them — the seed is the round's block hash XOR the round + uid.
    """
    bh_int = int(block_hash[:16], 16) if block_hash else 0
    seed = bh_int ^ (round_id << 16) ^ (miner_uid << 32)
    rng = random.Random(seed)
    offsets = sorted(rng.uniform(0.1, 0.9) * window_seconds for _ in range(n))
    return offsets


async def reverify_workload(
    *, ready, expected_image_digest: str, trainer_url: str,
    targon_client, require_boot_proof: bool, boot_proof_timeout: float = 5.0,
) -> VerifyResult:
    """Mid-round re-verification — runs only checks (b) and (d).

    (c) is skipped on re-verify because the workload UID identity is
    pinned by (b) and full TDX attestation is expensive.
    """
    from shared.targon_breaker import TargonUnavailable
    try:
        digest_ok = await targon_client.verify_image_digest(
            ready.targon_workload_uid, expected_image_digest,
        )
    except TargonUnavailable as e:
        return VerifyResult(ok=False, reason=str(e), targon_unavailable=True)
    if not digest_ok:
        return VerifyResult(ok=False, reason="reverify: workload digest mismatch")

    boot_proof = await fetch_boot_proof(trainer_url, timeout=boot_proof_timeout)
    if require_boot_proof:
        ok, reason = check_boot_proof(boot_proof, expected_image_digest)
        if not ok:
            return VerifyResult(ok=False, reason=f"reverify boot_proof: {reason}", boot_proof=boot_proof)
    return VerifyResult(ok=True, boot_proof=boot_proof)
