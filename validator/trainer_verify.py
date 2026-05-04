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

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


def _canonical_json(obj) -> bytes:
    """Stable JSON encoding — must match runner/boot_proof.py exactly."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


@dataclass
class VerifyResult:
    ok: bool
    reason: str = ""
    targon_unavailable: bool = False
    boot_proof: Optional[dict] = None
    gpu_class: str = ""
    # True when the round was hosted on a backend with hardware
    # attestation (currently: only Targon). RunPod / Basilica deploys
    # set this to False so scoring can apply NON_ATTESTED_SCORE_MULTIPLIER.
    attested: bool = False


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

    # Sanity: miner self-declared digest should match the validator's
    # expected digest. Mismatch is a hard fail — Targon's verify would
    # also catch it but failing early gives a cleaner error.
    declared = getattr(ready, "deployed_image_digest", "")
    if declared and expected_image_digest and declared != expected_image_digest:
        return VerifyResult(
            ok=False,
            reason=f"declared digest {declared} != expected {expected_image_digest}",
        )

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
        ok, reason = check_boot_proof(
            boot_proof,
            expected_signer_hotkey=miner_hotkey,
            expected_hashes_root=_expected_hashes_root(),
        )
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
        attested=True,
    )


async def verify_trainer_runpod(
    *,
    ready,
    miner_hotkey: str,
    expected_image_digest: str,
    trainer_url: str,
    runpod_client,
    require_boot_proof: bool = False,
    boot_proof_timeout: float = 5.0,
) -> VerifyResult:
    """Verify a RunPod-deployed trainer.

    Two checks (no TDX/NRAS path on this backend):

      1. Image-digest pin via RunPod API (``runpod_client.verify_pod_image``).
      2. Boot proof — feature-flagged. Soft on RunPod since the proof
         signer is the miner-controlled pod; we pair this with the
         non-attested score multiplier rather than treating it as a
         hardware-rooted check.

    Result always carries ``attested=False`` — caller propagates this to
    scoring so honest RunPod miners receive the
    ``NON_ATTESTED_SCORE_MULTIPLIER`` discount.
    """
    declared = getattr(ready, "deployed_image_digest", "")
    if declared and expected_image_digest and declared != expected_image_digest:
        return VerifyResult(
            ok=False,
            reason=f"declared digest {declared} != expected {expected_image_digest}",
        )

    pod_id = getattr(ready, "runpod_pod_id", "")
    if not pod_id:
        return VerifyResult(ok=False, reason="TrainerReady missing runpod_pod_id")

    try:
        ok, why = await runpod_client.verify_pod_image(pod_id, expected_image_digest)
    except Exception as e:
        return VerifyResult(ok=False, reason=f"runpod verify_pod_image: {e}")
    if not ok:
        return VerifyResult(ok=False, reason=why)

    boot_proof = await fetch_boot_proof(trainer_url, timeout=boot_proof_timeout)
    if require_boot_proof:
        proof_ok, reason = check_boot_proof(
            boot_proof,
            expected_signer_hotkey=miner_hotkey,
            expected_hashes_root=_expected_hashes_root(),
        )
        if not proof_ok:
            return VerifyResult(
                ok=False,
                reason=f"boot_proof: {reason}",
                boot_proof=boot_proof,
            )
    elif boot_proof is None:
        logger.warning(
            "Boot proof unavailable for RunPod pod %s — feature-flagged off, treating as pass",
            pod_id,
        )

    return VerifyResult(
        ok=True,
        boot_proof=boot_proof,
        gpu_class=getattr(ready, "gpu_class", ""),
        attested=False,
    )


def _expected_hashes_root() -> str:
    """Read the pinned root from Config at call time so tests can monkeypatch."""
    try:
        from config import Config
        return Config.EXPECTED_BOOT_HASHES_ROOT
    except Exception:
        return ""


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


def check_boot_proof(
    boot_proof: Optional[dict],
    *,
    expected_signer_hotkey: str = "",
    expected_hashes_root: str = "",
) -> tuple[bool, str]:
    """Validate a boot-proof envelope.

    Returns ``(ok, reason)``. All checks below must pass:

    1. Envelope exists and contains ``proof`` + ``signature`` + ``signer_hotkey``.
    2. ``signer_hotkey`` equals the expected miner hotkey (when supplied) —
       a forged proof signed by some other key fails here.
    3. The SR25519 signature verifies over the canonical JSON encoding of
       ``proof``. Without this the prior version was a no-op — anyone could
       return ``{"signature": "x"}`` and pass.
    4. ``proof.hashes_root_sha256`` matches the pinned root (when supplied) —
       a tampered image with a different file table fails here even if the
       signature is valid.
    """
    if boot_proof is None:
        return False, "missing"
    if not isinstance(boot_proof, dict):
        return False, "malformed"
    proof = boot_proof.get("proof")
    if not isinstance(proof, dict):
        return False, "no proof"
    signature_hex = boot_proof.get("signature") or ""
    signer = boot_proof.get("signer_hotkey") or ""
    if not signature_hex or not signer:
        return False, "no signature"

    if expected_signer_hotkey and signer != expected_signer_hotkey:
        return False, f"signer mismatch: got {signer[:16]} expected {expected_signer_hotkey[:16]}"

    # Verify the SR25519 signature over the canonical JSON of `proof`.
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False, "signature not hex"
    payload = _canonical_json(proof)
    try:
        import bittensor as bt
        keypair = bt.Keypair(ss58_address=signer)
        if not keypair.verify(payload, signature):
            return False, "signature invalid"
    except ImportError:
        return False, "bittensor unavailable for signature verify"
    except Exception as e:
        return False, f"signature verify error: {e}"

    if expected_hashes_root:
        got = proof.get("hashes_root_sha256") or ""
        if got != expected_hashes_root:
            return False, f"hashes_root mismatch: got {got[:16]} expected {expected_hashes_root[:16]}"

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
    expected_signer_hotkey: str = "",
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
        ok, reason = check_boot_proof(
            boot_proof,
            expected_signer_hotkey=expected_signer_hotkey or ready.miner_hotkey,
            expected_hashes_root=_expected_hashes_root(),
        )
        if not ok:
            return VerifyResult(ok=False, reason=f"reverify boot_proof: {reason}", boot_proof=boot_proof)
    return VerifyResult(ok=True, boot_proof=boot_proof)
