"""Boot proof helpers — the validator-facing handshake for image hardening.

The bootstrap (``runner/_bootstrap.py``) writes ``/tmp/boot_proof.json``
on a successful integrity check. This module reads that file, signs its
canonical-JSON encoding with the shared HMAC secret, and returns the
payload ``runner/server.py`` exposes at GET /boot_proof.

Kept out of ``runner/server.py`` so that file stays under the project's
300-line cap. No FastAPI imports here — the caller wraps the dict in a
JSONResponse with the returned status.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

BOOT_PROOF_PATH = "/tmp/boot_proof.json"


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def build_boot_proof_response(
    proof_path: Optional[str] = None,
    wallet=None,
) -> Tuple[int, dict]:
    """Build the (status, body) tuple the /boot_proof endpoint returns.

    ``proof_path`` defaults to the module-level BOOT_PROOF_PATH and is
    resolved at call time so monkeypatched test paths take effect.
    ``wallet`` is accepted for backward compatibility and ignored — the
    proof is signed with the process-wide ``RADAR_SHARED_SECRET`` HMAC.
    The signature is empty when no secret is configured; validators that
    require attestation treat that as a hard fail.
    """
    if proof_path is None:
        proof_path = BOOT_PROOF_PATH
    if not os.path.isfile(proof_path):
        return 503, {
            "error": "boot proof missing — bootstrap did not run",
            "reason": "missing_boot_proof",
        }
    try:
        with open(proof_path, "rb") as f:
            proof = json.loads(f.read())
    except Exception as e:
        logger.error("Failed to read boot proof: %s", e)
        return 500, {"error": f"boot proof unreadable: {e}"}

    payload = _canonical_json(proof)
    signature = ""
    try:
        from shared.auth import sign_request_hmac
        signature = sign_request_hmac(payload)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Boot proof HMAC signing failed: %s", e)

    signer = os.getenv("RADAR_TRAINER_ID", "")

    return 200, {
        "proof": proof,
        "canonical_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "signer_hotkey": signer,
        "signature": signature,
    }
