"""TDX + NRAS attestation against Targon's tower.

Split out of ``shared/targon_client.py`` so each file stays under the
project's 300-line cap. The flow:

  1. POST a fresh nonce to the CVM at ``http://{cvm_ip}:8080/api/v1/evidence``,
     optionally Epistula-signed by the validator's hotkey so the CVM
     can prove the evidence was bound to a specific validator.
  2. CVM returns ``{quote, user_data: {nvcc_response, ...}}``.
  3. Forward the bundle to ``tower.targon.com/api/v1/verify-attestation``.
  4. Tower verifies the Intel TDX signature on the quote and the
     NVIDIA NRAS GPU token, returns parsed CPU / GPU info.

The tower call uses the validator's Targon API key (Bearer auth);
the CVM call does not (the CVM has no notion of the operator's
Targon account).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AttestationResult:
    """Outcome of ``verify_attestation``."""

    verified: bool
    gpu_class: str = ""
    gpu_count: int = 0
    cpu_model: str = ""
    error: str = ""
    raw: dict = field(default_factory=dict)


async def fetch_cvm_evidence(
    cvm_ip: str,
    nonce: str,
    *,
    timeout: float,
    wallet=None,
) -> dict:
    """Ask the CVM for a fresh attestation evidence bundle.

    ``wallet`` is the validator's bittensor wallet — used to
    Epistula-sign the nonce request so the CVM can bind the evidence
    to a specific validator. Optional; if omitted the request goes
    out unsigned.
    """
    url = f"http://{cvm_ip}:8080/api/v1/evidence"
    body = ('{"nonce":"' + nonce + '"}').encode()
    headers = {"Content-Type": "application/json"}
    if wallet is not None:
        try:
            from shared.auth import sign_request
            headers.update(sign_request(wallet, body))
        except Exception as e:
            logger.warning("Could not Epistula-sign attestation nonce: %s", e)

    async with httpx.AsyncClient(timeout=timeout) as http:
        resp = await http.post(url, content=body, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def verify_with_tower(
    tower_url: str,
    auth_headers: dict,
    *,
    evidence: dict,
    cvm_ip: str,
    miner_hotkey: str,
    validator_hotkey: str,
    nonce: str,
    timeout: float,
) -> dict:
    """POST the CVM evidence to Targon's tower and return the verdict dict."""
    url = f"{tower_url.rstrip('/')}/api/v1/verify-attestation"
    payload = {
        "attestation": evidence,
        "ip_address": cvm_ip,
        "miner_hotkey": miner_hotkey,
        "validator_hotkey": validator_hotkey,
        "nonce": nonce,
    }
    async with httpx.AsyncClient(timeout=timeout) as http:
        resp = await http.post(url, json=payload, headers=auth_headers)
        resp.raise_for_status()
        return resp.json()


def parse_tower_response(verdict: dict) -> AttestationResult:
    """Map tower's verify-attestation response into an AttestationResult.

    Tower's exact field names are still in flux — accept either
    ``{gpu: {class, count}, cpu: {model}}`` or ``{gpu_class, gpu_count, cpu_model}``
    so we don't break when they tweak the schema.
    """
    if not isinstance(verdict, dict):
        return AttestationResult(verified=False, error="tower response not a dict")
    verified = bool(verdict.get("verified", False))
    if not verified:
        return AttestationResult(
            verified=False,
            error=str(verdict.get("error") or verdict.get("message") or "tower rejected"),
            raw=verdict,
        )
    gpu = verdict.get("gpu") if isinstance(verdict.get("gpu"), dict) else {}
    cpu = verdict.get("cpu") if isinstance(verdict.get("cpu"), dict) else {}
    return AttestationResult(
        verified=True,
        gpu_class=str(gpu.get("class") or gpu.get("model") or verdict.get("gpu_class") or ""),
        gpu_count=int(gpu.get("count") or verdict.get("gpu_count") or 0),
        cpu_model=str(cpu.get("model") or verdict.get("cpu_model") or ""),
        raw=verdict,
    )


def fresh_nonce() -> str:
    """Hex string suitable for binding into evidence requests."""
    return uuid.uuid4().hex
