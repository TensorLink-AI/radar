"""
Pod lifecycle manager for the validator.

Handles launching and connecting to containers via affinetes.
Supports three modes:
  - mode="docker"   -> local Docker (via affinetes)
  - mode="basilica" -> Basilica cloud (via affinetes)
  - mode="url"      -> connect to miner-hosted pod (future)
"""

from __future__ import annotations

import ast
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _af():
    """Lazy import of affinetes to avoid hard dependency at import time."""
    import affinetes as af_env
    return af_env


def get_mode() -> str:
    """Get pod mode from config."""
    mode = os.getenv("RADAR_AFFINETES_MODE", "docker").lower()
    if mode not in ("docker", "basilica", "url"):
        logger.warning("Unknown RADAR_AFFINETES_MODE='%s', falling back to 'docker'", mode)
        return "docker"
    return mode


def _build_env_vars() -> dict[str, str]:
    """Build env vars to pass into containers."""
    env_vars = {}
    # Forward Basilica auth
    token = os.getenv("BASILICA_API_TOKEN", "")
    if token:
        env_vars["BASILICA_API_TOKEN"] = token
    # Forward subtensor network config so pods connect to the right chain
    for key in ("SUBTENSOR_NETWORK", "NETUID"):
        val = os.getenv(key, "")
        if val:
            env_vars[key] = val
    # Note: R2 credentials are NOT forwarded to pods. Trainer pods
    # receive presigned URLs in the dispatch payload instead.
    # Forward user-specified vars
    extra = os.getenv("RADAR_BASILICA_ENV", "")
    if extra:
        for key in extra.split(","):
            key = key.strip()
            if key and key in os.environ:
                env_vars[key] = os.environ[key]
    return env_vars


# ── Static Pre-validation ────────────────────────────────────────

def pre_validate_code(code: str) -> tuple[bool, str]:
    """Validate miner submission has required functions."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "build_model" not in names:
        return False, "Missing build_model"
    if "build_optimizer" not in names:
        return False, "Missing build_optimizer"
    return True, ""


# ── Agent Pod Operations ─────────────────────────────────────────

async def launch_agent_pod(
    image_url: str,
    mode: Optional[str] = None,
    mem_limit: str = "8192Mi",
    miner_hosted_url: Optional[str] = None,
):
    """
    Launch a miner's agent pod via affinetes.

    Args:
        image_url: Docker image to launch
        mode: "docker", "basilica", or "url"
        mem_limit: Memory limit for the container
        miner_hosted_url: If set, connect to miner's pod (future mode)

    Returns: environment object with process_challenge() and cleanup()
    """
    if mode is None:
        mode = get_mode()

    # Future: miner-hosted mode
    if miner_hosted_url:
        logger.info("Connecting to miner-hosted agent at %s", miner_hosted_url)
        return _af().load_env(
            mode="url",
            base_url=miner_hosted_url,
        )

    # Docker or Basilica: both go through affinetes
    logger.info("Launching agent pod: image=%s mode=%s", image_url, mode)
    return _af().load_env(
        image=image_url,
        mode=mode,
        env_vars=_build_env_vars(),
        mem_limit=mem_limit,
        cleanup=True,
    )


async def run_agent_on_pod(env, challenge_json: str, timeout: int = 300) -> Optional[dict]:
    """
    Send a Challenge to a running agent pod and get the Proposal back.

    Returns: dict with code/name/motivation, or dict with "error" key, or None.
    """
    try:
        result = await env.process_challenge(
            challenge_json=challenge_json,
            _timeout=timeout,
        )
        return result
    except Exception as e:
        logger.error("Agent pod call failed: %s: %s", type(e).__name__, e)
        return None


# ── Pod Verification (Basilica attestation) ──────────────────────

async def verify_miner_pod(
    pod_url: str,
    attestation_id: str,
    expected_digest: str = "",
) -> tuple[bool, str]:
    """Ask Basilica if this pod is running the official image."""
    try:
        from config import Config
        if not expected_digest:
            expected_digest = Config.OFFICIAL_TRAINING_IMAGE_DIGEST
        if not expected_digest:
            return True, "ok (no digest configured)"

        # Lazy import basilica SDK
        import basilica
        attestation = basilica.verify_attestation(
            pod_url=pod_url,
            attestation_id=attestation_id,
        )
        if attestation.image_digest != expected_digest:
            return False, f"Wrong image: {attestation.image_digest[:16]}..."
        if not attestation.running:
            return False, "Pod not running"
        return True, "ok"
    except ImportError:
        logger.warning("basilica SDK not installed, skipping attestation check")
        return True, "ok (basilica not available)"
    except Exception as e:
        return False, f"Attestation failed: {e}"
