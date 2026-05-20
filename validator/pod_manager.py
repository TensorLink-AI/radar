"""
Pod lifecycle manager for the validator.

Handles launching and connecting to containers via affinetes.
Supports three modes:
  - mode="docker"   -> local Docker (via affinetes)
  - mode="basilica" -> Basilica cloud (via affinetes)
  - mode="url"      -> connect to miner-hosted pod (future)

Agent pods use the OFFICIAL agent image (subnet-owner controlled).
Miner code (.py files) is injected at launch time — miners never
supply their own Docker images for agents.
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import os
import tempfile
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
    """Build env vars to pass into trainer/agent containers."""
    env_vars = {}
    # Forward Basilica auth
    token = os.getenv("BASILICA_API_TOKEN", "")
    if token:
        env_vars["BASILICA_API_TOKEN"] = token
    # Forward the HMAC shared secret so the trainer pod can verify
    # incoming /train requests, and the static peer registry path so
    # callers can resolve hotkeys → endpoints.
    secret = os.getenv("RADAR_SHARED_SECRET", "")
    if secret:
        env_vars["RADAR_SHARED_SECRET"] = secret
    miners_path = os.getenv("MINERS_CONFIG_PATH", "")
    if miners_path:
        env_vars["MINERS_CONFIG_PATH"] = miners_path
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
    """Validate miner architecture submission has required functions."""
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


def pre_validate_agent_code(code: str) -> tuple[bool, str]:
    """Validate miner agent code has required entry point.

    Agent modules must define ``design_architecture(challenge, client)``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "design_architecture" not in names:
        return False, "Missing design_architecture"
    return True, ""


# ── Agent Pod Operations ─────────────────────────────────────────

def _build_agent_env_vars(allowed_urls: str = "") -> dict[str, str]:
    """Build env vars for agent pods (more restricted than trainer)."""
    env_vars = {}
    # URL allowlist — the harness reads this to build the GatedClient
    # (app-layer gate); the entrypoint reads RADAR_ALLOWED_URLS to program
    # iptables (network-layer gate) before the harness starts.
    if allowed_urls:
        env_vars["AGENT_ALLOWED_URLS"] = allowed_urls
        env_vars["RADAR_ALLOWED_URLS"] = allowed_urls
    # Note: NO Basilica token, NO R2 credentials, NO RADAR_BASILICA_ENV
    return env_vars


def _write_agent_code(agent_code: dict | str) -> str:
    """Write miner's agent code to a temp dir for volume-mounting.

    Accepts either a bundle dict (``{"files": {"agent.py": "..."}, ...}``)
    or a single string (written as ``agent.py``).

    Returns the temp directory path containing agent/*.py.
    """
    tmpdir = tempfile.mkdtemp(prefix="radar_agent_")
    agent_dir = os.path.join(tmpdir, "agent")
    os.makedirs(agent_dir, exist_ok=True)

    if isinstance(agent_code, dict) and "files" in agent_code:
        for filename, code in agent_code["files"].items():
            with open(os.path.join(agent_dir, filename), "w") as f:
                f.write(code)
    else:
        with open(os.path.join(agent_dir, "agent.py"), "w") as f:
            f.write(str(agent_code))
    return tmpdir


def _normalise_agent_code(agent_code: dict | str) -> dict:
    """Convert agent_code to a bundle dict suitable for inline delivery."""
    if isinstance(agent_code, dict) and "files" in agent_code:
        return agent_code
    # Single string → wrap as agent.py bundle
    return {"files": {"agent.py": str(agent_code)}, "entry_point": "agent.py"}


def _inject_allowed_urls_into_challenge(
    challenge_json: str, allowed_urls: str,
) -> str:
    """Add allowed_urls field to challenge JSON so harness can read it."""
    try:
        data = json.loads(challenge_json)
        data["allowed_urls"] = allowed_urls
        return json.dumps(data)
    except (json.JSONDecodeError, TypeError):
        return challenge_json


async def launch_agent_pod(
    image_url: str = "",
    mode: Optional[str] = None,
    mem_limit: str = "8192Mi",
    agent_code: dict | str = "",
    allowed_urls: str = "",
):
    """
    Launch a sandboxed agent pod via affinetes.

    Uses the official agent image (subnet-owner controlled).  The miner's
    .py files are delivered inline via ``process_challenge`` — the Actor
    writes them to /workspace/agent/ before running the harness.  This
    works identically for both Docker and Basilica modes.

    Args:
        image_url: Override official image (testing only).
        mode: "docker" or "basilica".
        mem_limit: Memory limit for the container.
        agent_code: Bundle dict (``{"files": {...}, "entry_point": "..."}``),
            or a single code string.
        allowed_urls: Comma-separated URL prefixes the agent may access.

    Returns: environment object with process_challenge() and cleanup().
        The returned wrapper carries ``_agent_code`` so
        ``run_agent_on_pod`` can pass code inline.
    """
    if mode is None:
        mode = get_mode()

    from config import Config
    official_image = image_url or Config.OFFICIAL_AGENT_IMAGE

    env_vars = _build_agent_env_vars(allowed_urls)

    # Tell the harness which file is the entry point
    if isinstance(agent_code, dict):
        entry = agent_code.get("entry_point", "agent.py")
        env_vars["AGENT_MODULE"] = f"/workspace/agent/{entry}"

    logger.info(
        "Launching sandboxed agent pod: image=%s mode=%s allowed_urls=%d prefixes",
        official_image, mode, len(allowed_urls.split(",")) if allowed_urls else 0,
    )

    env = _af().load_env(
        image=official_image,
        mode=mode,
        env_vars=env_vars,
        mem_limit=mem_limit,
        cleanup=True,
    )
    # Stash code bundle — run_agent_on_pod passes it inline to process_challenge
    env._agent_code = _normalise_agent_code(agent_code)
    return env



async def run_agent_on_pod(
    env, challenge_json: str, timeout: int = 300,
    allowed_urls: str = "",
) -> Optional[dict]:
    """
    Send a Challenge to a running agent pod and get the Proposal back.

    If *allowed_urls* is set, injects it into the challenge JSON so the
    frozen harness can build its GatedClient allowlist.

    Agent code (stashed on *env* by ``launch_agent_pod``) is always
    passed inline so the Actor can write the files before running the
    harness.  This works identically for Docker and Basilica modes.

    Retries up to ``RADAR_AGENT_POD_RETRIES`` times (default 2) with
    exponential backoff for transient Basilica deployment failures.

    Returns: dict with code/name/motivation, or dict with "error" key, or None.
    """
    if allowed_urls:
        challenge_json = _inject_allowed_urls_into_challenge(
            challenge_json, allowed_urls,
        )

    call_kwargs: dict = dict(
        challenge_json=challenge_json,
        _timeout=timeout,
        # Pass timeout as regular kwarg so BasilicaBackend uses it for
        # TTL calculation instead of its 1800s default.
        timeout=timeout,
    )
    agent_code = getattr(env, "_agent_code", None)
    if agent_code:
        call_kwargs["agent_code"] = agent_code

    from config import Config
    max_retries = Config.AGENT_POD_RETRIES
    for attempt in range(1 + max_retries):
        try:
            result = await env.process_challenge(**call_kwargs)
            if attempt > 0:
                logger.info(
                    "Agent pod succeeded on attempt %d/%d",
                    attempt + 1, 1 + max_retries,
                )
            return result
        except Exception as e:
            err_str = str(e)
            is_502 = "502" in err_str
            if attempt < max_retries:
                # Basilica deployments can take 30-60s to become fully
                # responsive even after reporting "ready".  Use a longer
                # base backoff (30s) so the next deployment has time to
                # warm up.  Double on subsequent retries.
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s
                logger.warning(
                    "Agent pod attempt %d/%d failed (%s): %s — retrying in %ds",
                    attempt + 1, 1 + max_retries,
                    "502 Bad Gateway" if is_502 else type(e).__name__,
                    e, wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "Agent pod call failed after %d attempt(s): %s: %s",
                    1 + max_retries, type(e).__name__, e,
                )
    return None


# ── Pod Verification (Basilica public metadata) ──────────────────


def _parse_image_ref(image_ref: str) -> tuple[str, str]:
    """Split 'registry/repo:tag' into (image, tag). Empty tag if none."""
    if ":" in image_ref and not image_ref.endswith(":"):
        parts = image_ref.rsplit(":", 1)
        return parts[0], parts[1]
    return image_ref.rstrip(":"), ""


async def verify_miner_pod(
    instance_name: str,
    expected_image: str = "",
) -> tuple[bool, str]:
    """Check Basilica public metadata to verify a trainer pod.

    Args:
        instance_name: Basilica deployment instance name.
        expected_image: Expected image reference (e.g. 'ghcr.io/org/repo:tag').
            Defaults to Config.OFFICIAL_TRAINING_IMAGE.

    Returns:
        (ok, reason) tuple.
    """
    try:
        from config import Config
        if not expected_image:
            expected_image = Config.OFFICIAL_TRAINING_IMAGE
        if not expected_image:
            return True, "ok (no expected image configured)"

        # Lazy import basilica SDK
        from basilica import BasilicaClient

        meta = BasilicaClient().get_public_deployment_metadata(
            instance_name=instance_name,
        )

        # Compare image
        exp_image, exp_tag = _parse_image_ref(expected_image)
        if meta.image != exp_image:
            return False, f"Wrong image: expected {exp_image}, got {meta.image}"
        if exp_tag and meta.image_tag != exp_tag:
            return False, f"Wrong tag: expected {exp_tag}, got {meta.image_tag}"

        # Check pod is running (case-insensitive — Basilica returns "Active")
        state_lower = (meta.state or "").lower()
        if state_lower not in ("running", "active"):
            return False, f"Pod not running: state={meta.state}"

        # Check replicas — skip if state is active but replicas not yet ready
        # (Basilica "Active" means deployment exists, replicas may still be scaling)
        ready = getattr(meta.replicas, "ready", meta.replicas) if hasattr(meta, "replicas") else 0
        ready_count = ready if isinstance(ready, int) else getattr(meta.replicas, "ready", 0)
        if isinstance(ready_count, int) and ready_count < 1 and state_lower == "running":
            return False, f"No ready replicas: {ready_count}"

        return True, "ok"
    except ImportError:
        logger.warning("basilica SDK not installed, skipping attestation check")
        return True, "ok (basilica not available)"
    except Exception as e:
        return False, f"Attestation failed: {e}"
