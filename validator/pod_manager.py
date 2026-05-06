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
import functools
import json
import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Default prefix used to identify validator-owned agent deployments on
# Basilica. Anything matching this prefix that's older than the reap age
# is assumed to be an orphan (the validator that created it crashed,
# was redeployed, or hit an HTTP error path that skipped cleanup).
_DEFAULT_AGENT_PREFIX = "radar-agent"


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
    # Subtensor for read-only chain queries
    for key in ("SUBTENSOR_NETWORK", "NETUID"):
        val = os.getenv(key, "")
        if val:
            env_vars[key] = val
    # URL allowlist — the harness reads this to build the GatedClient
    # (app-layer gate); the entrypoint reads RADAR_ALLOWED_URLS to program
    # iptables (network-layer gate) before the harness starts.
    if allowed_urls:
        env_vars["AGENT_ALLOWED_URLS"] = allowed_urls
        env_vars["RADAR_ALLOWED_URLS"] = allowed_urls
    # Note: NO Basilica token, NO Hippius/R2 credentials, NO RADAR_BASILICA_ENV
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
    task_id: Optional[int] = None,
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

    ``task_id`` disambiguates the Basilica deployment name so concurrent
    ``process_challenge`` calls from ``asyncio.gather`` do not collide on
    the same ``{image}-{method}-{timestamp}`` slug.  Pass the miner UID.

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
    if task_id is not None:
        # affinetes reads task_id from kwargs to disambiguate deployment
        # names; without it, two concurrent calls to the same method in
        # the same wallclock second collide and one serialises behind
        # the other.  The remote Actor accepts **kwargs so this is a no-op
        # on the pod side.
        call_kwargs["task_id"] = int(task_id)
    agent_code = getattr(env, "_agent_code", None)
    if agent_code:
        call_kwargs["agent_code"] = agent_code

    from config import Config
    max_retries = Config.AGENT_POD_RETRIES
    for attempt in range(1 + max_retries):
        try:
            result = await env.process_challenge(**call_kwargs)
            _record_deployment_name(env)
            if attempt > 0:
                logger.info(
                    "Agent pod succeeded on attempt %d/%d",
                    attempt + 1, 1 + max_retries,
                )
            return result
        except Exception as e:
            # Each Basilica retry spawns a NEW deployment (timestamp suffix
            # in the name). Record whatever name was assigned this attempt
            # so finally-block cleanup can delete every pod we created,
            # not just the last one.
            _record_deployment_name(env)
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


# ── Cleanup ──────────────────────────────────────────────────────

def _record_deployment_name(env) -> None:
    """Capture the Basilica deployment name affinetes assigned to ``env``.

    affinetes exposes the active deployment under one of a few attribute
    names depending on backend version (``_deployment_name``,
    ``deployment_name``, or a ``_deployment`` object with a ``.name``).
    We accumulate every distinct name we observe across retries so
    cleanup can hit them all.
    """
    name = None
    for attr in ("_deployment_name", "deployment_name", "instance_name"):
        candidate = getattr(env, attr, None)
        if isinstance(candidate, str) and candidate:
            name = candidate
            break
    if name is None:
        dep = getattr(env, "_deployment", None) or getattr(env, "deployment", None)
        candidate = getattr(dep, "name", None)
        if isinstance(candidate, str) and candidate:
            name = candidate
    if not name:
        return
    seen = getattr(env, "_radar_deployment_names", None)
    if seen is None:
        seen = []
        try:
            env._radar_deployment_names = seen
        except Exception:
            return
    if name not in seen:
        seen.append(name)


async def cleanup_agent_env(env) -> None:
    """Best-effort teardown of every Basilica deployment tied to ``env``.

    Calls ``env.cleanup()`` first (the affinetes path), then force-deletes
    every deployment name we recorded during retries via the Basilica
    SDK. Always swallows exceptions — this runs in ``finally`` blocks and
    must never mask the original error.
    """
    if env is None:
        return
    cleanup = getattr(env, "cleanup", None)
    if cleanup is not None:
        try:
            res = cleanup()
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:
            logger.debug("env.cleanup() failed: %s", e)

    names = list(getattr(env, "_radar_deployment_names", []) or [])
    if not names:
        return
    try:
        from basilica import BasilicaClient
    except ImportError:
        return
    try:
        client = BasilicaClient()
    except Exception as e:
        logger.debug("BasilicaClient init failed during cleanup: %s", e)
        return
    loop = asyncio.get_event_loop()
    for name in names:
        try:
            await loop.run_in_executor(
                None, functools.partial(client.delete_deployment, name),
            )
            logger.info("Force-deleted leaked agent deployment: %s", name)
        except Exception as e:
            logger.debug(
                "Force-delete of %s failed (TTL will reap): %s", name, e,
            )


# ── Orphan Reaper ────────────────────────────────────────────────

def _list_basilica_deployments(client):
    """Call whichever list method the installed basilica SDK exposes.

    Returns an iterable of deployment objects, or [] if no list API is
    available (older SDKs). Each object is expected to have ``.name`` and
    one of ``.created_at`` / ``.uptime_seconds`` we can use to age it.
    """
    for method in ("list_deployments", "deployments", "list"):
        fn = getattr(client, method, None)
        if callable(fn):
            try:
                return list(fn())
            except Exception as e:
                logger.debug("%s() failed: %s", method, e)
                return []
    return []


def _deployment_age_seconds(dep) -> Optional[float]:
    """Return the age of a Basilica deployment in seconds, or None.

    Tries ``uptime_seconds`` first (a single number), then ``created_at``
    (epoch or ISO 8601). Returns None when neither is parseable so the
    caller can skip rather than mistakenly delete a fresh pod.
    """
    uptime = getattr(dep, "uptime_seconds", None)
    if isinstance(uptime, (int, float)) and uptime >= 0:
        return float(uptime)
    created = getattr(dep, "created_at", None)
    if isinstance(created, (int, float)):
        return max(0.0, time.time() - float(created))
    if isinstance(created, str):
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(created.replace("Z", "+00:00"))
            return max(0.0, time.time() - ts.timestamp())
        except Exception:
            return None
    return None


async def reap_orphan_agent_pods(
    prefix: str = _DEFAULT_AGENT_PREFIX,
    max_age_seconds: int = 1800,
) -> int:
    """Delete Basilica deployments matching ``prefix`` older than the cap.

    Safety net for pods leaked by validator crashes, retry storms, or
    cleanup paths that didn't run. Mirrors ``Miner._reap_stale_deployments``
    on the miner side. Returns the number of deployments deleted.

    Skips deployments whose age can't be determined to avoid deleting
    fresh pods on SDK versions that don't expose timestamps.
    """
    if get_mode() != "basilica":
        return 0
    try:
        from basilica import BasilicaClient
    except ImportError:
        return 0
    try:
        client = BasilicaClient()
    except Exception as e:
        logger.debug("BasilicaClient init failed during reap: %s", e)
        return 0

    loop = asyncio.get_event_loop()
    try:
        deps = await loop.run_in_executor(
            None, functools.partial(_list_basilica_deployments, client),
        )
    except Exception as e:
        logger.debug("Basilica list failed: %s", e)
        return 0

    deleted = 0
    for dep in deps:
        name = getattr(dep, "name", None) or getattr(dep, "instance_name", None)
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        age = _deployment_age_seconds(dep)
        if age is None or age < max_age_seconds:
            continue
        try:
            await loop.run_in_executor(
                None, functools.partial(client.delete_deployment, name),
            )
            deleted += 1
            logger.info(
                "Reaped orphan agent deployment %s (%.0f min old)",
                name, age / 60,
            )
        except Exception as e:
            logger.debug("Reap delete of %s failed: %s", name, e)
    if deleted:
        logger.info("Orphan reaper deleted %d agent deployments", deleted)
    return deleted


# ── Pod Verification (Basilica public metadata) ──────────────────


def _parse_runpod_allowlist() -> set[str]:
    """Comma-separated env-var → set. Empty means "any endpoint accepted"."""
    from config import Config
    raw = (Config.OFFICIAL_RUNPOD_ENDPOINTS or "").strip()
    if not raw:
        return set()
    return {e.strip() for e in raw.split(",") if e.strip()}


async def verify_runpod_endpoint(
    endpoint_id: str,
    expected_image_digest: str = "",
    declared_image_digest: str = "",
    *,
    runpod_client=None,
) -> tuple[bool, str]:
    """Verify a miner's RunPod Serverless endpoint before accepting dispatch.

    Mirrors ``verify_miner_pod`` for Basilica — one control-plane API
    call, fast-fail on mismatch. The RunPod backend has no SSH/exec
    surface and no per-job hardware attestation, so the trust model is:

      1. The endpoint id must be on the subnet-owner allowlist (when one
         is configured) — keeps a miner from pointing their relay at
         some unrelated workload.
      2. The endpoint's template image must pin to the official trainer
         digest (when one is configured). Tag-form references aren't
         verifiable; the helper returns False with a clear reason so
         operators know to switch their template to digest-form.
      3. The miner's self-declared digest in TrainerReady must match
         the validator's expected digest (a sanity check that catches
         operator misconfig before we even hit the RunPod API).

    Boot-proof / mid-round re-verification are intentionally NOT
    performed here — see CLAUDE.md "Targon migration" comparison; the
    RunPod backend matches Basilica's verification depth.
    """
    if not endpoint_id:
        return False, "no endpoint_id provided"

    allowlist = _parse_runpod_allowlist()
    if allowlist and endpoint_id not in allowlist:
        return False, f"endpoint {endpoint_id} not in OFFICIAL_RUNPOD_ENDPOINTS allowlist"

    # Cheap sanity check before hitting the API.
    if (
        expected_image_digest
        and declared_image_digest
        and declared_image_digest != expected_image_digest
    ):
        return False, (
            f"declared digest {declared_image_digest} != expected {expected_image_digest}"
        )

    if runpod_client is None:
        # Lazy default — keeps Basilica/Targon validators from importing
        # the RunPod client / breaker.
        from miner.runpod_lifecycle import make_runpod_client
        try:
            runpod_client = make_runpod_client()
        except Exception as e:
            return False, f"could not construct RunpodClient: {e}"

    from shared.runpod_breaker import RunpodUnavailable
    try:
        info = await runpod_client.get_endpoint(endpoint_id)
    except RunpodUnavailable as e:
        # Caller treats this as the runpod_unavailable soft-fail path.
        return False, f"runpod unavailable: {e}"
    except Exception as e:
        return False, f"endpoint metadata fetch failed: {e}"

    if not info.template_id and not info.image_name:
        return False, f"endpoint {endpoint_id} not visible on this account"

    if expected_image_digest:
        if not info.image_digest:
            return False, (
                f"endpoint template uses tag-form image {info.image_name!r}; "
                "RunPod templates must reference @sha256:... for verification"
            )
        if info.image_digest != expected_image_digest:
            return False, (
                f"endpoint image digest {info.image_digest} != expected {expected_image_digest}"
            )
    if info.workers_max <= 0:
        return False, f"endpoint {endpoint_id} has workers_max=0 (no capacity allocated)"

    return True, "ok"


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

    The Targon equivalent lives in ``validator/trainer_verify.py`` —
    this function stays as the legacy / default check while
    ``RADAR_HOSTING_BACKEND=basilica``.

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
