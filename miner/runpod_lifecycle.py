"""RunPod-specific lifecycle helpers — mirrors miner/targon_lifecycle.py.

Holds the bits coupled to RunPod's pod lifecycle:

  - ``make_runpod_client()``           — config-driven lazy construction
  - ``deploy_runpod()``                — per-round pod create
  - ``teardown_runpod_with_retry()``   — pod delete with exponential
                                         backoff (RunPod bills by uptime)
  - ``wait_for_runpod_ready()``        — poll until RUNNING + /health
  - ``validate_and_reap_orphans_runpod()`` — startup credential check +
    orphan pod cleanup (matches the Targon flow).
  - ``get_runpod_registry_creds()``    — private-registry creds from env

Trust note: RunPod is non-attested. Trust falls back to digest-pinned
image refs (validators check ``imageName`` contains ``@sha256:...``)
plus the hardened-image bootstrap chain. ``Config.NON_ATTESTED_SCORE_MULTIPLIER``
discounts honest miners' score to price in residual risk.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import httpx

from config import Config
from miner.hosting import Deployment, TargonReadinessTimeout

logger = logging.getLogger(__name__)


# ── Client factory ──────────────────────────────────────────────────


def make_runpod_client():
    """Build a RunPodClient using Config defaults. Caller caches the result."""
    from shared.runpod_client import RunPodClient
    return RunPodClient(
        base_url=Config.RUNPOD_API_BASE_URL,
        timeout=Config.RUNPOD_VERIFICATION_TIMEOUT,
    )


def get_runpod_registry_creds() -> Optional[object]:
    """Build RegistryAuth from RADAR_RUNPOD_REGISTRY_* env vars (or None)."""
    user = os.environ.get("RADAR_RUNPOD_REGISTRY_USERNAME", "")
    pw = os.environ.get("RADAR_RUNPOD_REGISTRY_PASSWORD", "")
    if not user or not pw:
        return None
    from shared.runpod_client import RegistryAuth
    return RegistryAuth(
        server=os.environ.get("RADAR_RUNPOD_REGISTRY_SERVER", "https://index.docker.io/v1/"),
        username=user,
        password=pw,
    )


# ── Per-round deploy / teardown ─────────────────────────────────────


async def deploy_runpod(
    *,
    runpod_client,
    request,
    image: str,
    deployed_image_digest: str,
    hotkey: str,
    netuid: int,
    subtensor_network: str,
    gpu_type_ids: list[str],
    cloud_type: str,
    container_disk_gb: int,
    registry=None,
) -> Deployment:
    """Deploy a RunPod pod with the digest-pinned image.

    The image reference is built as ``image@sha256:<digest>`` so RunPod
    refuses to start if the digest doesn't resolve, and validators
    reject any pod whose ``imageName`` is missing the pin.
    """
    deploy_name = f"radar-trainer-{hotkey[:8]}-{request.round_id}"
    env: dict[str, str] = {}
    if subtensor_network:
        env["SUBTENSOR_NETWORK"] = subtensor_network
    if netuid:
        env["NETUID"] = str(netuid)
    tolerance = os.environ.get("RADAR_EPISTULA_TOLERANCE", "")
    if tolerance:
        env["RADAR_EPISTULA_TOLERANCE"] = tolerance

    if deployed_image_digest and "@sha256:" not in image:
        image_ref = f"{image}@{deployed_image_digest}"
    else:
        image_ref = image

    handle = await runpod_client.deploy_pod(
        image=image_ref,
        gpu_type_ids=gpu_type_ids,
        gpu_count=request.gpu_count,
        name=deploy_name,
        port=8081,
        env=env,
        cloud_type=cloud_type,
        container_disk_gb=container_disk_gb,
        registry=registry,
    )
    return Deployment(
        name=handle.name or deploy_name,
        url=handle.url,
        backend="runpod",
        runpod_pod_id=handle.pod_id,
        gpu_class=handle.gpu_type_id,
        deployed_image_digest=deployed_image_digest,
        raw=handle,
    )


async def teardown_runpod_with_retry(
    runpod_client, pod_id: str, *, attempts: int = 3,
) -> bool:
    """Sync teardown with exponential backoff (1s/2s/4s).

    RunPod bills by uptime; leaks cost real money. Returns True on
    success, False after all retries fail.
    """
    for i in range(attempts):
        try:
            await runpod_client.teardown_pod(pod_id)
            logger.info("RunPod teardown ok for %s (attempt %d)", pod_id, i + 1)
            return True
        except Exception as e:
            wait = 2 ** i
            logger.warning(
                "RunPod teardown failed for %s (attempt %d/%d): %s — retrying in %ds",
                pod_id, i + 1, attempts, e, wait,
            )
            if i + 1 < attempts:
                await asyncio.sleep(wait)
    logger.error("RunPod teardown gave up after %d attempts for %s", attempts, pod_id)
    return False


async def wait_for_runpod_ready(
    *,
    runpod_client,
    pod_id: str,
    trainer_url: str,
    timeout_s: float,
    poll_interval_s: float = 5.0,
) -> str:
    """Poll until pod transitions to RUNNING and trainer /health responds.

    Returns the resolved trainer URL (proxy URL once assigned).
    Reuses ``TargonReadinessTimeout`` so callers don't grow a parallel
    exception type. Tear-down on timeout is the caller's job.
    """
    deadline = time.monotonic() + timeout_s
    last_err = ""
    resolved_url = trainer_url
    healthy = False
    async with httpx.AsyncClient(timeout=5.0) as http:
        while time.monotonic() < deadline:
            try:
                pod = await runpod_client.get_pod(pod_id)
            except Exception as e:
                last_err = f"get_pod: {e}"
                pod = None
            if pod is not None:
                if pod.url:
                    resolved_url = pod.url
                if pod.is_running and resolved_url:
                    try:
                        resp = await http.get(f"{resolved_url.rstrip('/')}/health")
                        if resp.status_code == 200:
                            healthy = True
                    except Exception as e:
                        last_err = f"/health: {e}"
            if healthy:
                return resolved_url
            await asyncio.sleep(poll_interval_s)

    raise TargonReadinessTimeout(
        f"RunPod pod {pod_id} not ready in {timeout_s:.0f}s (last_err={last_err})"
    )


# ── Startup ─────────────────────────────────────────────────────────


async def validate_and_reap_orphans_runpod(client) -> None:
    """Validate credentials then tear down pods from prior processes.

    Raises RuntimeError on auth failure / unreachable API. Logs but
    does not raise on individual orphan teardown failures — operator
    can clean up manually if needed.
    """
    try:
        await client.validate_credentials()
        logger.info("RunPod credentials validated")
    except Exception as e:
        raise RuntimeError(
            f"RunPod credentials invalid or API unreachable at startup: {e}. "
            "Check RUNPOD_API_KEY at https://docs.runpod.io."
        ) from e

    try:
        pods = await client.list_active_pods(name_prefix="radar-trainer-")
    except Exception as e:
        logger.warning("Could not list pods at startup (continuing): %s", e)
        return
    if not pods:
        return
    logger.warning(
        "Found %d orphan RunPod pods from prior process — tearing down",
        len(pods),
    )
    for pod in pods:
        ok = await teardown_runpod_with_retry(client, pod.pod_id)
        logger.info("ORPHAN_TEARDOWN backend=runpod pod_id=%s ok=%s", pod.pod_id, ok)


def parse_gpu_types(env_value: str) -> list[str]:
    """Parse RADAR_RUNPOD_GPU_TYPES into a clean list."""
    return [g.strip() for g in (env_value or "").split(",") if g.strip()]
