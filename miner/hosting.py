"""Miner-side trainer deployment — Basilica or Targon, picked by config.

Keeps ``miner/neuron.py`` slim. Each backend exposes the same shape
of return (a ``Deployment`` namedtuple-ish object with the fields the
TrainerReady envelope cares about) so ``handle_prepare`` can be
backend-agnostic.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Deployment:
    """Backend-agnostic handle returned by the deploy_* helpers."""

    name: str
    url: str
    # Targon-only — empty for Basilica deploys.
    targon_workload_uid: str = ""
    cvm_ip: str = ""
    gpu_class: str = ""
    image_digest: str = ""
    # The backend's native handle (basilica deployment object, or
    # the WorkloadHandle from targon_client). Used for teardown.
    raw: object = None


# ── Basilica path (legacy default) ──────────────────────────────────


async def deploy_basilica(
    *,
    request,
    image: str,
    hotkey: str,
    netuid: int,
    subtensor_network: str,
    ttl: int,
    deploy_timeout: int = 900,
) -> Deployment:
    """Deploy a Basilica pod and enroll it for public metadata."""
    from basilica import BasilicaClient
    client = BasilicaClient()

    deploy_name = f"radar-trainer-{hotkey[:8]}-{request.round_id}"
    pod_env: dict[str, str] = {}
    if subtensor_network:
        pod_env["SUBTENSOR_NETWORK"] = subtensor_network
    if netuid:
        pod_env["NETUID"] = str(netuid)

    deploy_kwargs = dict(
        name=deploy_name,
        image=image,
        port=8081,
        public=True,
        replicas=1,
        ttl_seconds=ttl,
        gpu_count=request.gpu_count,
        min_gpu_memory_gb=request.min_gpu_memory_gb,
        memory=request.memory,
        env=pod_env,
        timeout=deploy_timeout,
    )
    gpu_models = os.environ.get("RADAR_TRAINER_GPU_MODELS", "")
    if gpu_models:
        deploy_kwargs["gpu_models"] = [
            m.strip() for m in gpu_models.split(",") if m.strip()
        ]

    loop = asyncio.get_event_loop()
    deployment = await loop.run_in_executor(
        None, functools.partial(client.deploy, **deploy_kwargs),
    )

    # Enroll for public metadata so validators can verify the pod.
    try:
        await loop.run_in_executor(
            None,
            functools.partial(client.enroll_metadata, deployment.name, enabled=True),
        )
    except Exception as e:
        logger.warning("Failed to enroll public metadata for %s: %s", deployment.name, e)

    return Deployment(name=deployment.name, url=deployment.url, raw=deployment)


def teardown_basilica_sync(deployment) -> None:
    """Synchronous teardown — Basilica's deployment object exposes ``.delete()``."""
    deployment.delete()


# ── Targon path ─────────────────────────────────────────────────────


async def deploy_targon(
    *,
    targon_client,
    request,
    image: str,
    image_digest: str,
    hotkey: str,
    netuid: int,
    subtensor_network: str,
    gpu_class: str,
    registry=None,
) -> Deployment:
    """Deploy a Targon confidential GPU pod with the hardened ENTRYPOINT."""
    deploy_name = f"radar-trainer-{hotkey[:8]}-{request.round_id}"
    env: dict[str, str] = {}
    if subtensor_network:
        env["SUBTENSOR_NETWORK"] = subtensor_network
    if netuid:
        env["NETUID"] = str(netuid)

    handle = await targon_client.deploy_workload(
        image=image,
        gpu_class=gpu_class,
        gpu_count=request.gpu_count,
        name=deploy_name,
        port=8081,
        env=env,
        registry=registry,
    )
    return Deployment(
        name=handle.name or deploy_name,
        url=handle.url,
        targon_workload_uid=handle.uid,
        cvm_ip=handle.cvm_ip,
        gpu_class=gpu_class,
        image_digest=image_digest,
        raw=handle,
    )


async def teardown_targon(targon_client, uid: str) -> None:
    await targon_client.teardown_workload(uid)


# ── Backend dispatch ────────────────────────────────────────────────


def get_targon_registry_creds() -> Optional[object]:
    """Build RegistryCreds from RADAR_TARGON_REGISTRY_* env vars (or None)."""
    user = os.environ.get("RADAR_TARGON_REGISTRY_USERNAME", "")
    pw = os.environ.get("RADAR_TARGON_REGISTRY_PASSWORD", "")
    if not user or not pw:
        return None
    from shared.targon_client import RegistryCreds
    return RegistryCreds(
        server=os.environ.get("RADAR_TARGON_REGISTRY_SERVER", "https://index.docker.io/v1/"),
        username=user,
        password=pw,
    )
