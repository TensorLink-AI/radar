"""Miner-side trainer deployment — Basilica, Targon, or RunPod, by config.

Keeps ``miner/neuron.py`` slim. Each backend exposes the same shape
of return (a ``Deployment`` dataclass with the fields the
TrainerReady envelope cares about) so ``handle_prepare`` can be
backend-agnostic. RunPod-specific deploy / teardown lives in
``miner/hosting_runpod.py`` to keep this file focused on the
long-lived-pod backends.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Deployment:
    """Backend-agnostic handle returned by the deploy_* helpers."""

    name: str
    url: str
    # Targon-only — empty for Basilica / RunPod deploys.
    targon_workload_uid: str = ""
    cvm_ip: str = ""
    # Targon + RunPod populate these; Basilica leaves them empty.
    gpu_class: str = ""
    deployed_image_digest: str = ""      # what the miner actually deployed
    # RunPod-only — empty for Basilica / Targon deploys. The endpoint
    # persists across rounds (RunPod manages worker lifecycle); jobs
    # are submitted lazily when the validator dispatches /train.
    runpod_endpoint_id: str = ""
    runpod_template_id: str = ""
    # The backend's native handle:
    #   basilica → BasilicaClient.Deployment object (has ``.delete()``)
    #   targon   → ``WorkloadHandle`` from targon_client
    #   runpod   → ``EndpointInfo`` from runpod_client (no per-round
    #              teardown — endpoints persist; jobs cancel separately)
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
    # Propagate Epistula tolerance so operators can widen the auth window
    # without rebuilding the trainer image. Trainer pods commonly drift
    # from the miner host's clock; the default baked into the image may
    # not be enough on noisy infrastructure.
    tolerance = os.environ.get("RADAR_EPISTULA_TOLERANCE", "")
    if tolerance:
        pod_env["RADAR_EPISTULA_TOLERANCE"] = tolerance

    # Phase-transition callback — gives operators real-time visibility
    # into "pulling / scheduling / health_check / ..." instead of 15
    # minutes of silence before a DeploymentTimeout.
    def _on_progress(status):
        logger.info(
            "BASILICA_PROGRESS name=%s state=%s phase=%s replicas=%d/%d msg=%s",
            deploy_name, status.state, status.phase,
            status.replicas_ready, status.replicas_desired,
            (status.message or "")[:200],
        )

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

    # Split create + wait so a wait timeout/failure still leaves us
    # a handle to fetch logs+events and to tear down the dead pod —
    # client.deploy() raises before returning, losing both.
    def _create_and_wait():
        response = client.create_deployment(
            instance_name=deploy_name,
            image=image,
            replicas=1,
            port=8081,
            env=pod_env,
            memory=request.memory,
            gpu_count=request.gpu_count,
            min_gpu_memory_gb=request.min_gpu_memory_gb,
            gpu_models=deploy_kwargs.get("gpu_models"),
            ttl_seconds=ttl,
            public=True,
            public_metadata=True,
        )
        from basilica._deployment import Deployment as _BDep
        dep = _BDep._from_response(client, response)
        try:
            dep.wait_until_ready(
                timeout=deploy_timeout,
                on_progress=_on_progress,
                silent=True,
            )
            dep.refresh()
            return dep
        except Exception:
            _dump_basilica_failure(client, deploy_name)
            # Tear down the dead pod so it doesn't sit consuming quota
            # until TTL expires.
            try:
                client.delete_deployment(deploy_name)
                logger.info("BASILICA_TEARDOWN name=%s after failed deploy", deploy_name)
            except Exception as te:
                logger.warning("BASILICA_TEARDOWN_FAILED name=%s err=%s", deploy_name, te)
            raise

    deployment = await loop.run_in_executor(None, _create_and_wait)

    return Deployment(name=deployment.name, url=deployment.url, raw=deployment)


def _dump_basilica_failure(client, name: str) -> None:
    """Pull events + tail logs from Basilica and surface them in the miner log."""
    try:
        events = client.get_deployment_events(name, limit=50)
        ev_list = events.get("events", []) if isinstance(events, dict) else []
        for ev in ev_list[-20:]:
            logger.error(
                "BASILICA_EVENT name=%s type=%s reason=%s count=%s ts=%s msg=%s",
                name, ev.get("event_type"), ev.get("reason"),
                ev.get("count"), ev.get("last_timestamp"),
                (ev.get("message") or "")[:300],
            )
    except Exception as e:
        logger.warning("BASILICA_EVENTS_UNAVAILABLE name=%s err=%s", name, e)
    try:
        logs = client.get_deployment_logs(name, tail=200)
        if logs:
            for line in str(logs).splitlines()[-200:]:
                logger.error("BASILICA_LOG name=%s | %s", name, line)
    except Exception as e:
        logger.warning("BASILICA_LOGS_UNAVAILABLE name=%s err=%s", name, e)


def teardown_basilica_sync(deployment) -> None:
    """Synchronous teardown — Basilica's deployment object exposes ``.delete()``."""
    deployment.delete()


# ── Targon path ─────────────────────────────────────────────────────


async def deploy_targon(
    *,
    targon_client,
    request,
    image: str,
    deployed_image_digest: str,
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
    tolerance = os.environ.get("RADAR_EPISTULA_TOLERANCE", "")
    if tolerance:
        env["RADAR_EPISTULA_TOLERANCE"] = tolerance

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
        deployed_image_digest=deployed_image_digest,
        raw=handle,
    )


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


# ── Post-deploy readiness ───────────────────────────────────────────


class TargonReadinessTimeout(Exception):
    """Workload deployed but never became responsive in the budget window."""


async def wait_for_trainer_ready(
    *,
    trainer_url: str,
    cvm_ip: str,
    timeout_s: float = 180.0,
    poll_interval_s: float = 5.0,
    health_path: str = "/health",
) -> None:
    """Poll the trainer's health and the CVM's evidence endpoint until both respond.

    Raises ``TargonReadinessTimeout`` if either is still unresponsive
    after ``timeout_s``. The miner tears down the workload on
    timeout — callers must handle the exception.

    TDX boot takes 60–120s on top of normal container start, so
    default budget is 180s.
    """
    deadline = time.monotonic() + timeout_s
    health_url = f"{trainer_url.rstrip('/')}{health_path}"
    evidence_url = f"http://{cvm_ip}:8080/api/v1/evidence" if cvm_ip else ""

    healthy = False
    # No CVM IP means there's no evidence endpoint to wait for (Basilica,
    # private deploys without raw-IP exposure). Treat as already up.
    evidence_up = not evidence_url
    last_err = ""
    async with httpx.AsyncClient(timeout=5.0) as http:
        while time.monotonic() < deadline:
            if not healthy:
                try:
                    resp = await http.get(health_url)
                    if resp.status_code == 200:
                        healthy = True
                        logger.info("Trainer /health up at %s", health_url)
                except Exception as e:
                    last_err = f"/health: {e}"
            if not evidence_up:
                # We don't actually post a quote — we just probe that the
                # endpoint exists. A 405 / 400 also counts (the server is up).
                try:
                    resp = await http.post(evidence_url, json={"nonce": "ping"})
                    if resp.status_code < 500:
                        evidence_up = True
                        logger.info("CVM evidence endpoint up at %s", evidence_url)
                except Exception as e:
                    last_err = f"evidence: {e}"
            if healthy and evidence_up:
                return
            await asyncio.sleep(poll_interval_s)

    raise TargonReadinessTimeout(
        f"trainer not ready in {timeout_s:.0f}s "
        f"(healthy={healthy}, evidence={evidence_up}, last_err={last_err})"
    )


# ── Teardown with retry ─────────────────────────────────────────────


async def teardown_targon_with_retry(
    targon_client, uid: str, *, attempts: int = 3,
) -> bool:
    """Sync teardown of a Targon workload with exponential backoff (1s/2s/4s).

    Returns True on success. Targon bills by uptime; leaks cost real
    money so we retry harder than the fire-and-forget path. If all
    attempts fail the caller should at least know — surface False
    instead of swallowing.
    """
    for i in range(attempts):
        try:
            await targon_client.teardown_workload(uid)
            logger.info("Targon teardown ok for %s (attempt %d)", uid, i + 1)
            return True
        except Exception as e:
            wait = 2 ** i
            logger.warning(
                "Targon teardown failed for %s (attempt %d/%d): %s — retrying in %ds",
                uid, i + 1, attempts, e, wait,
            )
            if i + 1 < attempts:
                await asyncio.sleep(wait)
    logger.error("Targon teardown gave up after %d attempts for %s", attempts, uid)
    return False
