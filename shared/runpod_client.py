"""RunPod hosting client — async wrapper around the REST API.

RunPod is a non-attested commodity GPU cloud. We use it as a third
backend (alongside Basilica and Targon) for operators who need wider
GPU selection (A100, RTX 4090, L40S, etc.) at lower price points
than confidential compute. Trade-offs vs Targon:

  - No TDX quote, no NRAS GPU token, no hardware-rooted attestation.
  - Image-bytes verification is "RunPod's API says the pod is running
    image@sha256:...", not a TDX measurement. Sufficient against lazy
    cheaters; insufficient against a determined attacker who
    compromises RunPod or the registry.
  - Covered by ``Config.NON_ATTESTED_SCORE_MULTIPLIER`` so honest
    RunPod miners get reduced (not zero) weight.

The verification chain we expose:

  1. Validator queries ``GET /pods/{id}`` and confirms ``imageName``
     contains the expected ``@sha256:`` pin.
  2. Validator hits the trainer's ``/boot_proof`` (cross-check; soft).

No CVM, no tower, no breaker — failures are returned as ``False`` /
exceptions and the caller decides whether to exclude the miner.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://rest.runpod.io/v1"
DEFAULT_TIMEOUT = 10.0


class RunPodError(Exception):
    """Base for everything raised by this module."""


@dataclass(slots=True)
class PodHandle:
    """Backend-agnostic snapshot of a RunPod pod."""

    pod_id: str
    name: str = ""
    image_name: str = ""           # full ref: "repo:tag@sha256:..."
    status: str = ""               # RUNNING / EXITED / etc.
    public_ip: str = ""
    public_port: int = 0
    proxy_url: str = ""            # https://{pod_id}-{port}.proxy.runpod.net
    gpu_type_id: str = ""

    @property
    def is_running(self) -> bool:
        return self.status.upper() in ("RUNNING", "READY")

    @property
    def url(self) -> str:
        """Best trainer URL: prefer the proxy URL (TLS) over raw IP."""
        if self.proxy_url:
            return self.proxy_url
        if self.public_ip and self.public_port:
            return f"http://{self.public_ip}:{self.public_port}"
        return ""


@dataclass(slots=True)
class RegistryAuth:
    """Optional private-registry credentials for image pulls."""
    username: str = ""
    password: str = ""
    server: str = ""


def _proxy_url(pod_id: str, port: int) -> str:
    if not pod_id or not port:
        return ""
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def _parse_pod(payload: dict) -> PodHandle:
    """Map a RunPod pod payload into a PodHandle."""
    pod_id = str(payload.get("id") or "")
    image_name = str(payload.get("imageName") or "")
    status = str(
        payload.get("desiredStatus")
        or payload.get("status")
        or ""
    )
    name = str(payload.get("name") or "")

    public_ip = ""
    public_port = 0
    proxy_url = ""
    runtime = payload.get("runtime")
    if isinstance(runtime, dict):
        for p in (runtime.get("ports") or []):
            if not isinstance(p, dict):
                continue
            if p.get("isIpPublic") and p.get("ip") and p.get("publicPort"):
                public_ip = str(p.get("ip"))
                try:
                    public_port = int(p.get("publicPort"))
                except (TypeError, ValueError):
                    public_port = 0
                break
    # Default RunPod proxy port matches the container port we expose (8081).
    proxy_url = _proxy_url(pod_id, 8081)

    gpu_type_id = ""
    machine = payload.get("machine") or payload.get("gpu") or {}
    if isinstance(machine, dict):
        gpu_type_id = str(
            machine.get("gpuTypeId")
            or machine.get("gpuType")
            or machine.get("displayName")
            or ""
        )

    return PodHandle(
        pod_id=pod_id,
        name=name,
        image_name=image_name,
        status=status,
        public_ip=public_ip,
        public_port=public_port,
        proxy_url=proxy_url,
        gpu_type_id=gpu_type_id,
    )


class RunPodClient:
    """Public client. One per process; ``api_key`` defaults to env."""

    def __init__(
        self,
        api_key: str = "",
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        if not self.api_key:
            raise RunPodError(
                "RUNPOD_API_KEY not set (required for RADAR_HOSTING_BACKEND=runpod)"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _request(
        self, method: str, path: str, *, json_body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        url = f"{self.base_url}{path}"
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as http:
                    resp = await http.request(
                        method, url,
                        headers=self._headers(),
                        json=json_body,
                        params=params,
                    )
                    if 500 <= resp.status_code < 600:
                        last_exc = httpx.HTTPStatusError(
                            f"5xx from {url}", request=resp.request, response=resp,
                        )
                        if attempt + 1 < self.max_retries:
                            await asyncio.sleep(min(2 ** attempt, 4))
                            continue
                    resp.raise_for_status()
                    if resp.status_code == 204 or not resp.content:
                        return {}
                    return resp.json()
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 4))
                    continue
            except httpx.HTTPStatusError:
                raise
        raise RunPodError(f"RunPod {method} {path} failed: {last_exc!r}") from last_exc

    # ── Pod management (miners) ────────────────────────────────────

    async def deploy_pod(
        self,
        *,
        image: str,
        gpu_type_ids: list[str],
        gpu_count: int = 1,
        name: str = "",
        port: int = 8081,
        env: Optional[dict] = None,
        cloud_type: str = "SECURE",
        container_disk_gb: int = 50,
        registry: Optional[RegistryAuth] = None,
    ) -> PodHandle:
        """Create a pod. Picks the first available ``gpu_type_ids``.

        We pin ``imageName`` exactly as supplied (caller should pass
        ``image@sha256:...``). RunPod will refuse to start if the
        digest doesn't resolve. Validators will reject any pod whose
        ``imageName`` is missing the ``@sha256:`` pin.
        """
        body: dict = {
            "name": name or "radar-trainer",
            "imageName": image,
            "gpuTypeIds": [g for g in gpu_type_ids if g],
            "gpuCount": gpu_count,
            "containerDiskInGb": container_disk_gb,
            "ports": f"{port}/http",
            "env": env or {},
            "cloudType": cloud_type.upper(),
        }
        if registry and registry.username:
            body["containerRegistryAuth"] = {
                "username": registry.username,
                "password": registry.password,
                "registry": registry.server or "https://index.docker.io/v1/",
            }
        payload = await self._request("POST", "/pods", json_body=body)
        return _parse_pod(payload)

    async def get_pod(self, pod_id: str) -> Optional[PodHandle]:
        try:
            payload = await self._request("GET", f"/pods/{pod_id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        return _parse_pod(payload)

    async def teardown_pod(self, pod_id: str) -> None:
        """Best-effort delete. 404 is treated as success (already gone)."""
        try:
            await self._request("DELETE", f"/pods/{pod_id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return
            raise

    async def list_active_pods(self, name_prefix: str = "") -> list[PodHandle]:
        params = {"name": name_prefix} if name_prefix else None
        payload = await self._request("GET", "/pods", params=params)
        items = payload.get("pods") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return []
        out = []
        for item in items:
            if isinstance(item, dict):
                out.append(_parse_pod(item))
        return out

    async def validate_credentials(self) -> None:
        """Cheap startup check. Raises RunPodError on auth failure."""
        try:
            await self._request("GET", "/pods")
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise RunPodError(
                    f"RUNPOD_API_KEY rejected by RunPod (HTTP {e.response.status_code})"
                ) from e
            raise

    # ── Verification (validators) ──────────────────────────────────

    async def verify_pod_image(self, pod_id: str, expected_digest: str) -> tuple[bool, str]:
        """Confirm the pod is running the digest-pinned image.

        Returns ``(ok, reason)``. ``ok=True`` requires:

          1. RunPod API returns the pod (not 404 / not torn down).
          2. ``imageName`` contains a ``@sha256:`` pin.
          3. The pin matches ``expected_digest``.

        Trust note: this is "RunPod says so", not a hardware
        measurement. A determined attacker who compromises RunPod can
        defeat it. Sufficient to catch lazy tampering; pair with
        ``NON_ATTESTED_SCORE_MULTIPLIER`` for residual risk.
        """
        if not expected_digest:
            return False, "no expected digest pinned (set OFFICIAL_TRAINING_IMAGE_DIGEST)"
        pod = await self.get_pod(pod_id)
        if pod is None:
            return False, f"pod {pod_id} not found"
        image_ref = pod.image_name
        if "@sha256:" not in image_ref:
            return False, f"image {image_ref!r} not digest-pinned"
        actual = image_ref.split("@", 1)[1]
        if actual != expected_digest:
            return False, f"digest mismatch: pod has {actual} expected {expected_digest}"
        return True, ""
