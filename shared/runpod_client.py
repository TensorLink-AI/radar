"""RunPod Serverless control-plane wrapper — async, breaker-guarded.

Async wrapper around RunPod's HTTP API for the three operations the
miner and validator need:

  - ``submit_job(endpoint_id, payload)`` — POST /v2/{id}/run
  - ``get_status(endpoint_id, job_id)`` — GET /v2/{id}/status/{job_id}
  - ``cancel_job(endpoint_id, job_id)`` — POST /v2/{id}/cancel/{job_id}
  - ``get_endpoint(endpoint_id)`` — GET /graphql, returns template digest
  - ``list_in_flight_jobs(endpoint_id)`` — GET /v2/{id}/health-style polling
  - ``validate_credentials()`` — cheap call used at miner startup

All calls route through a circuit breaker so a RunPod outage surfaces
as ``RunpodUnavailable`` instead of bubbling up mid-round. Validators
tag ``runpod_unavailable`` rounds with reduced scoring weight rather
than halting (parallel to the Targon hybrid-fallback).

Import-safe: callers don't pay any RunPod-SDK import cost unless they
construct a ``RunpodClient``.

Why the bare-httpx implementation rather than a SDK wrapper: RunPod's
official Python SDK is a thin shim over these HTTP endpoints, and
shipping it as a hard dependency for every miner / validator that
might never enable the backend is a worse trade than the ~150 lines
here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

from shared.runpod_breaker import RunpodCircuitBreaker, RunpodUnavailable

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.runpod.ai"
DEFAULT_TIMEOUT = 10.0


class RunpodError(Exception):
    """Base for everything raised by this module (other than RunpodUnavailable)."""


@dataclass(slots=True)
class JobHandle:
    """Reference to a submitted RunPod serverless job."""

    job_id: str
    endpoint_id: str
    status: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status.upper() in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")


@dataclass(slots=True)
class EndpointInfo:
    """Subset of the RunPod endpoint metadata we care about for verification."""

    endpoint_id: str
    template_id: str = ""
    image_name: str = ""
    image_digest: str = ""   # extracted from image_name when present (sha256:...)
    workers_running: int = 0
    workers_max: int = 0


class RunpodClient:
    """One client per process. ``api_key`` defaults to the env var."""

    def __init__(
        self,
        api_key: str = "",
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        breaker: Optional[RunpodCircuitBreaker] = None,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        if not self.api_key:
            raise RunpodError(
                "RUNPOD_API_KEY not set (required for RADAR_HOSTING_BACKEND=runpod)"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.breaker = breaker or RunpodCircuitBreaker()

    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _call(self, fn):
        """Run ``fn`` (an async callable) through the breaker with retries.

        Retries on transport errors and 5xx; does not retry 4xx. Mirrors
        ``TargonClient._call`` so a future shared base class can absorb
        both without behaviour changes.
        """
        await self.breaker.before_call()
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                result = await fn()
                await self.breaker.on_success()
                return result
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 4))
                    continue
            except httpx.HTTPStatusError as e:
                if 500 <= e.response.status_code < 600:
                    last_exc = e
                    if attempt + 1 < self.max_retries:
                        await asyncio.sleep(min(2 ** attempt, 4))
                        continue
                    break
                # 4xx — caller's problem, don't trip the breaker.
                await self.breaker.on_success()
                raise
            except Exception as e:
                last_exc = e
                break
        await self.breaker.on_failure()
        raise RunpodUnavailable(f"RunPod call failed: {last_exc!r}") from last_exc

    # ── Job control (miner) ────────────────────────────────────

    async def submit_job(
        self,
        endpoint_id: str,
        input_payload: dict,
        *,
        webhook: str = "",
    ) -> JobHandle:
        """POST /v2/{endpoint_id}/run.

        RunPod returns immediately with ``{"id": "...", "status": "IN_QUEUE"}``.
        The actual training happens asynchronously on a worker; the
        worker uploads its result to R2 (via the presigned URLs in the
        input payload) and the validator polls R2 for completion.
        """
        if not endpoint_id:
            raise RunpodError("submit_job: endpoint_id is required")
        url = f"{self.base_url}/v2/{endpoint_id}/run"
        body = {"input": input_payload}
        if webhook:
            body["webhook"] = webhook

        async def _post():
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                resp = await http.post(url, json=body, headers=self._auth_headers())
                resp.raise_for_status()
                return resp.json()

        data = await self._call(_post)
        return JobHandle(
            job_id=data.get("id", ""),
            endpoint_id=endpoint_id,
            status=data.get("status", ""),
        )

    async def get_status(self, endpoint_id: str, job_id: str) -> JobHandle:
        """GET /v2/{endpoint_id}/status/{job_id}."""
        if not endpoint_id or not job_id:
            raise RunpodError("get_status: endpoint_id and job_id are required")
        url = f"{self.base_url}/v2/{endpoint_id}/status/{job_id}"

        async def _get():
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                resp = await http.get(url, headers=self._auth_headers())
                resp.raise_for_status()
                return resp.json()

        data = await self._call(_get)
        return JobHandle(
            job_id=job_id,
            endpoint_id=endpoint_id,
            status=data.get("status", ""),
        )

    async def cancel_job(self, endpoint_id: str, job_id: str) -> None:
        """POST /v2/{endpoint_id}/cancel/{job_id}.

        Best-effort: a job in IN_PROGRESS can take a few seconds to
        actually stop on the worker. We don't block waiting for the
        confirmation — that's what mid-round status polling is for.
        """
        if not endpoint_id or not job_id:
            return
        url = f"{self.base_url}/v2/{endpoint_id}/cancel/{job_id}"

        async def _post():
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                resp = await http.post(url, headers=self._auth_headers())
                # 4xx on cancel commonly means "job already terminal" — fine.
                if resp.status_code >= 500:
                    resp.raise_for_status()
                return None

        try:
            await self._call(_post)
        except RunpodUnavailable:
            logger.warning(
                "RunPod unavailable during cancel of job %s — letting endpoint TTL it",
                job_id,
            )

    # ── Endpoint metadata (verification) ─────────────────────

    async def get_endpoint(self, endpoint_id: str) -> EndpointInfo:
        """Read endpoint metadata via the GraphQL API.

        RunPod's REST surface for endpoints is read-only and limited;
        the supported metadata read is GraphQL ``myself.endpoints``.
        We pull the full list and filter by id — sound for the typical
        case (a miner has a handful of endpoints) and lets us avoid
        carrying a per-endpoint API URL.

        Returns a parsed ``EndpointInfo``; the caller compares
        ``image_digest`` against ``Config.OFFICIAL_TRAINING_IMAGE_DIGEST``.
        """
        if not endpoint_id:
            raise RunpodError("get_endpoint: endpoint_id is required")
        url = f"{self.base_url}/graphql"
        query = """
        query Endpoints {
          myself {
            endpoints {
              id
              templateId
              workersRunning
              workersMax
              template { imageName }
            }
          }
        }
        """

        async def _post():
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                resp = await http.post(
                    url,
                    json={"query": query},
                    headers=self._auth_headers(),
                )
                resp.raise_for_status()
                return resp.json()

        data = await self._call(_post)
        endpoints = (
            data.get("data", {})
            .get("myself", {})
            .get("endpoints", [])
        ) or []
        for ep in endpoints:
            if ep.get("id") == endpoint_id:
                image = (ep.get("template") or {}).get("imageName", "") or ""
                return EndpointInfo(
                    endpoint_id=endpoint_id,
                    template_id=ep.get("templateId", "") or "",
                    image_name=image,
                    image_digest=_extract_digest(image),
                    workers_running=int(ep.get("workersRunning", 0) or 0),
                    workers_max=int(ep.get("workersMax", 0) or 0),
                )
        return EndpointInfo(endpoint_id=endpoint_id)

    async def validate_credentials(self) -> None:
        """Cheap startup check. Raises ``RunpodError`` on auth failure;
        raises ``RunpodUnavailable`` if the API is down at boot."""
        url = f"{self.base_url}/graphql"

        async def _post():
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                resp = await http.post(
                    url,
                    json={"query": "query { myself { id } }"},
                    headers=self._auth_headers(),
                )
                resp.raise_for_status()
                return resp.json()

        try:
            await self._call(_post)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise RunpodError(
                    f"RunPod credentials invalid (HTTP {e.response.status_code}). "
                    f"Check RUNPOD_API_KEY at https://www.runpod.io/console/user/settings."
                ) from e
            raise


def _extract_digest(image_name: str) -> str:
    """Pull the ``sha256:...`` digest from an OCI image reference, if any.

    RunPod accepts both tag-form (``ghcr.io/foo/bar:v1``) and digest-form
    (``ghcr.io/foo/bar@sha256:abc...``) image references. Only the
    digest form is verifiable — tag-form returns empty so the caller
    can detect "not pinned" and fail closed when verification matters.
    """
    if "@sha256:" not in image_name:
        return ""
    return "sha256:" + image_name.split("@sha256:", 1)[1]
