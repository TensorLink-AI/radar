"""Targon hosting client — async wrapper around the SDK + tower endpoints.

Wraps the targon-sdk serverless API (deploy / teardown / list), the
``/tha/v2/workloads/verify`` endpoint (raw httpx — not in the SDK),
and TDX+NRAS attestation (see ``shared/targon_attest.py``).

Every call routes through a circuit breaker so a Targon outage
surfaces as ``TargonUnavailable`` instead of bubbling up mid-round.
Validators tag ``targon_unavailable`` rounds with reduced scoring
weight rather than halting. Import-safe in Basilica deployments —
``import targon`` is deferred until first use.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

from shared.targon_attest import (
    AttestationResult,
    fetch_cvm_evidence,
    fresh_nonce,
    parse_tower_response,
    verify_with_tower,
)
from shared.targon_breaker import CircuitBreaker, TargonUnavailable

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.targon.com"
DEFAULT_TOWER_URL = "https://tower.targon.com"
DEFAULT_TIMEOUT = 5.0


class TargonError(Exception):
    """Base for everything raised by this module (other than TargonUnavailable)."""


@dataclass(slots=True)
class WorkloadHandle:
    uid: str
    url: str
    cvm_ip: str
    name: str = ""
    status: str = ""
    gpu_class: str = ""

    @property
    def is_running(self) -> bool:
        return self.status.lower() in ("running", "active", "ready")


@dataclass(slots=True)
class RegistryCreds:
    server: str = "https://index.docker.io/v1/"
    username: str = ""
    password: str = ""


def _extract_cvm_ip(url: str) -> str:
    """Hostname from a Targon workload URL (subdomain or raw IP)."""
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        return urlparse(url).hostname or ""
    except Exception:
        return ""


class TargonClient:
    """Public client. One per process; ``api_key`` defaults to env."""

    def __init__(
        self,
        api_key: str = "",
        *,
        base_url: str = DEFAULT_BASE_URL,
        tower_url: str = DEFAULT_TOWER_URL,
        timeout: float = DEFAULT_TIMEOUT,
        breaker: Optional[CircuitBreaker] = None,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("TARGON_API_KEY", "")
        if not self.api_key:
            raise TargonError("TARGON_API_KEY not set (required for RADAR_HOSTING_BACKEND=targon)")
        self.base_url = base_url.rstrip("/")
        self.tower_url = tower_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.breaker = breaker or CircuitBreaker()
        self._sdk = None

    def _get_sdk_client(self):
        if self._sdk is None:
            import targon
            self._sdk = targon.Client(api_key=self.api_key, timeout=int(self.timeout))
        return self._sdk

    async def aclose(self) -> None:
        if self._sdk is not None:
            try:
                await self._sdk.aclose()
            except Exception:
                pass

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def _call(self, fn, *args, **kwargs):
        # Run fn through the breaker with retries on transport / 5xx.
        # Does not retry on 4xx (auth, bad request — caller's problem).
        await self.breaker.before_call()
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                result = fn(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
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
                    # Final 5xx attempt — fall through to breaker open.
                    break
                # 4xx — not transient. Don't trip the breaker.
                await self.breaker.on_success()
                raise
            except Exception as e:
                last_exc = e
                break
        await self.breaker.on_failure()
        raise TargonUnavailable(f"Targon call failed: {last_exc!r}") from last_exc

    # ── Workload management (miners) ───────────────────────────────

    async def deploy_workload(
        self,
        *,
        image: str,
        gpu_class: str,
        gpu_count: int = 1,
        name: str = "",
        port: int = 8081,
        env: Optional[dict] = None,
        registry: Optional[RegistryCreds] = None,
    ) -> WorkloadHandle:
        # Always passes empty command/args so the image's hardened
        # ENTRYPOINT runs unchanged. Bypass is detected by missing /boot_proof.
        from targon.client.serverless import (
            AsyncServerlessClient, RegistryConfig as SDKRegistry,
        )

        sdk = self._get_sdk_client()
        sl = AsyncServerlessClient(sdk)

        sdk_registry = None
        if registry and registry.username:
            sdk_registry = SDKRegistry(
                server=registry.server,
                username=registry.username,
                password=registry.password,
            )

        async def _deploy():
            return await sl.deploy_container(
                name=name or None,
                image=image,
                resource=gpu_class.lower() if gpu_class else None,
                command=[],
                args=[],
                env=env or {},
                port=port,
                registry=sdk_registry,
            )

        resp = await self._call(_deploy)
        return WorkloadHandle(
            uid=resp.uid,
            url=resp.url,
            cvm_ip=_extract_cvm_ip(resp.url),
            name=resp.name,
            status=resp.status,
            gpu_class=gpu_class,
        )

    async def teardown_workload(self, uid: str) -> None:
        """Delete a workload (best-effort cleanup)."""
        from targon.client.serverless import AsyncServerlessClient
        sdk = self._get_sdk_client()
        sl = AsyncServerlessClient(sdk)

        async def _delete():
            return await sl.delete_container(uid)

        try:
            await self._call(_delete)
        except TargonUnavailable:
            logger.warning("Targon unavailable during teardown of %s — TTL will reap", uid)

    async def list_active_workloads(self) -> list[WorkloadHandle]:
        from targon.client.serverless import AsyncServerlessClient
        sdk = self._get_sdk_client()
        sl = AsyncServerlessClient(sdk)

        async def _list():
            return await sl.list_container()

        items = await self._call(_list)
        out = []
        for item in items:
            urls = getattr(item, "urls", None) or []
            url = urls[0].url if urls else ""
            out.append(WorkloadHandle(
                uid=getattr(item, "uid", ""),
                name=getattr(item, "name", ""),
                url=url,
                cvm_ip=_extract_cvm_ip(url),
                status=getattr(item, "status", ""),
            ))
        return out

    # ── Verification (validators) ──────────────────────────────────

    async def verify_image_digest(self, uid: str, expected_digest: str) -> bool:
        # Hits /tha/v2/workloads/verify directly — not in the SDK.
        url = f"{self.base_url}/tha/v2/workloads/verify"
        payload = {"workload_uid": uid, "expected_digest": expected_digest}

        async def _verify():
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                resp = await http.post(url, json=payload, headers=self._auth_headers())
                resp.raise_for_status()
                return resp.json()

        try:
            result = await self._call(_verify)
        except TargonUnavailable:
            raise
        except Exception as e:
            logger.warning("verify_image_digest failed for %s: %s", uid, e)
            return False
        return bool(result.get("verified", False))

    async def verify_attestation(
        self,
        cvm_ip: str,
        miner_hotkey: str,
        validator_hotkey: str,
        nonce: str = "",
        wallet=None,
    ) -> AttestationResult:
        # Raises TargonUnavailable on outage; returns verified=False on
        # any other failure so the caller can keep exclusion-vs-soft-fail
        # distinct. Per-step protocol lives in shared/targon_attest.py.
        nonce = nonce or fresh_nonce()

        async def _evidence():
            return await fetch_cvm_evidence(
                cvm_ip, nonce, timeout=self.timeout, wallet=wallet,
            )

        try:
            evidence = await self._call(_evidence)
        except TargonUnavailable:
            raise
        except Exception as e:
            return AttestationResult(verified=False, error=f"evidence fetch: {e}")

        async def _verify():
            return await verify_with_tower(
                self.tower_url, self._auth_headers(),
                evidence=evidence,
                cvm_ip=cvm_ip,
                miner_hotkey=miner_hotkey,
                validator_hotkey=validator_hotkey,
                nonce=nonce,
                timeout=self.timeout,
            )

        try:
            verdict = await self._call(_verify)
        except TargonUnavailable:
            raise
        except Exception as e:
            return AttestationResult(verified=False, error=f"tower verify: {e}")

        return parse_tower_response(verdict)
