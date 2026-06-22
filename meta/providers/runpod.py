"""RunPod provider — minimal REST + GraphQL surface.

This is intentionally thin: only the verbs the orchestrator needs
(price-list, create, status, terminate). It expects an API key in
``RUNPOD_API_KEY``. If the key is missing, the methods raise
``ProviderError`` rather than silently producing bad results — the
scheduler will surface the failure to the user.

The price-filter prefers spot/community-cloud instances when available.
"""

from __future__ import annotations

import json
import os
import time

from .base import Pod, PodProvider, PodSpecRT, PodStatus, ProviderError


_API = "https://api.runpod.io/graphql"


def _http_post(url: str, body: dict, *, headers: dict) -> dict:
    try:
        import httpx  # type: ignore
    except ImportError as e:
        raise ProviderError(
            "RunPod provider needs httpx; install `pip install -e .[meta]`"
        ) from e
    resp = httpx.post(url, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


class RunPodProvider:
    name = "runpod"

    def __init__(self) -> None:
        self._key = os.environ.get("RUNPOD_API_KEY", "")
        if not self._key:
            # Lazy — only raise on first use, so importing the provider
            # for tests/CI doesn't hard-fail.
            pass

    def _headers(self) -> dict[str, str]:
        if not self._key:
            raise ProviderError(
                "RUNPOD_API_KEY is not set; cannot talk to RunPod API"
            )
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }

    def _gql(self, query: str, variables: dict | None = None) -> dict:
        body = {"query": query}
        if variables is not None:
            body["variables"] = variables
        out = _http_post(_API, body, headers=self._headers())
        if "errors" in out:
            raise ProviderError(
                f"RunPod API error: {json.dumps(out['errors'])[:500]}"
            )
        return out.get("data") or {}

    # ── price discovery ────────────────────────────────────────────────

    _PRICE_QUERY = """
    query GpuTypes {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        lowestPrice(input: { gpuCount: 1, minMemoryInGb: 0 }) {
          uninterruptablePrice
          minimumBidPrice
        }
      }
    }
    """

    def cheapest_matching(
        self, *, gpu: str, max_usd_per_hour: float
    ) -> tuple[str, float] | None:
        """Walk RunPod's GPU types and return the cheapest one whose
        displayName/id matches ``gpu`` family at ≤ max_usd_per_hour.

        Prefers the lower of (spot minimumBidPrice, on-demand
        uninterruptablePrice).
        """
        data = self._gql(self._PRICE_QUERY)
        gpu_lc = gpu.lower()
        candidates: list[tuple[str, float]] = []
        for g in data.get("gpuTypes", []) or []:
            name = (g.get("displayName") or "") + " " + (g.get("id") or "")
            if gpu_lc not in name.lower():
                continue
            lp = g.get("lowestPrice") or {}
            spot = lp.get("minimumBidPrice")
            on_demand = lp.get("uninterruptablePrice")
            prices = [p for p in (spot, on_demand) if p is not None]
            if not prices:
                continue
            cheapest = min(prices)
            if cheapest <= max_usd_per_hour:
                candidates.append((g["id"], float(cheapest)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0]

    # ── pod lifecycle ──────────────────────────────────────────────────

    _CREATE_MUT = """
    mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        machineId
        runtime {
          ports { ip publicPort privatePort type isIpPublic }
        }
      }
    }
    """

    def create(self, spec: PodSpecRT) -> Pod:
        env_pairs = [
            {"key": k, "value": v} for k, v in (spec.env or {}).items()
        ]
        variables = {
            "input": {
                "cloudType": "ALL",
                "gpuCount": 1,
                "volumeInGb": 0,
                "containerDiskInGb": spec.disk_gb,
                "minVcpuCount": 4,
                "minMemoryInGb": 16,
                "gpuTypeId": spec.pod_type_id,
                "name": f"radar-{int(time.time())}",
                "imageName": spec.image or "runpod/pytorch:2.1.0-py3.10-cuda12.1",
                "dockerArgs": "",
                "ports": "22/tcp,8765/http",
                "env": env_pairs,
            }
        }
        data = self._gql(self._CREATE_MUT, variables)
        pod = data.get("podFindAndDeployOnDemand")
        if not pod or not pod.get("id"):
            raise ProviderError(f"RunPod deploy returned no pod: {data!r}")
        pod_id = pod["id"]
        host, port = self._wait_for_ssh(pod_id)
        return Pod(
            pod_id=pod_id,
            provider=self.name,
            ssh_host=host,
            ssh_port=port,
            ssh_user="root",
            hourly_usd=spec.hourly_usd,
        )

    _POD_QUERY = """
    query Pod($input: PodFilter!) {
      pod(input: $input) {
        id
        desiredStatus
        lastStatusChange
        runtime {
          ports { ip publicPort privatePort type isIpPublic }
        }
      }
    }
    """

    def _fetch_pod(self, pod_id: str) -> dict:
        data = self._gql(self._POD_QUERY, {"input": {"podId": pod_id}})
        return data.get("pod") or {}

    def _wait_for_ssh(
        self, pod_id: str, *, timeout: float = 600.0
    ) -> tuple[str, int]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            p = self._fetch_pod(pod_id)
            for port in (p.get("runtime") or {}).get("ports") or []:
                if port.get("privatePort") == 22 and port.get("isIpPublic"):
                    return port["ip"], int(port["publicPort"])
            time.sleep(5.0)
        raise ProviderError(
            f"RunPod pod {pod_id} did not expose an SSH port within "
            f"{timeout}s"
        )

    def status(self, pod_id: str) -> PodStatus:
        try:
            p = self._fetch_pod(pod_id)
        except ProviderError:
            return PodStatus(pod_id=pod_id, state="unknown")
        if not p:
            return PodStatus(pod_id=pod_id, state="dead")
        desired = (p.get("desiredStatus") or "").lower()
        state = {
            "running": "running",
            "starting": "starting",
            "exited": "dead",
            "terminated": "dead",
        }.get(desired, desired or "unknown")
        return PodStatus(pod_id=pod_id, state=state, raw=p)

    _TERMINATE_MUT = """
    mutation Terminate($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """

    def terminate(self, pod_id: str) -> None:
        self._gql(self._TERMINATE_MUT, {"input": {"podId": pod_id}})
