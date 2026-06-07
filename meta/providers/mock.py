"""In-process mock provider for tests + dry runs.

Tracks pods in memory; ``status()`` returns 'running' immediately after
``create()`` and 'dead' after ``terminate()``. Cheapest-matching is a
static price table keyed on GPU family so the cost-filter logic in the
scheduler can be tested without a network call.
"""

from __future__ import annotations

import itertools

from .base import Pod, PodProvider, PodSpecRT, PodStatus


# Static price table for the unit tests. Real providers query their API.
_PRICES: dict[str, list[tuple[str, float]]] = {
    "A100-40GB": [("a100-40gb-spot", 0.29), ("a100-40gb", 0.79)],
    "A100-80GB": [("a100-80gb-spot", 0.45), ("a100-80gb", 1.19)],
    "H100": [("h100-spot", 0.95), ("h100", 2.49)],
    "RTX-4090": [("4090-spot", 0.22), ("4090", 0.39)],
}


class MockProvider:
    name = "mock"

    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._pods: dict[str, PodStatus] = {}

    def cheapest_matching(
        self, *, gpu: str, max_usd_per_hour: float
    ) -> tuple[str, float] | None:
        candidates = sorted(_PRICES.get(gpu, []), key=lambda x: x[1])
        for pod_type_id, hourly in candidates:
            if hourly <= max_usd_per_hour:
                return pod_type_id, hourly
        return None

    def create(self, spec: PodSpecRT) -> Pod:
        pod_id = f"mock-{next(self._counter):04d}"
        self._pods[pod_id] = PodStatus(pod_id=pod_id, state="running")
        return Pod(
            pod_id=pod_id,
            provider=self.name,
            ssh_host=f"127.0.0.1",
            ssh_port=2200 + (next(self._counter) % 100),
            ssh_user="root",
            hourly_usd=spec.hourly_usd,
        )

    def status(self, pod_id: str) -> PodStatus:
        return self._pods.get(
            pod_id, PodStatus(pod_id=pod_id, state="unknown")
        )

    def terminate(self, pod_id: str) -> None:
        if pod_id in self._pods:
            self._pods[pod_id] = PodStatus(pod_id=pod_id, state="dead")
