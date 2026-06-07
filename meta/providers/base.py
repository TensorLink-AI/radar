"""Provider interface.

A provider knows how to create, query, and terminate a pod, and how to
turn a pod_id into an SSH target. Pricing is provider-specific so each
backend implements its own ``cheapest_matching`` to satisfy the
``cost.max_usd_per_hour`` filter from the experiment spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderError(RuntimeError):
    """Raised when the provider API is unavailable or refuses a request."""


@dataclass
class PodSpecRT:
    """Runtime pod spec — the merged result of ``ExperimentSpec.pod`` +
    the resolved pod-type choice from ``cheapest_matching``."""

    gpu: str
    image: str
    disk_gb: int
    max_hours: float
    pod_type_id: str = ""
    hourly_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    # Public key text to inject as authorized_keys on the pod.
    ssh_pubkey: str = ""
    # Env vars to set in the pod's environment (R2 creds, instance id,
    # backup/artifact config). Goes into the pod env at create time.
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Pod:
    pod_id: str
    provider: str
    ssh_host: str
    ssh_port: int = 22
    ssh_user: str = "root"
    hourly_usd: float = 0.0


@dataclass
class PodStatus:
    pod_id: str
    state: str  # one of: starting | running | stopping | dead | unknown
    raw: dict[str, Any] = field(default_factory=dict)


class PodProvider(Protocol):
    """The minimal surface the orchestrator needs from a compute backend."""

    name: str

    def cheapest_matching(
        self, *, gpu: str, max_usd_per_hour: float
    ) -> tuple[str, float] | None:
        """Return (pod_type_id, hourly_usd) for the cheapest pod that
        matches ``gpu`` and costs ≤ ``max_usd_per_hour``, or None if
        nothing qualifies. Spot/community-cloud preferred."""
        ...

    def create(self, spec: PodSpecRT) -> Pod:
        """Provision a pod. Blocks until the pod has an SSH endpoint."""
        ...

    def status(self, pod_id: str) -> PodStatus:
        ...

    def terminate(self, pod_id: str) -> None:
        ...
