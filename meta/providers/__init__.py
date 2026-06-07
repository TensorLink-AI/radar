"""Pluggable pod providers.

Each provider implements ``meta.providers.base.PodProvider``. The
``mock`` provider is dependency-free and used in tests; ``runpod`` is
the first real one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Pod, PodProvider, PodStatus, ProviderError

if TYPE_CHECKING:
    pass


def get_provider(name: str) -> PodProvider:
    if name == "mock":
        from .mock import MockProvider
        return MockProvider()
    if name == "runpod":
        from .runpod import RunPodProvider
        return RunPodProvider()
    raise ProviderError(f"unknown provider: {name!r}")


__all__ = ["Pod", "PodProvider", "PodStatus", "ProviderError", "get_provider"]
