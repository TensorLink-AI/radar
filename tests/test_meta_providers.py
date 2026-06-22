"""Tests for the provider abstraction (using MockProvider)."""

from __future__ import annotations

import pytest

from meta.providers import get_provider
from meta.providers.base import PodSpecRT, ProviderError


def test_get_provider_unknown_raises():
    with pytest.raises(ProviderError):
        get_provider("does-not-exist")


def test_mock_cheapest_filters_by_price():
    p = get_provider("mock")
    # A100-40GB spot is $0.29, on-demand is $0.79.
    assert p.cheapest_matching(gpu="A100-40GB", max_usd_per_hour=0.30) \
        == ("a100-40gb-spot", 0.29)
    # At $0.20 nothing qualifies.
    assert p.cheapest_matching(gpu="A100-40GB", max_usd_per_hour=0.20) is None
    # At $1.00 we still pick the cheaper (spot) option.
    assert p.cheapest_matching(gpu="A100-40GB", max_usd_per_hour=1.00) \
        == ("a100-40gb-spot", 0.29)


def test_mock_unknown_gpu_family_returns_none():
    p = get_provider("mock")
    assert p.cheapest_matching(
        gpu="MADE-UP-3000", max_usd_per_hour=10.0
    ) is None


def test_mock_lifecycle():
    p = get_provider("mock")
    spec = PodSpecRT(
        gpu="A100-40GB", image="img", disk_gb=100, max_hours=1,
        pod_type_id="a100-40gb-spot", hourly_usd=0.29,
    )
    pod = p.create(spec)
    assert pod.pod_id.startswith("mock-")
    assert pod.hourly_usd == 0.29

    st = p.status(pod.pod_id)
    assert st.state == "running"

    p.terminate(pod.pod_id)
    st = p.status(pod.pod_id)
    assert st.state == "dead"
