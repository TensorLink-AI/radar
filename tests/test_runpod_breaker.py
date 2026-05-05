"""Tests for shared/runpod_breaker.py — parallel to test_targon_breaker.

Same state machine, different exception class.
"""

from __future__ import annotations

import asyncio

import pytest

from shared.runpod_breaker import RunpodCircuitBreaker, RunpodUnavailable


class _Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.mark.asyncio
async def test_breaker_starts_closed_and_passes_calls():
    clk = _Clock()
    b = RunpodCircuitBreaker(threshold=3, reset_after=10.0, clock=clk)
    assert b.state == "closed"
    await b.before_call()
    await b.on_success()
    assert b.state == "closed"


@pytest.mark.asyncio
async def test_breaker_opens_after_threshold_failures():
    clk = _Clock()
    b = RunpodCircuitBreaker(threshold=2, reset_after=10.0, clock=clk)
    await b.before_call(); await b.on_failure()
    await b.before_call(); await b.on_failure()
    assert b.state == "open"
    with pytest.raises(RunpodUnavailable):
        await b.before_call()


@pytest.mark.asyncio
async def test_breaker_recovers_via_half_open_success():
    clk = _Clock()
    b = RunpodCircuitBreaker(threshold=1, reset_after=5.0, clock=clk)
    await b.before_call(); await b.on_failure()
    assert b.state == "open"
    clk.t = 6.0
    assert b.state == "half_open"
    await b.before_call()
    await b.on_success()
    assert b.state == "closed"


@pytest.mark.asyncio
async def test_half_open_failure_reopens_breaker():
    clk = _Clock()
    b = RunpodCircuitBreaker(threshold=1, reset_after=5.0, clock=clk)
    await b.before_call(); await b.on_failure()
    clk.t = 6.0
    await b.before_call()
    await b.on_failure()
    assert b.state == "open"


@pytest.mark.asyncio
async def test_half_open_only_allows_one_trial():
    clk = _Clock()
    b = RunpodCircuitBreaker(threshold=1, reset_after=5.0, clock=clk)
    await b.before_call(); await b.on_failure()
    clk.t = 6.0
    await b.before_call()  # claim the trial
    with pytest.raises(RunpodUnavailable):
        await b.before_call()
