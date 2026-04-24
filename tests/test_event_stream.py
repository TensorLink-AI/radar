"""Tests for validator.event_stream — buffer, handler, install helper.

No Postgres required; uses an in-memory fake DatabaseClient.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from validator.event_stream import (
    EventBuffer, EventLoggingHandler, attach_to_loggers,
)


class FakeDBClient:
    def __init__(self, succeed: bool = True):
        self.succeed = succeed
        self.batches: list[list[dict]] = []
        self.calls = 0

    async def post_events(self, batch):
        self.calls += 1
        if not self.succeed:
            return False
        self.batches.append(list(batch))
        return True


def test_log_metric_phase_enqueue():
    db = FakeDBClient()
    buf = EventBuffer(db, hotkey="hk", flush_interval_s=60)
    buf.set_round(7)
    buf.log("info", "hi", k=1)
    buf.metric("loss", 0.5)
    buf.phase("phase_b", n=3)
    assert len(buf._buffer) == 3
    kinds = [e["kind"] for e in buf._buffer]
    assert kinds == ["log", "metric", "phase"]
    assert all(e["round_id"] == 7 for e in buf._buffer)
    assert buf._buffer[0]["payload"]["message"] == "hi"
    assert buf._buffer[1]["payload"]["name"] == "loss"
    assert buf._buffer[1]["payload"]["value"] == 0.5


def test_buffer_drops_oldest_when_full():
    db = FakeDBClient()
    buf = EventBuffer(
        db, hotkey="hk", flush_interval_s=60,
        buffer_max=3, batch_max=2,
    )
    for i in range(10):
        buf.log("info", f"msg-{i}")
    assert len(buf._buffer) == 3
    assert buf.dropped_full == 7
    msgs = [e["payload"]["message"] for e in buf._buffer]
    assert msgs == ["msg-7", "msg-8", "msg-9"]


@pytest.mark.asyncio
async def test_flush_sends_and_drains():
    db = FakeDBClient()
    buf = EventBuffer(
        db, hotkey="hk", flush_interval_s=60,
        buffer_max=100, batch_max=50,
    )
    for i in range(5):
        buf.log("info", f"m{i}")
    await buf._flush_once()
    assert len(buf._buffer) == 0
    assert len(db.batches) == 1
    assert len(db.batches[0]) == 5
    assert buf.flushed_total == 5


@pytest.mark.asyncio
async def test_flush_failure_requeues():
    db = FakeDBClient(succeed=False)
    buf = EventBuffer(
        db, hotkey="hk", flush_interval_s=60,
        buffer_max=10, batch_max=4,
    )
    for i in range(3):
        buf.log("info", f"m{i}")
    await buf._flush_once()
    # Failed: events are requeued, counter incremented.
    assert buf.failed_flushes == 1
    assert len(buf._buffer) == 3


@pytest.mark.asyncio
async def test_flush_failure_caps_requeue_when_buffer_filled_during_flush():
    db = FakeDBClient(succeed=False)
    buf = EventBuffer(
        db, hotkey="hk", flush_interval_s=60,
        buffer_max=4, batch_max=2,
    )
    # Pre-fill batch_max=2 events that the flush will pop out.
    buf.log("info", "a")
    buf.log("info", "b")

    # Patch post_events to add new events into the buffer mid-flush, so by
    # the time the requeue runs there's no room for the original batch.
    original = db.post_events

    async def post_with_side_effect(batch):
        # Simulate concurrent producers filling the buffer during the
        # network round-trip.
        for ch in "cdef":
            buf.log("info", ch)
        return await original(batch)

    db.post_events = post_with_side_effect
    await buf._flush_once()
    # Requeue had room=0, so both popped events were dropped.
    assert buf.dropped_full == 2
    assert len(buf._buffer) == 4
    assert buf.failed_flushes >= 1


def test_logging_handler_forwards():
    db = FakeDBClient()
    buf = EventBuffer(db, hotkey="hk", flush_interval_s=60)
    handler = EventLoggingHandler(buf, level=logging.INFO)
    lg = logging.getLogger("radar.test_event_stream_forward")
    lg.handlers = [handler]
    lg.setLevel(logging.INFO)
    lg.info("hello %s", "world")
    assert len(buf._buffer) == 1
    ev = buf._buffer[0]
    assert ev["kind"] == "log"
    assert ev["level"] == "info"
    assert ev["payload"]["message"] == "hello world"
    assert ev["payload"]["logger"] == "radar.test_event_stream_forward"


def test_logging_handler_skips_blocked_loggers():
    db = FakeDBClient()
    buf = EventBuffer(db, hotkey="hk", flush_interval_s=60)
    handler = EventLoggingHandler(buf, level=logging.DEBUG)
    # validator.event_stream is in the block list — must not feed back.
    lg = logging.getLogger("validator.event_stream")
    lg.handlers = [handler]
    lg.setLevel(logging.DEBUG)
    lg.warning("recursion would be bad")
    assert len(buf._buffer) == 0


def test_attach_to_loggers_root_when_empty():
    db = FakeDBClient()
    buf = EventBuffer(db, hotkey="hk", flush_interval_s=60)
    handler = EventLoggingHandler(buf, level=logging.INFO)
    targets = attach_to_loggers(handler, names="", level=logging.INFO)
    try:
        assert targets == [logging.getLogger()]
    finally:
        for t in targets:
            t.removeHandler(handler)


def test_attach_to_loggers_named():
    db = FakeDBClient()
    buf = EventBuffer(db, hotkey="hk", flush_interval_s=60)
    handler = EventLoggingHandler(buf, level=logging.INFO)
    targets = attach_to_loggers(
        handler, names="alpha, beta", level=logging.INFO,
    )
    try:
        assert [t.name for t in targets] == ["alpha", "beta"]
    finally:
        for t in targets:
            t.removeHandler(handler)


@pytest.mark.asyncio
async def test_start_stop_flushes_remaining():
    db = FakeDBClient()
    buf = EventBuffer(
        db, hotkey="hk", flush_interval_s=0.05,
        buffer_max=100, batch_max=100,
    )
    buf.start()
    buf.log("info", "first")
    # Let the periodic flush fire at least once.
    await asyncio.sleep(0.15)
    buf.log("info", "second")
    await buf.stop()
    flushed = [e["payload"]["message"] for batch in db.batches for e in batch]
    assert "first" in flushed
    assert "second" in flushed
