"""Tests for agent_events logging end-to-end.

Covers:
  - direct store insert + read round-trip
  - truncation of oversized payloads
  - that an /llm/chat POST through the running service writes a row
  - JSONL export reproduces what's in the DB
"""

from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

import pytest

from local import export_events
from local.export_events import flush_round_to_r2, write_jsonl
from local.services import ServicesServer
from local.store import LocalStore


@pytest.fixture
def store(tmp_path: Path) -> LocalStore:
    s = LocalStore(tmp_path / "events.db")
    yield s
    s.close()


def test_record_and_iter_roundtrip(store: LocalStore):
    rid = store.record_agent_event(
        kind="llm_chat", miner_id="m1", round_id=3, endpoint="/llm/chat",
        status=200, latency_ms=42.5,
        request={"content": "hi"}, response={"text": "hello"},
    )
    assert rid > 0
    rows = list(store.iter_agent_events())
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "llm_chat"
    assert row["miner_id"] == "m1"
    assert row["round_id"] == 3
    assert row["request"] == {"content": "hi"}
    assert row["response"] == {"text": "hello"}
    assert row["latency_ms"] == 42.5


def test_filters(store: LocalStore):
    store.record_agent_event(kind="llm_chat", miner_id="m1", round_id=1)
    store.record_agent_event(kind="desearch", miner_id="m1", round_id=1)
    store.record_agent_event(kind="llm_chat", miner_id="m2", round_id=2)
    assert len(list(store.iter_agent_events(kind="llm_chat"))) == 2
    assert len(list(store.iter_agent_events(miner_id="m2"))) == 1
    assert len(list(store.iter_agent_events(round_id=1))) == 2


def test_truncation(store: LocalStore):
    huge = {"blob": "x" * 10_000}
    store.record_agent_event(
        kind="llm_chat", miner_id="m1", request=huge, max_bytes=1024,
    )
    row = next(iter(store.iter_agent_events()))
    assert isinstance(row["request"], dict)
    assert row["request"].get("_truncated") is True
    assert row["request"]["_original_bytes"] > 1024


def test_non_serializable_request(store: LocalStore):
    class Weird:
        def __repr__(self) -> str:
            return "Weird()"
    # ``default=str`` lets this through; the row should round-trip cleanly.
    store.record_agent_event(kind="llm_chat", request={"obj": Weird()})
    row = next(iter(store.iter_agent_events()))
    assert "Weird()" in json.dumps(row["request"])


def test_stats(store: LocalStore):
    store.record_agent_event(kind="llm_chat", miner_id="m1")
    store.record_agent_event(kind="wiki_read", miner_id="m1")
    s = store.agent_event_stats()
    assert s["total"] == 2
    assert s["by_kind"] == {"llm_chat": 1, "wiki_read": 1}


def test_export_jsonl_roundtrip(store: LocalStore, tmp_path: Path):
    store.record_agent_event(kind="llm_chat", miner_id="m1", round_id=1,
                             request={"q": "a"}, response={"a": 1})
    store.record_agent_event(kind="desearch", miner_id="m1", round_id=1,
                             request={"q": "b"}, response={"hits": []})
    out = tmp_path / "events.jsonl"
    n = write_jsonl(store.iter_agent_events(), out)
    assert n == 2
    lines = out.read_text().strip().splitlines()
    parsed = [json.loads(line) for line in lines]
    assert {p["kind"] for p in parsed} == {"llm_chat", "desearch"}


def test_service_llm_chat_writes_event(tmp_path: Path):
    """An end-to-end check: spinning up the service, POSTing /llm/chat
    with an X-Miner-Id header should create one agent_events row."""
    store = LocalStore(tmp_path / "events.db")
    # Seed an open challenge so the handler can resolve round_id.
    store.post_challenge("ch-1", round_id=7, payload={})

    server = ServicesServer(store=store, wiki_dir=None, port=0)
    url = server.start()
    try:
        body = json.dumps({"content": "hello", "model": "stub"}).encode()
        req = urllib.request.Request(
            url + "/llm/chat", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "X-Miner-Id": "miner-abc"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            resp.read()
    finally:
        server.stop()

    rows = list(store.iter_agent_events())
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "llm_chat"
    assert row["miner_id"] == "miner-abc"
    assert row["round_id"] == 7
    assert row["endpoint"] == "/llm/chat"
    assert row["status"] == 200
    assert row["latency_ms"] is not None and row["latency_ms"] >= 0
    assert row["request"] == {"content": "hello", "model": "stub"}
    store.close()


def test_delete_agent_events_by_round(store: LocalStore):
    store.record_agent_event(kind="llm_chat", miner_id="m1", round_id=1)
    store.record_agent_event(kind="llm_chat", miner_id="m1", round_id=1)
    store.record_agent_event(kind="llm_chat", miner_id="m1", round_id=2)
    n = store.delete_agent_events(round_id=1)
    assert n == 2
    remaining = list(store.iter_agent_events())
    assert [r["round_id"] for r in remaining] == [2]


def test_delete_agent_events_requires_filter(store: LocalStore):
    with pytest.raises(ValueError):
        store.delete_agent_events()


def test_flush_round_to_r2_uploads_and_prunes(
    store: LocalStore, tmp_path: Path, monkeypatch,
):
    """End-to-end flush: rows for round 5 go to R2 as a single JSONL
    shard and are deleted from SQLite; rows for round 6 stay."""
    store.record_agent_event(kind="llm_chat", miner_id="m1", round_id=5,
                             request={"q": "a"}, response={"a": 1})
    store.record_agent_event(kind="desearch", miner_id="m1", round_id=5,
                             request={"q": "b"}, response={"hits": []})
    store.record_agent_event(kind="llm_chat", miner_id="m1", round_id=6,
                             request={"q": "c"}, response={"a": 2})

    uploads: dict[str, str] = {}

    class FakeStorage:
        def __init__(self, bucket: str = "") -> None:
            self.bucket = bucket

        def upload_text(self, key: str, text: str) -> bool:
            uploads[key] = text
            return True

    # Patch the import inside flush_round_to_r2.
    import shared.r2_audit as r2_audit
    monkeypatch.setattr(r2_audit, "HippiusStorage", FakeStorage)

    ok, n = flush_round_to_r2(store, 5, bucket="test-bucket")
    assert ok is True
    assert n == 2

    # Round 5 rows are gone; round 6 row survived.
    rounds_left = [r["round_id"] for r in store.iter_agent_events()]
    assert rounds_left == [6]

    # The shard key follows agent-events/round=NNNNNN.jsonl and contains
    # exactly two JSON lines.
    assert list(uploads.keys()) == ["agent-events/round=000005.jsonl"]
    lines = uploads["agent-events/round=000005.jsonl"].strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert {p["kind"] for p in parsed} == {"llm_chat", "desearch"}


def test_flush_keeps_rows_on_upload_failure(
    store: LocalStore, monkeypatch,
):
    """A failed upload must NOT delete the local rows — they're the only
    copy until R2 confirms receipt."""
    store.record_agent_event(kind="llm_chat", miner_id="m1", round_id=9)

    class FailStorage:
        def __init__(self, bucket: str = "") -> None:
            pass

        def upload_text(self, key: str, text: str) -> bool:
            return False

    import shared.r2_audit as r2_audit
    monkeypatch.setattr(r2_audit, "HippiusStorage", FailStorage)

    ok, n = flush_round_to_r2(store, 9, bucket="test-bucket")
    assert ok is False
    assert n == 1
    # Row is still there for the next retry.
    assert len(list(store.iter_agent_events(round_id=9))) == 1


def test_flush_empty_round_is_noop(store: LocalStore):
    ok, n = flush_round_to_r2(store, 42, bucket="test-bucket")
    assert ok is True
    assert n == 0


def test_service_skips_logging_without_miner_header(tmp_path: Path):
    """Internal/dashboard callers (no X-Miner-Id) must not pollute the
    event log."""
    store = LocalStore(tmp_path / "events.db")
    store.post_challenge("ch-1", round_id=1, payload={})
    server = ServicesServer(store=store, wiki_dir=None, port=0)
    url = server.start()
    try:
        body = json.dumps({"content": "hi"}).encode()
        req = urllib.request.Request(
            url + "/llm/chat", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
    finally:
        server.stop()

    assert list(store.iter_agent_events()) == []
    store.close()
