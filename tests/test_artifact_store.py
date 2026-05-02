"""Tests for shared/artifact_store.py — dual-write R2 + Hippius."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.artifact_store import ArtifactStore, DualWriteResult


# ── Fakes ─────────────────────────────────────────────────────────────


class _FakeR2:
    """In-memory stand-in for R2AuditLog covering the ArtifactStore surface."""

    def __init__(self, *, fail: bool = False):
        self._store: dict[str, bytes] = {}
        self.fail = fail

    def upload_text(self, key: str, text: str) -> bool:
        if self.fail:
            return False
        self._store[key] = text.encode()
        return True

    def upload_json(self, key: str, data: dict) -> bool:
        if self.fail:
            return False
        self._store[key] = json.dumps(data).encode()
        return True

    def upload_file_from_disk(self, path: str, key: str) -> bool:
        if self.fail:
            return False
        with open(path, "rb") as f:
            self._store[key] = f.read()
        return True

    def download_text(self, key: str):
        if self.fail:
            return None
        if key not in self._store:
            return None
        try:
            return self._store[key].decode()
        except UnicodeDecodeError:
            return None


class _FakeHippius:
    """Async stand-in for HippiusClient.upload/download (Phase 2 structured)."""

    def __init__(self, *, fail: bool = False, return_none: bool = False):
        self._store: dict[str, bytes] = {}
        self._meta: dict[str, dict] = {}
        self.fail = fail
        self.return_none = return_none
        self._next_id = 0

    async def upload_bundle(
        self, data: bytes, *,
        app_tag: str = "radar", phase: str, run_id: str,
        netuid: int = 0, extra_metadata=None,
    ):
        if self.fail:
            raise RuntimeError("hippius down")
        if self.return_none:
            return None
        self._next_id += 1
        # Mirror the real client's content-addressed key shape so tests can
        # assert against it. Also stash everything for inspection.
        import hashlib as _hl
        sha = _hl.sha256(data).hexdigest()
        key = f"{app_tag}/{netuid}/{phase}/{run_id}/{sha[:16]}.tar"
        self._store[key] = data
        self._meta[key] = {
            "app_tag": app_tag, "phase": phase, "run_id": run_id,
            "netuid": str(netuid), **(extra_metadata or {}),
        }
        return MagicMock(
            key=key, cid=key, size=len(data), etag="fakeetag",
            metadata=self._meta[key],
            tags={"app_tag": app_tag, "phase": phase, "netuid": str(netuid)},
        )

    async def download_bundle(self, key_or_cid: str) -> bytes:
        if self.fail or key_or_cid not in self._store:
            raise RuntimeError(f"missing key {key_or_cid}")
        return self._store[key_or_cid]


# ── put_bytes / put_json ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_put_bytes_both_succeed():
    r2, hip = _FakeR2(), _FakeHippius()
    store = ArtifactStore(r2=r2, hippius=hip)
    res = await store.put_bytes("k.json", b'{"a":1}', {"kind": "test"})
    assert res.r2_ok and res.hippius_ok
    assert res.r2_key == "k.json"
    # Content-addressed key shape: {app_tag}/{netuid}/{phase}/{run_id}/...
    assert "/" in res.hippius_cid and res.hippius_cid.endswith(".tar")
    assert not res.both_failed
    assert r2._store["k.json"] == b'{"a":1}'
    counters = store.snapshot_counters()
    assert counters["r2_writes_ok"] == 1
    assert counters["hippius_writes_ok"] == 1
    assert counters["bytes_written"] == len(b'{"a":1}')


@pytest.mark.asyncio
async def test_put_bytes_r2_fails_hippius_succeeds():
    """A flaky R2 must not block the Hippius copy."""
    r2 = _FakeR2(fail=True)
    hip = _FakeHippius()
    store = ArtifactStore(r2=r2, hippius=hip)
    res = await store.put_bytes("k.json", b"hello", {})
    assert not res.r2_ok
    assert res.hippius_ok
    assert not res.both_failed
    counters = store.snapshot_counters()
    assert counters["r2_writes_fail"] == 1
    assert counters["hippius_writes_ok"] == 1


@pytest.mark.asyncio
async def test_put_bytes_hippius_fails_r2_succeeds():
    r2 = _FakeR2()
    hip = _FakeHippius(fail=True)
    store = ArtifactStore(r2=r2, hippius=hip)
    res = await store.put_bytes("k.json", b"hello", {})
    assert res.r2_ok
    assert not res.hippius_ok
    assert not res.both_failed
    counters = store.snapshot_counters()
    assert counters["hippius_writes_fail"] == 1


@pytest.mark.asyncio
async def test_put_bytes_both_fail_flags_result():
    r2 = _FakeR2(fail=True)
    hip = _FakeHippius(fail=True)
    store = ArtifactStore(r2=r2, hippius=hip)
    res = await store.put_bytes("k.json", b"hello", {})
    assert res.both_failed
    assert not res.r2_ok and not res.hippius_ok
    counters = store.snapshot_counters()
    assert counters["both_failed"] == 1


@pytest.mark.asyncio
async def test_put_bytes_no_backends_fails_loudly():
    """Constructed without either side, every write trips both_failed."""
    store = ArtifactStore(r2=None, hippius=None)
    res = await store.put_bytes("k.json", b"hello", {})
    assert res.both_failed


@pytest.mark.asyncio
async def test_put_bytes_only_r2_configured():
    r2 = _FakeR2()
    store = ArtifactStore(r2=r2, hippius=None)
    res = await store.put_bytes("k.json", b"hello", {})
    assert res.r2_ok
    assert not res.hippius_ok
    assert not res.both_failed


@pytest.mark.asyncio
async def test_put_bytes_only_hippius_configured():
    hip = _FakeHippius()
    store = ArtifactStore(r2=None, hippius=hip)
    res = await store.put_bytes("k.json", b"hello", {})
    assert res.hippius_ok
    assert not res.r2_ok
    assert not res.both_failed


@pytest.mark.asyncio
async def test_put_bytes_handles_hippius_not_implemented():
    """Phase 2 stop: upload_bundle raises NotImplementedError → treated as
    a disabled backend, not a failure that masks R2's success."""
    class _NotImplHippius:
        async def upload_bundle(self, data, metadata):
            raise NotImplementedError("see TEN-242")

    r2 = _FakeR2()
    store = ArtifactStore(r2=r2, hippius=_NotImplHippius())
    res = await store.put_bytes("k.json", b"hello", {})
    assert res.r2_ok
    assert not res.hippius_ok
    # NotImplementedError isn't counted as a failure (it's a disabled
    # backend), but R2's success keeps both_failed False.
    assert not res.both_failed


@pytest.mark.asyncio
async def test_put_json_serialises_deterministically():
    r2, hip = _FakeR2(), _FakeHippius()
    store = ArtifactStore(r2=r2, hippius=hip)
    await store.put_json("k.json", {"b": 2, "a": 1}, {"kind": "test"})
    # Sorted keys → deterministic across runs.
    assert r2._store["k.json"] == b'{"a":1,"b":2}'


# ── get_bytes ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_bytes_r2_first():
    r2, hip = _FakeR2(), _FakeHippius()
    store = ArtifactStore(r2=r2, hippius=hip)
    await store.put_bytes("k.json", b"hello", {})
    data = await store.get_bytes(key="k.json")
    assert data == b"hello"


@pytest.mark.asyncio
async def test_get_bytes_by_cid():
    r2, hip = _FakeR2(), _FakeHippius()
    store = ArtifactStore(r2=r2, hippius=hip)
    res = await store.put_bytes("k.json", b"hello", {})
    data = await store.get_bytes(cid=res.hippius_cid)
    assert data == b"hello"


@pytest.mark.asyncio
async def test_get_bytes_fallback_off_r2_miss_raises():
    """Default behaviour: a missed R2 read raises rather than silently
    falling through. Loud failures during rollout."""
    hip = _FakeHippius()
    store = ArtifactStore(
        r2=_FakeR2(fail=True), hippius=hip, allow_fallback=False,
    )
    with pytest.raises(FileNotFoundError):
        await store.get_bytes(key="k.json", cid="bafy1")


@pytest.mark.asyncio
async def test_get_bytes_fallback_on_falls_through_to_hippius():
    r2, hip = _FakeR2(), _FakeHippius()
    store = ArtifactStore(r2=r2, hippius=hip, allow_fallback=True)
    res = await store.put_bytes("k.json", b"hello", {})
    # Wipe R2's copy to force fallback.
    r2._store.clear()
    data = await store.get_bytes(key="k.json", cid=res.hippius_cid)
    assert data == b"hello"


@pytest.mark.asyncio
async def test_get_bytes_requires_key_or_cid():
    store = ArtifactStore(r2=_FakeR2(), hippius=_FakeHippius())
    with pytest.raises(ValueError):
        await store.get_bytes()


@pytest.mark.asyncio
async def test_get_bytes_both_miss_raises():
    r2, hip = _FakeR2(), _FakeHippius()
    store = ArtifactStore(r2=r2, hippius=hip, allow_fallback=True)
    with pytest.raises(FileNotFoundError):
        await store.get_bytes(key="absent", cid="bafyabsent")


# ── Roundtrip integration ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_coordinator_routes_through_artifact_store_when_present():
    """TrainingCoordinator.write_dispatch_record honours artifact_store."""
    from validator.coordinator import TrainingCoordinator, TrainingResult

    r2, hip = _FakeR2(), _FakeHippius()
    store = ArtifactStore(r2=r2, hippius=hip)

    # Build a coordinator without invoking __init__'s broader concerns.
    coord = TrainingCoordinator.__new__(TrainingCoordinator)
    coord.wallet = MagicMock()
    coord.wallet.hotkey.ss58_address = "5Vali"
    coord.metagraph = MagicMock()
    coord.r2 = r2
    coord.my_uid = 0
    coord.artifact_store = store
    coord._fallback_uids = {}

    results = [TrainingResult(
        arch_owner=1, trainer_uid=2, status="success",
        flops_equivalent_size=1_000_000, training_time_seconds=42.0,
        checkpoint_key="round_7/miner_x/checkpoint.safetensors",
    )]
    await coord.write_dispatch_record(round_id=7, results=results)

    # Dual-written: R2 carries the JSON, Hippius carries an identical copy.
    r2_payload = json.loads(r2._store["round_7/dispatch/vali_5Vali.json"])
    assert r2_payload["dispatcher"] == "5Vali"
    assert r2_payload["round_id"] == 7
    assert len(hip._store) == 1
    key = next(iter(hip._store))
    # Translation: artifact_store metadata.kind -> Hippius phase tag.
    assert hip._meta[key]["phase"] == "dispatch"
    assert hip._meta[key]["validator_hotkey"] == "5Vali"


@pytest.mark.asyncio
async def test_coordinator_falls_back_to_r2_only_when_no_store():
    """Backward compat: artifact_store=None preserves the R2-only path."""
    from validator.coordinator import TrainingCoordinator, TrainingResult

    r2 = _FakeR2()
    coord = TrainingCoordinator.__new__(TrainingCoordinator)
    coord.wallet = MagicMock()
    coord.wallet.hotkey.ss58_address = "5Vali"
    coord.metagraph = MagicMock()
    coord.r2 = r2
    coord.my_uid = 0
    coord.artifact_store = None
    coord._fallback_uids = {}

    await coord.write_dispatch_record(
        round_id=7, results=[TrainingResult(
            arch_owner=1, trainer_uid=2, status="success",
            flops_equivalent_size=1_000_000, training_time_seconds=1.0,
            checkpoint_key="x",
        )],
    )
    assert "round_7/dispatch/vali_5Vali.json" in r2._store


@pytest.mark.asyncio
async def test_dual_write_then_fetch_via_cid_matches_bytes():
    r2, hip = _FakeR2(), _FakeHippius()
    store = ArtifactStore(r2=r2, hippius=hip, allow_fallback=True)
    payload = b'{"frontier": [1, 2, 3]}'
    res = await store.put_bytes("frontier/ts/latest.json", payload, {
        "app": "radar", "kind": "frontier", "task": "ts",
    })
    assert res.r2_ok and res.hippius_ok
    # Both backends carry identical bytes — the dual-write contract.
    via_r2 = await store.get_bytes(key="frontier/ts/latest.json")
    via_hippius = await store.get_bytes(cid=res.hippius_cid)
    assert via_r2 == via_hippius == payload
