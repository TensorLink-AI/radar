"""Tests for shared/hippius_client.py — boto3-backed Hippius S3 client.

We mock the boto3 S3 client (no real network), exercise the four public
methods (``upload_bundle``, ``download_bundle``, ``head_bundle``,
``list_by_metadata``), and confirm:
  * the content-addressed key shape is right;
  * tags only carry the queryable subset (``app_tag``, ``phase``, ``netuid``);
  * user metadata is ASCII-clamped to ~2 KiB;
  * NoSuchKey maps to FileNotFoundError;
  * IPFS-gateway fallback kicks in when no S3 creds are configured.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import httpx
import pytest

from shared.hippius_client import (
    BundleRef,
    HippiusClient,
    _build_prefix,
    _sanitize_metadata,
)


# ── Helpers / fixtures ───────────────────────────────────────────────


class _FakeS3:
    """In-memory stand-in that records put/get/head/list/tagging calls.

    Only models the bits ``HippiusClient`` actually exercises. Real boto3
    would round-trip an S3 server; here we keep just enough state to
    answer the same questions.
    """

    def __init__(self, *, raise_on_get: str = ""):
        self._objects: dict[str, dict] = {}  # key -> {"body", "metadata", "etag"}
        self._tags: dict[str, dict] = {}     # key -> {tag_key: tag_value}
        self.raise_on_get = raise_on_get
        self.calls: list = []

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        body = kwargs["Body"]
        meta = kwargs.get("Metadata") or {}
        tagging = kwargs.get("Tagging") or ""
        # Tagging is x-www-form-urlencoded "k=v&k=v".
        from urllib.parse import parse_qs
        tags = {k: v[0] for k, v in parse_qs(tagging).items()} if tagging else {}
        etag = '"' + hashlib.md5(body).hexdigest() + '"'
        self._objects[key] = {"body": body, "metadata": dict(meta), "etag": etag}
        self._tags[key] = tags
        self.calls.append(("put_object", kwargs))
        return {"ETag": etag}

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        if key == self.raise_on_get or key not in self._objects:
            from botocore.exceptions import ClientError
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )
        rec = self._objects[key]
        body = MagicMock()
        body.read = MagicMock(return_value=rec["body"])
        return {"Body": body, "ETag": rec["etag"], "Metadata": rec["metadata"]}

    def head_object(self, **kwargs):
        key = kwargs["Key"]
        if key not in self._objects:
            from botocore.exceptions import ClientError
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"}}, "HeadObject",
            )
        rec = self._objects[key]
        return {
            "ETag": rec["etag"],
            "Metadata": rec["metadata"],
            "ContentLength": len(rec["body"]),
        }

    def get_object_tagging(self, **kwargs):
        key = kwargs["Key"]
        return {"TagSet": [
            {"Key": k, "Value": v} for k, v in (self._tags.get(key) or {}).items()
        ]}

    def list_objects_v2(self, **kwargs):
        prefix = kwargs.get("Prefix", "") or ""
        contents = []
        for key, rec in self._objects.items():
            if key.startswith(prefix):
                contents.append({
                    "Key": key, "Size": len(rec["body"]), "ETag": rec["etag"],
                })
        return {"Contents": contents}


def _make_client(*, fake_s3: _FakeS3) -> HippiusClient:
    """Build a HippiusClient with a pre-injected fake S3."""
    c = HippiusClient(
        access_key="ak", secret_key="sk", bucket="radar-test",
    )
    c._s3 = fake_s3  # bypass real boto3
    return c


# ── Helpers under test ───────────────────────────────────────────────


def test_sanitize_metadata_drops_non_ascii():
    out = _sanitize_metadata({"ok": "hello", "bad": "héllo", "n": 7})
    assert "ok" in out and "bad" not in out
    assert out["n"] == "7"  # values get stringified


def test_sanitize_metadata_caps_total_size():
    huge = {f"k{i}": "x" * 200 for i in range(40)}
    out = _sanitize_metadata(huge, max_bytes=1900)
    # Total bytes must stay under the cap (with the +4 overhead per entry).
    total = sum(len(k) + len(v) + 4 for k, v in out.items())
    assert total <= 1900
    assert len(out) < len(huge)  # some were dropped


def test_build_prefix_is_leftmost_contiguous():
    assert _build_prefix("radar", 1, "phase_c", "42") == "radar/1/phase_c/42/"
    assert _build_prefix("radar", 1, None, "42") == "radar/1/"  # gap → stop
    assert _build_prefix(None, 1, "phase_c", None) == ""        # no leading


# ── upload_bundle ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upload_bundle_content_addressed_key():
    fake = _FakeS3()
    client = _make_client(fake_s3=fake)
    data = b"hello phase c"
    sha = hashlib.sha256(data).hexdigest()

    ref = await client.upload_bundle(
        data, app_tag="radar", phase="phase_c", run_id="42", netuid=1,
    )

    assert isinstance(ref, BundleRef)
    assert ref.key == f"radar/1/phase_c/42/{sha[:16]}.tar"
    assert ref.size == len(data)
    # Tags carry only the queryable subset.
    assert ref.tags == {"app_tag": "radar", "phase": "phase_c", "netuid": "1"}
    # Metadata carries the full schema view + caller's extra fields.
    assert ref.metadata["sha256"] == sha
    assert ref.metadata["run_id"] == "42"
    # Backwards-compat: BundleRef.cid aliases .key for legacy substrate
    # audit entries that read ``.cid``.
    assert ref.cid == ref.key


@pytest.mark.asyncio
async def test_upload_bundle_passes_extra_metadata_through():
    fake = _FakeS3()
    client = _make_client(fake_s3=fake)
    ref = await client.upload_bundle(
        b"x", phase="phase_c", run_id="1",
        extra_metadata={"validator_hotkey": "5G", "record_count": "3"},
    )
    assert ref.metadata["validator_hotkey"] == "5G"
    assert ref.metadata["record_count"] == "3"


@pytest.mark.asyncio
async def test_upload_bundle_without_creds_raises():
    """No access key / secret → no S3 client wired → upload refuses."""
    client = HippiusClient()  # no creds, no env
    with pytest.raises(RuntimeError, match="S3 credentials"):
        await client.upload_bundle(b"x", phase="phase_c", run_id="1")


# ── download_bundle ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_download_bundle_via_s3_returns_bytes():
    fake = _FakeS3()
    client = _make_client(fake_s3=fake)
    ref = await client.upload_bundle(b"hello", phase="phase_c", run_id="1")
    data = await client.download_bundle(ref.key)
    assert data == b"hello"


@pytest.mark.asyncio
async def test_download_bundle_nosuchkey_falls_through_to_ipfs_gateway():
    """When S3 misses, we silently try the gateway path. Default behaviour
    so legacy IPFS CIDs in old audit entries keep resolving."""
    fake = _FakeS3()
    client = _make_client(fake_s3=fake)

    # Stub the IPFS gateway side.
    async def _ipfs(self, cid):
        return b"ipfs-bytes"

    with patch.object(HippiusClient, "_download_from_ipfs", _ipfs):
        data = await client.download_bundle("bafymissing")
    assert data == b"ipfs-bytes"


@pytest.mark.asyncio
async def test_download_bundle_no_backend_raises():
    """No creds, no gateway URL → loud failure."""
    client = HippiusClient()
    client._ipfs_gateway = ""  # explicitly disable gateway fallback
    with pytest.raises(FileNotFoundError):
        await client.download_bundle("bafyany")


@pytest.mark.asyncio
async def test_download_bundle_empty_arg_raises():
    client = HippiusClient(access_key="ak", secret_key="sk", bucket="b")
    with pytest.raises(ValueError):
        await client.download_bundle("")


# ── head_bundle ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_head_bundle_returns_metadata_and_tags():
    fake = _FakeS3()
    client = _make_client(fake_s3=fake)
    ref = await client.upload_bundle(
        b"x", phase="phase_c", run_id="1",
        extra_metadata={"validator_hotkey": "5G"},
    )
    head = await client.head_bundle(ref.key)
    assert head.key == ref.key
    assert head.size == 1
    assert head.metadata["validator_hotkey"] == "5G"
    assert head.tags == ref.tags


@pytest.mark.asyncio
async def test_head_bundle_missing_raises_file_not_found():
    fake = _FakeS3()
    client = _make_client(fake_s3=fake)
    with pytest.raises(FileNotFoundError):
        await client.head_bundle("not/a/real/key.tar")


# ── list_by_metadata ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_by_metadata_prefix_filters():
    fake = _FakeS3()
    client = _make_client(fake_s3=fake)
    await client.upload_bundle(b"a", phase="phase_c", run_id="1", netuid=1)
    await client.upload_bundle(b"b", phase="phase_c", run_id="2", netuid=1)
    await client.upload_bundle(b"c", phase="dispatch", run_id="1", netuid=2)

    # Prefix scope: app_tag=radar netuid=1 → only the two phase_c objects.
    refs = await client.list_by_metadata(app_tag="radar", netuid=1)
    keys = sorted(r.key for r in refs)
    assert len(keys) == 2
    assert all(k.startswith("radar/1/") for k in keys)


@pytest.mark.asyncio
async def test_list_by_metadata_tag_filter_keeps_matches():
    fake = _FakeS3()
    client = _make_client(fake_s3=fake)
    await client.upload_bundle(b"a", phase="phase_c", run_id="1")
    await client.upload_bundle(b"b", phase="dispatch", run_id="2")
    refs = await client.list_by_metadata(
        app_tag="radar", match_tags={"phase": "dispatch"},
    )
    assert len(refs) == 1
    assert refs[0].tags["phase"] == "dispatch"


# ── End-to-end smoke (per Phase 2 spec) ──────────────────────────────


@pytest.mark.asyncio
async def test_smoke_upload_list_head_download_sha_matches():
    """Per spec: upload a 1 KB bundle, list, head, download, sha256 matches."""
    fake = _FakeS3()
    client = _make_client(fake_s3=fake)

    payload = b"x" * 1024
    expected_sha = hashlib.sha256(payload).hexdigest()

    ref = await client.upload_bundle(
        payload, app_tag="radar", phase="phase_c", run_id="42", netuid=1,
    )
    listed = await client.list_by_metadata(app_tag="radar", netuid=1)
    assert any(r.key == ref.key for r in listed)
    head = await client.head_bundle(ref.key)
    assert head.metadata["sha256"] == expected_sha
    fetched = await client.download_bundle(ref.key)
    assert hashlib.sha256(fetched).hexdigest() == expected_sha
