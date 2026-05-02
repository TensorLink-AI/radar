"""Async client for Hippius S3 (https://s3.hippius.com).

Per docs.hippius.com/cli/usage the legacy ``hippius`` PyPI SDK is deprecated;
all integrations target the S3 data plane, which supports object metadata,
tagging, and prefix-filtered listing — the surfaces Phase 2 needs.

Constraints (docs.hippius.com/storage/s3/compatibility):
  * Path-style addressing, region ``decentralized``, signature v4.
  * No object versioning. Overwrites are silent — we use content-addressed
    keys ``{app_tag}/{netuid}/{phase}/{run_id}/{sha256[:16]}.tar`` so a
    write never collides with a different payload.
  * Object tags = small queryable subset; user metadata (``x-amz-meta-*``)
    carries the full per-bundle schema, ASCII-only, ~2 KiB total.

Reads fall back to the IPFS gateway for clients without S3 creds (the
verify CLI). Without either, ``download_bundle`` raises ``FileNotFoundError``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)


HIPPIUS_S3_DEFAULT_ENDPOINT = "https://s3.hippius.com"
HIPPIUS_S3_DEFAULT_REGION = "decentralized"
HIPPIUS_IPFS_DEFAULT_GATEWAY = "https://get.hippius.network"


@dataclass(frozen=True)
class BundleRef:
    """Pointer + provenance for a bundle stored on Hippius S3."""
    key: str
    metadata: dict
    tags: dict
    size: int
    etag: str = ""

    @property
    def cid(self) -> str:
        """Backwards-compat alias used by older substrate audit entries."""
        return self.key


# Stable export name for callers that imported the placeholder UploadResult.
UploadResult = BundleRef


def _sanitize_metadata(meta: dict, *, max_bytes: int = 1900) -> dict:
    """Keep only ASCII-safe keys/values, total under ~2 KiB (S3 header limit)."""
    out: dict[str, str] = {}
    used = 0
    for raw_k, raw_v in (meta or {}).items():
        try:
            k = str(raw_k).encode("ascii").decode("ascii")
            v = str(raw_v).encode("ascii").decode("ascii")
        except UnicodeEncodeError:
            continue
        cost = len(k) + len(v) + 4
        if used + cost > max_bytes:
            break
        out[k] = v
        used += cost
    return out


def _build_prefix(app_tag, netuid, phase, run_id) -> str:
    """Leftmost contiguous filter prefix for ListObjectsV2."""
    parts: list[str] = []
    for value in (app_tag, netuid, phase, run_id):
        if value is None or value == "":
            break
        parts.append(str(value))
    return "/".join(parts) + ("/" if parts else "")


class HippiusClient:
    """Async wrapper over boto3 S3 against Hippius.

    Constructor falls back through env in order:
      * ``HIPPIUS_S3_ACCESS_KEY`` → ``HIPPIUS_ACCESS_KEY_ID``
      * ``HIPPIUS_S3_SECRET_KEY`` → ``HIPPIUS_SECRET_ACCESS_KEY``
      * ``HIPPIUS_S3_BUCKET`` → ``HIPPIUS_BUCKET``
      * ``HIPPIUS_S3_ENDPOINT`` (default https://s3.hippius.com)
      * ``HIPPIUS_S3_REGION`` (default decentralized)

    Read-only callers can omit creds; download falls back to the IPFS
    gateway.
    """

    def __init__(
        self,
        access_key: str = "", secret_key: str = "", bucket: str = "",
        endpoint_url: str = "", region: str = "",
        *,
        ipfs_gateway_url: str = "",
        # Legacy kwargs honoured by older lazy imports.
        ipfs_api_url: str = "", hippius_key: str = "", substrate_rpc: str = "",
    ):
        env = os.getenv
        self._access_key = access_key or env("HIPPIUS_S3_ACCESS_KEY", "") \
            or env("HIPPIUS_ACCESS_KEY_ID", "")
        self._secret_key = secret_key or env("HIPPIUS_S3_SECRET_KEY", "") \
            or env("HIPPIUS_SECRET_ACCESS_KEY", "")
        self.bucket = bucket or env("HIPPIUS_S3_BUCKET", "") or env("HIPPIUS_BUCKET", "")
        self.endpoint_url = endpoint_url or env("HIPPIUS_S3_ENDPOINT", "") \
            or HIPPIUS_S3_DEFAULT_ENDPOINT
        self.region = region or env("HIPPIUS_S3_REGION", "") or HIPPIUS_S3_DEFAULT_REGION
        self._ipfs_gateway = (
            ipfs_gateway_url or ipfs_api_url or env("HIPPIUS_IPFS_GATEWAY", "")
            or HIPPIUS_IPFS_DEFAULT_GATEWAY
        ).rstrip("/")
        self._http: Optional[httpx.AsyncClient] = None
        self._s3 = self._build_s3() if self._access_key and self._secret_key else None

    def _build_s3(self):
        import boto3
        from botocore.config import Config as BotoConfig
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            region_name=self.region,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

    @property
    def has_s3(self) -> bool:
        """True when S3 creds were supplied — required for writes."""
        return self._s3 is not None and bool(self.bucket)

    async def upload_bundle(
        self,
        data: bytes,
        *,
        app_tag: str = "radar",
        phase: str,
        run_id: str,
        netuid: int = 0,
        extra_metadata: Optional[dict] = None,
    ) -> BundleRef:
        """Content-addressed PUT to Hippius S3."""
        if not self.has_s3:
            raise RuntimeError(
                "HippiusClient: S3 credentials not configured "
                "(set HIPPIUS_S3_ACCESS_KEY / _SECRET_KEY / _BUCKET)"
            )
        sha = hashlib.sha256(data).hexdigest()
        key = f"{app_tag}/{netuid}/{phase}/{run_id}/{sha[:16]}.tar"
        tags = {"app_tag": app_tag, "phase": phase, "netuid": str(netuid)}
        metadata = _sanitize_metadata({
            "app_tag": app_tag, "phase": phase, "run_id": run_id,
            "netuid": str(netuid), "sha256": sha,
            **(extra_metadata or {}),
        })
        tagging = urlencode(tags)

        def _put():
            return self._s3.put_object(
                Bucket=self.bucket, Key=key, Body=data,
                Metadata=metadata, Tagging=tagging,
            )

        resp = await asyncio.to_thread(_put)
        return BundleRef(
            key=key, metadata=metadata, tags=tags, size=len(data),
            etag=(resp.get("ETag") or "").strip('"'),
        )

    async def download_bundle(self, key_or_cid: str) -> bytes:
        """Fetch by S3 key (preferred) or IPFS CID gateway fallback."""
        if not key_or_cid:
            raise ValueError("download_bundle requires a key or cid")
        if self.has_s3:
            try:
                return await self._download_from_s3(key_or_cid)
            except FileNotFoundError:
                logger.debug("S3 miss for %s — trying IPFS gateway", key_or_cid)
        if self._ipfs_gateway:
            return await self._download_from_ipfs(key_or_cid)
        raise FileNotFoundError(
            f"download_bundle: no backend able to resolve {key_or_cid!r}"
        )

    async def _download_from_s3(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        def _get():
            return self._s3.get_object(Bucket=self.bucket, Key=key)

        try:
            resp = await asyncio.to_thread(_get)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404"}:
                raise FileNotFoundError(key) from e
            raise
        return resp["Body"].read()

    async def _download_from_ipfs(self, cid: str) -> bytes:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30.0)
        resp = await self._http.get(f"{self._ipfs_gateway}/ipfs/{cid}")
        if resp.status_code == 404:
            raise FileNotFoundError(cid)
        resp.raise_for_status()
        return resp.content

    async def head_bundle(self, key: str) -> BundleRef:
        """``head_object`` + ``get_object_tagging``, returned as a BundleRef."""
        if not self.has_s3:
            raise RuntimeError("HippiusClient: S3 credentials required for head_bundle")
        from botocore.exceptions import ClientError

        try:
            head = await asyncio.to_thread(
                lambda: self._s3.head_object(Bucket=self.bucket, Key=key),
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404", "NotFound"}:
                raise FileNotFoundError(key) from e
            raise
        tag_resp = await asyncio.to_thread(
            lambda: self._s3.get_object_tagging(Bucket=self.bucket, Key=key),
        )
        tags = {t["Key"]: t["Value"] for t in tag_resp.get("TagSet", [])}
        return BundleRef(
            key=key, metadata=dict(head.get("Metadata") or {}),
            tags=tags, size=int(head.get("ContentLength") or 0),
            etag=(head.get("ETag") or "").strip('"'),
        )

    async def list_by_metadata(
        self,
        *,
        app_tag: Optional[str] = None,
        netuid: Optional[int] = None,
        phase: Optional[str] = None,
        run_id: Optional[str] = None,
        match_tags: Optional[dict] = None,
        max_keys: int = 1000,
    ) -> list[BundleRef]:
        """Server-side prefix filter + optional client-side tag match."""
        if not self.has_s3:
            raise RuntimeError("HippiusClient: S3 credentials required for list_by_metadata")
        prefix = _build_prefix(app_tag, netuid, phase, run_id)
        resp = await asyncio.to_thread(
            lambda: self._s3.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix, MaxKeys=max_keys,
            ),
        )
        out: list[BundleRef] = []
        for obj in (resp.get("Contents") or []):
            key = obj["Key"]
            ref = BundleRef(
                key=key, metadata={}, tags={},
                size=int(obj.get("Size") or 0),
                etag=(obj.get("ETag") or "").strip('"'),
            )
            if match_tags:
                tag_resp = await asyncio.to_thread(
                    lambda k=key: self._s3.get_object_tagging(Bucket=self.bucket, Key=k),
                )
                tags = {t["Key"]: t["Value"] for t in tag_resp.get("TagSet", [])}
                if not all(tags.get(k) == v for k, v in match_tags.items()):
                    continue
                ref = BundleRef(
                    key=ref.key, metadata=ref.metadata, tags=tags,
                    size=ref.size, etag=ref.etag,
                )
            out.append(ref)
        return out

    async def close(self):
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
            self._http = None


__all__ = [
    "BundleRef", "HippiusClient", "UploadResult",
    "HIPPIUS_S3_DEFAULT_ENDPOINT", "HIPPIUS_S3_DEFAULT_REGION",
    "HIPPIUS_IPFS_DEFAULT_GATEWAY",
]
