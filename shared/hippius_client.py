"""Async client for Hippius (Substrate-based decentralized object storage).

Phase 1 of TEN-242 ships only the *download* path because that's what the
``/experiments/{id}/verify`` endpoint and the standalone verify CLI need to
operate. The upload + metadata-discovery surface is intentionally
``NotImplementedError`` until the upstream SDK gains metadata-tagged uploads
(see TEN-242 investigation log).

Downloads use the IPFS HTTP gateway directly so the client is dependency-light
and works against any IPFS-compatible endpoint — operators don't have to
install the full ``hippius`` SDK just to verify someone else's experiments.

The class signature matches the placeholders in
``validator/substrate_publisher.UploadResult`` and the lazy imports inside
``Validator._init_hippius`` / ``DatabaseNeuron._init_hippius`` so the rest of
the substrate plumbing finds it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


HIPPIUS_DEFAULT_GATEWAY = "https://get.hippius.network"


@dataclass(frozen=True)
class UploadResult:
    """Outcome of a successful Hippius upload.

    Mirrors ``validator.substrate_publisher.UploadResult`` so callers can
    swap freely. Once upload support lands the publisher's local placeholder
    will be retired in favour of this one.
    """
    cid: str
    size_bytes: int
    block_number: Optional[int] = None
    block_timestamp: Optional[float] = None


class HippiusClient:
    """Minimal async wrapper for Hippius IPFS reads.

    Constructor signature is stable across the eventual upload work, so
    operators who set ``HIPPIUS_*`` env vars today can keep the same config
    when the upload path lands.
    """

    def __init__(
        self,
        ipfs_api_url: str = "",
        hippius_key: str = "",
        substrate_rpc: str = "",
        timeout: float = 30.0,
    ):
        # ``ipfs_api_url`` is treated as an IPFS HTTP gateway root (i.e. a
        # server that serves /ipfs/<cid>). The Hippius SDK's bundled
        # gateway works; any IPFS gateway does. Empty falls back to the
        # public Hippius gateway.
        self.ipfs_api_url = (ipfs_api_url or HIPPIUS_DEFAULT_GATEWAY).rstrip("/")
        self.hippius_key = hippius_key
        self.substrate_rpc = substrate_rpc
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def download_bundle(self, cid: str) -> bytes:
        """Fetch a bundle's bytes from IPFS by CID.

        Raises on empty CID, transport error, or non-2xx response. Callers
        (the verify endpoint and the CLI tool) wrap this in a ``try`` and
        surface failures as user-visible diagnostics.
        """
        if not cid:
            raise ValueError("cid is empty")
        client = await self._http()
        url = f"{self.ipfs_api_url}/ipfs/{cid}"
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content

    async def upload_bundle(self, data: bytes, metadata: dict) -> UploadResult:
        """Not yet implemented — see TEN-242.

        The upstream Hippius SDK doesn't currently expose metadata-tagged
        uploads, which the substrate publisher relies on for /verify-style
        discovery. Once that lands this method will write to IPFS and
        attach the metadata via the SDK's storage_request flow.
        """
        raise NotImplementedError(
            "HippiusClient.upload_bundle is not implemented yet (TEN-242). "
            "Substrate publishing on the validator side is gated by "
            "Config.HIPPIUS_ENABLED — leave it off until this lands."
        )

    async def list_by_metadata(
        self,
        app_tag: str,
        additional_filters: Optional[dict] = None,
    ) -> list[dict]:
        """Not yet implemented — see TEN-242."""
        raise NotImplementedError(
            "HippiusClient.list_by_metadata is not implemented yet (TEN-242)."
        )

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


__all__ = ["HippiusClient", "UploadResult", "HIPPIUS_DEFAULT_GATEWAY"]
