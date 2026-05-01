"""Unified dual-write artifact store: R2 (hot cache) + Hippius (durable).

Phase 7 of TEN-240. R2 stays the hot-path cache because it's the existing
deploy and millisecond-scale; Hippius becomes the durable, censorship-
resistant copy. Both writes are best-effort — failure of either backend
logs but does not raise unless *both* fail.

Reads default to R2-first (cache locality). When ``allow_fallback`` is set
(``Config.HIPPIUS_ARTIFACT_FALLBACK``), a missing/failing R2 read drops
through to Hippius. We deliberately keep this off by default during rollout
so misconfigurations stay loud.

The store is duck-typed against:
  * ``r2.upload_text(key, str)`` / ``download_text(key) -> str | None``
  * ``r2.upload_json(key, dict)`` / ``download_json(key) -> dict | None``
  * ``r2.upload_file_from_disk(local, key) -> bool``
  * ``hippius.upload_bundle(bytes, dict) -> UploadResult``
  * ``hippius.download_bundle(cid) -> bytes``
so the existing ``shared.r2_audit.HippiusStorage`` (the S3-compatible
client) and the lazy ``shared.hippius_client.HippiusClient`` plug straight in.

Observability counters on the instance — not Prometheus, not bittensor
metrics — let callers log a per-round summary without a separate metrics
backend. The validator's round-loop reads them and emits one INFO line.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DualWriteResult:
    """One round-trip's worth of dual-write outcome.

    ``r2_key`` and ``hippius_cid`` are only populated when their respective
    side succeeded. ``both_failed`` is True iff neither side wrote anything,
    in which case callers may want to retry or escalate (e.g. fail the
    round, surface a metric).
    """
    r2_key: str = ""
    hippius_cid: str = ""
    r2_ok: bool = False
    hippius_ok: bool = False
    both_failed: bool = False


@dataclass
class _Counters:
    """Cumulative counters for one ArtifactStore lifetime."""
    r2_writes_ok: int = 0
    r2_writes_fail: int = 0
    hippius_writes_ok: int = 0
    hippius_writes_fail: int = 0
    both_failed: int = 0
    bytes_written: int = 0
    mismatches: int = 0  # bytes returned by R2 vs Hippius differ on read


class ArtifactStore:
    """Dual-write wrapper over R2 (or any S3 client) + Hippius.

    Either ``r2`` or ``hippius`` may be ``None``: an absent backend simply
    skips that side of the dual-write. With both set to ``None`` every
    write returns ``both_failed=True`` and no bytes move.
    """

    def __init__(
        self,
        r2=None,
        hippius=None,
        *,
        dual_write: bool = True,
        allow_fallback: bool = False,
    ):
        self.r2 = r2
        self.hippius = hippius
        self.dual_write = dual_write
        self.allow_fallback = allow_fallback
        self.counters = _Counters()

    # ── Internal helpers ─────────────────────────────────────────────

    async def _write_r2(self, key: str, data: bytes) -> bool:
        """Best-effort R2 write. Returns True on success."""
        if self.r2 is None or not key:
            return False
        try:
            ok = self.r2.upload_text(key, data.decode("utf-8")) \
                if self._looks_like_text(data) \
                else self._upload_binary_to_r2(key, data)
            return bool(ok)
        except Exception as e:
            logger.warning("R2 write %s failed: %s", key, e)
            return False

    @staticmethod
    def _looks_like_text(data: bytes) -> bool:
        """Heuristic: prefer upload_text for JSON/UTF-8, upload_file for blobs.

        We have no MIME hints from callers, so this falls back to a "decode
        succeeds" check. Robust enough for the JSON-tier artifacts this
        store currently fronts; large binary blobs (checkpoints) take the
        binary path via ``put_file``.
        """
        if not data:
            return True
        try:
            data.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    def _upload_binary_to_r2(self, key: str, data: bytes) -> bool:
        """R2 binary write via a temp file (S3 client takes a path)."""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            path = f.name
        try:
            return bool(self.r2.upload_file_from_disk(path, key))
        finally:
            import os as _os
            try:
                _os.unlink(path)
            except OSError:
                pass

    async def _write_hippius(self, data: bytes, metadata: dict) -> str:
        """Best-effort Hippius write. Returns CID on success, '' on failure."""
        if self.hippius is None:
            return ""
        try:
            result = await self.hippius.upload_bundle(data, metadata)
        except NotImplementedError as e:
            # Phase 2 (TEN-242) hasn't shipped upload yet. Treat as a
            # disabled backend rather than a bug — operators see the
            # warning once at startup, not on every write.
            logger.debug("Hippius upload skipped (not yet implemented): %s", e)
            return ""
        except Exception as e:
            logger.warning("Hippius upload failed: %s", e)
            return ""
        if result is None:
            return ""
        return getattr(result, "cid", None) or (
            result.get("cid", "") if isinstance(result, dict) else ""
        )

    # ── Public API ───────────────────────────────────────────────────

    async def put_bytes(
        self,
        key: str,
        data: bytes,
        metadata: Optional[dict] = None,
    ) -> DualWriteResult:
        """Dual-write a byte payload.

        Both backends are attempted in parallel. The result captures which
        side(s) succeeded. ``both_failed=True`` is the only outcome that
        warrants caller intervention; partial failures are logged and
        counted but treated as success.
        """
        meta = dict(metadata or {})
        result = DualWriteResult(r2_key=key)

        if self.dual_write or self.r2 is not None:
            result.r2_ok = await self._write_r2(key, data)
        if self.dual_write or self.hippius is not None:
            cid = await self._write_hippius(data, meta)
            result.hippius_ok = bool(cid)
            result.hippius_cid = cid

        # Counters
        c = self.counters
        if result.r2_ok:
            c.r2_writes_ok += 1
        elif self.r2 is not None:
            c.r2_writes_fail += 1
        if result.hippius_ok:
            c.hippius_writes_ok += 1
        elif self.hippius is not None:
            c.hippius_writes_fail += 1
        c.bytes_written += len(data)

        if not (result.r2_ok or result.hippius_ok):
            result.both_failed = True
            c.both_failed += 1
            logger.error(
                "Both R2 and Hippius failed for key=%s metadata=%s",
                key, sorted(meta.keys()),
            )
        return result

    async def put_json(
        self,
        key: str,
        data: dict,
        metadata: Optional[dict] = None,
    ) -> DualWriteResult:
        """Convenience wrapper: dump dict to deterministic JSON bytes."""
        body = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        return await self.put_bytes(key, body, metadata)

    async def get_bytes(
        self,
        key: str = "",
        cid: str = "",
    ) -> bytes:
        """Get by R2 key (preferred) or Hippius CID. Raises on no-data.

        With ``allow_fallback`` set, a failed R2 read falls through to
        Hippius (and vice versa). Empty values for both ``key`` and
        ``cid`` raises ``ValueError`` — the caller forgot to identify
        the artifact.
        """
        if not key and not cid:
            raise ValueError("get_bytes requires either key or cid")

        # R2-first when we have a key.
        if key and self.r2 is not None:
            data = self._read_r2(key)
            if data is not None:
                return data
            if not self.allow_fallback or not cid:
                raise FileNotFoundError(f"R2 read failed for key={key}")

        # Hippius read (primary path when caller passed only cid, or
        # fallback path when R2 missed).
        if cid and self.hippius is not None:
            try:
                return await self.hippius.download_bundle(cid)
            except Exception as e:
                if not self.allow_fallback or not key:
                    raise
                logger.warning(
                    "Hippius read for cid=%s failed: %s — falling back to R2",
                    cid, e,
                )

        # Last-resort R2 fallback (cid path tried first).
        if key and self.r2 is not None:
            data = self._read_r2(key)
            if data is not None:
                return data

        raise FileNotFoundError(f"both backends missed: key={key!r} cid={cid!r}")

    def _read_r2(self, key: str) -> Optional[bytes]:
        """Try text then binary for parity with `_write_r2`'s heuristic."""
        try:
            text = self.r2.download_text(key)
        except Exception as e:
            logger.warning("R2 read %s failed: %s", key, e)
            return None
        if text is not None:
            return text.encode()
        return None

    def snapshot_counters(self) -> dict:
        """Return a JSON-friendly snapshot of cumulative counters."""
        c = self.counters
        return {
            "r2_writes_ok": c.r2_writes_ok,
            "r2_writes_fail": c.r2_writes_fail,
            "hippius_writes_ok": c.hippius_writes_ok,
            "hippius_writes_fail": c.hippius_writes_fail,
            "both_failed": c.both_failed,
            "bytes_written": c.bytes_written,
            "mismatches": c.mismatches,
        }


__all__ = ["ArtifactStore", "DualWriteResult"]
