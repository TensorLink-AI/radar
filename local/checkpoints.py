"""Durable checkpoint store for continuation training.

The ts_forecasting trainer writes ``model.safetensors`` into an ephemeral
``tempfile.mkdtemp`` workdir that the validator ``shutil.rmtree``s right
after scoring. Continuation training needs the *parent's* checkpoint to
survive into a later round, so this module copies the checkpoint out of
the doomed workdir into a durable location keyed by experiment id.

A ``ref`` is an opaque string stored on the experiment row
(``checkpoint_ref``). ``resolve(ref)`` turns it back into a local path
(downloading from R2 first if the local copy is gone). Weights never
reach miner agents — only the *signature* (tensor names + shapes, read
from the safetensors header without torch) is exposed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "local/checkpoints"


def _default_dir() -> Path:
    return Path(os.environ.get("RADAR_CHECKPOINT_DIR", _DEFAULT_DIR))


class CheckpointStore:
    """File-backed checkpoint store, optionally mirrored to R2.

    ``base_dir`` holds ``{exp_id}.safetensors`` files. When ``sink`` is an
    R2-enabled ``ArtifactSink`` the checkpoint is also uploaded so it
    survives container recycling; ``resolve`` re-downloads on a local miss.
    """

    def __init__(self, base_dir: str | Path | None = None, sink: object = None):
        self.base_dir = Path(base_dir) if base_dir else _default_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sink = sink

    def _local_path(self, exp_id: int) -> Path:
        return self.base_dir / f"{int(exp_id)}.safetensors"

    def save(self, exp_id: int, src_path: str | Path) -> Optional[str]:
        """Copy ``src_path`` into the store keyed by ``exp_id``.

        Returns the ``checkpoint_ref`` to persist, or ``None`` if the
        source is missing.
        """
        src = Path(src_path)
        if not src.is_file():
            logger.warning("checkpoint save: source missing at %s", src)
            return None
        dst = self._local_path(exp_id)
        try:
            shutil.copy2(src, dst)
        except OSError as e:  # noqa: BLE001
            logger.warning("checkpoint save failed (%s); not persisted", e)
            return None
        ref = f"ckpt:{int(exp_id)}"
        if self.sink is not None and getattr(self.sink, "r2_enabled", False):
            try:
                key = f"checkpoints/{int(exp_id)}.safetensors"
                self.sink._client.upload_file_from_disk(str(dst), key)  # type: ignore[attr-defined]
            except Exception as e:  # noqa: BLE001
                logger.debug("checkpoint R2 mirror failed: %s", e)
        return ref

    def resolve(self, ref: Optional[str]) -> Optional[str]:
        """Turn a ``checkpoint_ref`` back into a local path, or ``None``."""
        if not ref or not ref.startswith("ckpt:"):
            return None
        try:
            exp_id = int(ref.split(":", 1)[1])
        except ValueError:
            return None
        local = self._local_path(exp_id)
        if local.is_file():
            return str(local)
        # Local miss — try pulling from R2 if available.
        if self.sink is not None and getattr(self.sink, "r2_enabled", False):
            try:
                key = f"checkpoints/{exp_id}.safetensors"
                body = self.sink.fetch_bytes(key)  # type: ignore[attr-defined]
                if body:
                    local.write_bytes(body)
                    return str(local)
            except Exception as e:  # noqa: BLE001
                logger.debug("checkpoint R2 fetch failed: %s", e)
        return None

    def gc(self, keep_ids: set[int]) -> int:
        """Delete local checkpoints whose experiment id is not in ``keep_ids``.

        Returns the number of files removed. R2 copies (if any) are left
        intact — they're cheap and re-downloadable.
        """
        removed = 0
        keep = {int(i) for i in keep_ids}
        for f in self.base_dir.glob("*.safetensors"):
            try:
                exp_id = int(f.stem)
            except ValueError:
                continue
            if exp_id not in keep:
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed


def read_signature(path: str | Path) -> dict[str, list[int]]:
    """Read tensor name → shape from a safetensors file header.

    Parses only the JSON header (no torch, no weight load), so it's cheap
    and safe to call at request time. Returns ``{}`` on any parse failure.
    """
    p = Path(path)
    try:
        with p.open("rb") as f:
            n_bytes = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n_bytes))
    except (OSError, ValueError, struct.error) as e:  # noqa: BLE001
        logger.debug("could not read safetensors header from %s: %s", p, e)
        return {}
    out: dict[str, list[int]] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta.get("shape") if isinstance(meta, dict) else None
        if isinstance(shape, list):
            out[name] = [int(x) for x in shape]
    return out
