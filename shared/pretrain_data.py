"""Generic shard-based pretrain infrastructure.

Task-agnostic utilities for streaming large pretrain datasets from R2:
  - ShuffleBuffer: fixed-capacity reservoir for approximate shuffling
  - download_shard: HTTP download of a single shard into memory
  - PretrainBenchmark: manifest loading, shard selection, presigned URL generation

Task-specific logic (parsing shard contents, windowing, batching) lives
in each task's runner directory, e.g. runner/timeseries_forecast/pretrain_loader.py.
"""

from __future__ import annotations

import logging
import random
from typing import Iterator

logger = logging.getLogger(__name__)


class ShuffleBuffer:
    """Fixed-capacity reservoir for approximate shuffling without full materialization.

    When full, adding a new item randomly evicts one (returned to caller).
    At the end, drain() yields remaining items in random order.
    """

    def __init__(self, capacity: int = 10_000, seed: int = 42):
        self._buf: list = []
        self._capacity = max(1, capacity)
        self._rng = random.Random(seed)

    def add(self, item):
        """Add item. Returns evicted item if full, else None."""
        if len(self._buf) < self._capacity:
            self._buf.append(item)
            return None
        idx = self._rng.randint(0, self._capacity - 1)
        evicted = self._buf[idx]
        self._buf[idx] = item
        return evicted

    def drain(self) -> Iterator:
        """Yield all remaining items in random order, then clear."""
        self._rng.shuffle(self._buf)
        yield from self._buf
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


def download_shard(url: str, timeout: int = 120) -> bytes:
    """Download a shard from a presigned URL into memory."""
    import httpx
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


class PretrainBenchmark:
    """Manages pretrain data: R2 manifest, shard selection, presigned URL generation.

    Task-agnostic — just knows about shard files listed in a manifest.json.
    """

    def __init__(
        self,
        r2=None,
        r2_prefix: str = "datasets/radar/v1",
    ):
        self.r2 = r2
        self.r2_prefix = r2_prefix
        self._manifest: dict | None = None

    def _load_manifest(self) -> dict:
        """Load manifest.json from R2."""
        if self._manifest is not None:
            return self._manifest
        if self.r2 is None:
            logger.warning("Pretrain R2 client is None, cannot load manifest")
            return {}
        key = f"{self.r2_prefix}/manifest.json"
        logger.info("Loading pretrain manifest from bucket=%s key=%s", self.r2.bucket, key)
        manifest = self.r2.download_json(key)
        if manifest:
            n_shards = len(manifest.get("shards", []))
            logger.info("Pretrain manifest loaded: %d shards", n_shards)
            self._manifest = manifest
        else:
            logger.warning(
                "Failed to load pretrain manifest from bucket=%s key=%s",
                self.r2.bucket, key,
            )
        return manifest or {}

    def get_shard_keys(self) -> list[str]:
        """Return all shard R2 keys from the manifest."""
        manifest = self._load_manifest()
        shards = manifest.get("shards", [])
        return [
            s["s3_key"]
            for s in shards
            if "s3_key" in s
        ]

    def select_shards(self, seed: int, n: int) -> list[str]:
        """Deterministic selection of N shard keys for a round."""
        all_keys = self.get_shard_keys()
        if n <= 0 or n >= len(all_keys):
            return all_keys
        rng = random.Random(seed)
        chosen = rng.sample(all_keys, n)
        return sorted(chosen)

    def generate_presigned_shard_urls(
        self,
        shard_keys: list[str],
        ttl: int = 5400,
    ) -> list[str]:
        """Generate presigned GET URLs for selected shards."""
        if self.r2 is None:
            return []
        urls = []
        for key in shard_keys:
            url = self.r2.generate_presigned_get_url(key, ttl=ttl)
            if url:
                urls.append(url)
        logger.info(
            "Generated %d presigned URLs for %d pretrain shards",
            len(urls), len(shard_keys),
        )
        return urls
