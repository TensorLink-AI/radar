"""Streaming shard-based pretrain dataloader for time-series forecasting.

Downloads parquet shards one at a time from presigned URLs (or R2 directly),
extracts series, cuts context/target windows, and yields shuffled batches.
Designed for ~1TB pretrain data that cannot fit in memory or on disk at once.

Architecture:
  _iter_series()              → yields (values, freq) from parquet shards
  _cut_windows_from_series()  → slides context+horizon windows, yields (x, y)
  _ShuffleBuffer              → fixed-capacity reservoir for approximate shuffling
  pretrain_dataloader()       → top-level generator yielding {"context", "target"} batches
"""

from __future__ import annotations

import io
import logging
import random
from typing import Iterator

import torch

logger = logging.getLogger(__name__)

# Same frequency → prediction-length mapping as gift_eval.py
FREQ_TO_PRED_LEN: dict[str, int] = {
    "H": 48, "T": 60, "D": 14, "W": 8, "M": 12, "Q": 8, "Y": 4,
    "B": 14, "5T": 60, "10T": 60, "15T": 48, "30T": 48,
}

DEFAULT_PRED_LEN = 96
DEFAULT_CONTEXT_LEN = 512
DEFAULT_STRIDE = 64
MAX_SERIES_LEN = 8192


class _ShuffleBuffer:
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


def _download_shard(url: str) -> bytes:
    """Download a shard from a presigned URL into memory."""
    import httpx
    resp = httpx.get(url, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def _iter_series(
    shard_urls: list[str],
    seed: int = 42,
) -> Iterator[tuple[list[float], str]]:
    """Download shards one at a time, yield (values, freq) per series.

    Each shard is a parquet file with at least a 'target' column (list[float])
    and a 'freq' column (string). Shards are shuffled, then rows within each
    shard are shuffled.
    """
    import pandas as pd

    rng = random.Random(seed)
    order = list(range(len(shard_urls)))
    rng.shuffle(order)

    for shard_idx in order:
        url = shard_urls[shard_idx]
        try:
            raw = _download_shard(url)
            df = pd.read_parquet(io.BytesIO(raw))
        except Exception as e:
            logger.warning("Failed to load shard %d: %s", shard_idx, e)
            continue

        if "target" not in df.columns:
            logger.warning("Shard %d missing 'target' column, skipping", shard_idx)
            continue

        has_freq = "freq" in df.columns

        # Shuffle rows within shard
        indices = list(range(len(df)))
        rng.shuffle(indices)

        for row_idx in indices:
            row = df.iloc[row_idx]
            values = row["target"]
            if not isinstance(values, list):
                try:
                    values = list(values)
                except (TypeError, ValueError):
                    continue

            # Clamp to max length to avoid memory issues
            if len(values) > MAX_SERIES_LEN:
                values = values[:MAX_SERIES_LEN]

            freq = str(row["freq"]) if has_freq else "H"
            yield values, freq

        del df, raw  # free memory before next shard


def _cut_windows_from_series(
    values: list[float],
    freq: str,
    context_len: int = DEFAULT_CONTEXT_LEN,
    stride: int = DEFAULT_STRIDE,
) -> Iterator[tuple[list[float], list[float]]]:
    """Slide a (context_len + pred_len) window over a series.

    Yields (context_window, target_window) pairs.
    pred_len is determined from the frequency string.
    """
    pred_len = FREQ_TO_PRED_LEN.get(freq, DEFAULT_PRED_LEN)
    window_len = context_len + pred_len
    n = len(values)

    if n < window_len:
        return

    # Slide with stride
    for start in range(0, n - window_len + 1, stride):
        ctx = values[start : start + context_len]
        tgt = values[start + context_len : start + window_len]
        yield ctx, tgt


def pretrain_dataloader(
    shard_urls: list[str],
    batch_size: int = 64,
    context_len: int = DEFAULT_CONTEXT_LEN,
    stride: int = DEFAULT_STRIDE,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
) -> Iterator[dict[str, torch.Tensor]]:
    """Streaming pretrain dataloader.

    Downloads parquet shards one at a time, extracts series, cuts windows,
    shuffles via a fixed-capacity buffer, and yields batches of
    {"context": (B, context_len, 1), "target": (B, pred_len, 1)}.

    Because different series have different prediction lengths (based on freq),
    batches are grouped by pred_len so all items in a batch share the same
    target shape.
    """
    buf = _ShuffleBuffer(capacity=shuffle_buffer_size, seed=seed)

    # Accumulate windows by pred_len for batching
    pending: dict[int, list[tuple[list[float], list[float]]]] = {}

    def _flush_pending(pred_len: int) -> Iterator[dict[str, torch.Tensor]]:
        """Yield full batches from pending[pred_len]."""
        items = pending.get(pred_len, [])
        while len(items) >= batch_size:
            batch_items = items[:batch_size]
            items = items[batch_size:]
            ctx = torch.tensor(
                [w[0] for w in batch_items], dtype=torch.float32,
            ).unsqueeze(-1)  # (B, context_len, 1)
            tgt = torch.tensor(
                [w[1] for w in batch_items], dtype=torch.float32,
            ).unsqueeze(-1)  # (B, pred_len, 1)
            yield {"context": ctx, "target": tgt}
        pending[pred_len] = items

    def _process_window(window: tuple[list[float], list[float]]):
        """Add window to pending, yield any full batches."""
        pred_len = len(window[1])
        if pred_len not in pending:
            pending[pred_len] = []
        pending[pred_len].append(window)
        yield from _flush_pending(pred_len)

    # Main loop: iterate shards → series → windows → shuffle buffer → batches
    for values, freq in _iter_series(shard_urls, seed=seed):
        for window in _cut_windows_from_series(values, freq, context_len, stride):
            evicted = buf.add(window)
            if evicted is not None:
                yield from _process_window(evicted)

    # Drain remaining items from shuffle buffer
    for window in buf.drain():
        yield from _process_window(window)

    # Flush any remaining partial batches
    for pred_len in list(pending.keys()):
        items = pending[pred_len]
        if items:
            ctx = torch.tensor(
                [w[0] for w in items], dtype=torch.float32,
            ).unsqueeze(-1)
            tgt = torch.tensor(
                [w[1] for w in items], dtype=torch.float32,
            ).unsqueeze(-1)
            yield {"context": ctx, "target": tgt}
            pending[pred_len] = []


class PretrainBenchmark:
    """Manages pretrain data: R2 manifest, presigned URL generation."""

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
            f"{self.r2_prefix}/{s['filename']}"
            for s in shards
            if "filename" in s
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
