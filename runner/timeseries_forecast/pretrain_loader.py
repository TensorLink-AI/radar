"""Streaming pretrain dataloader for time-series forecasting.

Task-specific: parses parquet shards with 'target' (list[float]) and 'freq'
(string) columns, cuts context/target windows based on frequency, and yields
shuffled batches. Uses generic shard infrastructure from shared/pretrain_data.py.
"""

from __future__ import annotations

import io
import logging
import random
from typing import Iterator

import torch

from shared.pretrain_data import ShuffleBuffer, download_shard

logger = logging.getLogger(__name__)

# ── Time-series-specific constants ──────────────────────────────────

FREQ_TO_PRED_LEN: dict[str, int] = {
    "H": 48, "T": 60, "D": 14, "W": 8, "M": 12, "Q": 8, "Y": 4,
    "B": 14, "5T": 60, "10T": 60, "15T": 48, "30T": 48,
}

DEFAULT_PRED_LEN = 96
DEFAULT_CONTEXT_LEN = 512
DEFAULT_STRIDE = 64
MAX_SERIES_LEN = 8192


# ── Shard iteration ─────────────────────────────────────────────────

def iter_series(
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
            raw = download_shard(url)
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

            if len(values) > MAX_SERIES_LEN:
                values = values[:MAX_SERIES_LEN]

            freq = str(row["freq"]) if has_freq else "H"
            yield values, freq

        del df, raw  # free memory before next shard


# ── Window cutting ──────────────────────────────────────────────────

def cut_windows(
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

    for start in range(0, n - window_len + 1, stride):
        ctx = values[start : start + context_len]
        tgt = values[start + context_len : start + window_len]
        yield ctx, tgt


# ── Dataloader ──────────────────────────────────────────────────────

def pretrain_dataloader(
    shard_urls: list[str],
    batch_size: int = 64,
    context_len: int = DEFAULT_CONTEXT_LEN,
    stride: int = DEFAULT_STRIDE,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
) -> Iterator[dict[str, torch.Tensor]]:
    """Streaming pretrain dataloader for time-series forecasting.

    Downloads parquet shards one at a time, extracts series, cuts windows,
    shuffles via a fixed-capacity buffer, and yields batches of
    {"context": (B, context_len, 1), "target": (B, pred_len, 1)}.

    Batches are grouped by pred_len so all items in a batch share the same
    target shape.
    """
    buf = ShuffleBuffer(capacity=shuffle_buffer_size, seed=seed)

    pending: dict[int, list[tuple[list[float], list[float]]]] = {}

    def _flush_pending(pred_len: int) -> Iterator[dict[str, torch.Tensor]]:
        items = pending.get(pred_len, [])
        while len(items) >= batch_size:
            batch_items = items[:batch_size]
            items = items[batch_size:]
            ctx = torch.tensor(
                [w[0] for w in batch_items], dtype=torch.float32,
            ).unsqueeze(-1)
            tgt = torch.tensor(
                [w[1] for w in batch_items], dtype=torch.float32,
            ).unsqueeze(-1)
            yield {"context": ctx, "target": tgt}
        pending[pred_len] = items

    def _process_window(window: tuple[list[float], list[float]]):
        pred_len = len(window[1])
        if pred_len not in pending:
            pending[pred_len] = []
        pending[pred_len].append(window)
        yield from _flush_pending(pred_len)

    for values, freq in iter_series(shard_urls, seed=seed):
        for window in cut_windows(values, freq, context_len, stride):
            evicted = buf.add(window)
            if evicted is not None:
                yield from _process_window(evicted)

    for window in buf.drain():
        yield from _process_window(window)

    # Flush remaining partial batches
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
