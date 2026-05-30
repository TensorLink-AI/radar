"""Streaming pretrain dataloader for time-series forecasting.

Task-specific: parses parquet shards with 'target' (list[float]) and 'freq'
(string) columns, cuts fixed-size context/target windows, and yields shuffled
batches. Uses generic shard infrastructure from shared/pretrain_data.py.

All windows use the same PREDICTION_LEN (96) to match the model output shape.
Frequency-specific prediction lengths are a GIFT-Eval evaluation convention
and don't apply during training — miners build models with a fixed
(context_len=512, prediction_len=96) interface.
"""

from __future__ import annotations

import io
import logging
import math
import random
from typing import Iterator

import torch

from shared.pretrain_data import ShuffleBuffer, download_shard

logger = logging.getLogger(__name__)

# ── Constants (must match prepare.py) ───────────────────────────────

CONTEXT_LEN = 512
PREDICTION_LEN = 96
DEFAULT_STRIDE = 64
MAX_SERIES_LEN = 8192


# ── Shard iteration ─────────────────────────────────────────────────

def iter_series(
    shard_urls: list[str] | None = None,
    seed: int = 42,
    shard_paths: list[str] | None = None,
) -> Iterator[list[float]]:
    """Yield raw value arrays from parquet shards.

    Reads either prefetched local files (``shard_paths``, preferred when
    set — used by the sandboxed trainer) or presigned URLs that get
    downloaded on-the-fly.  Shards are shuffled, then rows within each
    shard are shuffled.
    """
    import pandas as pd

    sources: list[str] = list(shard_paths or shard_urls or [])
    use_local = bool(shard_paths)

    rng = random.Random(seed)
    order = list(range(len(sources)))
    rng.shuffle(order)

    for shard_idx in order:
        source = sources[shard_idx]
        try:
            if use_local:
                df = pd.read_parquet(source)
            else:
                raw = download_shard(source)
                df = pd.read_parquet(io.BytesIO(raw))
        except Exception as e:
            logger.warning("Failed to load shard %d: %s", shard_idx, e)
            continue

        if "target" not in df.columns:
            logger.warning("Shard %d missing 'target' column, skipping", shard_idx)
            continue

        # Shuffle rows within shard
        indices = list(range(len(df)))
        rng.shuffle(indices)

        for row_idx in indices:
            values = df.iloc[row_idx]["target"]
            if not isinstance(values, list):
                try:
                    values = list(values)
                except (TypeError, ValueError):
                    continue

            if len(values) > MAX_SERIES_LEN:
                values = values[:MAX_SERIES_LEN]

            # Drop series containing any non-finite values — they propagate
            # NaN/inf through the model and trip the divergence guard.
            if any(not math.isfinite(v) for v in values):
                continue

            yield values

        del df  # free memory before next shard


# ── Window cutting ──────────────────────────────────────────────────

def cut_windows(
    values: list[float],
    context_len: int = CONTEXT_LEN,
    prediction_len: int = PREDICTION_LEN,
    stride: int = DEFAULT_STRIDE,
) -> Iterator[tuple[list[float], list[float]]]:
    """Slide a fixed (context_len + prediction_len) window over a series.

    Yields (context_window, target_window) pairs. All windows have the
    same shape so they're compatible with the miner's model output.
    """
    window_len = context_len + prediction_len
    n = len(values)

    if n < window_len:
        return

    for start in range(0, n - window_len + 1, stride):
        ctx = values[start : start + context_len]
        tgt = values[start + context_len : start + window_len]
        yield ctx, tgt


# ── Dataloader ──────────────────────────────────────────────────────

def pretrain_dataloader(
    shard_urls: list[str] | None = None,
    batch_size: int = 64,
    context_len: int = CONTEXT_LEN,
    prediction_len: int = PREDICTION_LEN,
    stride: int = DEFAULT_STRIDE,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
    shard_paths: list[str] | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Streaming pretrain dataloader for time-series forecasting.

    Reads parquet shards (from local paths or presigned URLs), extracts
    series, cuts fixed-size windows, shuffles, and yields batches of
    {"context": (B, context_len, 1), "target": (B, prediction_len, 1)}.

    All batches have the same target shape (matching PREDICTION_LEN=96)
    so they're compatible with models built via build_model().
    """
    buf = ShuffleBuffer(capacity=shuffle_buffer_size, seed=seed)
    pending: list[tuple[list[float], list[float]]] = []

    def _flush() -> Iterator[dict[str, torch.Tensor]]:
        nonlocal pending
        while len(pending) >= batch_size:
            batch_items = pending[:batch_size]
            pending = pending[batch_size:]
            ctx = torch.tensor(
                [w[0] for w in batch_items], dtype=torch.float32,
            ).unsqueeze(-1)  # (B, context_len, 1)
            tgt = torch.tensor(
                [w[1] for w in batch_items], dtype=torch.float32,
            ).unsqueeze(-1)  # (B, prediction_len, 1)
            yield {"context": ctx, "target": tgt}

    for values in iter_series(
        shard_urls=shard_urls, seed=seed, shard_paths=shard_paths,
    ):
        for window in cut_windows(values, context_len, prediction_len, stride):
            evicted = buf.add(window)
            if evicted is not None:
                pending.append(evicted)
                yield from _flush()

    for window in buf.drain():
        pending.append(window)
        yield from _flush()

    # Flush remaining partial batch
    if pending:
        ctx = torch.tensor(
            [w[0] for w in pending], dtype=torch.float32,
        ).unsqueeze(-1)
        tgt = torch.tensor(
            [w[1] for w in pending], dtype=torch.float32,
        ).unsqueeze(-1)
        yield {"context": ctx, "target": tgt}
