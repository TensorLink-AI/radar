"""Tests for pretrain infrastructure (shared) and ts-forecast loader (runner)."""

import io
import os
import random
import sys
import tempfile

import torch

# Add runner/timeseries_forecast to path so we can import pretrain_loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))

from shared.pretrain_data import ShuffleBuffer, PretrainBenchmark
from pretrain_loader import (
    cut_windows,
    pretrain_dataloader,
    CONTEXT_LEN,
    PREDICTION_LEN,
)


# ── ShuffleBuffer (generic, shared/) ──


def test_shuffle_buffer_fills_before_evicting():
    buf = ShuffleBuffer(capacity=5, seed=42)
    for i in range(5):
        result = buf.add(i)
        assert result is None
    assert len(buf) == 5

    result = buf.add(99)
    assert result is not None
    assert len(buf) == 5


def test_shuffle_buffer_drain():
    buf = ShuffleBuffer(capacity=10, seed=42)
    for i in range(7):
        buf.add(i)
    drained = list(buf.drain())
    assert sorted(drained) == list(range(7))
    assert len(buf) == 0


def test_shuffle_buffer_eviction_contains_all():
    """All items (evicted + drained) should equal all items added."""
    buf = ShuffleBuffer(capacity=5, seed=42)
    all_evicted = []
    for i in range(20):
        ev = buf.add(i)
        if ev is not None:
            all_evicted.append(ev)
    drained = list(buf.drain())
    all_items = sorted(all_evicted + drained)
    assert all_items == list(range(20))


# ── Window cutting (ts-specific, fixed prediction_len) ──


def test_cut_windows_basic():
    values = list(range(1000))
    windows = list(cut_windows(values, context_len=512, prediction_len=96, stride=64))
    window_len = 512 + 96
    expected_count = (1000 - window_len) // 64 + 1
    assert len(windows) == expected_count
    ctx, tgt = windows[0]
    assert len(ctx) == 512
    assert len(tgt) == 96
    assert ctx == list(range(512))
    assert tgt == list(range(512, 608))


def test_cut_windows_short_series():
    """Series shorter than context+pred yields no windows."""
    values = list(range(100))
    windows = list(cut_windows(values, context_len=512, prediction_len=96))
    assert len(windows) == 0


def test_cut_windows_fixed_prediction_len():
    """All windows have the same target length regardless of series content."""
    for series_len in [700, 1000, 2000]:
        values = list(range(series_len))
        windows = list(cut_windows(values, context_len=512, prediction_len=96, stride=64))
        for ctx, tgt in windows:
            assert len(ctx) == 512
            assert len(tgt) == 96


def test_cut_windows_matches_model_interface():
    """Window shapes match what build_model(512, 96, 1, quantiles) expects."""
    values = list(range(800))
    windows = list(cut_windows(values))
    assert len(windows) > 0
    for ctx, tgt in windows:
        assert len(ctx) == CONTEXT_LEN  # 512
        assert len(tgt) == PREDICTION_LEN  # 96


# ── Pretrain dataloader with mock shards ──


def _create_mock_parquet_shard(tmpdir: str, shard_name: str, n_series: int = 10, series_len: int = 700):
    """Create a mock parquet shard with random time series data."""
    import pandas as pd

    rng = random.Random(42)
    data = {
        "target": [
            [rng.gauss(0, 1) for _ in range(series_len)]
            for _ in range(n_series)
        ],
        "freq": ["H"] * n_series,
    }
    df = pd.DataFrame(data)
    path = os.path.join(tmpdir, shard_name)
    df.to_parquet(path)
    return path


def test_pretrain_dataloader_basic(tmp_path):
    """Pretrain dataloader yields batches with correct shapes."""
    shard_paths = []
    for i in range(2):
        path = _create_mock_parquet_shard(str(tmp_path), f"shard_{i}.parquet", n_series=20, series_len=700)
        shard_paths.append(path)

    import pretrain_loader as pl
    original_download = pl.download_shard

    def mock_download(url: str, timeout: int = 120) -> bytes:
        with open(url, "rb") as f:
            return f.read()

    pl.download_shard = mock_download
    try:
        batches = list(pretrain_dataloader(
            shard_urls=shard_paths,
            batch_size=8,
            context_len=512,
            shuffle_buffer_size=100,
            seed=42,
        ))
        assert len(batches) > 0
        for batch in batches:
            assert "context" in batch
            assert "target" in batch
            assert batch["context"].shape[1] == 512
            assert batch["context"].shape[2] == 1
            assert batch["target"].shape[1] == 96  # fixed prediction_len
            assert batch["target"].shape[2] == 1
    finally:
        pl.download_shard = original_download


def test_pretrain_dataloader_uniform_target_shape(tmp_path):
    """All batches have the same target shape — no variable pred_len."""
    import pandas as pd

    rng = random.Random(42)
    # Mix of frequencies — but target shape should still be (B, 96, 1)
    data = {
        "target": [
            [rng.gauss(0, 1) for _ in range(700)]
            for _ in range(20)
        ],
        "freq": ["H"] * 10 + ["D"] * 10,
    }
    df = pd.DataFrame(data)
    path = str(tmp_path / "mixed.parquet")
    df.to_parquet(path)

    import pretrain_loader as pl
    original_download = pl.download_shard

    def mock_download(url: str, timeout: int = 120) -> bytes:
        with open(url, "rb") as f:
            return f.read()

    pl.download_shard = mock_download
    try:
        batches = list(pretrain_dataloader(
            shard_urls=[path],
            batch_size=4,
            context_len=512,
            shuffle_buffer_size=50,
            seed=42,
        ))
        assert len(batches) > 0
        for batch in batches:
            # Every batch should have prediction_len=96, regardless of freq
            assert batch["target"].shape[1] == 96
            assert batch["context"].shape[1] == 512
    finally:
        pl.download_shard = original_download


# ── PretrainBenchmark (generic, shared/) ──


def test_pretrain_benchmark_select_shards_deterministic():
    """Same seed produces same shard selection."""

    class MockR2:
        bucket = "test-bucket"
        def download_json(self, key):
            return {
                "shards": [
                    {"s3_key": f"datasets/radar/v1/shard_{i:05d}.parquet", "shard_id": f"{i:05d}"}
                    for i in range(284)
                ],
            }

        def generate_presigned_get_url(self, key, ttl=900):
            return f"https://example.com/{key}?ttl={ttl}"

    bench = PretrainBenchmark(r2=MockR2(), r2_prefix="datasets/radar/v1")
    a = bench.select_shards(seed=42, n=8)
    b = bench.select_shards(seed=42, n=8)
    assert a == b
    assert len(a) == 8

    c = bench.select_shards(seed=999, n=8)
    assert c != a


def test_pretrain_benchmark_generate_urls():
    """Generates presigned URLs for selected shards."""
    generated = []

    class MockR2:
        bucket = "test-bucket"
        def download_json(self, key):
            return {
                "shards": [
                    {"s3_key": f"datasets/radar/v1/shard_{i:05d}.parquet", "shard_id": f"{i:05d}"}
                    for i in range(10)
                ],
            }

        def generate_presigned_get_url(self, key, ttl=900):
            generated.append(key)
            return f"https://example.com/{key}"

    bench = PretrainBenchmark(r2=MockR2(), r2_prefix="datasets/radar/v1")
    keys = bench.select_shards(seed=42, n=3)
    urls = bench.generate_presigned_shard_urls(keys)
    assert len(urls) == 3
    assert len(generated) == 3
    for key in generated:
        assert key.startswith("datasets/radar/v1/shard_")
