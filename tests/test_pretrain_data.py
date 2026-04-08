"""Tests for streaming pretrain dataloader."""

import io
import os
import random
import tempfile

import torch

from shared.pretrain_data import (
    _ShuffleBuffer,
    _cut_windows_from_series,
    pretrain_dataloader,
    PretrainBenchmark,
    FREQ_TO_PRED_LEN,
)


# ── ShuffleBuffer ──


def test_shuffle_buffer_fills_before_evicting():
    buf = _ShuffleBuffer(capacity=5, seed=42)
    evicted = []
    for i in range(5):
        result = buf.add(i)
        assert result is None  # no eviction while filling
    assert len(buf) == 5

    # 6th item should evict one
    result = buf.add(99)
    assert result is not None
    assert len(buf) == 5


def test_shuffle_buffer_drain():
    buf = _ShuffleBuffer(capacity=10, seed=42)
    for i in range(7):
        buf.add(i)
    drained = list(buf.drain())
    assert sorted(drained) == list(range(7))
    assert len(buf) == 0


def test_shuffle_buffer_eviction_contains_all():
    """All items (evicted + drained) should equal all items added."""
    buf = _ShuffleBuffer(capacity=5, seed=42)
    all_evicted = []
    for i in range(20):
        ev = buf.add(i)
        if ev is not None:
            all_evicted.append(ev)
    drained = list(buf.drain())
    all_items = sorted(all_evicted + drained)
    assert all_items == list(range(20))


# ── Window cutting ──


def test_cut_windows_basic():
    values = list(range(1000))
    windows = list(_cut_windows_from_series(values, "H", context_len=512, stride=64))
    pred_len = FREQ_TO_PRED_LEN["H"]  # 48
    window_len = 512 + pred_len
    expected_count = (1000 - window_len) // 64 + 1
    assert len(windows) == expected_count
    # Check first window
    ctx, tgt = windows[0]
    assert len(ctx) == 512
    assert len(tgt) == pred_len
    assert ctx == list(range(512))
    assert tgt == list(range(512, 512 + pred_len))


def test_cut_windows_short_series():
    """Series shorter than context+pred yields no windows."""
    values = list(range(100))
    windows = list(_cut_windows_from_series(values, "H", context_len=512))
    assert len(windows) == 0


def test_cut_windows_daily_freq():
    """Daily frequency should use pred_len=14."""
    values = list(range(600))
    windows = list(_cut_windows_from_series(values, "D", context_len=512, stride=32))
    for ctx, tgt in windows:
        assert len(tgt) == 14


def test_cut_windows_unknown_freq():
    """Unknown frequency should use default pred_len=96."""
    values = list(range(700))
    windows = list(_cut_windows_from_series(values, "UNKNOWN", context_len=512, stride=64))
    for ctx, tgt in windows:
        assert len(tgt) == 96


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
    # Create mock shards as local files and use file:// URLs
    shard_paths = []
    for i in range(2):
        path = _create_mock_parquet_shard(str(tmp_path), f"shard_{i}.parquet", n_series=20, series_len=700)
        shard_paths.append(path)

    # Monkey-patch _download_shard to read local files
    import shared.pretrain_data as ptd
    original_download = ptd._download_shard

    def mock_download(url: str) -> bytes:
        with open(url, "rb") as f:
            return f.read()

    ptd._download_shard = mock_download
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
            assert batch["target"].shape[2] == 1
            # Pred len should be 48 for "H" frequency
            assert batch["target"].shape[1] == 48
    finally:
        ptd._download_shard = original_download


def test_pretrain_dataloader_mixed_freq(tmp_path):
    """Dataloader handles mixed frequencies correctly."""
    import pandas as pd

    rng = random.Random(42)
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

    import shared.pretrain_data as ptd
    original_download = ptd._download_shard

    def mock_download(url: str) -> bytes:
        with open(url, "rb") as f:
            return f.read()

    ptd._download_shard = mock_download
    try:
        batches = list(pretrain_dataloader(
            shard_urls=[path],
            batch_size=4,
            context_len=512,
            shuffle_buffer_size=50,
            seed=42,
        ))
        pred_lens = set()
        for batch in batches:
            pred_lens.add(batch["target"].shape[1])
        # Should have both H (48) and D (14) pred lens
        assert 48 in pred_lens
        assert 14 in pred_lens
    finally:
        ptd._download_shard = original_download


# ── PretrainBenchmark ──


def test_pretrain_benchmark_select_shards_deterministic():
    """Same seed produces same shard selection."""

    class MockR2:
        def download_json(self, key):
            return {
                "shards": [
                    {"filename": f"shard_{i:05d}.parquet"}
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
        def download_json(self, key):
            return {
                "shards": [
                    {"filename": f"shard_{i:05d}.parquet"}
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
