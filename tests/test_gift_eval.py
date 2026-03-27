"""Tests for GIFT-Eval integration and NUM_VARIATES=1 changes."""

import os
import struct
import tempfile

import torch

from shared.gift_eval import select_datasets, GIFT_EVAL_DATASETS


# ── Deterministic dataset selection ──


def test_gift_eval_select_deterministic():
    """Same seed produces same dataset selection."""
    a = select_datasets(eval_split_seed=42, n=5)
    b = select_datasets(eval_split_seed=42, n=5)
    assert a == b
    assert len(a) == 5
    # Different seeds should (almost certainly) produce different selections
    c = select_datasets(eval_split_seed=999, n=5)
    assert c != a  # extremely unlikely to match


def test_gift_eval_select_sorted():
    """Selected datasets are sorted for deterministic ordering."""
    result = select_datasets(eval_split_seed=123, n=5)
    assert result == sorted(result)


def test_gift_eval_select_from_known_list():
    """All selected datasets are from the known dataset list."""
    result = select_datasets(eval_split_seed=42, n=10)
    for name in result:
        assert name in GIFT_EVAL_DATASETS


def test_gift_eval_select_respects_n():
    """select_datasets returns exactly n datasets."""
    for n in [1, 3, 5, 10]:
        result = select_datasets(eval_split_seed=42, n=n)
        assert len(result) == n


# ── Arrow loading and windowing ──


def _create_mock_arrow_file(tmpdir: str, dataset_name: str, series_data: list[list[float]]):
    """Create a minimal Arrow IPC file matching GIFT-Eval format."""
    import pyarrow as pa

    ds_dir = os.path.join(tmpdir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)
    arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

    arrays = [pa.array(s) for s in series_data]
    target_col = pa.ListArray.from_arrays(
        offsets=[0] + [len(s) for s in series_data],
        values=pa.concat_arrays(arrays),
    )
    # Fix offsets calculation
    import numpy as np
    offsets = np.cumsum([0] + [len(s) for s in series_data])
    target_col = pa.ListArray.from_arrays(offsets.tolist(), pa.concat_arrays(arrays))

    table = pa.table({"target": target_col})
    with pa.ipc.new_file(arrow_path, table.schema) as writer:
        writer.write_table(table)

    return arrow_path


def test_gift_eval_arrow_load():
    """Load mock Arrow file and verify shapes."""
    from shared.gift_eval import load_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 series of length 700 (enough for context=512 + pred=96)
        series_data = [list(range(700)) for _ in range(3)]
        _create_mock_arrow_file(tmpdir, "test_dataset", series_data)

        samples = load_dataset(
            "test_dataset", context_len=512, prediction_len=96,
            cache_dir=tmpdir,
        )
        assert len(samples) == 3
        assert len(samples[0]["context"]) == 512
        assert len(samples[0]["target"]) == 96


def test_gift_eval_windowing():
    """Context/target sliced from END of each series."""
    from shared.gift_eval import load_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Series: [0, 1, 2, ..., 699]
        series_data = [list(range(700))]
        _create_mock_arrow_file(tmpdir, "test_window", series_data)

        samples = load_dataset(
            "test_window", context_len=512, prediction_len=96,
            cache_dir=tmpdir,
        )
        assert len(samples) == 1
        # Target should be the last 96 values: [604, 605, ..., 699]
        assert samples[0]["target"][-1] == 699
        assert samples[0]["target"][0] == 604
        # Context should be the preceding 512 values: [92, ..., 603]
        assert samples[0]["context"][-1] == 603
        assert samples[0]["context"][0] == 92


def test_gift_eval_short_series_skipped():
    """Series shorter than context_len + pred_len are skipped."""
    from shared.gift_eval import load_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        series_data = [
            list(range(100)),   # too short
            list(range(700)),   # long enough
        ]
        _create_mock_arrow_file(tmpdir, "test_skip", series_data)

        samples = load_dataset(
            "test_skip", context_len=512, prediction_len=96,
            cache_dir=tmpdir,
        )
        assert len(samples) == 1


def test_gift_eval_batches_univariate():
    """get_eval_batches yields (B, T, 1) and (B, P, 1) tensors."""
    from shared.gift_eval import load_dataset, get_eval_batches

    with tempfile.TemporaryDirectory() as tmpdir:
        series_data = [list(range(700)) for _ in range(5)]
        _create_mock_arrow_file(tmpdir, "test_batches", series_data)

        samples = load_dataset(
            "test_batches", context_len=512, prediction_len=96,
            cache_dir=tmpdir,
        )
        batches = list(get_eval_batches(samples, batch_size=3))
        assert len(batches) == 2  # 5 samples / 3 batch_size = 2 batches

        # First batch
        assert batches[0]["context"].shape == (3, 512, 1)
        assert batches[0]["target"].shape == (3, 96, 1)

        # Second batch (2 remaining)
        assert batches[1]["context"].shape == (2, 512, 1)
        assert batches[1]["target"].shape == (2, 96, 1)


# ── NUM_VARIATES = 1 ──


def test_univariate_constant():
    """prepare.py NUM_VARIATES is 1."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import NUM_VARIATES
    assert NUM_VARIATES == 1


def test_univariate_random_dataloader_shapes():
    """Random dataloader produces (B, T, 1) shapes."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import _random_dataloader, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES
    loader = _random_dataloader(batch_size=4)
    batch = next(loader)
    assert batch["context"].shape == (4, CONTEXT_LEN, NUM_VARIATES)
    assert batch["target"].shape == (4, PREDICTION_LEN, NUM_VARIATES)
    assert NUM_VARIATES == 1


def test_univariate_model_output_shape():
    """Model output (B, P, 1, 9) with NUM_VARIATES=1."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES

    # Simple model matching expected interface
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(CONTEXT_LEN, PREDICTION_LEN * NUM_VARIATES * len(QUANTILES))

        def forward(self, x):
            B = x.shape[0]
            out = self.fc(x.mean(dim=2))
            return out.view(B, PREDICTION_LEN, NUM_VARIATES, len(QUANTILES))

    model = TinyModel()
    x = torch.randn(2, CONTEXT_LEN, NUM_VARIATES)
    out = model(x)
    assert out.shape == (2, PREDICTION_LEN, 1, 9)


# ── Fallback mode ──


def test_gift_eval_fallback_random():
    """RADAR_EVAL_DATA=random still works via _random_validate."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import _random_validate, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES

    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(CONTEXT_LEN, PREDICTION_LEN * NUM_VARIATES * len(QUANTILES))

        def forward(self, x):
            B = x.shape[0]
            out = self.fc(x.mean(dim=2))
            return out.view(B, PREDICTION_LEN, NUM_VARIATES, len(QUANTILES))

    model = TinyModel()
    result = _random_validate(model, n_batches=2, batch_size=4)
    assert "crps" in result
    assert "ncrps" in result
    assert "mase" in result
    assert result["crps"] > 0
    assert result["crps"] < float("inf")
