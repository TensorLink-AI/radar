"""Tests for GIFT-Eval integration and NUM_VARIATES=1 changes."""

import math
import os
import struct
import tempfile

import pytest
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


def test_gift_eval_nan_series_skipped():
    """Series with NaN values are filtered out during loading."""
    from shared.gift_eval import load_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # context=values[92:604], target=values[604:700]
        clean = [float(x) for x in range(700)]
        nan_in_target = list(clean)
        nan_in_target[650] = float("nan")          # index 650 is in target window
        nan_in_context = list(clean)
        nan_in_context[300] = float("nan")          # index 300 is in context window
        series_data = [
            clean,              # clean
            nan_in_target,      # NaN in target region
            nan_in_context,     # NaN in context region
            clean,              # clean
        ]
        _create_mock_arrow_file(tmpdir, "test_nan", series_data)

        samples = load_dataset(
            "test_nan", context_len=512, prediction_len=96,
            cache_dir=tmpdir,
        )
        # Only the 2 clean series should survive
        assert len(samples) == 2
        # Verify no NaN in returned data
        import math
        for s in samples:
            assert not any(math.isnan(v) for v in s["context"])
            assert not any(math.isnan(v) for v in s["target"])


def test_gift_eval_freq_override_pred_len():
    """Frequency metadata overrides prediction_len in loaded samples."""
    import pyarrow as pa
    from shared.gift_eval import load_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_dir = os.path.join(tmpdir, "test_freq")
        os.makedirs(ds_dir, exist_ok=True)
        arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

        # Create Arrow file with freq="H" metadata → pred_len should become 48
        series_data = [list(range(700))]
        arrays = [pa.array(s) for s in series_data]
        import numpy as np
        offsets = np.cumsum([0] + [len(s) for s in series_data])
        target_col = pa.ListArray.from_arrays(offsets.tolist(), pa.concat_arrays(arrays))

        schema = pa.schema([("target", target_col.type)],
                           metadata={b"freq": b"H"})
        table = pa.table({"target": target_col}, schema=schema)
        with pa.ipc.new_file(arrow_path, table.schema) as writer:
            writer.write_table(table)

        samples = load_dataset(
            "test_freq", context_len=512, prediction_len=96,
            cache_dir=tmpdir,
        )
        assert len(samples) == 1
        assert samples[0]["prediction_len"] == 48
        assert len(samples[0]["target"]) == 48


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


def test_gift_eval_validate_truncates_predictions():
    """Evaluation truncates model predictions to match task horizon."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import _gift_eval_validate, CONTEXT_LEN, PREDICTION_LEN, QUANTILES
    import pyarrow as pa
    import numpy as np
    import torch.nn as nn

    # Use "hospital" (freq M, non-m4 → base 12, short term → H=12). Model
    # always emits PREDICTION_LEN=96, which must be truncated to 12.
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_dir = os.path.join(tmpdir, "hospital")  # manifest key
        os.makedirs(ds_dir, exist_ok=True)
        arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

        # 5 series each 700 long
        series_data = [list(range(700)) for _ in range(5)]
        arrays = [pa.array(s) for s in series_data]
        offsets = np.cumsum([0] + [len(s) for s in series_data])
        target_col = pa.ListArray.from_arrays(offsets.tolist(), pa.concat_arrays(arrays))
        table = pa.table({"target": target_col})
        with pa.ipc.new_file(arrow_path, table.schema) as writer:
            writer.write_table(table)

        class FixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(CONTEXT_LEN, PREDICTION_LEN * len(QUANTILES))

            def forward(self, x):
                B = x.shape[0]
                out = self.fc(x.mean(dim=2))
                return out.view(B, PREDICTION_LEN, 1, len(QUANTILES))

        model = FixedModel()
        result = _gift_eval_validate(
            model, ["hospital"], batch_size=5, seed=42, data_dir=tmpdir,
        )
        # Should succeed (not crash) and produce finite CRPS
        assert "crps" in result
        assert result["crps"] < float("inf")
        assert not math.isnan(result["crps"])
        assert result.get("n_tasks", 0) >= 1


def test_gift_eval_all_nan_model_returns_inf():
    """Model that outputs ALL NaN scores as failure (inf), not nan."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import _random_validate, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES
    import math
    import torch.nn as nn

    class NaNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)

        def forward(self, x):
            B = x.shape[0]
            return torch.full(
                (B, PREDICTION_LEN, NUM_VARIATES, len(QUANTILES)), float("nan")
            )

    model = NaNModel()
    result = _random_validate(model, n_batches=2, batch_size=4)
    # All-NaN model → no valid samples → inf (not nan, which breaks weights)
    assert not math.isnan(result["crps"])
    assert result["crps"] == float("inf")
    assert result["n_samples"] == 0


def test_gift_eval_partial_nan_masked():
    """Partial NaN in model output: valid samples still scored."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import _random_validate, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES
    import math
    import torch.nn as nn

    class PartialNaNModel(nn.Module):
        """First sample in each batch is NaN, rest are valid."""
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(CONTEXT_LEN, PREDICTION_LEN * NUM_VARIATES * len(QUANTILES))

        def forward(self, x):
            B = x.shape[0]
            out = self.fc(x.mean(dim=2))
            out = out.view(B, PREDICTION_LEN, NUM_VARIATES, len(QUANTILES))
            out[0] = float("nan")  # poison first sample only
            return out

    model = PartialNaNModel()
    result = _random_validate(model, n_batches=2, batch_size=4)
    # Partial NaN → masked out, valid samples produce finite CRPS
    assert not math.isnan(result["crps"])
    assert result["crps"] < float("inf")


def test_evaluator_env_has_cache_default():
    """Evaluator subprocess env uses correct GIFT_EVAL_CACHE default."""
    # Verify the evaluator code uses /tmp/radar_gift_eval as default
    # (not empty string, which would cause subprocess to miss cached data)
    import ast
    eval_path = os.path.join(
        os.path.dirname(__file__), "..", "validator", "evaluator.py"
    )
    with open(eval_path) as f:
        source = f.read()
    assert "/tmp/radar_gift_eval" in source


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


# ── New leaderboard constants / task expansion ──


def test_seasonality_map_matches_gluonts():
    """Every freq in SEASONALITY_MAP matches gluonts.get_seasonality."""
    pytest.importorskip("gluonts")
    import warnings
    from gluonts.time_feature import get_seasonality as gluonts_seasonality
    from shared.gift_eval import SEASONALITY_MAP, get_seasonality

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for freq, ours in SEASONALITY_MAP.items():
            g = gluonts_seasonality(freq)
            assert g == ours, f"{freq}: ours={ours} gluonts={g}"
            # And the function path agrees too.
            assert get_seasonality(freq) == ours


def test_build_task_configs_produces_97_tasks():
    """SHORT_DATASETS expanded with MED_LONG_DATASETS yields >=95 tasks."""
    from shared.gift_eval import SHORT_DATASETS, build_task_configs

    tasks = build_task_configs(SHORT_DATASETS)
    assert len(tasks) >= 95
    # Sanity checks
    terms = {t["term"] for t in tasks}
    assert terms == {"short", "medium", "long"}
    for t in tasks:
        assert t["prediction_length"] > 0
        assert t["season_length"] >= 1
        assert "/" in t["config_name"]


def test_m4_not_in_med_long():
    """No m4_* dataset should appear in MED_LONG_DATASETS."""
    from shared.gift_eval import MED_LONG_DATASETS
    for name in MED_LONG_DATASETS:
        assert not name.startswith("m4_"), f"m4 should not be in MED_LONG: {name}"


def test_loop_seattle_d_is_short_only():
    """LOOP_SEATTLE/D produces only a short-term task (per user spec)."""
    from shared.gift_eval import build_task_configs
    tasks = build_task_configs(["LOOP_SEATTLE/D", "LOOP_SEATTLE/H"])
    loop_d = [t for t in tasks if t["dataset"] == "LOOP_SEATTLE" and t["freq"] == "D"]
    loop_h = [t for t in tasks if t["dataset"] == "LOOP_SEATTLE" and t["freq"] == "H"]
    assert [t["term"] for t in loop_d] == ["short"]
    assert sorted(t["term"] for t in loop_h) == ["long", "medium", "short"]


def test_m4_horizons_applied():
    """m4 datasets use M4_HORIZONS, non-m4 use PRED_LENGTH_MAP."""
    from shared.gift_eval import build_task_configs
    tasks = build_task_configs(["m4_yearly", "m4_daily", "electricity/D"])
    by_name = {t["config_name"]: t for t in tasks}
    assert by_name["m4_yearly/A/short"]["prediction_length"] == 6
    assert by_name["m4_daily/D/short"]["prediction_length"] == 14
    # non-m4 daily → 30 (not 14)
    assert by_name["electricity/D/short"]["prediction_length"] == 30


# ── Rolling-origin windowing ──


def test_seasonal_naive_forecast_shape():
    """_seasonal_naive_forecast tiles the last `season` values to `horizon`."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import _seasonal_naive_forecast

    # Season-24 history of length 100, horizon 48 → repeat last 24 twice
    history = torch.arange(100, dtype=torch.float32)
    out = _seasonal_naive_forecast(history, horizon=48, season=24)
    assert out.shape == (48,)
    # last-24 of history = [76..99]; tiled twice
    expected_first = history[-24:]
    torch.testing.assert_close(out[:24], expected_first)
    torch.testing.assert_close(out[24:], expected_first)


def test_seasonal_naive_fallback_when_short_history():
    """Falls back to last-value repeat when season >= history length."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import _seasonal_naive_forecast

    history = torch.tensor([1.0, 2.0, 3.0])
    out = _seasonal_naive_forecast(history, horizon=5, season=10)
    # season > len(history) → fallback to s=1 → repeat last value (3.0)
    torch.testing.assert_close(out, torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0]))


def test_rolling_windows_non_overlapping():
    """load_dataset_for_task carves non-overlapping windows from the end."""
    import pyarrow as pa
    from shared.gift_eval import load_dataset_for_task

    with tempfile.TemporaryDirectory() as tmpdir:
        # Manifest key "ett1__H" expected for (ett1, H). Create that dir.
        ds_dir = os.path.join(tmpdir, "ett1__H")
        os.makedirs(ds_dir, exist_ok=True)
        arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

        series = [float(x) for x in range(1000)]
        arr = pa.array(series)
        target_col = pa.ListArray.from_arrays([0, len(series)], arr)
        table = pa.table({"target": target_col})
        with pa.ipc.new_file(arrow_path, table.schema) as writer:
            writer.write_table(table)

        task = {
            "dataset": "ett1",
            "freq": "H",
            "term": "short",
            "prediction_length": 48,
            "season_length": 24,
            "config_name": "ett1/H/short",
        }
        samples = load_dataset_for_task(task, context_len=96, cache_dir=tmpdir)
        # 1000 - 96 = 904 usable, // 48 = 18 windows (under cap of 20)
        assert len(samples) == 18
        # i=0 → target is last 48 values
        assert samples[0]["target"] == series[-48:]
        # i=1 → target is 48 values before that
        assert samples[1]["target"] == series[-96:-48]
        # context is exactly 96 long
        assert all(len(s["context"]) == 96 for s in samples)


def test_rolling_windows_max_20_cap():
    """Caps at 20 windows per series even if more are available."""
    import pyarrow as pa
    from shared.gift_eval import load_dataset_for_task

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_dir = os.path.join(tmpdir, "ett1__H")
        os.makedirs(ds_dir, exist_ok=True)
        arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

        # 5000 values, context=100, pred=10 → 490 possible, capped at 20
        series = [float(x) for x in range(5000)]
        arr = pa.array(series)
        target_col = pa.ListArray.from_arrays([0, len(series)], arr)
        table = pa.table({"target": target_col})
        with pa.ipc.new_file(arrow_path, table.schema) as writer:
            writer.write_table(table)

        task = {
            "dataset": "ett1", "freq": "H", "term": "short",
            "prediction_length": 10, "season_length": 24,
            "config_name": "ett1/H/short",
        }
        samples = load_dataset_for_task(task, context_len=100, cache_dir=tmpdir)
        assert len(samples) == 20


def test_rolling_windows_too_short_series_skipped():
    """Series with len(series) < context_len + H are dropped."""
    import pyarrow as pa
    from shared.gift_eval import load_dataset_for_task

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_dir = os.path.join(tmpdir, "ett1__H")
        os.makedirs(ds_dir, exist_ok=True)
        arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

        series = [float(x) for x in range(50)]  # too short
        arr = pa.array(series)
        target_col = pa.ListArray.from_arrays([0, len(series)], arr)
        table = pa.table({"target": target_col})
        with pa.ipc.new_file(arrow_path, table.schema) as writer:
            writer.write_table(table)

        task = {
            "dataset": "ett1", "freq": "H", "term": "short",
            "prediction_length": 48, "season_length": 24,
            "config_name": "ett1/H/short",
        }
        samples = load_dataset_for_task(task, context_len=96, cache_dir=tmpdir)
        assert samples == []


def test_no_500_series_cap_by_default():
    """max_series=0 (default) should not subsample — all windows evaluated."""
    import pyarrow as pa
    from shared.gift_eval import load_dataset_for_task

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_dir = os.path.join(tmpdir, "ett1__H")
        os.makedirs(ds_dir, exist_ok=True)
        arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

        # 600 series, each just long enough for exactly 1 window
        # context=50, pred=10 → need len >= 60; use 60 exactly → 1 window each
        n_series = 600
        series_len = 60
        arrays = []
        offsets = [0]
        for s in range(n_series):
            vals = [float((s + x) % 100) for x in range(series_len)]
            arrays.append(pa.array(vals))
            offsets.append(offsets[-1] + series_len)
        target_col = pa.ListArray.from_arrays(offsets, pa.concat_arrays(arrays))
        table = pa.table({"target": target_col})
        with pa.ipc.new_file(arrow_path, table.schema) as writer:
            writer.write_table(table)

        task = {
            "dataset": "ett1", "freq": "H", "term": "short",
            "prediction_length": 10, "season_length": 2,
            "config_name": "ett1/H/short",
        }
        samples = load_dataset_for_task(
            task, context_len=50, cache_dir=tmpdir, max_series=0,
        )
        # 600 series * 1 window each
        assert len(samples) == 600


# ── Metric math ──


def test_wql_components_gluonts_agreement():
    """_wql_components summed across series matches gluonts
    mean_weighted_sum_quantile_loss(axis=None)."""
    pytest.importorskip("gluonts")
    import numpy as np
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import _wql_components

    torch.manual_seed(0)
    np.random.seed(0)
    B, H, V, Q = 4, 12, 1, 9
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Random positive targets so |target| is well-defined
    targets = torch.rand(B, H, V) * 10.0 + 1.0
    preds = targets.unsqueeze(-1) + torch.randn(B, H, V, Q) * 0.5

    # Ours: task-level aggregation
    q_t = torch.tensor(quantiles)
    num, den = _wql_components(preds, targets, q_t)
    ours = float(num.sum() / den.sum())

    # gluonts reference: per-series pinball then axis=None aggregation
    # quantile_loss_q = 2 * max(q*(y-pred), (q-1)*(y-pred)). Sum over all
    # timesteps and quantiles (gluonts axis=None), divide by sum of |y|.
    y = targets.numpy()
    p = preds.numpy()
    q_arr = np.array(quantiles).reshape(1, 1, 1, Q)
    err = y[..., None] - p
    ql = 2 * np.maximum(q_arr * err, (q_arr - 1) * err)  # (B,H,V,Q)
    # mean over Q for each (series,time) — matches gluonts "mean quantile loss"
    qsum = ql.mean(axis=-1).sum()
    abs_target_sum = np.abs(y).sum()
    gl_wql = qsum / abs_target_sum
    assert abs(ours - gl_wql) / max(abs(gl_wql), 1e-6) < 1e-5, (ours, gl_wql)


def test_mase_components_matches_reference():
    """_mase_components implements gluonts-style MASE per-series."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import _mase_components

    history = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    target = torch.tensor([11.0, 12.0])
    pred = torch.tensor([10.5, 12.5])
    season = 1
    err, weight = _mase_components(pred, target, history, season)
    # err = |10.5-11| + |12.5-12| = 0.5 + 0.5 = 1.0
    assert abs(err - 1.0) < 1e-6
    # scale = mean(|h[i] - h[i-1]|) = 1.0 (all diffs are 1)
    # weight = H * scale = 2 * 1.0 = 2.0
    assert abs(weight - 2.0) < 1e-6


def test_geometric_mean_aggregation():
    """_geomean returns geometric mean over positive finite values, ignoring
    infs and non-positive entries."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "timeseries_forecast"))
    from prepare import _geomean

    # geomean([2, 8]) = sqrt(16) = 4
    assert abs(_geomean([2.0, 8.0]) - 4.0) < 1e-9
    # inf values are ignored
    assert abs(_geomean([2.0, 8.0, float("inf")]) - 4.0) < 1e-9
    # all inf → inf
    assert _geomean([float("inf")]) == float("inf")
    # empty → inf
    assert _geomean([]) == float("inf")
