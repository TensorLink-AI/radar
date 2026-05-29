"""Tests for shared.gift_eval (selection, Arrow loading, windowing, task expansion).

These tests cover only `shared.gift_eval` directly. The trainer/eval-side tests
that lived alongside (depending on the removed `runner/timeseries_forecast/`)
are intentionally omitted; restore that runner if you need them back.
"""

import os
import tempfile

import pytest

from shared.gift_eval import select_datasets, GIFT_EVAL_DATASETS


# ── Deterministic dataset selection ──


def test_gift_eval_select_deterministic():
    """Same seed produces same dataset selection."""
    a = select_datasets(eval_split_seed=42, n=5)
    b = select_datasets(eval_split_seed=42, n=5)
    assert a == b
    assert len(a) == 5
    c = select_datasets(eval_split_seed=999, n=5)
    assert c != a


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
    pa = pytest.importorskip("pyarrow")
    import numpy as np

    ds_dir = os.path.join(tmpdir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)
    arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

    arrays = [pa.array(s) for s in series_data]
    offsets = np.cumsum([0] + [len(s) for s in series_data])
    target_col = pa.ListArray.from_arrays(offsets.tolist(), pa.concat_arrays(arrays))

    table = pa.table({"target": target_col})
    with pa.ipc.new_file(arrow_path, table.schema) as writer:
        writer.write_table(table)

    return arrow_path


def test_gift_eval_arrow_load():
    """Load mock Arrow file and verify shapes."""
    pytest.importorskip("pyarrow")
    from shared.gift_eval import load_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
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
    pytest.importorskip("pyarrow")
    from shared.gift_eval import load_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        series_data = [list(range(700))]
        _create_mock_arrow_file(tmpdir, "test_window", series_data)

        samples = load_dataset(
            "test_window", context_len=512, prediction_len=96,
            cache_dir=tmpdir,
        )
        assert len(samples) == 1
        assert samples[0]["target"][-1] == 699
        assert samples[0]["target"][0] == 604
        assert samples[0]["context"][-1] == 603
        assert samples[0]["context"][0] == 92


def test_gift_eval_short_series_skipped():
    """Series shorter than context_len + pred_len are skipped."""
    pytest.importorskip("pyarrow")
    from shared.gift_eval import load_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        series_data = [list(range(100)), list(range(700))]
        _create_mock_arrow_file(tmpdir, "test_skip", series_data)

        samples = load_dataset(
            "test_skip", context_len=512, prediction_len=96,
            cache_dir=tmpdir,
        )
        assert len(samples) == 1


def test_gift_eval_nan_series_skipped():
    """Series with NaN values are filtered out during loading."""
    pytest.importorskip("pyarrow")
    import math
    from shared.gift_eval import load_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        clean = [float(x) for x in range(700)]
        nan_in_target = list(clean)
        nan_in_target[650] = float("nan")
        nan_in_context = list(clean)
        nan_in_context[300] = float("nan")
        series_data = [clean, nan_in_target, nan_in_context, clean]
        _create_mock_arrow_file(tmpdir, "test_nan", series_data)

        samples = load_dataset(
            "test_nan", context_len=512, prediction_len=96,
            cache_dir=tmpdir,
        )
        assert len(samples) == 2
        for s in samples:
            assert not any(math.isnan(v) for v in s["context"])
            assert not any(math.isnan(v) for v in s["target"])


def test_gift_eval_freq_override_pred_len():
    """Frequency metadata overrides prediction_len in loaded samples."""
    pa = pytest.importorskip("pyarrow")
    from shared.gift_eval import load_dataset
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_dir = os.path.join(tmpdir, "test_freq")
        os.makedirs(ds_dir, exist_ok=True)
        arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

        series_data = [list(range(700))]
        arrays = [pa.array(s) for s in series_data]
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
    pytest.importorskip("pyarrow")
    pytest.importorskip("torch")
    from shared.gift_eval import load_dataset, get_eval_batches

    with tempfile.TemporaryDirectory() as tmpdir:
        series_data = [list(range(700)) for _ in range(5)]
        _create_mock_arrow_file(tmpdir, "test_batches", series_data)

        samples = load_dataset(
            "test_batches", context_len=512, prediction_len=96,
            cache_dir=tmpdir,
        )
        batches = list(get_eval_batches(samples, batch_size=3))
        assert len(batches) == 2
        assert batches[0]["context"].shape == (3, 512, 1)
        assert batches[0]["target"].shape == (3, 96, 1)
        assert batches[1]["context"].shape == (2, 512, 1)
        assert batches[1]["target"].shape == (2, 96, 1)


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
            assert get_seasonality(freq) == ours


def test_build_task_configs_produces_97_tasks():
    """SHORT_DATASETS expanded with MED_LONG_DATASETS yields >=95 tasks."""
    from shared.gift_eval import SHORT_DATASETS, build_task_configs

    tasks = build_task_configs(SHORT_DATASETS)
    assert len(tasks) >= 95
    terms = {t["term"] for t in tasks}
    assert terms == {"short", "medium", "long"}
    for t in tasks:
        assert t["prediction_length"] > 0
        assert t["season_length"] >= 1
        assert "/" in t["config_name"]


def test_parse_dataset_spec_accepts_manifest_keys():
    """_parse_dataset_spec handles both leaderboard and manifest-key spellings.

    Regression: `--all` iterates GIFT_EVAL_DATASETS (manifest keys using the
    `name__freq` form), which previously fell through to the single-freq
    lookup and raised KeyError for multi-freq datasets like LOOP_SEATTLE__5T.
    """
    from shared.gift_eval import _parse_dataset_spec

    assert _parse_dataset_spec("LOOP_SEATTLE/5T") == ("LOOP_SEATTLE", "5T")
    assert _parse_dataset_spec("LOOP_SEATTLE__5T") == ("LOOP_SEATTLE", "5T")
    assert _parse_dataset_spec("ett1__15T") == ("ett1", "15T")
    # Single-freq datasets still infer freq from _DATASET_PROPERTIES.
    assert _parse_dataset_spec("hospital") == ("hospital", "M")


def test_parse_dataset_spec_resolves_every_manifest_key():
    """Every key in GIFT_EVAL_DATASETS parses and maps back to the manifest.

    Guards the `--all` code path end to end (parse → manifest bridge).
    """
    from shared.gift_eval import (
        GIFT_EVAL_DATASETS,
        GIFT_EVAL_MANIFEST,
        _dataset_key_to_manifest,
        _parse_dataset_spec,
    )

    for name in GIFT_EVAL_DATASETS:
        dataset, freq = _parse_dataset_spec(name)
        assert _dataset_key_to_manifest(dataset, freq) in GIFT_EVAL_MANIFEST


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
    assert by_name["electricity/D/short"]["prediction_length"] == 30


# ── Rolling-origin windowing ──


def test_rolling_windows_match_gift_eval_formula():
    """Window count follows ceil(0.1 * min_series_len / H), capped [1, 20]."""
    pa = pytest.importorskip("pyarrow")
    from shared.gift_eval import load_dataset_for_task, _gift_eval_window_count

    assert _gift_eval_window_count(1000, 48, "ett1") == 3
    assert _gift_eval_window_count(5000, 10, "ett1") == 20
    assert _gift_eval_window_count(50, 48, "ett1") == 1
    assert _gift_eval_window_count(1000, 48, "m4_hourly") == 1

    with tempfile.TemporaryDirectory() as tmpdir:
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
            "dataset": "ett1", "freq": "H", "term": "short",
            "prediction_length": 48, "season_length": 24,
            "config_name": "ett1/H/short",
        }
        samples = load_dataset_for_task(task, context_len=96, cache_dir=tmpdir)
        assert len(samples) == 3
        assert samples[0]["target"] == series[-48:]
        assert samples[1]["target"] == series[-96:-48]
        assert samples[2]["target"] == series[-144:-96]
        assert all(len(s["context"]) == 96 for s in samples)


def test_rolling_windows_m4_fixed_at_one():
    """M4 datasets always produce 1 window regardless of series length."""
    pa = pytest.importorskip("pyarrow")
    from shared.gift_eval import load_dataset_for_task

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_dir = os.path.join(tmpdir, "m4_hourly")
        os.makedirs(ds_dir, exist_ok=True)
        arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

        series = [float(x) for x in range(5000)]
        arr = pa.array(series)
        target_col = pa.ListArray.from_arrays([0, len(series)], arr)
        table = pa.table({"target": target_col})
        with pa.ipc.new_file(arrow_path, table.schema) as writer:
            writer.write_table(table)

        task = {
            "dataset": "m4_hourly", "freq": "H", "term": "short",
            "prediction_length": 48, "season_length": 24,
            "config_name": "m4_hourly/H/short",
        }
        samples = load_dataset_for_task(task, context_len=96, cache_dir=tmpdir)
        assert len(samples) == 1


def test_rolling_windows_cap_at_20():
    """Window count clamped at 20 even when formula would exceed."""
    pa = pytest.importorskip("pyarrow")
    from shared.gift_eval import load_dataset_for_task

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_dir = os.path.join(tmpdir, "ett1__H")
        os.makedirs(ds_dir, exist_ok=True)
        arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

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


def test_rolling_windows_series_too_short_for_even_one_window():
    """Series with L < H are dropped entirely (can't fit one target)."""
    pa = pytest.importorskip("pyarrow")
    from shared.gift_eval import load_dataset_for_task

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_dir = os.path.join(tmpdir, "ett1__H")
        os.makedirs(ds_dir, exist_ok=True)
        arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

        series = [float(x) for x in range(20)]
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
    pa = pytest.importorskip("pyarrow")
    from shared.gift_eval import load_dataset_for_task

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_dir = os.path.join(tmpdir, "ett1__H")
        os.makedirs(ds_dir, exist_ok=True)
        arrow_path = os.path.join(ds_dir, "data-00000-of-00001.arrow")

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
        assert len(samples) == 600
