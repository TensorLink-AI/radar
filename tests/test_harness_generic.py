"""Tests for runner/harness.py — the generic training harness."""

from runner.harness import (
    TrainingConfig, _read_config, _read_amp_config,
    _has_callable, _check_size_gate, _load_submission,
)


class TestTrainingConfig:
    def test_from_dict(self):
        cfg = TrainingConfig.from_dict({
            "seed": 123, "round_id": 5, "min_flops": 1000,
            "max_flops": 5000, "miner_hotkey": "hk", "time_budget": 60,
        })
        assert cfg.seed == 123
        assert cfg.round_id == 5
        assert cfg.min_flops == 1000
        assert cfg.time_budget == 60

    def test_from_dict_defaults(self):
        cfg = TrainingConfig.from_dict({})
        assert cfg.seed == 42
        assert cfg.miner_hotkey == "unknown"

    def test_from_dict_ignores_extra_keys(self):
        cfg = TrainingConfig.from_dict({"seed": 1, "extra_key": "ignored"})
        assert cfg.seed == 1


class TestReadConfig:
    def test_defaults(self):
        class _FakeSub:
            pass
        cfg = _read_config(_FakeSub())
        assert cfg["batch_size"] == 64
        assert cfg["grad_accum_steps"] == 1

    def test_clamps_values(self):
        class _FakeSub:
            def training_config(self):
                return {"batch_size": 9999, "grad_clip": -5.0}
        cfg = _read_config(_FakeSub())
        assert cfg["batch_size"] == 512  # clamped to max
        assert cfg["grad_clip"] == 0.0   # clamped to min


class TestReadAmpConfig:
    def test_defaults(self):
        class _FakeSub:
            pass
        amp = _read_amp_config(_FakeSub())
        assert amp["enabled"] is True
        assert amp["dtype"] == "bfloat16"

    def test_invalid_dtype_fallback(self):
        class _FakeSub:
            def configure_amp(self):
                return {"dtype": "invalid_type"}
        amp = _read_amp_config(_FakeSub())
        assert amp["dtype"] == "bfloat16"


class TestSizeGate:
    def test_passes_within_range(self):
        cfg = TrainingConfig(min_flops=100_000, max_flops=500_000)
        assert _check_size_gate(cfg, 300_000) is None

    def test_fails_outside_range(self):
        cfg = TrainingConfig(min_flops=100_000, max_flops=500_000)
        result = _check_size_gate(cfg, 1_000_000)
        assert result is not None
        assert result["status"] == "size_violation"

    def test_skips_when_no_range(self):
        cfg = TrainingConfig(min_flops=0, max_flops=0)
        assert _check_size_gate(cfg, 999_999) is None


class TestHasCallable:
    def test_true(self):
        class _Obj:
            def foo(self): pass
        assert _has_callable(_Obj(), "foo") is True

    def test_false_missing(self):
        class _Obj:
            pass
        assert _has_callable(_Obj(), "foo") is False

    def test_false_not_callable(self):
        class _Obj:
            foo = 42
        assert _has_callable(_Obj(), "foo") is False


class TestTSForecastingRunner:
    """Test that the ts_forecasting runner implements TaskRunner correctly."""

    def test_runner_has_required_methods(self):
        from runner.timeseries_forecast.train import TSForecastingRunner
        runner = TSForecastingRunner()
        assert hasattr(runner, "build_model")
        assert hasattr(runner, "get_dataloader")
        assert hasattr(runner, "default_loss")
        assert hasattr(runner, "measure_flops")

    def test_run_training_callable(self):
        from runner.timeseries_forecast.train import run_training
        assert callable(run_training)
