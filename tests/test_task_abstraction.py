"""Tests for the task-abstraction refactor.

Covers the three consumer-side generalizations:
  - shared.eval_result.build_error_result — error dicts keyed on TaskSpec.objectives
  - validator.evaluator._get_eval_template — importlib-based loader
  - runner.<task>.dispatch.build_dispatch_extras — per-task dispatch hook
  - validator.neuron._build_dispatch_extras — importlib lookup wrapper
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from shared.eval_result import build_error_result
from shared.task import Objective, TaskSpec


# ── build_error_result ────────────────────────────────────────────────


def _task_with_objectives(primary_name: str, *others: str) -> TaskSpec:
    objectives = [
        Objective(name=primary_name, pattern=r"x:\s*([\d.]+)", primary=True, default=0.0),
    ]
    for name in others:
        objectives.append(
            Objective(name=name, pattern=r"x:\s*([\d.]+)", default=0.0)
        )
    return TaskSpec(name="fake", objectives=objectives)


def test_build_error_result_uses_objective_names():
    task = _task_with_objectives("perplexity", "loss")
    result = build_error_result(task, "something broke")
    assert result["error"] == "something broke"
    assert result["flops_equivalent_size"] == 0
    assert result["perplexity"] == 0.0
    assert result["loss"] == 0.0
    # Must NOT contain forecasting-specific keys
    assert "crps" not in result
    assert "mase" not in result


def test_build_error_result_uses_objective_defaults():
    objectives = [
        Objective(name="accuracy", pattern=r"x", primary=True, default=0.0),
        Objective(name="latency_ms", pattern=r"x", default=9999.0),
    ]
    task = TaskSpec(name="fake", objectives=objectives)
    result = build_error_result(task, "oops")
    assert result["accuracy"] == 0.0
    assert result["latency_ms"] == 9999.0


def test_build_error_result_none_task_falls_back_to_ts_forecasting():
    """When called without a task (direct unit-test path), keep the legacy shape."""
    result = build_error_result(None, "legacy path")
    assert math.isinf(result["crps"])
    assert math.isinf(result["mase"])
    assert result["flops_equivalent_size"] == 0
    assert result["error"] == "legacy path"


# ── TaskSpec size_buckets round-trip ──────────────────────────────────


def test_task_size_buckets_round_trip_via_dict():
    """size_buckets survive to_dict/from_dict as lists of (min, max) tuples."""
    spec = TaskSpec(
        name="custom",
        size_buckets=[(1_000_000, 10_000_000), (10_000_000, 100_000_000)],
    )
    d = spec.to_dict()
    # YAML-friendly representation: list of lists.
    assert d["size_buckets"] == [[1_000_000, 10_000_000], [10_000_000, 100_000_000]]

    spec2 = TaskSpec.from_dict(d)
    assert spec2.size_buckets == [(1_000_000, 10_000_000), (10_000_000, 100_000_000)]


def test_task_size_buckets_accepts_dict_form():
    """`{min, max}` dict entries also parse — friendlier to some YAMLs."""
    spec = TaskSpec.from_dict({
        "name": "custom",
        "objectives": [],
        "size_buckets": [{"min": 500, "max": 5000}],
    })
    assert spec.size_buckets == [(500, 5000)]


def test_task_size_buckets_default_empty():
    """Default is an empty list — generate_challenge falls back to globals."""
    spec = TaskSpec(name="custom")
    assert spec.size_buckets == []


def test_ts_forecasting_yaml_declares_size_buckets():
    """The ts_forecasting YAML now carries explicit size buckets."""
    from pathlib import Path
    from shared.task import load_task
    yaml_path = Path(__file__).parent.parent / "tasks" / "ts_forecasting" / "ts_forecasting.yaml"
    task = load_task(str(yaml_path))
    assert len(task.size_buckets) == 5
    # Every bucket must be a (lo, hi) pair with lo < hi.
    for lo, hi in task.size_buckets:
        assert 0 < lo < hi


def test_nanogpt_yaml_declares_broader_buckets():
    """The nanogpt YAML declares bigger ranges than the global defaults."""
    from pathlib import Path
    from shared.challenge import SIZE_BUCKETS
    from shared.task import load_task
    yaml_path = Path(__file__).parent.parent / "tasks" / "nanogpt" / "nanogpt.yaml"
    task = load_task(str(yaml_path))
    assert task.size_buckets, "nanogpt should declare custom size_buckets"
    max_global = max(hi for _, hi in SIZE_BUCKETS)
    max_nanogpt = max(hi for _, hi in task.size_buckets)
    assert max_nanogpt > max_global


# ── evaluator eval-template loader ────────────────────────────────────


def test_eval_template_importlib_loads_ts_forecasting():
    from validator.evaluator import _get_eval_template
    template = _get_eval_template("runner/timeseries_forecast")
    # The template is a str with `{arch_path}`, `{checkpoint_path}` etc. substitutions
    assert isinstance(template, str)
    assert "{arch_path}" in template
    assert "{checkpoint_path}" in template
    assert "{eval_split_seed}" in template


def test_eval_template_unknown_runner_dir_falls_back():
    from validator.evaluator import _get_eval_template
    # Non-existent task module → fallback to ts_forecasting (logged as warning)
    template = _get_eval_template("runner/does_not_exist")
    assert isinstance(template, str)
    assert "{arch_path}" in template


def test_eval_template_empty_runner_dir_uses_default():
    from validator.evaluator import _get_eval_template
    template = _get_eval_template("")
    assert isinstance(template, str)
    assert "{arch_path}" in template


# ── ts_forecasting dispatch extras ────────────────────────────────────


def _fake_task() -> TaskSpec:
    return TaskSpec(name="ts_forecasting", runner_dir="runner/timeseries_forecast")


def test_dispatch_extras_empty_when_no_r2_clients():
    from runner.timeseries_forecast.dispatch import build_dispatch_extras
    extras = build_dispatch_extras(
        _fake_task(),
        gift_r2=None, pretrain_r2=None,
        seed=42, shards_per_round=2, r2_prefixes={},
    )
    assert extras == {}


def test_dispatch_extras_ts_forecasting_produces_urls(monkeypatch):
    """Mocked R2 clients → extras contains the three expected URL keys."""
    import shared.gift_eval as gift_eval_mod
    import shared.pretrain_data as pretrain_mod
    from runner.timeseries_forecast.dispatch import build_dispatch_extras

    fake_gift = MagicMock()
    fake_gift.generate_presigned_get_urls.return_value = {"dataset_a": "https://r2/a"}
    monkeypatch.setattr(gift_eval_mod, "GiftEvalBenchmark", lambda **kw: fake_gift)

    fake_pretrain = MagicMock()
    fake_pretrain.select_shards.return_value = ["shards/000.parquet"]
    fake_pretrain.generate_presigned_shard_urls.side_effect = [
        ["https://r2/train/000"],  # train URLs
        ["https://r2/val/000"],    # val URLs
    ]
    fake_pretrain.get_val_shard_keys.return_value = ["shards/val.parquet"]
    monkeypatch.setattr(pretrain_mod, "PretrainBenchmark", lambda **kw: fake_pretrain)

    extras = build_dispatch_extras(
        _fake_task(),
        gift_r2=MagicMock(), pretrain_r2=MagicMock(),
        seed=42, shards_per_round=1,
        r2_prefixes={"gift": "gift/", "pretrain": "pretrain/"},
    )
    assert extras["gift_eval_urls"] == {"dataset_a": "https://r2/a"}
    assert extras["pretrain_shard_urls"] == ["https://r2/train/000"]
    assert extras["pretrain_val_shard_urls"] == ["https://r2/val/000"]


def test_dispatch_extras_handles_gift_failure_gracefully(monkeypatch):
    """GIFT-Eval errors shouldn't explode the whole extras call."""
    import shared.gift_eval as gift_eval_mod
    from runner.timeseries_forecast.dispatch import build_dispatch_extras

    def _boom(**kw):
        raise RuntimeError("r2 bucket missing")

    monkeypatch.setattr(gift_eval_mod, "GiftEvalBenchmark", _boom)
    extras = build_dispatch_extras(
        _fake_task(),
        gift_r2=MagicMock(), pretrain_r2=None,
        seed=42, shards_per_round=1, r2_prefixes={},
    )
    # GIFT failed → key omitted, but call still returned cleanly
    assert "gift_eval_urls" not in extras


# ── Validator._build_dispatch_extras ──────────────────────────────────


def test_validator_build_dispatch_extras_missing_module_returns_empty():
    """runner_dir without a `dispatch` module yields {}."""
    from validator.neuron import Validator

    class _Challenge:
        task = {"runner_dir": "runner/no_such_runner"}
        seed = 7

    # Stub out __init__ so we don't need a real wallet/subtensor
    validator = Validator.__new__(Validator)
    validator.gift_r2 = None
    validator.pretrain_r2 = None
    extras = validator._build_dispatch_extras(_Challenge(), _fake_task())
    assert extras == {}


def test_validator_build_dispatch_extras_empty_runner_dir_returns_empty():
    from validator.neuron import Validator

    class _Challenge:
        task = {"runner_dir": ""}
        seed = 7

    validator = Validator.__new__(Validator)
    validator.gift_r2 = None
    validator.pretrain_r2 = None
    extras = validator._build_dispatch_extras(_Challenge(), _fake_task())
    assert extras == {}
