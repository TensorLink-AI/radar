"""Tests for the optimizer plugin registry."""

from __future__ import annotations

import pytest

from miner_template import optimizers
from miner_template.optimizers import ResultRow


def test_list_builtins_has_gepa_and_random():
    names = optimizers.list_builtins()
    assert "gepa" in names
    assert "random_mutate" in names


def test_resolve_builtin_returns_callable():
    fn = optimizers.resolve("random_mutate")
    assert callable(fn)


def test_resolve_unknown_alias_raises():
    with pytest.raises(ValueError):
        optimizers.resolve("nonexistent_optimizer")


def test_resolve_dotted_path():
    fn = optimizers.resolve("miner_template.optimizers.random_mutate:optimize")
    assert callable(fn)


def test_resolve_missing_module_raises_importerror():
    with pytest.raises(ImportError):
        optimizers.resolve("not_a_real_module.somewhere:func")


def test_resolve_missing_function_raises_attributeerror():
    with pytest.raises(AttributeError):
        optimizers.resolve("miner_template.optimizers.random_mutate:not_there")


def test_resultrow_from_dict_safe_with_missing_fields():
    r = ResultRow.from_dict({})
    assert r.round_id == 0
    assert r.prompt_id is None
    assert r.scores == {}


def test_resultrow_from_dict_full():
    r = ResultRow.from_dict({
        "round_id": 12,
        "submission_id": "sub-1",
        "task_name": "ts_forecasting",
        "prompt_id": "p-abc",
        "architecture_code": "class M: pass",
        "scores": {"raw_score": 0.7, "crps": 0.3},
        "created_at": "2026-05-19T00:00:00Z",
    })
    assert r.round_id == 12
    assert r.prompt_id == "p-abc"
    assert r.scores["raw_score"] == 0.7
