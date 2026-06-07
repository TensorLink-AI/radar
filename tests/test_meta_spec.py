"""Tests for meta.spec — schema validation + render_run_argv."""

from __future__ import annotations

import json

import pytest

from meta.spec import (
    ExperimentSpec,
    load_spec,
    parse_spec,
    render_run_argv,
)


def _ok_spec_dict() -> dict:
    return {
        "name": "test-experiment-1",
        "base": {
            "task": "ts_forecasting",
            "rounds": 10,
            "miners": 1,
        },
        "units": [
            {"id": "a", "overrides": {"continuation_equilibrium": 0.5}},
            {"id": "b", "overrides": {"continuation_equilibrium": 0.7}},
        ],
        "pod": {"provider": "mock", "gpu": "A100-40GB", "max_hours": 2},
        "cost": {"max_usd_per_hour": 0.30, "budget_usd": 5.0},
    }


def test_parse_ok():
    spec = parse_spec(_ok_spec_dict())
    assert isinstance(spec, ExperimentSpec)
    assert spec.name == "test-experiment-1"
    assert len(spec.units) == 2
    assert spec.cost.max_usd_per_hour == 0.30
    assert spec.pod.provider == "mock"


def test_merged_overrides_apply():
    spec = parse_spec(_ok_spec_dict())
    cfg_a = spec.merged(spec.units[0])
    assert cfg_a["continuation_equilibrium"] == 0.5
    # base flags preserved
    assert cfg_a["task"] == "ts_forecasting"
    assert cfg_a["rounds"] == 10


def test_unit_id_validation():
    d = _ok_spec_dict()
    d["units"][0]["id"] = "BAD ID"
    with pytest.raises(ValueError, match="unit"):
        parse_spec(d)


def test_duplicate_unit_id():
    d = _ok_spec_dict()
    d["units"][1]["id"] = "a"
    with pytest.raises(ValueError, match="duplicate"):
        parse_spec(d)


def test_unknown_flag_rejected():
    d = _ok_spec_dict()
    d["base"]["totally_unknown_flag"] = 42
    with pytest.raises(ValueError, match="unknown flag"):
        parse_spec(d)


def test_unknown_override_rejected():
    d = _ok_spec_dict()
    d["units"][0]["overrides"]["nope"] = 1
    with pytest.raises(ValueError, match="unknown flag"):
        parse_spec(d)


def test_cost_must_be_positive():
    d = _ok_spec_dict()
    d["cost"]["max_usd_per_hour"] = 0
    with pytest.raises(ValueError):
        parse_spec(d)


def test_render_argv_basic():
    cfg = {"task": "ts_forecasting", "rounds": 10, "miners": 2}
    argv = render_run_argv(cfg, db_path="/tmp/x.db")
    assert argv[:4] == ["python", "local/run.py", "--db", "/tmp/x.db"]
    assert "--task" in argv
    assert "ts_forecasting" in argv
    assert "--rounds" in argv
    assert "10" in argv


def test_render_argv_bool_handled():
    cfg = {"task": "ts_forecasting"}
    argv = render_run_argv(cfg, db_path="x.db")
    # bool=False should be omitted
    assert "--continuation" not in argv


def test_load_spec_from_json(tmp_path):
    # PyYAML may not be installed; JSON path is the fallback.
    f = tmp_path / "spec.json"
    f.write_text(json.dumps(_ok_spec_dict()))
    spec = load_spec(f)
    assert spec.name == "test-experiment-1"
