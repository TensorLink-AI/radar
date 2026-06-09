"""Tests for local/noise.py — replicate pairing + noise-floor estimate."""

from __future__ import annotations

import math

import pytest

from local.noise import noise_floor, replicate_pairs, within_noise


def _exp(eid, metric, *, mode="new", replicate_of=None, success=True,
         task="ts_forecasting"):
    objs = {}
    if replicate_of is not None:
        objs["replicate_of"] = replicate_of
    return {"id": eid, "metric": metric, "mode": mode, "success": success,
            "task": task, "objectives": objs}


def test_replicate_pairs_join_on_source():
    exps = [
        _exp(1, 0.50),
        _exp(2, 0.52, mode="replicate", replicate_of=1),
        _exp(3, 0.70),
        _exp(4, 0.69, mode="replicate", replicate_of=3),
        # Failed replicate ignored.
        _exp(5, None, mode="replicate", replicate_of=1, success=False),
        # Replicate of a missing source ignored.
        _exp(6, 0.4, mode="replicate", replicate_of=999),
    ]
    pairs = replicate_pairs(exps)
    assert pairs == [(0.50, 0.52), (0.70, 0.69)]


def test_noise_floor_estimate():
    exps = [
        _exp(1, 0.50), _exp(2, 0.52, mode="replicate", replicate_of=1),
        _exp(3, 0.70), _exp(4, 0.66, mode="replicate", replicate_of=3),
    ]
    floor = noise_floor(exps)
    assert floor["n_pairs"] == 2
    # sigma = sqrt(mean(d^2)/2), d = [-0.02, 0.04]
    expected = math.sqrt(((0.02 ** 2) + (0.04 ** 2)) / 2 / 2)
    assert floor["sigma_metric"] == pytest.approx(expected)
    assert floor["threshold"] == pytest.approx(1.645 * expected)
    assert floor["sigma_rel"] > 0


def test_noise_floor_empty():
    floor = noise_floor([_exp(1, 0.5)])
    assert floor["n_pairs"] == 0
    assert floor["sigma_metric"] is None
    assert within_noise(0.01, floor) is None


def test_within_noise():
    floor = {"threshold": 0.05}
    assert within_noise(0.01, floor) is True
    assert within_noise(-0.01, floor) is True
    assert within_noise(0.10, floor) is False


def test_task_filter():
    exps = [
        _exp(1, 0.5, task="a"),
        _exp(2, 0.6, mode="replicate", replicate_of=1, task="a"),
        _exp(3, 0.5, task="b"),
        _exp(4, 0.6, mode="replicate", replicate_of=3, task="b"),
    ]
    assert len(replicate_pairs(exps, task="a")) == 1
    assert len(replicate_pairs(exps)) == 2
