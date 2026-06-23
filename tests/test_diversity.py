"""Tests for local/diversity.py — technique signatures + novelty bonus.
Pure-python, no torch."""

from __future__ import annotations

import pytest

from local.diversity import (
    novelty_bonus_weight,
    novelty_multiplier,
    novelty_score,
    pipeline_exemplars,
    technique_signature,
)


def test_technique_signature_detects_families():
    assert "spectral" in technique_signature("x = np.fft.rfft(series)")
    assert "gp_kernel" in technique_signature("build an RBF kernel covariance")
    assert "regime_hmm" in technique_signature("a hidden markov regime switch")
    assert technique_signature("") == set()
    assert technique_signature("plain linear ramp") == set()


def test_novelty_score_against_frontier():
    frontier = ["np.fft.fft(x)", "spectral periodogram with changepoint"]
    # A pure-GP pipeline is fully novel vs a spectral frontier.
    assert novelty_score("gaussian_process rbf kernel", frontier) == 1.0
    # A spectral pipeline is not novel at all.
    assert novelty_score("np.fft.rfft(x)", frontier) == 0.0
    # Half its techniques are new.
    half = novelty_score("np.fft.fft(x); matern kernel", frontier)
    assert half == pytest.approx(0.5)


def test_novelty_score_empty_frontier_is_novel():
    assert novelty_score("arima ar_order", []) == 1.0
    # No recognised technique → no novelty signal.
    assert novelty_score("just a constant", []) == 0.0


def test_novelty_multiplier_opt_in():
    # weight 0 → identity no-op.
    mult, nov = novelty_multiplier("gp kernel", ["np.fft.fft(x)"], 0.0)
    assert mult == 1.0 and nov == 0.0
    # weight>0 → 1 + weight*novelty.
    mult, nov = novelty_multiplier("gp kernel", ["np.fft.fft(x)"], 0.5)
    assert nov == 1.0 and mult == pytest.approx(1.5)


def test_novelty_bonus_weight_env(monkeypatch):
    monkeypatch.delenv("RADAR_NOVELTY_BONUS_WEIGHT", raising=False)
    assert novelty_bonus_weight() == 0.0
    monkeypatch.setenv("RADAR_NOVELTY_BONUS_WEIGHT", "0.5")
    assert novelty_bonus_weight() == 0.5
    monkeypatch.setenv("RADAR_NOVELTY_BONUS_WEIGHT", "9")
    assert novelty_bonus_weight() == 2.0  # clamped
    monkeypatch.setenv("RADAR_NOVELTY_BONUS_WEIGHT", "junk")
    assert novelty_bonus_weight() == 0.0


def test_pipeline_exemplars_are_non_spectral():
    fams = {e["family"] for e in pipeline_exemplars()}
    # The seeded families deliberately exclude the spectral monoculture.
    assert "spectral" not in fams
    assert {"gp_kernel", "regime_hmm", "arima", "trend_seasonal"} <= fams
