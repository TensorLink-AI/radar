"""Structural-diversity signal for breaking the pipeline monoculture.

A 416-run analysis found 78% FFT / 72% spectral / 61% changepoint pipelines:
miners converge on re-tuning one spectral recipe instead of exploring
structurally different generators (GP/kernel, regime-switching/HMM, ARIMA,
structural trend+seasonality). This module turns a submission's source into a
coarse *technique signature* and scores how distinct it is from the current
frontier, so the validator can hand out an optional, additive novelty bonus.

Pure-python, numpy-free, importable everywhere. The signature is a keyword
scan (lower-cased substring match) — deliberately cheap and robust to
formatting; it is a diversity nudge, not a classifier.
"""

from __future__ import annotations

import os

# technique → marker substrings (lower-cased). A pipeline "uses" a technique
# when any of its markers appears in the source.
TECHNIQUE_MARKERS: dict[str, tuple[str, ...]] = {
    "spectral": ("fft", "rfft", "irfft", "fourier", "spectral", "periodogram",
                 "welch", "psd"),
    "wavelet": ("wavelet", "pywt", " dwt", " cwt", "haar"),
    "changepoint": ("changepoint", "change_point", "cusum", "bocpd",
                    "ruptures", "breakpoint"),
    "gp_kernel": ("gaussian_process", "gaussianprocess", "kernel", "rbf",
                  "matern", "covariance", "cholesky"),
    "regime_hmm": ("hmm", "markov", "regime", "viterbi", "transition_matrix",
                   "transition matrix"),
    "arima": ("arima", "sarimax", "autoregress", "autoregressive", "ar_order",
              "ma_order"),
    "trend_seasonal": ("trend", "seasonal", "seasonality", "stl", "holt",
                       "winters", "deseason"),
    "stochastic": ("brownian", "ornstein", "uhlenbeck", "geometric_brownian",
                   "gbm", "random_walk", "random walk", "levy", "poisson"),
    "latent_generative": ("gan", "vae", "diffusion", "discriminator",
                          "latent", "decoder", "encoder"),
}

NOVELTY_BONUS_ENV = "RADAR_NOVELTY_BONUS_WEIGHT"


def novelty_bonus_weight() -> float:
    """Additive novelty-bonus weight from env (default 0 = off).

    A fully novel fresh design earns up to ``(1 + weight)×`` its base score.
    """
    try:
        w = float(os.environ.get(NOVELTY_BONUS_ENV, "0") or "0")
    except ValueError:
        return 0.0
    return max(0.0, min(2.0, w))


def technique_signature(code: str) -> set[str]:
    """Set of technique tokens present in ``code`` (lower-cased scan)."""
    if not code:
        return set()
    low = code.lower()
    return {name for name, markers in TECHNIQUE_MARKERS.items()
            if any(m in low for m in markers)}


def frontier_signature(frontier_codes: list[str]) -> set[str]:
    """Union of technique tokens across the current frontier's pipelines."""
    out: set[str] = set()
    for c in frontier_codes or []:
        out |= technique_signature(c)
    return out


def novelty_score(code: str, frontier_codes: list[str]) -> float:
    """Fraction of this pipeline's techniques absent from the frontier.

    1.0 = every technique it uses is new to the frontier; 0.0 = all of its
    techniques are already represented (or it uses none we recognise). The
    frontier being empty makes any recognised technique fully novel.
    """
    sig = technique_signature(code)
    if not sig:
        return 0.0
    front = frontier_signature(frontier_codes)
    novel = sig - front
    return len(novel) / len(sig)


def novelty_multiplier(code: str, frontier_codes: list[str],
                       weight: float) -> tuple[float, float]:
    """Return ``(multiplier, novelty)``.

    ``multiplier = 1 + weight * novelty`` (≥ 1). ``weight <= 0`` is the
    no-op identity path so the bonus is fully opt-in.
    """
    if weight <= 0.0:
        return 1.0, 0.0
    nov = novelty_score(code, frontier_codes)
    return 1.0 + weight * nov, nov


# Non-spectral exemplars surfaced to miners to break the spectral monoculture.
# Descriptions only (no full code) — they seed the design prompt with
# structurally distinct families the frontier under-explores.
PIPELINE_EXEMPLARS: list[dict] = [
    {
        "family": "gp_kernel",
        "name": "Gaussian-process / kernel synthesis",
        "idea": "Sample series from a GP prior — sum of RBF (smooth), Matérn "
                "(rough), and periodic kernels — so correlation structure, not "
                "a frequency comb, drives realism.",
    },
    {
        "family": "regime_hmm",
        "name": "Regime-switching / HMM",
        "idea": "A hidden Markov chain switches between regimes (trend / "
                "mean-revert / volatile); each regime has its own dynamics. "
                "Captures structural breaks the spectral recipe smears over.",
    },
    {
        "family": "arima",
        "name": "ARIMA / state-space",
        "idea": "Drive an AR(p)+MA(q) (optionally seasonal/integrated) process "
                "with parameter ranges sampled per series. Classic, cheap, and "
                "structurally orthogonal to FFT mixing.",
    },
    {
        "family": "trend_seasonal",
        "name": "Structural trend + seasonality + noise",
        "idea": "Compose explicit components — piecewise/logistic trend, "
                "multiple seasonal cycles, holiday spikes, heteroscedastic "
                "noise — STL-style, so each component is independently tunable.",
    },
    {
        "family": "stochastic",
        "name": "Stochastic differential / jump processes",
        "idea": "Ornstein–Uhlenbeck mean reversion, geometric Brownian drift, "
                "and Poisson jumps. Produces realistic volatility clustering "
                "without any spectral construction.",
    },
]


def pipeline_exemplars() -> list[dict]:
    """The exemplar list (copy) for seeding the challenge."""
    return [dict(e) for e in PIPELINE_EXEMPLARS]
