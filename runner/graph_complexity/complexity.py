"""Complexity profiling for random graph walk sequences.

Computes intrinsic complexity measures that characterise how hard a dataset
is to predict. Called once by prepare.py during data generation; results
cached to complexity_profile.json for evaluate.py to include in metrics.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def knn_conditional_entropy(
    x: np.ndarray, y: np.ndarray, k: int = 5,
) -> float:
    """KSG estimator for H(Y|X) using k-nearest neighbours.

    Args:
        x: (N, d_x) conditioning variable
        y: (N, d_y) target variable
        k: number of neighbours

    Returns:
        Estimated conditional entropy in nats.
    """
    from scipy.spatial import cKDTree

    n = x.shape[0]
    if n < k + 1:
        return 0.0

    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)

    # k+1 because the point itself is included
    dists, _ = tree_xy.query(xy, k=k + 1)
    eps = dists[:, -1]  # distance to k-th neighbour

    # Count neighbours within eps in the x-marginal
    nx = np.array([
        tree_x.query_ball_point(x[i], r=eps[i] + 1e-10, return_length=True) - 1
        for i in range(n)
    ], dtype=float)
    nx = np.maximum(nx, 1.0)

    from scipy.special import digamma
    h_cond = -digamma(k) + digamma(n) - np.mean(np.log(nx))
    return max(float(h_cond), 0.0)


def knn_multi_horizon_entropy(
    sequences: np.ndarray, horizons: list[int] | None = None,
) -> dict[int, float]:
    """Compute H(delta_x_{t+h} | x_t) at multiple horizons.

    Args:
        sequences: (n_walks, walk_len) token or value sequences
        horizons: list of horizon steps (default: [1, 8, 32, 64])

    Returns:
        Dict mapping horizon h -> conditional entropy in nats.
    """
    if horizons is None:
        horizons = [1, 8, 32, 64]

    n_walks, walk_len = sequences.shape
    result = {}

    for h in horizons:
        if h >= walk_len - 1:
            result[h] = 0.0
            continue
        # Sample pairs (x_t, x_{t+h} - x_t)
        max_t = walk_len - h
        sample_size = min(5000, n_walks * max_t)
        rng = np.random.RandomState(42)
        walk_idx = rng.randint(0, n_walks, size=sample_size)
        t_idx = rng.randint(0, max_t, size=sample_size)

        x_t = sequences[walk_idx, t_idx].reshape(-1, 1).astype(float)
        delta = (sequences[walk_idx, t_idx + h] - sequences[walk_idx, t_idx])
        delta = delta.reshape(-1, 1).astype(float)

        result[h] = knn_conditional_entropy(x_t, delta, k=min(5, sample_size - 1))

    return result


def spectral_entropy(sequences: np.ndarray) -> float:
    """Spectral entropy of first differences (normalised).

    Higher values indicate more complex / less predictable dynamics.
    """
    diffs = np.diff(sequences.ravel().astype(float))
    if len(diffs) < 2:
        return 0.0

    # Compute power spectral density via FFT
    fft_vals = np.fft.rfft(diffs - diffs.mean())
    psd = np.abs(fft_vals) ** 2
    psd = psd / (psd.sum() + 1e-12)
    psd = psd[psd > 0]

    # Shannon entropy normalised by log(N)
    h = -np.sum(psd * np.log(psd))
    h_max = np.log(len(psd)) if len(psd) > 1 else 1.0
    return float(h / max(h_max, 1e-12))


def stationarity_score(
    sequences: np.ndarray, n_segments: int = 4,
) -> dict[str, float]:
    """Measure non-stationarity via segment-wise mean and volatility drift.

    Returns dict with 'mean_drift' and 'vol_drift' in [0, inf).
    Lower = more stationary.
    """
    flat = sequences.ravel().astype(float)
    seg_len = len(flat) // max(n_segments, 1)
    if seg_len < 2:
        return {"mean_drift": 0.0, "vol_drift": 0.0}

    means = []
    vols = []
    for i in range(n_segments):
        seg = flat[i * seg_len: (i + 1) * seg_len]
        means.append(seg.mean())
        vols.append(seg.std())

    mean_drift = float(np.std(means) / (np.mean(np.abs(means)) + 1e-8))
    vol_drift = float(np.std(vols) / (np.mean(vols) + 1e-8))
    return {"mean_drift": mean_drift, "vol_drift": vol_drift}


def cross_series_diversity(sequences: np.ndarray, max_pairs: int = 500) -> float:
    """Mean pairwise correlation distance between walks.

    1.0 = perfectly uncorrelated, 0.0 = identical series.
    """
    n_walks = sequences.shape[0]
    if n_walks < 2:
        return 0.0

    rng = np.random.RandomState(42)
    n_pairs = min(max_pairs, n_walks * (n_walks - 1) // 2)

    dists = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_walks, size=2, replace=False)
        a = sequences[i].astype(float)
        b = sequences[j].astype(float)
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))
        if denom < 1e-12:
            dists.append(1.0)
        else:
            corr = np.sum(a * b) / denom
            dists.append(1.0 - abs(float(corr)))

    return float(np.mean(dists))


def marginal_entropy(tokens: np.ndarray) -> float:
    """H(marginal) = -sum(p_i * log(p_i)) from token frequency counts.

    Args:
        tokens: flat array of integer token IDs

    Returns:
        Marginal entropy in nats.
    """
    flat = tokens.ravel()
    _, counts = np.unique(flat, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def composite_score(
    multi_horizon: dict[int, float],
    spec_entropy: float,
    station: dict[str, float],
    diversity: float,
) -> float:
    """Weighted sum of normalised complexity axes. Higher = harder."""
    # Normalise each axis to roughly [0, 1]
    h_mean = np.mean(list(multi_horizon.values())) if multi_horizon else 0.0
    # Cap at reasonable maxima for normalisation
    h_norm = min(h_mean / 3.0, 1.0)
    s_norm = spec_entropy  # already in [0, 1]
    drift_norm = min((station["mean_drift"] + station["vol_drift"]) / 2.0, 1.0)
    div_norm = diversity  # already in [0, 1]

    return float(0.4 * h_norm + 0.3 * s_norm + 0.15 * drift_norm + 0.15 * div_norm)


def normalise_windows(
    data: np.ndarray, window_size: int,
) -> np.ndarray:
    """Per-window z-score normalisation.

    Args:
        data: (n_walks, walk_len) float array
        window_size: size of normalisation windows

    Returns:
        z-scored array of same shape.
    """
    result = data.copy().astype(float)
    n_walks, walk_len = result.shape
    for i in range(0, walk_len, window_size):
        end = min(i + window_size, walk_len)
        window = result[:, i:end]
        mu = window.mean()
        std = window.std()
        if std < 1e-8:
            result[:, i:end] = 0.0
        else:
            result[:, i:end] = (window - mu) / std
    return result


def compute_complexity_profile(
    token_sequences: np.ndarray,
    raw_sequences: Optional[np.ndarray] = None,
) -> dict:
    """Compute full complexity profile for a dataset.

    Args:
        token_sequences: (n_walks, walk_len) integer token IDs
        raw_sequences: (n_walks, walk_len) float values (optional, for richer stats)

    Returns:
        Dict with all complexity measures.
    """
    seq = raw_sequences if raw_sequences is not None else token_sequences

    multi_h = knn_multi_horizon_entropy(seq)
    spec_ent = spectral_entropy(seq)
    station = stationarity_score(seq)
    diversity = cross_series_diversity(seq)
    h_marginal = marginal_entropy(token_sequences)

    return {
        "multi_horizon_entropy": multi_h,
        "spectral_entropy": spec_ent,
        "stationarity": station,
        "cross_series_diversity": diversity,
        "marginal_entropy": h_marginal,
        "composite_difficulty": composite_score(multi_h, spec_ent, station, diversity),
    }
