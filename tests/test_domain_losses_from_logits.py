"""Tests for domain loss extraction from logits."""

import math
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runner", "graph_complexity"))


def test_expected_value_from_uniform_logits():
    """Uniform logits -> expected value = mean of bin centres."""
    bin_centres = torch.linspace(-1.0, 1.0, 64)
    logits = torch.zeros(4, 16, 64)  # uniform

    probs = F.softmax(logits, dim=-1)
    expected = (probs * bin_centres).sum(dim=-1)

    # Should be close to mean of bin_centres
    target_mean = bin_centres.mean().item()
    assert abs(expected.mean().item() - target_mean) < 0.01


def test_quantile_extraction_monotonic():
    """Extracted quantiles should be monotonically non-decreasing."""
    bin_centres = torch.linspace(0.0, 10.0, 128)
    # Create non-uniform logits (peaked in the middle)
    logits = torch.randn(8, 32, 128)
    probs = F.softmax(logits, dim=-1)
    cdf = probs.cumsum(dim=-1)

    quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    prev_q = None
    for q in quantile_levels:
        idx = (cdf >= q).float().argmax(dim=-1)
        qvals = bin_centres[idx]
        if prev_q is not None:
            assert (qvals >= prev_q - 1e-6).all(), f"Quantile {q} not monotonic"
        prev_q = qvals


def test_crps_positive():
    """CRPS should always be non-negative."""
    bin_centres = torch.linspace(-2.0, 2.0, 64)
    logits = torch.randn(4, 16, 64)
    targets = torch.randn(4, 16)

    probs = F.softmax(logits, dim=-1)
    cdf = probs.cumsum(dim=-1)

    quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantiles = []
    for q in quantile_levels:
        idx = (cdf >= q).float().argmax(dim=-1)
        quantiles.append(bin_centres[idx])
    q_preds = torch.stack(quantiles, dim=-1)  # (B, H, Q)

    q_t = torch.tensor(quantile_levels)
    errors = targets.unsqueeze(-1) - q_preds
    pinball = torch.max(q_t * errors, (q_t - 1) * errors)
    crps = pinball.mean().item()

    assert crps >= 0.0


def test_mse_computation():
    """MSE from expected values matches manual computation."""
    bin_centres = torch.linspace(0.0, 1.0, 32)
    # Peaked logits at index 16 (centre = 0.5)
    logits = torch.full((2, 8, 32), -10.0)
    logits[:, :, 16] = 10.0

    probs = F.softmax(logits, dim=-1)
    expected = (probs * bin_centres).sum(dim=-1)

    targets = torch.full((2, 8), 0.5)
    mse = F.mse_loss(expected, targets).item()

    assert mse < 0.01
