"""Tests for normalised_ce metric properties."""

import math
import numpy as np
import torch
import torch.nn.functional as F


def test_uniform_predictor_normalised_ce_near_one():
    """Uniform logits -> normalised_ce ~ log(V) / H(marginal).

    When data is roughly uniform, H(marginal) ~ log(V), so normalised_ce ~ 1.0.
    """
    vocab_size = 256
    B, H = 32, 64

    # Uniform logits
    logits = torch.zeros(B, H, vocab_size)
    # Uniform target distribution
    targets = torch.randint(0, vocab_size, (B, H))

    ce = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1)).item()
    # H(marginal) for uniform is log(V)
    h_marginal = math.log(vocab_size)
    normalised = ce / h_marginal

    # Should be close to 1.0 (uniform predicts uniform data)
    assert 0.9 < normalised < 1.1


def test_perfect_predictor_normalised_ce_near_zero():
    """Perfect prediction -> normalised_ce ~ 0."""
    vocab_size = 64
    B, H = 16, 32

    # Target tokens
    targets = torch.randint(0, vocab_size, (B, H))

    # Perfect logits: very high value at correct position
    logits = torch.full((B, H, vocab_size), -100.0)
    for b in range(B):
        for h in range(H):
            logits[b, h, targets[b, h]] = 100.0

    ce = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1)).item()
    h_marginal = math.log(vocab_size)
    normalised = ce / h_marginal

    assert normalised < 0.01


def test_frequency_counting_normalised_ce_near_one():
    """A predictor that outputs marginal frequencies -> normalised_ce ~ 1.0."""
    vocab_size = 32
    B, H = 64, 32

    # Generate tokens with non-uniform distribution
    rng = np.random.RandomState(42)
    probs = rng.dirichlet(np.ones(vocab_size) * 0.5)
    targets = torch.from_numpy(
        rng.choice(vocab_size, size=(B, H), p=probs)
    ).long()

    # Frequency-counting logits: log(p_i) at each position
    log_probs = torch.log(torch.tensor(probs, dtype=torch.float32))
    logits = log_probs.unsqueeze(0).unsqueeze(0).expand(B, H, -1)

    ce = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1)).item()

    # H(marginal) from true distribution
    h_marginal = float(-np.sum(probs * np.log(probs)))
    normalised = ce / h_marginal

    # Should be close to 1.0 — frequency counting captures no transition structure
    assert 0.85 < normalised < 1.15
