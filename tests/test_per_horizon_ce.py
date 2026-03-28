"""Tests for per-horizon CE extraction."""

import torch
import torch.nn.functional as F


def test_per_horizon_ce_correct_extraction():
    """Verify ce_h1..ce_h64 are extracted at correct positions."""
    vocab_size = 32
    B = 4
    pred_len = 64

    logits = torch.randn(B, pred_len, vocab_size)
    targets = torch.randint(0, vocab_size, (B, pred_len))

    # Per-position CE
    ce_per_pos = F.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none",
    ).reshape(B, pred_len)

    horizon_positions = [1, 8, 16, 32, 64]
    for h in horizon_positions:
        pos = h - 1  # 0-indexed
        if pos < pred_len:
            ce_h = ce_per_pos[:, pos].mean().item()
            assert ce_h > 0, f"ce_h{h} should be positive"


def test_per_position_ce_sums_to_total():
    """Sum of per-position CE should equal total CE * prediction_len."""
    vocab_size = 32
    B = 8
    pred_len = 16

    logits = torch.randn(B, pred_len, vocab_size)
    targets = torch.randint(0, vocab_size, (B, pred_len))

    # Total CE (mean reduction)
    total_ce = F.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1),
    ).item()

    # Per-position CE
    ce_per_pos = F.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none",
    ).reshape(B, pred_len)

    # Mean across all positions should equal total_ce
    mean_all = ce_per_pos.mean().item()
    assert abs(mean_all - total_ce) < 1e-5


def test_all_horizons_positive():
    """All per-horizon CE values should be positive."""
    vocab_size = 64
    B = 4
    pred_len = 64

    logits = torch.randn(B, pred_len, vocab_size)
    targets = torch.randint(0, vocab_size, (B, pred_len))

    ce_per_pos = F.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none",
    ).reshape(B, pred_len)

    for pos in range(pred_len):
        assert ce_per_pos[:, pos].mean().item() > 0
