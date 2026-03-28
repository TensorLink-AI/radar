"""Frozen evaluation for graph complexity task.

Loads checkpoint, runs validation ALWAYS in teacher-forced mode regardless
of training prediction_mode. Computes normalised_ce (primary), per-horizon
CE breakdown, and domain-specific losses extracted from logits.
"""

import importlib.util
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from prepare import (
    CONTEXT_LEN, PREDICTION_LEN, VOCAB_SIZE, MODALITY,
    PREDICTION_MODE, MARGINAL_ENTROPY, BIN_CENTRES,
)

SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

HORIZON_POSITIONS = [1, 8, 16, 32, 64]


def _load_val_data():
    """Load validation data from prepare module."""
    from prepare import _prepare_data
    _, val = _prepare_data()
    return val["x"], val["y"]


def _extract_expected_value(logits: torch.Tensor, bin_centres: torch.Tensor) -> torch.Tensor:
    """Extract continuous point predictions from discrete logits.

    Args:
        logits: (B, H, V) raw logits
        bin_centres: (V,) centre values for each bin

    Returns:
        (B, H) expected value predictions.
    """
    probs = F.softmax(logits, dim=-1)
    return (probs * bin_centres.unsqueeze(0).unsqueeze(0)).sum(dim=-1)


def _extract_quantiles(
    logits: torch.Tensor, bin_centres: torch.Tensor,
    quantile_levels: list[float] | None = None,
) -> torch.Tensor:
    """Extract quantile predictions via CDF inversion.

    Returns:
        (B, H, Q) quantile predictions.
    """
    if quantile_levels is None:
        quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    probs = F.softmax(logits, dim=-1)
    cdf = probs.cumsum(dim=-1)  # (B, H, V)

    quantiles = []
    for q in quantile_levels:
        idx = (cdf >= q).float().argmax(dim=-1)  # (B, H)
        qpred = bin_centres[idx.long()]
        quantiles.append(qpred)
    return torch.stack(quantiles, dim=-1)  # (B, H, Q)


def _compute_crps(
    quantile_preds: torch.Tensor, targets: torch.Tensor,
    quantile_levels: list[float],
) -> float:
    """CRPS from quantile predictions via pinball loss."""
    q = torch.tensor(quantile_levels, device=targets.device)
    errors = targets.unsqueeze(-1) - quantile_preds  # (B, H, Q)
    pinball = torch.max(q * errors, (q - 1) * errors)
    return pinball.mean().item()


def _compute_mase(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """MASE: mean absolute error / naive seasonal MAE."""
    mae = (predictions - targets).abs().mean()
    naive = (targets[:, 1:] - targets[:, :-1]).abs().mean().clamp(min=1e-6)
    return (mae / naive).item()


def main():
    # Load submission and model
    spec = importlib.util.spec_from_file_location("sub", "/workspace/submission.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)

    model = sub.build_model(
        CONTEXT_LEN, PREDICTION_LEN, VOCAB_SIZE, PREDICTION_MODE,
    ).cuda()

    ckpt = "/workspace/checkpoints/model.safetensors"
    if not os.path.exists(ckpt):
        print("WARNING: no checkpoint found", file=sys.stderr)
        return

    model.load_state_dict(load_file(ckpt, device="cuda"))
    model.eval()

    # Load validation data
    val_x, val_y = _load_val_data()
    n = len(val_x)
    batch_size = 32

    # Bin centres for domain loss extraction
    bin_centres_t = torch.tensor(BIN_CENTRES, dtype=torch.float32, device="cuda")
    h_marginal = MARGINAL_ENTROPY if MARGINAL_ENTROPY > 0 else 1.0

    # Accumulate metrics
    total_ce = 0.0
    total_tokens = 0
    horizon_ce = {h: 0.0 for h in HORIZON_POSITIONS}
    horizon_counts = {h: 0 for h in HORIZON_POSITIONS}
    all_expected = []
    all_targets_cont = []
    all_quantile_preds = []
    quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]

    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            x = torch.from_numpy(val_x[i:end]).long().cuda()
            y = torch.from_numpy(val_y[i:end]).long().cuda()

            # ALWAYS teacher-forced regardless of PREDICTION_MODE
            inp = torch.cat([x, y[:, :-1]], dim=1)
            logits = model(inp)
            logits_pred = logits[:, -PREDICTION_LEN:]  # (B, pred_len, vocab)

            # Raw CE per position
            B, H, V = logits_pred.shape
            ce_per_pos = F.cross_entropy(
                logits_pred.reshape(-1, V), y.reshape(-1), reduction="none",
            ).reshape(B, H)

            total_ce += ce_per_pos.sum().item()
            total_tokens += y.numel()

            # Per-horizon CE
            for h in HORIZON_POSITIONS:
                pos = h - 1  # 0-indexed
                if pos < H:
                    horizon_ce[h] += ce_per_pos[:, pos].sum().item()
                    horizon_counts[h] += B

            # Domain losses from logits
            expected = _extract_expected_value(logits_pred, bin_centres_t)
            all_expected.append(expected)
            all_targets_cont.append(bin_centres_t[y])  # map tokens back to continuous

            qpreds = _extract_quantiles(logits_pred, bin_centres_t, quantile_levels)
            all_quantile_preds.append(qpreds)

    # Average metrics
    raw_ce = total_ce / max(total_tokens, 1)
    normalised_ce = raw_ce / h_marginal
    universal_ce = raw_ce / max(math.log(VOCAB_SIZE), 1e-8)

    # Print primary metric
    print(f"normalised_ce: {normalised_ce:.6f}")
    print(f"universal_ce: {universal_ce:.6f}")

    # Per-horizon CE
    for h in HORIZON_POSITIONS:
        if horizon_counts[h] > 0:
            ce_h = horizon_ce[h] / horizon_counts[h]
            print(f"ce_h{h}: {ce_h:.6f}")

    # Domain-specific losses
    all_exp = torch.cat(all_expected, dim=0)
    all_tgt = torch.cat(all_targets_cont, dim=0)
    all_qp = torch.cat(all_quantile_preds, dim=0)

    if MODALITY == "tokens":
        perplexity = math.exp(min(raw_ce, 20.0))  # cap to avoid overflow
        print(f"perplexity: {perplexity:.6f}")

    if MODALITY in ("continuous", "rms_energy"):
        crps = _compute_crps(all_qp, all_tgt, quantile_levels)
        mase = _compute_mase(all_exp, all_tgt)
        print(f"crps: {crps:.6f}")
        print(f"mase: {mase:.6f}")

    if MODALITY == "waveform":
        mse = F.mse_loss(all_exp, all_tgt).item()
        signal_power = (all_tgt ** 2).mean().item()
        error_power = ((all_exp - all_tgt) ** 2).mean().item()
        snr = 10.0 * math.log10(max(signal_power, 1e-12) / max(error_power, 1e-12))
        print(f"mse: {mse:.6f}")
        print(f"snr: {snr:.6f}")

    print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1e6:.1f}")

    # Load and print complexity profile if available
    profile_path = os.environ.get(
        "COMPLEXITY_PROFILE_PATH", "/workspace/complexity_profile.json",
    )
    if os.path.exists(profile_path):
        try:
            with open(profile_path) as f:
                profile = json.load(f)
            print(f"composite_difficulty: {profile.get('composite_difficulty', 0.0):.6f}")
        except Exception:
            pass

    # Print training seconds placeholder (actual comes from harness)
    print(f"training_seconds: 0.0")


if __name__ == "__main__":
    main()
