"""
Frozen data preparation and evaluation for time-series forecasting.

Placeholder for Phase 1 — provides get_dataloader(), validate(), and constants.
In production, replace with GIFT-Eval data pipeline that streams real
time-series data across diverse domains and frequencies.
"""

import math
import random

import torch

# ── Task constants (frozen, match the task YAML) ──────────────────
CONTEXT_LEN = 512       # Input context window length
PREDICTION_LEN = 96     # Forecast horizon
NUM_VARIATES = 7        # Number of variates (multivariate time series)
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def get_dataloader(batch_size: int = 64):
    """
    Placeholder dataloader yielding random time-series batches.

    In production, this streams real time-series data from GIFT-Eval
    across diverse domains and frequencies.

    Yields: dict with "context" (B, context_len, num_variates)
            and "target" (B, prediction_len, num_variates)
    """
    while True:
        batch = {
            "context": torch.randn(batch_size, CONTEXT_LEN, NUM_VARIATES),
            "target": torch.randn(batch_size, PREDICTION_LEN, NUM_VARIATES),
        }
        yield batch


def _crps_from_quantiles(predictions, targets, quantiles_t):
    """Compute per-sample CRPS from quantile predictions via pinball loss.

    Args:
        predictions: (B, P, V, Q) quantile predictions
        targets: (B, P, V) ground truth
        quantiles_t: (Q,) quantile levels tensor on same device

    Returns:
        (B,) per-sample CRPS — mean pinball loss over (P, V, Q) dimensions.
        NO 2x multiplier. Consistent with naive baseline (plain MAE).
    """
    target_expanded = targets.unsqueeze(-1)      # (B, P, V, 1)
    errors = target_expanded - predictions        # (B, P, V, Q)
    q = quantiles_t.view(1, 1, 1, -1)            # (1, 1, 1, Q)
    pinball = torch.max(q * errors, (q - 1) * errors)  # (B, P, V, Q)
    # Mean over prediction horizon, variates, and quantiles — keep batch dim
    return pinball.mean(dim=(1, 2, 3))            # (B,)


def _naive_crps(targets):
    """Compute per-sample naive (seasonal) baseline CRPS.

    Uses last-value-repeated forecast as the naive baseline.
    CRPS for a deterministic forecast = MAE. No 2x multiplier.

    Args:
        targets: (B, P, V) ground truth

    Returns:
        (B,) per-sample naive CRPS (MAE of last-context-value forecast).
    """
    # Naive forecast: repeat last known value across prediction horizon
    # For the placeholder, use first timestep as "last context value"
    naive_pred = targets[:, 0:1, :].expand_as(targets)  # (B, P, V)
    # MAE per sample, averaged over (P, V)
    return (targets - naive_pred).abs().mean(dim=(1, 2))  # (B,)


def validate(model, n_batches: int = 10, batch_size: int = 32) -> dict:
    """
    Evaluate a model on the validation set.

    Computes nCRPS (normalized CRPS) as the geometric mean of per-sample
    CRPS ratios: nCRPS = geomean(crps_sample / naive_sample).

    nCRPS < 1.0 means the model beats the naive baseline.

    Args:
        model: nn.Module that takes (B, context_len, num_variates)
               and returns (B, prediction_len, num_variates, num_quantiles)
        n_batches: number of eval batches
        batch_size: batch size

    Returns: dict with crps (raw), ncrps (normalized), and mase
    """
    all_log_ratios: list[torch.Tensor] = []
    total_crps = 0.0
    total_mase = 0.0
    total_samples = 0

    device = next(model.parameters()).device
    loader = get_dataloader(batch_size)
    quantiles_t = torch.tensor(QUANTILES, device=device)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            context = batch["context"].to(device)
            targets = batch["target"].to(device)
            predictions = model(context)

            # Per-sample CRPS (B,)
            sample_crps = _crps_from_quantiles(predictions, targets, quantiles_t)

            # Per-sample naive CRPS (B,)
            sample_naive = _naive_crps(targets).to(device)

            # nCRPS: accumulate log ratios for geometric mean
            # Clamp to avoid log(0) or division by zero
            ratio = sample_crps / sample_naive.clamp(min=1e-8)
            all_log_ratios.append(torch.log(ratio.clamp(min=1e-8)))

            # Raw CRPS (batch mean for backward compat)
            total_crps += sample_crps.sum().item()
            total_samples += sample_crps.shape[0]

            # MASE: simplified placeholder (|error| / naive_scale)
            median_idx = len(QUANTILES) // 2
            median_pred = predictions[..., median_idx]
            naive_scale = (targets[:, 1:] - targets[:, :-1]).abs().mean().clamp(min=1e-6)
            mase = ((median_pred - targets).abs().mean() / naive_scale).item()
            total_mase += mase

    count = max(i, 1)  # number of batches actually processed
    avg_crps = total_crps / max(total_samples, 1)

    # Geometric mean of per-sample ratios
    if all_log_ratios:
        all_logs = torch.cat(all_log_ratios)
        ncrps = torch.exp(all_logs.mean()).item()
    else:
        ncrps = float("inf")

    avg_mase = total_mase / max(count, 1)

    return {
        "crps": avg_crps,
        "ncrps": ncrps,
        "mase": avg_mase,
        "n_batches": count,
        "n_samples": total_samples,
    }
