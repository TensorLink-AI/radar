"""
Frozen data preparation and evaluation for time-series forecasting.

Dual-mode: GIFT-Eval (real benchmark data from Arrow files) or random
(placeholder for testing). Mode controlled by RADAR_EVAL_DATA env var.
"""

import math
import os
import random

import torch

# ── Task constants (frozen, match the task YAML) ──────────────────
CONTEXT_LEN = 512       # Input context window length
PREDICTION_LEN = 96     # Forecast horizon (default, overridden per dataset)
NUM_VARIATES = 1        # Univariate time series (GIFT-Eval is univariate)
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

EVAL_DATA_MODE = os.environ.get("RADAR_EVAL_DATA", "gift_eval")


def get_dataloader(
    batch_size: int = 64,
    seed: int = 42,
    data_dir: str | None = None,
    dataset_names: list[str] | None = None,
):
    """Yield training batches. GIFT-Eval mode loads from Arrow cache."""
    if EVAL_DATA_MODE == "random" or data_dir is None:
        yield from _random_dataloader(batch_size)
        return

    try:
        from shared.gift_eval import load_dataset, get_eval_batches, GIFT_EVAL_DATASETS
        names = dataset_names or _discover_cached_datasets(data_dir)
        if not names:
            yield from _random_dataloader(batch_size)
            return
        while True:
            for name in names:
                try:
                    samples = load_dataset(
                        name, CONTEXT_LEN, PREDICTION_LEN,
                        seed=seed, cache_dir=data_dir,
                    )
                    yield from get_eval_batches(samples, batch_size)
                except Exception:
                    continue
    except ImportError:
        yield from _random_dataloader(batch_size)


def _random_dataloader(batch_size: int = 64):
    """Placeholder dataloader yielding random time-series batches."""
    while True:
        batch = {
            "context": torch.randn(batch_size, CONTEXT_LEN, NUM_VARIATES),
            "target": torch.randn(batch_size, PREDICTION_LEN, NUM_VARIATES),
        }
        yield batch


def _discover_cached_datasets(data_dir: str) -> list[str]:
    """Find datasets available in the local cache directory."""
    from pathlib import Path
    cache = Path(data_dir)
    if not cache.exists():
        return []
    return sorted(
        d.name for d in cache.iterdir()
        if d.is_dir() and (d / "data-00000-of-00001.arrow").exists()
    )


def _crps_from_quantiles(predictions, targets, quantiles_t):
    """Compute per-sample CRPS from quantile predictions via pinball loss.

    Args:
        predictions: (B, P, V, Q) quantile predictions
        targets: (B, P, V) ground truth
        quantiles_t: (Q,) quantile levels tensor on same device

    Returns:
        (B,) per-sample CRPS — mean pinball loss over (P, V, Q) dimensions.
    """
    target_expanded = targets.unsqueeze(-1)      # (B, P, V, 1)
    errors = target_expanded - predictions        # (B, P, V, Q)
    q = quantiles_t.view(1, 1, 1, -1)            # (1, 1, 1, Q)
    pinball = torch.max(q * errors, (q - 1) * errors)  # (B, P, V, Q)
    return pinball.mean(dim=(1, 2, 3))            # (B,)


def _naive_crps(targets):
    """Compute per-sample naive baseline CRPS (last-value-repeated).

    Args:
        targets: (B, P, V) ground truth

    Returns:
        (B,) per-sample naive CRPS (MAE of last-context-value forecast).
    """
    naive_pred = targets[:, 0:1, :].expand_as(targets)  # (B, P, V)
    return (targets - naive_pred).abs().mean(dim=(1, 2))  # (B,)


def validate(
    model,
    n_batches: int = 10,
    batch_size: int = 32,
    seed: int = 42,
    data_dir: str | None = None,
    dataset_names: list[str] | None = None,
) -> dict:
    """Evaluate a model. GIFT-Eval mode evaluates per-dataset then aggregates."""
    if data_dir is None:
        data_dir = os.environ.get("RADAR_GIFT_EVAL_CACHE", "")

    if EVAL_DATA_MODE == "random" or not data_dir:
        return _random_validate(model, n_batches, batch_size)

    try:
        from shared.gift_eval import load_dataset, get_eval_batches
        names = dataset_names or _discover_cached_datasets(data_dir)
        if not names:
            return _random_validate(model, n_batches, batch_size)
        return _gift_eval_validate(model, names, batch_size, seed, data_dir)
    except ImportError:
        return _random_validate(model, n_batches, batch_size)


def _gift_eval_validate(
    model, dataset_names: list[str], batch_size: int,
    seed: int, data_dir: str,
) -> dict:
    """Evaluate across multiple GIFT-Eval datasets."""
    from shared.gift_eval import load_dataset, get_eval_batches

    device = next(model.parameters()).device
    quantiles_t = torch.tensor(QUANTILES, device=device)
    per_dataset = []
    all_crps = []
    all_ncrps_logs = []
    all_mase = []

    for name in dataset_names:
        try:
            samples = load_dataset(
                name, CONTEXT_LEN, PREDICTION_LEN,
                max_series=int(os.environ.get("RADAR_GIFT_EVAL_MAX_SERIES", "500")),
                seed=seed, cache_dir=data_dir,
            )
        except Exception:
            continue

        if not samples:
            continue

        ds_crps_sum = 0.0
        ds_mase_sum = 0.0
        ds_samples = 0
        ds_log_ratios = []

        with torch.no_grad():
            for batch in get_eval_batches(samples, batch_size):
                context = batch["context"].to(device)
                targets = batch["target"].to(device)
                predictions = model(context)

                sample_crps = _crps_from_quantiles(predictions, targets, quantiles_t)
                sample_naive = _naive_crps(targets).to(device)

                ratio = sample_crps / sample_naive.clamp(min=1e-8)
                ds_log_ratios.append(torch.log(ratio.clamp(min=1e-8)))

                ds_crps_sum += sample_crps.sum().item()
                ds_samples += sample_crps.shape[0]

                median_idx = len(QUANTILES) // 2
                median_pred = predictions[..., median_idx]
                naive_scale = (targets[:, 1:] - targets[:, :-1]).abs().mean().clamp(min=1e-6)
                ds_mase_sum += ((median_pred - targets).abs().mean() / naive_scale).item()

        if ds_samples == 0:
            continue

        n_batches_ds = max(len(ds_log_ratios), 1)
        ds_avg_crps = ds_crps_sum / ds_samples
        ds_ncrps = torch.exp(torch.cat(ds_log_ratios).mean()).item() if ds_log_ratios else float("inf")
        ds_avg_mase = ds_mase_sum / n_batches_ds

        per_dataset.append({
            "name": name, "crps": ds_avg_crps,
            "ncrps": ds_ncrps, "mase": ds_avg_mase,
            "n_series": ds_samples,
        })
        all_crps.append(ds_avg_crps)
        all_ncrps_logs.extend(
            [lr for t in ds_log_ratios for lr in t.tolist()]
        )
        all_mase.append(ds_avg_mase)

    if not per_dataset:
        return _random_validate(model, 10, batch_size)

    agg_crps = sum(all_crps) / len(all_crps)
    agg_ncrps = math.exp(sum(all_ncrps_logs) / len(all_ncrps_logs)) if all_ncrps_logs else float("inf")
    agg_mase = sum(all_mase) / len(all_mase)

    return {
        "crps": agg_crps,
        "ncrps": agg_ncrps,
        "mase": agg_mase,
        "n_datasets": len(per_dataset),
        "per_dataset": per_dataset,
    }


def _random_validate(model, n_batches: int = 10, batch_size: int = 32) -> dict:
    """Evaluate on random data (fallback/testing)."""
    all_log_ratios: list[torch.Tensor] = []
    total_crps = 0.0
    total_mase = 0.0
    total_samples = 0

    device = next(model.parameters()).device
    loader = _random_dataloader(batch_size)
    quantiles_t = torch.tensor(QUANTILES, device=device)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            context = batch["context"].to(device)
            targets = batch["target"].to(device)
            predictions = model(context)

            sample_crps = _crps_from_quantiles(predictions, targets, quantiles_t)
            sample_naive = _naive_crps(targets).to(device)

            ratio = sample_crps / sample_naive.clamp(min=1e-8)
            all_log_ratios.append(torch.log(ratio.clamp(min=1e-8)))

            total_crps += sample_crps.sum().item()
            total_samples += sample_crps.shape[0]

            median_idx = len(QUANTILES) // 2
            median_pred = predictions[..., median_idx]
            naive_scale = (targets[:, 1:] - targets[:, :-1]).abs().mean().clamp(min=1e-6)
            mase = ((median_pred - targets).abs().mean() / naive_scale).item()
            total_mase += mase

    count = max(i, 1)
    avg_crps = total_crps / max(total_samples, 1)

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
