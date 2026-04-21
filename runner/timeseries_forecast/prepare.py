"""
Frozen data preparation and evaluation for time-series forecasting.

Three training data modes:
  - pretrain: Streaming parquet shards from presigned URLs (production)
  - gift_eval: GIFT-Eval Arrow benchmark data (legacy/fallback)
  - random: Random data (placeholder for testing)

Evaluation always uses GIFT-Eval Arrow data or random fallback.
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
    pretrain_shard_urls: list[str] | None = None,
):
    """Yield training batches.

    Priority: pretrain shards (if URLs provided) → GIFT-Eval Arrow → random.
    """
    # Pretrain mode: streaming parquet shards
    if pretrain_shard_urls:
        try:
            from pretrain_loader import pretrain_dataloader
            shuffle_buf = int(os.environ.get("RADAR_PRETRAIN_SHUFFLE_BUFFER", "10000"))
            yield from pretrain_dataloader(
                shard_urls=pretrain_shard_urls,
                batch_size=batch_size,
                context_len=CONTEXT_LEN,
                shuffle_buffer_size=shuffle_buf,
                seed=seed,
            )
            return
        except ImportError:
            pass  # fall through to gift_eval / random

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


def _wql_components(predictions, targets, quantiles_t):
    """Per-series weighted quantile loss numerator and abs-target denominator.

    Implements gluonts mean_weighted_sum_quantile_loss axis=None aggregation:
    numerator = (2/Q) * Σ_{q,t} pinball(y, q_pred), denominator = Σ_t |y|.
    Caller sums across series then divides.

    Args:
        predictions: (B, P, V, Q) quantile predictions
        targets:     (B, P, V)    ground truth
        quantiles_t: (Q,)         quantile levels on same device

    Returns:
        (num_per_series, den_per_series): both (B,) tensors.
    """
    target_expanded = targets.unsqueeze(-1)               # (B, P, V, 1)
    errors = target_expanded - predictions                 # (B, P, V, Q)
    q = quantiles_t.view(1, 1, 1, -1)                     # (1, 1, 1, Q)
    pinball = torch.max(q * errors, (q - 1) * errors)     # (B, P, V, Q)
    num = (2.0 / quantiles_t.numel()) * pinball.sum(dim=(1, 2, 3))  # (B,)
    den = targets.abs().sum(dim=(1, 2))                   # (B,)
    return num, den


def _crps_from_quantiles(predictions, targets, quantiles_t):
    """Per-series weighted quantile loss (wQL), for back-compat callers."""
    num, den = _wql_components(predictions, targets, quantiles_t)
    return num / den.clamp(min=1e-8)


def _seasonal_naive_forecast(history: torch.Tensor, horizon: int, season: int) -> torch.Tensor:
    """Tile the last `season` values of history across `horizon` steps.

    If history is shorter than `season` (or season<=0), falls back to the
    last-value repeat. Returns a (horizon,) tensor on history's device.
    """
    L = history.numel()
    if L == 0:
        return torch.zeros(horizon, dtype=history.dtype, device=history.device)
    s = max(int(season), 1)
    if s > L:
        s = 1
    last_cycle = history[L - s:]                               # (s,)
    reps = (horizon + s - 1) // s
    return last_cycle.repeat(reps)[:horizon]


def _mase_components(
    median_pred: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
    season: int,
) -> tuple[float, float]:
    """Return (err_sum, horizon*scale) for a single series.

    err_sum  = Σ_t |target - median_pred|
    scale    = mean(|h[k] - h[k - season]|) over the history
    Caller sums both across series and divides.
    """
    err_sum = float((target - median_pred).abs().sum().item())
    H = int(target.numel())
    L = history.numel()
    s = max(int(season), 1)
    if s >= L:
        s = 1
    if L <= s:
        return err_sum, float(H)  # degenerate → scale=1
    diffs = (history[s:] - history[:-s]).abs()
    scale = float(diffs.mean().item()) if diffs.numel() else 1.0
    if scale <= 0.0 or not math.isfinite(scale):
        scale = 1.0
    return err_sum, float(H) * scale


def validate(
    model,
    n_batches: int = 10,
    batch_size: int = 32,
    seed: int = 42,
    data_dir: str | None = None,
    dataset_names: list[str] | None = None,
) -> dict:
    """Evaluate a model across GIFT-Eval tasks (dataset, freq, term).

    `dataset_names` accepts the leaderboard spelling (e.g. "ett1/15T"); when
    omitted we default to the full SHORT_DATASETS list (which expands to 97
    tasks via MED_LONG_DATASETS). Per-task metrics are geometric-mean
    normalized against seasonal-naive for leaderboard comparability.
    """
    if data_dir is None:
        data_dir = os.environ.get("RADAR_GIFT_EVAL_CACHE", "")

    if EVAL_DATA_MODE == "random" or not data_dir:
        return _random_validate(model, n_batches, batch_size)

    try:
        from shared.gift_eval import SHORT_DATASETS
        names = list(dataset_names) if dataset_names else list(SHORT_DATASETS)
        return _gift_eval_validate(model, names, batch_size, seed, data_dir)
    except ImportError:
        return _random_validate(model, n_batches, batch_size)


def _eval_one_task(
    model, task: dict, batch_size: int, data_dir: str,
    quantiles_t, device, max_series: int,
) -> dict | None:
    """Run a single (dataset, freq, term) task. Returns per-task metrics or None."""
    from shared.gift_eval import (
        load_dataset_for_task, get_eval_batches_with_history,
    )

    try:
        samples = load_dataset_for_task(
            task, CONTEXT_LEN, cache_dir=data_dir, max_series=max_series,
        )
    except (FileNotFoundError, KeyError, ValueError):
        return None
    if not samples:
        return None

    season = int(task.get("season_length", 1))
    H = int(task["prediction_length"])

    # Running sums for task-level aggregation (gluonts axis=None style).
    wql_num_sum = 0.0
    wql_den_sum = 0.0
    snaive_wql_num_sum = 0.0
    snaive_wql_den_sum = 0.0
    mase_err_sum = 0.0
    mase_scale_sum = 0.0
    snaive_err_sum = 0.0
    snaive_scale_sum = 0.0
    n_windows = 0

    median_idx = len(QUANTILES) // 2

    with torch.no_grad():
        for batch in get_eval_batches_with_history(samples, batch_size):
            context = batch["context"].to(device)
            targets = batch["target"].to(device)
            histories: list[list[float]] = batch["history"]

            predictions = model(context)
            # Truncate (model may emit CONTEXT-tied length larger than H).
            if predictions.shape[1] > H:
                predictions = predictions[:, :H, :, :]
            # Pad if too short (rare — only if a model underproduces).
            if predictions.shape[1] < H:
                pad = predictions.new_zeros(
                    predictions.shape[0], H - predictions.shape[1],
                    *predictions.shape[2:],
                )
                predictions = torch.cat([predictions, pad], dim=1)

            valid = (
                torch.isfinite(predictions).flatten(1).all(dim=1)
                & torch.isfinite(targets).flatten(1).all(dim=1)
            )
            if not valid.any():
                continue
            predictions = predictions[valid]
            targets = targets[valid]
            kept_idx = valid.nonzero(as_tuple=True)[0].tolist()

            # Model wQL components
            num, den = _wql_components(predictions, targets, quantiles_t)
            wql_num_sum += float(num.sum().item())
            wql_den_sum += float(den.sum().item())

            # Per-series seasonal-naive + MASE
            median_pred = predictions[..., median_idx]   # (B, H, V)
            B = predictions.shape[0]
            for bi in range(B):
                hist_list = histories[kept_idx[bi]]
                hist_t = torch.tensor(hist_list, dtype=torch.float32, device=device)
                target_s = targets[bi, :, 0]
                median_s = median_pred[bi, :, 0]

                # MASE for the model prediction
                err, w = _mase_components(median_s, target_s, hist_t, season)
                if math.isfinite(err) and math.isfinite(w) and w > 0:
                    mase_err_sum += err
                    mase_scale_sum += w

                # Seasonal-naive baseline forecast → wQL + MASE
                snaive = _seasonal_naive_forecast(hist_t, H, season)  # (H,)
                # wQL for a point forecast reduces to 2 * |y - q_0.5| * 0.5 per
                # quantile symmetrically; gluonts treats point forecast as the
                # same prediction at every quantile level (median). Equivalent:
                # num = Σ_t |y - m|, den = Σ_t |y|.
                sn_num = float((target_s - snaive).abs().sum().item())
                sn_den = float(target_s.abs().sum().item())
                snaive_wql_num_sum += sn_num
                snaive_wql_den_sum += sn_den

                sn_err, sn_w = _mase_components(snaive, target_s, hist_t, season)
                if math.isfinite(sn_err) and math.isfinite(sn_w) and sn_w > 0:
                    snaive_err_sum += sn_err
                    snaive_scale_sum += sn_w
                n_windows += 1

    if n_windows == 0 or wql_den_sum <= 0.0:
        return None

    task_wql = wql_num_sum / max(wql_den_sum, 1e-12)
    task_mase = mase_err_sum / max(mase_scale_sum, 1e-12)
    snaive_wql = snaive_wql_num_sum / max(snaive_wql_den_sum, 1e-12)
    snaive_mase = snaive_err_sum / max(snaive_scale_sum, 1e-12)

    norm_crps = task_wql / snaive_wql if snaive_wql > 0 else float("inf")
    norm_mase = task_mase / snaive_mase if snaive_mase > 0 else float("inf")

    return {
        "config_name": task["config_name"],
        "crps": task_wql,
        "mase": task_mase,
        "seasonal_naive_crps": snaive_wql,
        "seasonal_naive_mase": snaive_mase,
        "normalized_crps": norm_crps,
        "normalized_mase": norm_mase,
        "n_windows": n_windows,
    }


def _geomean(values: list[float]) -> float:
    logs = [math.log(v) for v in values if math.isfinite(v) and v > 0]
    if not logs:
        return float("inf")
    return math.exp(sum(logs) / len(logs))


def _gift_eval_validate(
    model, dataset_names: list[str], batch_size: int,
    seed: int, data_dir: str,
) -> dict:
    """Evaluate across GIFT-Eval tasks and aggregate by geometric mean."""
    from shared.gift_eval import build_task_configs

    device = next(model.parameters()).device
    quantiles_t = torch.tensor(QUANTILES, device=device)
    max_series = int(os.environ.get("RADAR_GIFT_EVAL_MAX_SERIES", "0"))
    max_tasks = int(os.environ.get("RADAR_GIFT_EVAL_MAX_TASKS", "0"))

    tasks = build_task_configs(dataset_names)
    if max_tasks > 0:
        tasks = tasks[:max_tasks]

    per_task: list[dict] = []
    for task in tasks:
        res = _eval_one_task(
            model, task, batch_size, data_dir,
            quantiles_t, device, max_series,
        )
        if res is not None:
            per_task.append(res)

    if not per_task:
        return _random_validate(model, 10, batch_size)

    agg_crps = _geomean([t["normalized_crps"] for t in per_task])
    agg_mase = _geomean([t["normalized_mase"] for t in per_task])

    return {
        "crps": agg_crps,
        "mase": agg_mase,
        "n_tasks": len(per_task),
        "per_task": per_task,
        # legacy mirrors for one-release back-compat:
        "ncrps": agg_crps,
        "n_datasets": len(per_task),
        "per_dataset": per_task,
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

            # Mask: keep samples where predictions are finite (no NaN/inf)
            valid = torch.isfinite(predictions).flatten(1).all(dim=1)
            if not valid.any():
                continue

            predictions = predictions[valid]
            targets = targets[valid]

            sample_crps = _crps_from_quantiles(predictions, targets, quantiles_t)
            naive_pred = targets[:, 0:1, :].expand_as(targets)
            sample_naive = (targets - naive_pred).abs().mean(dim=(1, 2))

            # Drop samples with non-finite CRPS (overflow from large preds)
            finite_mask = torch.isfinite(sample_crps)
            if not finite_mask.any():
                continue
            sample_crps = sample_crps[finite_mask]
            sample_naive = sample_naive[finite_mask]
            predictions = predictions[finite_mask]
            targets = targets[finite_mask]

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

    # No valid samples → model is fully broken, score as failure
    if total_samples == 0:
        return {
            "crps": float("inf"),
            "ncrps": float("inf"),
            "mase": float("inf"),
            "n_batches": count,
            "n_samples": 0,
        }

    avg_crps = total_crps / total_samples

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
