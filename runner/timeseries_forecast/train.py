"""Time-series forecasting TaskRunner.

Thin wrapper — only defines what's unique to ts_forecasting:
model signature, data loading, loss function, FLOPs measurement.
The generic harness handles everything else.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import inspect

from runner.harness import TaskRunner, TrainingConfig, run_training as generic_run_training

logger = logging.getLogger(__name__)


def _decode_json_list(raw: str) -> list[str] | None:
    """Decode a JSON list from an env var. Returns None when unset/invalid/empty."""
    import json
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, list) or not value:
        return None
    return value


class TSForecastingRunner:
    """TaskRunner implementation for time-series forecasting."""

    def __init__(self):
        # Lazy-loaded constants from frozen prepare.py
        self._constants = None

    def _load_constants(self):
        if self._constants is None:
            from prepare import CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES
            self._constants = {
                "context_len": CONTEXT_LEN,
                "prediction_len": PREDICTION_LEN,
                "num_variates": NUM_VARIATES,
                "quantiles": QUANTILES,
            }
        return self._constants

    def build_model(self, sub: Any, device: str) -> Any:
        c = self._load_constants()
        return sub.build_model(
            c["context_len"], c["prediction_len"],
            c["num_variates"], c["quantiles"],
        ).to(device)

    def get_dataloader(self, batch_size: int) -> Iterator:
        import json
        import os
        from prepare import get_dataloader

        data_dir = os.environ.get("RADAR_GIFT_EVAL_CACHE", "")

        # Local paths win — set by sandbox_runner.py from prefetched shards.
        pretrain_shard_paths = _decode_json_list(
            os.environ.get("RADAR_PRETRAIN_LOCAL_PATHS", ""),
        )
        # URLs are only consulted when local paths are absent (legacy /
        # in-process path used by tests).
        pretrain_shard_urls = None
        if not pretrain_shard_paths:
            pretrain_shard_urls = _decode_json_list(
                os.environ.get("RADAR_PRETRAIN_SHARD_URLS", ""),
            )

        return get_dataloader(
            batch_size=batch_size,
            data_dir=data_dir if data_dir else None,
            pretrain_shard_urls=pretrain_shard_urls,
            pretrain_shard_paths=pretrain_shard_paths,
        )

    def get_val_dataloader(self, batch_size: int):
        """Yield val batches from the reserved pretrain val shard.

        Reads ``RADAR_PRETRAIN_VAL_LOCAL_PATHS`` (set by the sandbox after
        prefetch) or, as a fallback for direct in-process callers,
        ``RADAR_PRETRAIN_VAL_SHARD_URLS``.  Returns None if neither is set.
        Val batches come from a shard that is fixed across rounds.

        Contract: finite + deterministic — same batches in same order every call.
        """
        import os

        shard_paths = _decode_json_list(
            os.environ.get("RADAR_PRETRAIN_VAL_LOCAL_PATHS", ""),
        )
        shard_urls = None
        if not shard_paths:
            shard_urls = _decode_json_list(
                os.environ.get("RADAR_PRETRAIN_VAL_SHARD_URLS", ""),
            )
        if not (shard_paths or shard_urls):
            return None

        from pretrain_loader import pretrain_dataloader
        from prepare import CONTEXT_LEN

        max_val_batches = int(os.environ.get("RADAR_VAL_MAX_BATCHES", "50"))

        def _val_iter():
            # shuffle_buffer_size=1 disables shuffling. seed=0 fixes the
            # deterministic shard/row iteration order so every call to this
            # function yields the same batches.
            loader = pretrain_dataloader(
                shard_urls=shard_urls,
                shard_paths=shard_paths,
                batch_size=batch_size,
                context_len=CONTEXT_LEN,
                shuffle_buffer_size=1,
                seed=0,
            )
            for i, batch in enumerate(loader):
                if i >= max_val_batches:
                    break
                yield batch

        return _val_iter()

    @staticmethod
    def _align_pred_target(predictions: Any, targets: Any):
        """Truncate predictions/targets to a common forecast horizon.

        Models always output PREDICTION_LEN=96 steps, but GIFT-Eval datasets
        each have their own native prediction length (e.g. hourly=48).
        Without this alignment, the loss call broadcasts 96 vs 48 at dim 1
        and PyTorch raises a tensor-size mismatch that fails training.
        """
        p, t = predictions.shape[1], targets.shape[1]
        if p == t:
            return predictions, targets
        n = min(p, t)
        return predictions[:, :n], targets[:, :n]

    def default_loss(self, predictions: Any, targets: Any) -> Any:
        """Quantile loss (pinball) — the ts_forecasting default."""
        import torch
        c = self._load_constants()
        quantiles = c["quantiles"]
        predictions, targets = self._align_pred_target(predictions, targets)
        target_expanded = targets.unsqueeze(-1)
        errors = target_expanded - predictions
        q = torch.tensor(quantiles, device=predictions.device)
        return torch.max(q * errors, (q - 1) * errors).mean()

    def compute_val_metrics(
        self, predictions: Any, targets: Any, inputs: Any,
    ) -> dict[str, tuple[float, float]]:
        """CRPS (weighted quantile loss) and MASE on the pretrain val shard.

        Per-batch components ``(num_sum, den_sum)`` are returned; the harness
        sums them across the val pass and reports ``num/den`` per metric.
        Pretrain shards mix frequencies, so MASE uses season=1 (1-step naive
        baseline scale).
        """
        import torch

        c = self._load_constants()
        quantiles = c["quantiles"]
        median_idx = len(quantiles) // 2
        predictions, targets = self._align_pred_target(predictions, targets)

        finite = (
            torch.isfinite(predictions).flatten(1).all(dim=1)
            & torch.isfinite(targets).flatten(1).all(dim=1)
        )
        if not finite.any():
            return {}
        predictions = predictions[finite]
        targets = targets[finite]
        if inputs.shape[0] == finite.shape[0]:
            inputs = inputs[finite]

        device = predictions.device
        quantiles_t = torch.tensor(quantiles, device=device)

        # CRPS components (gluonts mean_weighted_sum_quantile_loss axis=None):
        # num = (2/Q) Σ pinball(y, q_pred), den = Σ |y|.
        target_expanded = targets.unsqueeze(-1)
        errors = target_expanded - predictions
        q_view = quantiles_t.view(1, 1, 1, -1)
        pinball = torch.max(q_view * errors, (q_view - 1) * errors)
        crps_num = float(((2.0 / quantiles_t.numel()) * pinball).sum().item())
        crps_den = float(targets.abs().sum().item())

        # MASE components, season=1 (mixed-freq pretrain windows have no
        # known seasonality — fall back to the 1-step naive baseline).
        median_pred = predictions[..., median_idx]              # (B, H, V)
        H = targets.shape[1]
        err = (targets - median_pred).abs().sum(dim=1)          # (B, V)
        if inputs.shape[1] >= 2:
            diffs = (inputs[:, 1:] - inputs[:, :-1]).abs()      # (B, ctx-1, V)
            scale = diffs.mean(dim=1)                           # (B, V)
        else:
            scale = torch.ones_like(err)
        valid = torch.isfinite(scale) & (scale > 0)
        if valid.any():
            mase_err = float(err[valid].sum().item())
            mase_scale = float((H * scale[valid]).sum().item())
        else:
            mase_err, mase_scale = 0.0, 0.0

        return {
            "crps": (crps_num, crps_den),
            "mase": (mase_err, mase_scale),
        }

    def wrap_loss(self, sub_loss_fn):
        """Wrap miner's compute_loss for backward compat.

        Old ts_forecasting harness called compute_loss(preds, targets, QUANTILES).
        Generic harness calls loss_fn(preds, targets). This adapter handles both.
        Also aligns prediction/target horizons so miners don't have to worry
        about GIFT-Eval per-dataset prediction lengths.
        """
        c = self._load_constants()
        quantiles = c["quantiles"]
        sig = inspect.signature(sub_loss_fn)
        align = self._align_pred_target
        if len(sig.parameters) >= 3:
            # Old-style 3-arg: compute_loss(preds, targets, quantiles)
            def wrapped(predictions, targets):
                predictions, targets = align(predictions, targets)
                return sub_loss_fn(predictions, targets, quantiles)
            return wrapped

        def wrapped_2arg(predictions, targets):
            predictions, targets = align(predictions, targets)
            return sub_loss_fn(predictions, targets)
        return wrapped_2arg

    def measure_flops(self, model: Any, device: str) -> int:
        try:
            from flops import compute_flops_equivalent
            c = self._load_constants()
            cpu_model = model.to("cpu") if device != "cpu" else model
            flops = compute_flops_equivalent(cpu_model, c["context_len"], c["num_variates"], "cpu")
            if device != "cpu":
                model.to(device)
            return flops
        except Exception as e:
            logger.warning("FLOPs measurement failed: %s", e)
            if device != "cpu":
                try:
                    model.to(device)
                except Exception:
                    pass
            return 0


# Singleton instance
_runner = TSForecastingRunner()


def run_training(architecture_code: str, config: dict) -> dict:
    """Entry point called by runner/server.py."""
    tc = TrainingConfig.from_dict(config)
    return generic_run_training(_runner, architecture_code, tc)


# ── Eval template for Phase C ────────────────────────────────────────
# Kept in a zero-dep sibling module so the validator can load it via
# importlib without pulling torch/prepare/flops.
from runner.timeseries_forecast.eval_template import EVAL_TEMPLATE  # noqa: E402,F401
