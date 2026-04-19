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
        # Pretrain shard URLs passed via env var (JSON list of presigned URLs)
        pretrain_urls_raw = os.environ.get("RADAR_PRETRAIN_SHARD_URLS", "")
        pretrain_shard_urls = None
        if pretrain_urls_raw:
            try:
                pretrain_shard_urls = json.loads(pretrain_urls_raw)
            except (json.JSONDecodeError, TypeError):
                pass
        return get_dataloader(
            batch_size=batch_size,
            data_dir=data_dir if data_dir else None,
            pretrain_shard_urls=pretrain_shard_urls,
        )

    def get_val_dataloader(self, batch_size: int):
        """Yield val batches from the reserved pretrain val shard.

        Reads RADAR_PRETRAIN_VAL_SHARD_URLS (JSON list). Returns None if unset.
        Val batches come from a shard that is fixed across rounds (the
        coordinator presigns the same shard URL every round for val).

        Contract: finite + deterministic — same batches in same order every call.
        """
        import json
        import os

        urls_raw = os.environ.get("RADAR_PRETRAIN_VAL_SHARD_URLS", "")
        if not urls_raw:
            return None
        try:
            shard_urls = json.loads(urls_raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not shard_urls:
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

    def default_loss(self, predictions: Any, targets: Any) -> Any:
        """Quantile loss (pinball) — the ts_forecasting default."""
        import torch
        c = self._load_constants()
        quantiles = c["quantiles"]
        target_expanded = targets.unsqueeze(-1)
        errors = target_expanded - predictions
        q = torch.tensor(quantiles, device=predictions.device)
        return torch.max(q * errors, (q - 1) * errors).mean()

    def wrap_loss(self, sub_loss_fn):
        """Wrap miner's compute_loss for backward compat.

        Old ts_forecasting harness called compute_loss(preds, targets, QUANTILES).
        Generic harness calls loss_fn(preds, targets). This adapter handles both.
        """
        c = self._load_constants()
        quantiles = c["quantiles"]
        sig = inspect.signature(sub_loss_fn)
        if len(sig.parameters) >= 3:
            # Old-style 3-arg: compute_loss(preds, targets, quantiles)
            def wrapped(predictions, targets):
                return sub_loss_fn(predictions, targets, quantiles)
            return wrapped
        return sub_loss_fn

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
