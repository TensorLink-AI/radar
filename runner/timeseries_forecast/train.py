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

EVAL_TEMPLATE = '''
import json
import os
import random
import sys

import torch
from safetensors.torch import load_file

from prepare import validate, CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES
from flops import compute_flops_equivalent

random.seed({eval_split_seed})
torch.manual_seed({eval_split_seed})
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

arch_path = "{arch_path}"
checkpoint_path = "{checkpoint_path}"
device = "{device}"

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission", arch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "build_model") or not callable(mod.build_model):
        print(json.dumps({{"crps": float("inf"), "mase": float("inf"), "error": "Missing build_model()"}}))
        sys.exit(0)

    model = mod.build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES).to(device)
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict)

    flops_equiv = 0
    try:
        flops_equiv = compute_flops_equivalent(model, CONTEXT_LEN, NUM_VARIATES, device)
    except Exception:
        pass

    param_count = sum(p.numel() for p in model.parameters())
    if hasattr(model, "reset"):
        model.reset()
    model.eval()

    data_dir = os.environ.get("RADAR_GIFT_EVAL_CACHE", "")
    metrics = validate(model, seed={eval_split_seed},
                       data_dir=data_dir if data_dir else None)

    result = {{
        "crps": metrics["crps"],
        "ncrps": metrics.get("ncrps", float("inf")),
        "mase": metrics["mase"],
        "flops_equivalent_size": flops_equiv,
        "param_count": param_count,
    }}
    if "n_datasets" in metrics:
        result["n_datasets"] = metrics["n_datasets"]
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"crps": float("inf"), "mase": float("inf"), "error": str(e)}}))
'''
