"""GIFT-Eval benchmark data: R2 download, caching, Arrow parsing, windowing.

Handles deterministic dataset selection per round, downloading Arrow files
from R2, and slicing context/target windows from real time-series data.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.r2_audit import R2AuditLog

logger = logging.getLogger(__name__)

GIFT_EVAL_DATASETS = [
    "LOOP_SEATTLE", "M_DENSE", "SZ_TAXI",
    "bitbrains_fast_storage", "bitbrains_rnd",
    "bizitobs_application", "bizitobs_l2c", "bizitobs_service",
    "car_parts_with_missing", "covid_deaths", "electricity",
    "ett1", "ett2", "hierarchical_sales", "hospital",
    "jena_weather", "kdd_cup_2018_with_missing",
    "m4_daily", "m4_hourly", "m4_monthly", "m4_quarterly",
    "m4_weekly", "m4_yearly",
    "restaurant", "saugeenday", "solar",
    "temperature_rain_with_missing", "us_births",
]

FREQ_TO_PRED_LEN = {
    "H": 48, "T": 60, "D": 14, "W": 8, "M": 12, "Q": 8, "Y": 4,
    "B": 14, "5T": 60, "10T": 60, "15T": 48, "30T": 48,
}


def select_datasets(eval_split_seed: int, n: int) -> list[str]:
    """Deterministic selection of N datasets for a round."""
    rng = random.Random(eval_split_seed)
    chosen = rng.sample(GIFT_EVAL_DATASETS, min(n, len(GIFT_EVAL_DATASETS)))
    return sorted(chosen)


class GiftEvalBenchmark:
    """Manages GIFT-Eval data: R2 download, caching, Arrow loading."""

    def __init__(
        self,
        r2: "R2AuditLog | None" = None,
        cache_dir: str = "/tmp/radar_gift_eval",
        r2_prefix: str = "gift-eval-benchmark/gift-eval-full",
    ):
        self.r2 = r2
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.r2_prefix = r2_prefix

    def select_datasets(self, eval_split_seed: int, n: int) -> list[str]:
        return select_datasets(eval_split_seed, n)

    def download_dataset(self, dataset_name: str) -> Path:
        """Download Arrow file from R2 to local cache. Skip if cached."""
        if dataset_name not in GIFT_EVAL_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        local_dir = self.cache_dir / dataset_name
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / "data-00000-of-00001.arrow"

        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path

        if self.r2 is None:
            raise RuntimeError("R2 client required to download GIFT-Eval data")

        r2_key = f"{self.r2_prefix}/{dataset_name}/data-00000-of-00001.arrow"
        self.r2.download_file_to_disk(r2_key, str(local_path))
        logger.info("Downloaded %s (%d bytes)", dataset_name, local_path.stat().st_size)
        return local_path


def load_dataset(
    dataset_name: str,
    context_len: int,
    prediction_len: int,
    max_series: int = 500,
    seed: int = 42,
    cache_dir: str = "/tmp/radar_gift_eval",
) -> list[dict]:
    """Load Arrow file, extract series, slice context/target windows.

    Returns list of {"context": list[float], "target": list[float],
                      "prediction_len": int}.
    """
    import pyarrow.ipc as ipc

    # Validate dataset_name to prevent path traversal
    if "/" in dataset_name or "\\" in dataset_name or ".." in dataset_name:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    arrow_path = Path(cache_dir) / dataset_name / "data-00000-of-00001.arrow"
    if not arrow_path.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_path}")

    reader = ipc.open_file(str(arrow_path))
    table = reader.read_all()

    # Try to get prediction length from schema metadata
    pred_len = prediction_len
    meta = table.schema.metadata or {}
    for key in (b"prediction_length", "prediction_length"):
        if key in meta:
            try:
                pred_len = int(meta[key])
            except (ValueError, TypeError):
                pass
            break

    # Try frequency-based prediction length
    if pred_len == prediction_len:
        for key in (b"freq", "freq"):
            if key in meta:
                freq_str = meta[key].decode() if isinstance(meta[key], bytes) else str(meta[key])
                if freq_str in FREQ_TO_PRED_LEN:
                    pred_len = FREQ_TO_PRED_LEN[freq_str]
                break

    # Extract series
    required_len = context_len + pred_len
    samples = []

    for i in range(table.num_rows):
        target_col = table.column("target")[i].as_py()
        if isinstance(target_col, list):
            values = target_col
        else:
            values = list(target_col)

        # Reconstruct shape if flattened
        shape_col_name = "target._np_shape"
        if shape_col_name in table.column_names:
            shape = table.column(shape_col_name)[i].as_py()
            if shape and len(shape) == 1:
                pass  # already 1D
            elif shape and len(shape) >= 2:
                # Flattened multivariate — take first variate
                n_variates = shape[-1] if len(shape) == 2 else 1
                if n_variates > 1:
                    total = len(values)
                    series_len = total // n_variates
                    values = values[:series_len]

        if len(values) < required_len:
            continue

        context = values[-(context_len + pred_len):-pred_len]
        target = values[-pred_len:]
        samples.append({
            "context": context,
            "target": target,
            "prediction_len": pred_len,
        })

    # Deterministic subsample
    if len(samples) > max_series:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_series)

    return samples


def get_eval_batches(
    samples: list[dict],
    batch_size: int = 32,
):
    """Yield batches as torch tensors: context (B, T, 1), target (B, P, 1)."""
    import torch

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        ctx = torch.tensor(
            [s["context"] for s in batch], dtype=torch.float32,
        ).unsqueeze(-1)  # (B, context_len, 1)
        tgt = torch.tensor(
            [s["target"] for s in batch], dtype=torch.float32,
        ).unsqueeze(-1)  # (B, pred_len, 1)
        yield {"context": ctx, "target": tgt}
