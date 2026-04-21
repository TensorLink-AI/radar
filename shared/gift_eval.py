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

# Manifest: dataset key → R2 subpath under the prefix.
# Keys use format "name__freq" for multi-freq datasets, plain "name" otherwise.
# Every frequency variant is a separate eval target — full benchmark coverage.
GIFT_EVAL_MANIFEST: dict[str, str] = {
    # ── No freq subdirectory (single frequency) ──────────────
    "bizitobs_application": "bizitobs_application",
    "bizitobs_service": "bizitobs_service",
    "car_parts_with_missing": "car_parts_with_missing",
    "covid_deaths": "covid_deaths",
    "hospital": "hospital",
    "m4_daily": "m4_daily",
    "m4_hourly": "m4_hourly",
    "m4_monthly": "m4_monthly",
    "m4_quarterly": "m4_quarterly",
    "m4_weekly": "m4_weekly",
    "m4_yearly": "m4_yearly",
    "restaurant": "restaurant",
    "temperature_rain_with_missing": "temperature_rain_with_missing",
    # ── Multi-frequency datasets (all variants) ──────────────
    "LOOP_SEATTLE__5T": "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE__D": "LOOP_SEATTLE/D",
    "LOOP_SEATTLE__H": "LOOP_SEATTLE/H",
    "M_DENSE__D": "M_DENSE/D",
    "M_DENSE__H": "M_DENSE/H",
    "SZ_TAXI__15T": "SZ_TAXI/15T",
    "SZ_TAXI__H": "SZ_TAXI/H",
    "bitbrains_fast_storage__5T": "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage__H": "bitbrains_fast_storage/H",
    "bitbrains_rnd__5T": "bitbrains_rnd/5T",
    "bitbrains_rnd__H": "bitbrains_rnd/H",
    "bizitobs_l2c__5T": "bizitobs_l2c/5T",
    "bizitobs_l2c__H": "bizitobs_l2c/H",
    "electricity__15T": "electricity/15T",
    "electricity__D": "electricity/D",
    "electricity__H": "electricity/H",
    "electricity__W": "electricity/W",
    "ett1__15T": "ett1/15T",
    "ett1__D": "ett1/D",
    "ett1__H": "ett1/H",
    "ett1__W": "ett1/W",
    "ett2__15T": "ett2/15T",
    "ett2__D": "ett2/D",
    "ett2__H": "ett2/H",
    "ett2__W": "ett2/W",
    "hierarchical_sales__D": "hierarchical_sales/D",
    "hierarchical_sales__W": "hierarchical_sales/W",
    "jena_weather__10T": "jena_weather/10T",
    "jena_weather__D": "jena_weather/D",
    "jena_weather__H": "jena_weather/H",
    "kdd_cup_2018_with_missing__D": "kdd_cup_2018_with_missing/D",
    "kdd_cup_2018_with_missing__H": "kdd_cup_2018_with_missing/H",
    "saugeenday__D": "saugeenday/D",
    "saugeenday__M": "saugeenday/M",
    "saugeenday__W": "saugeenday/W",
    "solar__10T": "solar/10T",
    "solar__D": "solar/D",
    "solar__H": "solar/H",
    "solar__W": "solar/W",
    "us_births__D": "us_births/D",
    "us_births__M": "us_births/M",
    "us_births__W": "us_births/W",
}

GIFT_EVAL_DATASETS = sorted(GIFT_EVAL_MANIFEST.keys())

# Legacy freq→prediction-length (used by load_dataset). Preserved for back-compat.
# New code should use get_base_prediction_length() which matches the GIFT-Eval spec.
FREQ_TO_PRED_LEN = {
    "H": 48, "T": 60, "D": 14, "W": 8, "M": 12, "Q": 8, "Y": 4,
    "B": 14, "5T": 60, "10T": 60, "15T": 48, "30T": 48, "S": 60,
}

# ── GIFT-Eval leaderboard dataset lists ─────────────────────────
# Verbatim from SalesforceAIResearch/gift-eval notebooks/naive.ipynb.
# SHORT_DATASETS: every dataset/freq pair evaluated on the leaderboard.
# MED_LONG_DATASETS: the subset that also gets "medium" and "long" terms.
# Total tasks = len(SHORT) + 2 * len(MED_LONG) = 55 + 2*21 = 97.
SHORT_DATASETS: tuple[str, ...] = (
    "m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly", "m4_daily", "m4_hourly",
    "electricity/15T", "electricity/H", "electricity/D", "electricity/W",
    "solar/10T", "solar/H", "solar/D", "solar/W",
    "hospital", "covid_deaths",
    "us_births/D", "us_births/M", "us_births/W",
    "saugeenday/D", "saugeenday/M", "saugeenday/W",
    "temperature_rain_with_missing",
    "kdd_cup_2018_with_missing/H", "kdd_cup_2018_with_missing/D",
    "car_parts_with_missing", "restaurant",
    "hierarchical_sales/D", "hierarchical_sales/W",
    "LOOP_SEATTLE/5T", "LOOP_SEATTLE/H", "LOOP_SEATTLE/D",
    "SZ_TAXI/15T", "SZ_TAXI/H",
    "M_DENSE/H", "M_DENSE/D",
    "ett1/15T", "ett1/H", "ett1/D", "ett1/W",
    "ett2/15T", "ett2/H", "ett2/D", "ett2/W",
    "jena_weather/10T", "jena_weather/H", "jena_weather/D",
    "bitbrains_fast_storage/5T", "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T", "bitbrains_rnd/H",
    "bizitobs_application", "bizitobs_service",
    "bizitobs_l2c/5T", "bizitobs_l2c/H",
)

MED_LONG_DATASETS: tuple[str, ...] = (
    "electricity/15T", "electricity/H",
    "solar/10T", "solar/H",
    "kdd_cup_2018_with_missing/H",
    "LOOP_SEATTLE/5T", "LOOP_SEATTLE/H",
    "SZ_TAXI/15T",
    "M_DENSE/H",
    "ett1/15T", "ett1/H",
    "ett2/15T", "ett2/H",
    "jena_weather/10T", "jena_weather/H",
    "bitbrains_fast_storage/5T",
    "bitbrains_rnd/5T",
    "bizitobs_application", "bizitobs_service",
    "bizitobs_l2c/5T", "bizitobs_l2c/H",
)

TERMS: tuple[str, str, str] = ("short", "medium", "long")

# Term multiplier applied to the base prediction length.
_TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}

# Native freq for single-freq datasets (those without a `/freq` in the leaderboard
# name). Values from gift-eval notebooks/dataset_properties.json. Keys here are
# the POST-pretty-name form, matching the lowercased leaderboard name after
# _PRETTY_NAMES substitution.
_DATASET_PROPERTIES: dict[str, str] = {
    "m4_yearly": "A",
    "m4_quarterly": "Q",
    "m4_monthly": "M",
    "m4_weekly": "W",
    "m4_daily": "D",
    "m4_hourly": "H",
    "hospital": "M",
    "covid_deaths": "D",
    "us_births": "M",
    "saugeen": "M",
    "temperature_rain": "D",
    "kdd_cup_2018": "D",
    "jena_weather": "D",
    "car_parts": "M",
    "restaurant": "D",
    "hierarchical_sales": "W-WED",
    "loop_seattle": "D",
    "sz_taxi": "H",
    "m_dense": "D",
    "bitbrains_fast_storage": "H",
    "bitbrains_rnd": "H",
    "bizitobs_application": "10S",
    "bizitobs_service": "10S",
    "bizitobs_l2c": "H",
    "electricity": "W",
    "ett1": "W",
    "ett2": "W",
    "solar": "W",
}

# Lowercased-name aliases used by dataset_properties.json.
_PRETTY_NAMES: dict[str, str] = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# M4 base horizons (per M4 competition spec). Used when "m4" in dataset name.
M4_HORIZONS: dict[str, int] = {
    "A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48,
}

# Non-M4 base horizon per freq family (GIFT-Eval canonical PRED_LENGTH_MAP).
_PRED_LENGTH_MAP: dict[str, int] = {
    "M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60,
}

# Seasonality for each freq we may encounter. Values verified against
# gluonts.time_feature.get_seasonality (gluonts==0.16.2):
#   base(S)=3600, base(T)=1440, base(H)=24, base(D)=1, base(W)=1,
#   base(M)=12, base(Q)=4, base(A/Y)=1, base(B)=5
# For multiplied freqs like "5T", gluonts divides base by the multiplier.
SEASONALITY_MAP: dict[str, int] = {
    "S": 3600,
    "10S": 360,      # 3600 / 10
    "T": 1440,
    "1T": 1440,
    "5T": 288,       # 1440 / 5
    "10T": 144,      # 1440 / 10
    "15T": 96,       # 1440 / 15
    "30T": 48,       # 1440 / 30
    "H": 24,
    "6H": 4,         # 24 / 6
    "D": 1,
    "B": 5,
    "W": 1,
    "W-SUN": 1,
    "W-WED": 1,
    "M": 12,
    "MS": 12,
    "Q": 4,
    "Q-DEC": 4,
    "A": 1,
    "A-DEC": 1,
    "Y": 1,
}


def _normalize_freq_base(freq: str) -> str:
    """Strip multiplier prefix and map deprecated codes to legacy ones.

    "15T" → "T", "10S" → "S", "6H" → "H", "W-WED" → "W", "ME" → "M".
    """
    import re
    m = re.match(r"^(\d+)?([A-Za-z]+)", freq)
    if not m:
        return freq
    base = m.group(2)
    # pandas newer codes → gift-eval legacy (from maybe_reconvert_freq)
    reconvert = {
        "Y": "A", "YE": "A", "QE": "Q", "ME": "M",
        "h": "H", "min": "T", "s": "S", "us": "U",
    }
    return reconvert.get(base, base)


def get_seasonality(freq: str) -> int:
    """Return the seasonality period for a freq string. Matches gluonts.

    Unknown freqs fall back to 1 (matches gluonts default).
    """
    if freq in SEASONALITY_MAP:
        return SEASONALITY_MAP[freq]
    # Derive from base + multiplier (mimics gluonts divmod logic).
    import re
    m = re.match(r"^(\d+)?([A-Za-z]+)", freq)
    if not m:
        return 1
    n = int(m.group(1)) if m.group(1) else 1
    base = _normalize_freq_base(freq)
    base_season = SEASONALITY_MAP.get(base, 1)
    season, remainder = divmod(base_season, n)
    return season if remainder == 0 else 1


def get_base_prediction_length(freq: str, dataset_name: str) -> int:
    """Base prediction length for (dataset, freq) per GIFT-Eval spec.

    M4 datasets use M4_HORIZONS, all others use PRED_LENGTH_MAP. Caller
    multiplies by the term multiplier (short=1, medium=10, long=15).
    """
    base = _normalize_freq_base(freq)
    if "m4" in dataset_name.lower():
        return M4_HORIZONS.get(base, 14)
    return _PRED_LENGTH_MAP.get(base, 48)


def _parse_dataset_spec(name: str) -> tuple[str, str]:
    """Split a leaderboard name into (dataset, freq).

    "ett1/15T" → ("ett1", "15T")
    "hospital"  → ("hospital", "M") via _DATASET_PROPERTIES lookup
    """
    if "/" in name:
        dataset, freq = name.split("/", 1)
        return dataset, freq
    key = _PRETTY_NAMES.get(name.lower(), name.lower())
    freq = _DATASET_PROPERTIES.get(key)
    if freq is None:
        raise KeyError(f"No default freq for single-freq dataset: {name!r}")
    return name, freq


def _dataset_key_to_manifest(dataset: str, freq: str | None) -> str:
    """Bridge leaderboard spelling to GIFT_EVAL_MANIFEST key.

    Leaderboard uses "electricity/15T"; manifest uses "electricity__15T".
    For single-freq datasets (e.g. "hospital"), freq is inferred but the
    manifest key has no suffix.
    """
    if freq is not None:
        candidate = f"{dataset}__{freq}"
        if candidate in GIFT_EVAL_MANIFEST:
            return candidate
    if dataset in GIFT_EVAL_MANIFEST:
        return dataset
    # Fall through — caller will surface the KeyError when downloading.
    return f"{dataset}__{freq}" if freq else dataset


def build_task_configs(dataset_names: list[str] | tuple[str, ...]) -> list[dict]:
    """Expand leaderboard dataset names into full task configs.

    Each short-dataset entry yields one "short" task; if the same entry is
    also in MED_LONG_DATASETS, it additionally yields "medium" and "long"
    tasks. Returns list of dicts with keys: dataset, freq, term,
    prediction_length, season_length, config_name.
    """
    med_long = set(MED_LONG_DATASETS)
    tasks: list[dict] = []
    for name in dataset_names:
        dataset, freq = _parse_dataset_spec(name)
        base_pred = get_base_prediction_length(freq, dataset)
        season = get_seasonality(freq)
        for term in TERMS:
            if term != "short" and name not in med_long:
                continue
            pred_len = base_pred * _TERM_MULTIPLIER[term]
            tasks.append({
                "dataset": dataset,
                "freq": freq,
                "term": term,
                "prediction_length": pred_len,
                "season_length": season,
                "config_name": f"{dataset}/{freq}/{term}",
            })
    return tasks


def _has_nan(values: list) -> bool:
    """Check if a list of numeric values contains NaN or None."""
    import math
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return True
    return False


def select_datasets(eval_split_seed: int, n: int) -> list[str]:
    """Deterministic selection of N datasets for a round. 0 = all."""
    if n <= 0 or n >= len(GIFT_EVAL_DATASETS):
        return list(GIFT_EVAL_DATASETS)
    rng = random.Random(eval_split_seed)
    chosen = rng.sample(GIFT_EVAL_DATASETS, n)
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
        """Download Arrow file from R2 to local cache. Skip if cached.

        Uses GIFT_EVAL_MANIFEST to resolve the R2 subpath (some datasets
        have a frequency subdirectory like ett2/H/, some don't).
        """
        if dataset_name not in GIFT_EVAL_MANIFEST:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        local_dir = self.cache_dir / dataset_name
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / "data-00000-of-00001.arrow"

        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path

        if self.r2 is None:
            raise RuntimeError("R2 client required to download GIFT-Eval data")

        subpath = GIFT_EVAL_MANIFEST[dataset_name]
        r2_key = f"{self.r2_prefix}/{subpath}/data-00000-of-00001.arrow"
        if not self.r2.download_file_to_disk(r2_key, str(local_path)):
            raise FileNotFoundError(f"Failed to download {r2_key} from R2")

        size = local_path.stat().st_size
        logger.info("Downloaded %s from %s (%d bytes)", dataset_name, r2_key, size)
        return local_path

    def generate_presigned_get_urls(self, ttl: int = 5400) -> dict[str, str]:
        """Generate presigned GET URLs for all datasets.

        Returns {dataset_key: presigned_url} for passing to trainer pods
        so they can download GIFT-Eval data without R2 credentials.
        """
        if self.r2 is None:
            return {}
        urls = {}
        for name, subpath in GIFT_EVAL_MANIFEST.items():
            r2_key = f"{self.r2_prefix}/{subpath}/data-00000-of-00001.arrow"
            url = self.r2.generate_presigned_get_url(r2_key, ttl=ttl)
            if url:
                urls[name] = url
        logger.info("Generated %d presigned GET URLs for GIFT-Eval data", len(urls))
        return urls


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

        # Skip series with NaN/None — datasets like "car_parts_with_missing"
        # have missing values that poison CRPS computation
        if _has_nan(context) or _has_nan(target):
            continue

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


# ── Rolling-origin window loader (new evaluation path) ──────────────────


_MAX_WINDOWS = 20         # matches gluonts / GIFT-Eval MAX_WINDOW cap
_TEST_SPLIT = 0.1         # fraction of min_series_length used for windowing


def _gift_eval_window_count(
    min_series_length: int, prediction_length: int, dataset_name: str,
) -> int:
    """Compute number of rolling-origin windows per GIFT-Eval spec.

    Matches src/gift_eval/data.py::Dataset.windows:
      - M4 datasets: always 1 window.
      - Others: ceil(TEST_SPLIT * min_series_length / H), clamped to [1, 20].
    """
    if "m4" in dataset_name.lower():
        return 1
    import math as _math
    w = _math.ceil(_TEST_SPLIT * min_series_length / max(prediction_length, 1))
    return min(max(1, w), _MAX_WINDOWS)


def _extract_series(table) -> list[list[float]]:
    """Pull flat 1D series from an Arrow table, handling multivariate flatten."""
    out: list[list[float]] = []
    shape_col = "target._np_shape" if "target._np_shape" in table.column_names else None
    for i in range(table.num_rows):
        target_col = table.column("target")[i].as_py()
        values = target_col if isinstance(target_col, list) else list(target_col)
        if shape_col is not None:
            shape = table.column(shape_col)[i].as_py()
            if shape and len(shape) >= 2:
                n_variates = shape[-1]
                if n_variates > 1:
                    series_len = len(values) // n_variates
                    values = values[:series_len]
        if _has_nan(values):
            continue
        out.append(values)
    return out


def load_dataset_for_task(
    task: dict,
    context_len: int,
    cache_dir: str,
    max_series: int = 0,
) -> list[dict]:
    """Build rolling-origin evaluation windows for a single task.

    Window count is a DATASET-level property matching the GIFT-Eval benchmark:
      windows = ceil(0.1 * min_series_length / H), clamped to [1, 20]
      (M4 datasets fixed at 1).
    Every series in the dataset uses this same window count; any series too
    short to support it is skipped so the eval stays apples-to-apples.

    Context of `context_len` is taken immediately before each target. If the
    history preceding the context is shorter than context_len, the context is
    left-padded with zeros (matches predictor internal padding in gluonts).

    Returns [{context, target, history, freq}] where `history` is every
    value before the target (used for seasonal-naive + MASE scale).
    """
    import pyarrow.ipc as ipc

    dataset, freq = task["dataset"], task["freq"]
    manifest_key = _dataset_key_to_manifest(dataset, freq)
    # path-traversal guard (manifest keys are controlled, but be defensive)
    if "/" in manifest_key or "\\" in manifest_key or ".." in manifest_key:
        raise ValueError(f"Invalid manifest key: {manifest_key!r}")
    arrow_path = Path(cache_dir) / manifest_key / "data-00000-of-00001.arrow"
    if not arrow_path.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_path}")

    reader = ipc.open_file(str(arrow_path))
    table = reader.read_all()
    raw_series = _extract_series(table)
    if not raw_series:
        return []

    H = int(task["prediction_length"])
    if H <= 0:
        return []

    # Dataset-level window count, matching GIFT-Eval Dataset.windows.
    min_L = min(len(s) for s in raw_series)
    n_windows = _gift_eval_window_count(min_L, H, dataset)

    samples: list[dict] = []
    for series in raw_series:
        L = len(series)
        # Skip series too short to support the full rolling evaluation.
        if L < n_windows * H:
            continue
        for i in range(n_windows):
            end = L - i * H
            target = series[end - H:end]
            history = series[:end - H]
            if len(history) >= context_len:
                context = history[-context_len:]
            else:
                context = [0.0] * (context_len - len(history)) + history
            samples.append({
                "context": context,
                "target": target,
                "history": history,
                "freq": freq,
            })

    # Optional deterministic subsample AFTER windowing so every validator sees
    # the same subset. Default 0 = keep all.
    if max_series and len(samples) > max_series:
        rng = random.Random(hash((task.get("config_name", ""), max_series)))
        samples = rng.sample(samples, max_series)

    return samples


def get_eval_batches_with_history(
    samples: list[dict],
    batch_size: int = 32,
):
    """Batch rolling-origin samples. Yields dicts with context/target tensors
    plus a list-of-lists `history` (variable length per series).
    """
    import torch

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        ctx = torch.tensor(
            [s["context"] for s in batch], dtype=torch.float32,
        ).unsqueeze(-1)
        tgt = torch.tensor(
            [s["target"] for s in batch], dtype=torch.float32,
        ).unsqueeze(-1)
        yield {
            "context": ctx,
            "target": tgt,
            "history": [s["history"] for s in batch],
            "freq": batch[0].get("freq", ""),
        }
