"""FLOPs-equivalent measurement via wallclock calibration.

Used by both the trainer (Phase B) and validator evaluator (Phase C) to
measure a model's compute cost in a hardware-independent way.

The approach: time a reference dense transformer with known FLOPs count,
then measure the target model's forward-pass time relative to that reference.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Cached calibration result per device
_calibration_cache: dict[str, float] = {}


class _ReferenceDenseTransformer(nn.Module):
    """Small dense transformer for calibration. Known FLOPs count."""

    def __init__(self, d_model: int = 256, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.proj_in = nn.Linear(7, d_model)  # num_variates -> d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj_out = nn.Linear(d_model, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.encoder(self.proj_in(x)))

    @property
    def known_flops(self) -> int:
        """Approximate FLOPs for one forward pass with seq_len=512, batch=1."""
        seq_len = 512
        d = self.d_model
        # Linear projections: 2 * seq * d * 7
        proj_flops = 2 * seq_len * d * 7 * 2
        # Self-attention per layer: 4 * seq * d^2 + 2 * seq^2 * d
        attn_flops = (4 * seq_len * d * d + 2 * seq_len * seq_len * d)
        # FFN per layer: 2 * seq * d * 4d * 2
        ffn_flops = 2 * seq_len * d * 4 * d * 2
        num_layers = 2
        return proj_flops + num_layers * (attn_flops + ffn_flops)


_MEASURE_BATCH = 32  # Large enough to amortize CUDA kernel launch overhead


def _trimmed_mean(values: list[float], trim_fraction: float = 0.2) -> float:
    """Compute trimmed mean, dropping top/bottom trim_fraction of values."""
    s = sorted(values)
    n = len(s)
    trim_count = max(1, int(n * trim_fraction))
    trimmed = s[trim_count: n - trim_count]
    if not trimmed:
        trimmed = s  # fallback if too few values
    return sum(trimmed) / len(trimmed)


def calibrate(device: str = "cpu", seq_len: int = 512, num_variates: int = 7) -> float:
    """Run reference model with known FLOPs, return FLOPS_PER_SEC for this device.

    Uses batch_size=_MEASURE_BATCH to amortize per-op overhead (CUDA kernel
    launches, Python interpreter, etc.) so the ratio reflects actual compute
    rather than fixed overhead.

    Result is cached per device string.
    """
    if device in _calibration_cache:
        return _calibration_cache[device]

    ref = _ReferenceDenseTransformer().to(device).eval()
    dummy = torch.randn(_MEASURE_BATCH, seq_len, num_variates, device=device)
    known_flops_batch = ref.known_flops * _MEASURE_BATCH

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            ref(dummy)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(20):
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            ref(dummy)
            if device != "cpu":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    trimmed_time = _trimmed_mean(times, trim_fraction=0.2)
    flops_per_sec = known_flops_batch / max(trimmed_time, 1e-9)
    _calibration_cache[device] = flops_per_sec
    logger.info("Calibrated %s: %.2e FLOPS/sec (trimmed_mean=%.4fs, batch=%d)", device, flops_per_sec, trimmed_time, _MEASURE_BATCH)
    return flops_per_sec


def compute_flops_analytical(
    model: nn.Module,
    context_len: int,
    num_variates: int,
) -> int | None:
    """Analytical FLOPs counting via torch.utils.flop_counter.

    Returns integer FLOPs count for one forward pass, or None if
    the model uses ops that can't be counted analytically.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        dummy = torch.randn(1, context_len, num_variates)
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            model(dummy)
        total = flop_counter.get_total_flops()
        if total > 0:
            return int(total)
        return None
    except Exception:
        return None


def compute_flops_equivalent(
    model: nn.Module,
    context_len: int,
    num_variates: int,
    device: str = "cpu",
) -> int:
    """Measure FLOPs — analytical counting preferred, wallclock fallback.

    Analytical counting is exact for standard nn.Module ops.
    Wallclock calibration is used only when analytical counting fails
    (custom CUDA kernels, dynamic control flow, etc.)
    """
    # Try analytical first (device-independent, exact)
    analytical = compute_flops_analytical(model, context_len, num_variates)
    if analytical is not None:
        logger.info("FLOPs (analytical): %d", analytical)
        return analytical

    # Fallback to wallclock calibration
    logger.info("Analytical FLOPs failed, falling back to wallclock calibration")
    flops_per_sec = calibrate(device)
    dummy = torch.randn(_MEASURE_BATCH, context_len, num_variates, device=device)

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(20):
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device != "cpu":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    trimmed_time = _trimmed_mean(times, trim_fraction=0.2)
    total_flops = flops_per_sec * trimmed_time
    return int(total_flops / _MEASURE_BATCH)
