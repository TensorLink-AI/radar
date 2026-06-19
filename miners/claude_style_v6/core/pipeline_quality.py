"""Distribution-quality stats for v6's pipeline smoke test.

Split out of ``core.pipeline_probe`` to keep that module under the 300-line
cap. Given the batches a miner's ``build_pipeline`` produced, compute cheap
descriptive statistics (per-channel variance, constant fraction, lag-1
autocorrelation, value range, cross-batch variety) and flag degenerate
generators. The validator scores generators on held-out GIFT-Eval, where a
collapsed / zero-variance generator trains smoothly and then fails the
gate — surfacing the collapse in-agent lets the designer fix it before
burning a round. Every function is best-effort and never raises.
"""

from __future__ import annotations

from typing import Any


# Quality thresholds — conservative so warnings flag genuine degeneracy,
# not merely "low-noise". Each is gated on a known-bad signature.
_CONSTANT_STD_EPS = 1e-6      # a series with std below this is constant
_NEAR_ZERO_STD = 1e-3        # mean per-channel std below this ⇒ near-flat
_CONSTANT_FRAC_WARN = 0.5    # >half the channels constant ⇒ warn
_HIGH_AUTOCORR = 0.999       # lag-1 ac this high + low noise ⇒ too smooth


def _to_numpy(x):
    """Best-effort conversion of a batch tensor to a float32 numpy array.

    Returns ``None`` when numpy is unavailable or the object can't be
    coerced — callers treat that as "stats unavailable", never an error.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().to("cpu", dtype=torch.float32).numpy()
    except ImportError:
        pass
    try:
        return np.asarray(x, dtype="float32")
    except Exception:  # noqa: BLE001
        return None


def quality_stats(
    inputs: list, targets: list,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Cheap distribution stats + degeneracy warnings over pulled batches.

    Operates on the full series (context ⊕ horizon) per batch. Never
    raises — returns ``(None, [])`` if numpy is missing or the arrays
    can't be analysed, and otherwise ``(stats, warnings)``.
    """
    try:
        import numpy as np
    except ImportError:
        return None, []

    full: list = []
    for inp, tgt in zip(inputs, targets):
        a = _to_numpy(inp)
        b = _to_numpy(tgt)
        if a is None or b is None or a.ndim != 3 or b.ndim != 3:
            continue
        if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
            continue
        full.append(np.concatenate([a, b], axis=1))  # (B, ctx+pred, V)
    if not full:
        return None, []

    try:
        warnings: list[str] = []
        flat = np.concatenate([f.reshape(-1) for f in full])
        finite_frac = float(np.isfinite(flat).mean())
        if finite_frac < 1.0:
            warnings.append(
                f"{(1 - finite_frac) * 100:.1f}% of values are non-finite "
                "(nan/inf). The validator sanitizes these to 0, but it "
                "signals an overflow/zero-division in the generator — fix "
                "the cast or clamp before yielding."
            )

        # Per-series stats: stack batches → (N, T, V), one series per (n, v).
        series = np.concatenate(full, axis=0).astype("float64")
        series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
        std_ps = series.std(axis=1)            # (N, V)
        mean_std = float(np.mean(std_ps))
        min_std = float(np.min(std_ps))
        constant_frac = float(np.mean(std_ps < _CONSTANT_STD_EPS))

        # Lag-1 autocorrelation per series, averaged (temporal structure).
        xm = series - series.mean(axis=1, keepdims=True)
        num = (xm[:, :-1, :] * xm[:, 1:, :]).sum(axis=1)
        den = (xm * xm).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ac = np.where(den > 1e-12, num / den, 0.0)
        mean_ac = float(np.mean(ac))

        gmin, gmax = float(series.min()), float(series.max())
        mean_abs = float(np.mean(np.abs(series)))

        # Cross-batch variety — does each draw differ? Identical batches
        # (fixed seed not advanced) collapse the effective dataset to one.
        variety: float | None = None
        identical = False
        if len(full) > 1:
            if all(f.shape == full[0].shape for f in full):
                identical = all(np.array_equal(full[0], f) for f in full[1:])
            batch_means = np.array([f.mean() for f in full])
            spread = float(np.std(batch_means))
            scale = abs(float(np.mean(batch_means))) + 1e-8
            variety = spread / scale

        # ── Degeneracy warnings ──────────────────────────────────────
        if mean_std < _NEAR_ZERO_STD:
            warnings.append(
                f"near-constant output (mean per-channel std={mean_std:.2e}). "
                "The model will memorize one flat mode and collapse on the "
                "GIFT-Eval gate — inject real variation (trend, seasonality, "
                "noise)."
            )
        if constant_frac > _CONSTANT_FRAC_WARN:
            warnings.append(
                f"{constant_frac * 100:.0f}% of channel-series are exactly "
                "constant. Make every variate carry signal, or reduce "
                "num_variates of dead channels."
            )
        if identical:
            warnings.append(
                "every pulled batch is byte-identical — the iterator yields "
                "the same data each step. Advance the RNG per batch so the "
                "model sees a real distribution, not one repeated sample."
            )
        if mean_ac > _HIGH_AUTOCORR and mean_std > _NEAR_ZERO_STD:
            warnings.append(
                f"series are near-perfectly smooth (lag-1 autocorr="
                f"{mean_ac:.4f}). Clean signals let the arch overfit a "
                "single mode; add observation noise / heteroscedasticity."
            )

        stats: dict[str, Any] = {
            "finite_fraction": round(finite_frac, 4),
            "mean_channel_std": round(mean_std, 4),
            "min_channel_std": round(min_std, 4),
            "constant_channel_fraction": round(constant_frac, 4),
            "mean_lag1_autocorr": round(mean_ac, 4),
            "value_min": round(gmin, 4),
            "value_max": round(gmax, 4),
            "mean_abs": round(mean_abs, 4),
        }
        if variety is not None:
            stats["cross_batch_variety"] = round(variety, 4)
            stats["batches_identical"] = identical
        return stats, warnings
    except Exception:  # noqa: BLE001
        # Stats are best-effort — never let them turn a passing smoke test
        # into a failure.
        return None, []
