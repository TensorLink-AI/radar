"""Fixed reference architecture for the ``synthetic_data_generator`` task.

Unlike ``ts_data_pipeline`` — whose frozen architecture is the *current best*
ts_forecasting model and is re-snapshotted every N rounds — the
synthetic-data-generator task pins a **single, never-changing** model. Miners
compete purely on the synthetic data they feed it; the architecture is the same
``code`` every round, so a continuation round is just "keep training the same
weights on (maybe better) data".

The reference model is a miniature (~10M-parameter) Toto-2.0-style **causal
patch decoder**: contiguous non-overlapping patches over the context window with
an arcsinh robust scaler and missing-value masking, residual-MLP patch
projections, a stack of causal (decoder) transformer blocks with PerDimScale
attention, and learned forecast-query tokens that emit the horizon as contiguous
output patches in a single pass. The source lives in
``local/synthetic_arch_code.py`` as *text* (not an imported module) so the
validator runs with only numpy installed — torch is only imported when the
harness ``exec``s the string inside a synthetic_data_generator round.

The ``code`` satisfies the same ``build_model(context_len, prediction_len,
num_variates, quantiles)`` contract every ts_forecasting submission does, so the
existing harness / GIFT-Eval path drives it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from local.synthetic_arch_code import REFERENCE_ARCH_CODE

# Bumping this only makes sense if the reference model itself changes; it is
# stamped onto every experiment's objectives (``synth_arch_version``) and gates
# continuation lineages so a future model swap can't silently stitch
# incomparable checkpoints together.
#
# NOTE: the reference model was upgraded toward Toto-2.0 (arcsinh robust scaler,
# residual-MLP patch projections, missing-value masking, single-pass multi-patch
# forecast head, u-μP/PerDimScale). That is a parameter-incompatible change, so
# this version MUST be bumped to 2 before the new arch is merged — left at 1 here
# pending that decision so continuation lineages aren't silently stitched across
# the model swap.
REFERENCE_ARCH_VERSION = 1
REFERENCE_ARCH_NAME = "toto_mini_patch_decoder_10m"


@dataclass
class ReferenceArch:
    """The fixed architecture every synthetic_data_generator round trains.

    Shaped to plug into the same ``frozen_arch`` challenge card / miner tools
    the ts_data_pipeline task already uses, so the miner-facing surface is
    identical — only the validator-side scoring and the (never-changing)
    version differ.
    """

    version: int
    name: str
    code: str

    def to_card(self) -> dict:
        """Challenge-payload ``frozen_arch`` card (mirrors FrozenArch fields)."""
        return {
            "version": self.version,
            "source_experiment_id": 0,
            "source_metric": 0.0,
            "source_crps": 0.0,
            "source_mase": 0.0,
            "source_flops": 0,
            "source_name": self.name,
            "code": self.code,
            "fixed": True,
        }


def reference_arch() -> ReferenceArch:
    """Return the single fixed reference architecture (cheap; no torch)."""
    return ReferenceArch(
        version=REFERENCE_ARCH_VERSION,
        name=REFERENCE_ARCH_NAME,
        code=REFERENCE_ARCH_CODE,
    )
