"""Tiny CPU regression task for the local stack.

We mirror the real radar shape (Phase A → B → C, FLOPs-equivalent size
buckets, validator-side eval as trust anchor) on a problem that runs in
seconds on a laptop with nothing but numpy.

The miner's submission must define::

    def build_model(input_dim, output_dim):
        return Model

Where ``Model`` exposes:

* ``hidden_sizes`` — list[int], the hidden layer widths
* ``activation`` — str, one of {"relu", "tanh"}
* ``learning_rate`` — float
* ``epochs`` — int (capped by the harness)

The harness builds the actual numpy MLP, trains it under a fixed
gradient-descent loop, and reports MSE on a held-out test split.
Restricting the surface to a config object (not raw forward/backward)
means we don't have to ship a numpy autograd, but the miner still has
real design choices — width, depth, nonlinearity, optimization budget.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

INPUT_DIM = 8
OUTPUT_DIM = 1
N_TRAIN = 2000
N_TEST = 500
DATA_SEED = 12345

MAX_HIDDEN_LAYERS = 4
MAX_HIDDEN_WIDTH = 128
MAX_EPOCHS = 200


@dataclass
class TaskSpec:
    name: str = "synth_regression"
    input_dim: int = INPUT_DIM
    output_dim: int = OUTPUT_DIM
    n_train: int = N_TRAIN
    n_test: int = N_TEST
    data_seed: int = DATA_SEED
    max_hidden_layers: int = MAX_HIDDEN_LAYERS
    max_hidden_width: int = MAX_HIDDEN_WIDTH
    max_epochs: int = MAX_EPOCHS


# Size buckets in "FLOPs-equivalent" units (= 2 * forward-pass mac count
# per sample, summed across layers). Real radar uses analytical FLOPs
# counting; we approximate with parameter count × 2 for an MLP.
SIZE_BUCKETS: dict[str, tuple[int, int]] = {
    "tiny":   (200,    2_000),
    "small":  (2_000,  10_000),
    "medium": (10_000, 50_000),
    "large":  (50_000, 200_000),
}


def make_dataset(seed: int = DATA_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic regression: a smooth nonlinear target with light noise.

    Anyone who learns the right hidden representation will get close to
    the noise floor; tiny linear models will plateau well above it.
    """
    rng = np.random.default_rng(seed)
    X_all = rng.standard_normal((N_TRAIN + N_TEST, INPUT_DIM)).astype(np.float32)

    # Target: ground-truth is a small MLP. This guarantees there *is* a
    # learnable structure rather than testing the miner's ability to
    # overfit noise.
    W1 = rng.standard_normal((INPUT_DIM, 16)).astype(np.float32) * 0.5
    b1 = rng.standard_normal(16).astype(np.float32) * 0.1
    W2 = rng.standard_normal((16, OUTPUT_DIM)).astype(np.float32) * 0.5
    h = np.tanh(X_all @ W1 + b1)
    y_all = h @ W2 + rng.standard_normal((N_TRAIN + N_TEST, OUTPUT_DIM)).astype(np.float32) * 0.05

    X_tr, X_te = X_all[:N_TRAIN], X_all[N_TRAIN:]
    y_tr, y_te = y_all[:N_TRAIN], y_all[N_TRAIN:]
    return X_tr, y_tr, X_te, y_te


def estimate_flops_equivalent(hidden_sizes: list[int]) -> int:
    """Param count × 2 ≈ FLOPs per forward pass per sample.

    Matches the "FLOPs-equivalent" framing in shared/scoring.py without
    pulling in torch's flop_counter.
    """
    dims = [INPUT_DIM, *hidden_sizes, OUTPUT_DIM]
    params = 0
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        params += d_in * d_out + d_out  # weights + bias
    return 2 * params
