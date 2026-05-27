"""In-process trainer for the local stack.

Mirrors Phase B (train) + Phase C (evaluate) of the real validator,
collapsed into one function because there's no need to dispatch over
HTTP when everything is on the same machine. Returns the metrics
``shared/scoring.py`` would compute against — but we score with our
own minimal helpers in ``local/scoring.py``.

The submission is plain Python source: we ``exec`` it in a sandboxed
namespace (no real isolation — single-laptop only, you trust your own
miner). It must define ``build_model(input_dim, output_dim)`` returning
a config object with ``hidden_sizes``, ``activation``, ``learning_rate``,
``epochs``.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any

import numpy as np

from local.task import (
    MAX_EPOCHS, MAX_HIDDEN_LAYERS, MAX_HIDDEN_WIDTH,
    INPUT_DIM, OUTPUT_DIM, estimate_flops_equivalent, make_dataset,
)

logger = logging.getLogger(__name__)


def _exec_submission(code: str) -> Any:
    """Run the miner's code in a fresh namespace; return ``build_model``."""
    ns: dict[str, Any] = {"__name__": "submission"}
    exec(code, ns)
    bm = ns.get("build_model")
    if not callable(bm):
        raise ValueError("submission.code must define build_model(input_dim, output_dim)")
    return bm


def _clamp_config(cfg: Any) -> dict:
    hidden_sizes = list(getattr(cfg, "hidden_sizes", []) or [])
    if not hidden_sizes:
        raise ValueError("Model.hidden_sizes must be a non-empty list of ints")
    hidden_sizes = hidden_sizes[:MAX_HIDDEN_LAYERS]
    hidden_sizes = [max(1, min(MAX_HIDDEN_WIDTH, int(h))) for h in hidden_sizes]

    activation = str(getattr(cfg, "activation", "relu") or "relu").lower()
    if activation not in {"relu", "tanh"}:
        activation = "relu"

    lr = float(getattr(cfg, "learning_rate", 1e-2) or 1e-2)
    lr = max(1e-5, min(1.0, lr))

    epochs = int(getattr(cfg, "epochs", 50) or 50)
    epochs = max(1, min(MAX_EPOCHS, epochs))

    return {
        "hidden_sizes": hidden_sizes,
        "activation": activation,
        "learning_rate": lr,
        "epochs": epochs,
    }


def _init_params(hidden_sizes: list[int], rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    dims = [INPUT_DIM, *hidden_sizes, OUTPUT_DIM]
    params: list[tuple[np.ndarray, np.ndarray]] = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        # He init for relu, Xavier for tanh — close enough for both.
        scale = np.sqrt(2.0 / d_in)
        W = (rng.standard_normal((d_in, d_out)) * scale).astype(np.float32)
        b = np.zeros(d_out, dtype=np.float32)
        params.append((W, b))
    return params


def _forward(params, x, activation):
    """Forward through the MLP; record post-activation tensors for backward."""
    acts = [x]
    pre = []
    n = len(params)
    a = x
    for i, (W, b) in enumerate(params):
        z = a @ W + b
        pre.append(z)
        if i < n - 1:
            a = np.maximum(z, 0) if activation == "relu" else np.tanh(z)
        else:
            a = z  # linear output for regression
        acts.append(a)
    return acts, pre


def _backward(params, acts, pre, y_true, activation):
    """Compute MSE gradient by hand. Avoids depending on torch."""
    n = len(params)
    grads = [None] * n
    # dL/dy_pred for MSE = 2 * (y_pred - y_true) / N
    y_pred = acts[-1]
    batch = y_true.shape[0]
    delta = (2.0 / batch) * (y_pred - y_true)

    for i in reversed(range(n)):
        a_prev = acts[i]
        W = params[i][0]
        # dW, db
        dW = a_prev.T @ delta
        db = delta.sum(axis=0)
        grads[i] = (dW, db)
        if i > 0:
            # backprop through activation of layer i-1
            da_prev = delta @ W.T
            z_prev = pre[i - 1]
            if activation == "relu":
                delta = da_prev * (z_prev > 0)
            else:
                delta = da_prev * (1.0 - np.tanh(z_prev) ** 2)
    return grads


def _mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def run_training(code: str, *, seed: int = 0, log_every: int = 5) -> dict:
    """Phase B + Phase C, in one function. Returns the experiment record.

    Result keys:
      ``success`` (bool), ``metric`` (MSE on test, lower is better, NaN
      on failure), ``objectives`` (dict with ``flops_equivalent_size``,
      ``num_params``, ``train_seconds``), ``loss_curve`` (list[float],
      one entry per logged epoch), ``analysis`` (str), ``error`` (str
      on failure).
    """
    started = time.time()
    try:
        build_model = _exec_submission(code)
        cfg_obj = build_model(INPUT_DIM, OUTPUT_DIM)
        cfg = _clamp_config(cfg_obj)
    except Exception as e:  # noqa: BLE001 — surface the failure mode
        return {
            "success": False,
            "metric": None,
            "objectives": {"flops_equivalent_size": 0, "num_params": 0,
                            "train_seconds": 0.0},
            "loss_curve": [],
            "analysis": "submission failed to load",
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
        }

    rng = np.random.default_rng(seed)
    params = _init_params(cfg["hidden_sizes"], rng)
    activation = cfg["activation"]
    lr = cfg["learning_rate"]
    epochs = cfg["epochs"]
    num_params = sum(W.size + b.size for W, b in params)
    flops_equiv = estimate_flops_equivalent(cfg["hidden_sizes"])

    X_tr, y_tr, X_te, y_te = make_dataset()

    loss_curve: list[float] = []
    try:
        for ep in range(epochs):
            acts, pre = _forward(params, X_tr, activation)
            train_loss = _mse(acts[-1], y_tr)
            if not np.isfinite(train_loss):
                raise FloatingPointError(f"non-finite train loss at epoch {ep}")
            grads = _backward(params, acts, pre, y_tr, activation)
            for i, (g_W, g_b) in enumerate(grads):
                W, b = params[i]
                params[i] = (W - lr * g_W, b - lr * g_b)
            if ep % log_every == 0 or ep == epochs - 1:
                loss_curve.append(train_loss)

        # Phase C — evaluate on held-out test split (trust anchor).
        test_pred, _ = _forward(params, X_te, activation)
        test_mse = _mse(test_pred[-1], y_te)
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "metric": None,
            "objectives": {"flops_equivalent_size": flops_equiv,
                            "num_params": num_params,
                            "train_seconds": time.time() - started},
            "loss_curve": loss_curve,
            "analysis": "training crashed",
            "error": f"{type(e).__name__}: {e}",
        }

    train_seconds = time.time() - started
    return {
        "success": True,
        "metric": test_mse,
        "objectives": {
            "flops_equivalent_size": flops_equiv,
            "num_params": num_params,
            "train_seconds": train_seconds,
        },
        "loss_curve": loss_curve,
        "analysis": (
            f"hidden_sizes={cfg['hidden_sizes']} act={activation} "
            f"lr={lr:.4g} epochs={epochs} params={num_params} "
            f"flops_equiv={flops_equiv}"
        ),
        "error": "",
    }
