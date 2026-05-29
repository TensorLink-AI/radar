"""In-process trainer for the local stack.

Mirrors Phase B (train) + Phase C (evaluate) of the real validator,
collapsed into one function because there's no need to dispatch over
HTTP when everything is on the same machine. Returns the metrics
``shared/scoring.py`` would compute against — but we score with our
own minimal helpers in ``local/scoring.py``.

The submission is plain Python source: we ``exec`` it in a sandboxed
namespace (no real isolation — single-laptop only, you trust your own
miner).

Two dispatch paths:

* ``synth_regression`` (default) — numpy MLP, the submission must
  define ``build_model(input_dim, output_dim)`` returning a config
  object with ``hidden_sizes`` / ``activation`` / ``learning_rate`` /
  ``epochs``.

* ``ts_forecasting`` — torch model, the submission must define
  ``build_model(context_len, prediction_len, num_variates, quantiles)``
  returning an ``nn.Module``. The harness in ``runner.harness`` owns
  the loop; it can pretrain on parquet shards under
  ``$RADAR_PRETRAIN_CACHE`` and eval on GIFT-Eval Arrow data under
  ``$RADAR_GIFT_EVAL_CACHE``.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from local.task import (
    MAX_EPOCHS, MAX_HIDDEN_LAYERS, MAX_HIDDEN_WIDTH,
    INPUT_DIM, OUTPUT_DIM, estimate_flops_equivalent, make_dataset,
    TSForecastingSpec,
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


def run_training(
    code: str,
    *,
    seed: int = 0,
    log_every: int = 5,
    task: Any = None,
    min_flops: int = 0,
    max_flops: int = 0,
    parent_checkpoint_path: str | None = None,
    compute_offset: float = 0.0,
    step_offset: int = 0,
    shard_paths: list[str] | None = None,
    shard_reuse: bool = False,
) -> dict:
    """Phase B + Phase C, in one function. Returns the experiment record.

    When ``task`` is a ``TSForecastingSpec`` the call is forwarded to
    ``runner.harness.run_training`` (torch pretrain + eval). Otherwise
    the legacy numpy MLP path runs.

    Result keys:
      ``success`` (bool), ``metric`` (MSE on test, lower is better, NaN
      on failure), ``objectives`` (dict with ``flops_equivalent_size``,
      ``num_params``, ``train_seconds``), ``loss_curve`` (list[float],
      one entry per logged epoch), ``analysis`` (str), ``error`` (str
      on failure).
    """
    if isinstance(task, TSForecastingSpec):
        return _run_ts_forecasting(
            code, seed=seed, task=task,
            min_flops=min_flops, max_flops=max_flops,
            parent_checkpoint_path=parent_checkpoint_path,
            compute_offset=compute_offset, step_offset=step_offset,
            shard_paths=shard_paths, shard_reuse=shard_reuse,
        )
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
            "workdir": "",
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
            "workdir": "",
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
        "workdir": "",
    }


# ── ts_forecasting dispatch (torch path) ────────────────────────────


def _ts_runner_paths_setup() -> None:
    """The frozen runner uses sibling-style imports (``from prepare import
    ...``). Add the package dir to sys.path so it resolves when the harness
    is driven from outside the sandbox.
    """
    pkg = Path(__file__).resolve().parent.parent / "runner" / "timeseries_forecast"
    p = str(pkg)
    if p not in sys.path:
        sys.path.insert(0, p)


def _run_ts_forecasting(
    code: str,
    *,
    seed: int,
    task: TSForecastingSpec,
    min_flops: int,
    max_flops: int,
    parent_checkpoint_path: str | None = None,
    compute_offset: float = 0.0,
    step_offset: int = 0,
    shard_paths: list[str] | None = None,
    shard_reuse: bool = False,
) -> dict:
    """Drive ``runner.harness.run_training`` against a ts_forecasting submission.

    Sets the env vars the harness reads (CHECKPOINT_DIR, SUBMISSION_PATH,
    RADAR_GIFT_EVAL_CACHE, pretrain shard lists) and translates the harness
    result dict into the shape ``local/validator.py`` already expects.

    For continuation runs ``parent_checkpoint_path`` is forwarded to the
    harness (warm-start), and ``compute_offset`` / ``step_offset`` shift the
    recorded curve coordinates into lineage-absolute space so a stitched
    trajectory is monotonic. ``shard_paths`` overrides the cache glob with
    the validator's lineage-disjoint assignment.
    """
    started = time.time()
    _ts_runner_paths_setup()

    try:
        from runner.harness import TrainingConfig, run_training as harness_run
        from runner.timeseries_forecast.train import TSForecastingRunner
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "metric": None,
            "objectives": {"flops_equivalent_size": 0, "num_params": 0,
                           "train_seconds": 0.0},
            "loss_curve": [],
            "analysis": "ts_forecasting runner unavailable",
            "error": f"{type(e).__name__}: {e} (install torch + the [gift_eval] extra)",
            "workdir": "",
        }

    workdir = Path(tempfile.mkdtemp(prefix="radar_ts_"))
    submission_path = workdir / "submission.py"
    checkpoint_dir = workdir / "checkpoints"
    logs_dir = workdir / "logs"
    for d in (checkpoint_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # The harness reads these from os.environ. Keep originals to restore.
    cache_dir = os.environ.get("RADAR_GIFT_EVAL_CACHE", "/tmp/radar_gift_eval")
    pretrain_cache = os.environ.get("RADAR_PRETRAIN_CACHE", "/tmp/radar_pretrain")
    # The validator passes a lineage-disjoint shard assignment for
    # continuations; otherwise glob the cache (legacy fresh-run behaviour).
    if shard_paths is None:
        shard_paths = []
        if Path(pretrain_cache).is_dir():
            shard_paths = sorted(
                str(p) for p in Path(pretrain_cache).glob("*.parquet")
            )

    overrides = {
        "CHECKPOINT_DIR": str(checkpoint_dir),
        "SUBMISSION_PATH": str(submission_path),
        "RADAR_GIFT_EVAL_CACHE": cache_dir,
        "RADAR_STEP_OFFSET": str(int(step_offset)),
        "RADAR_FLOPS_OFFSET": str(int(compute_offset)),
    }
    if parent_checkpoint_path:
        overrides["PARENT_CHECKPOINT_PATH"] = str(parent_checkpoint_path)
    train_paths: list[str] = []
    if shard_paths:
        import json as _json
        # Reserve the last shard as val when we have >=2; rest is train.
        if len(shard_paths) >= 2:
            train_paths, val_paths = shard_paths[:-1], shard_paths[-1:]
        else:
            train_paths, val_paths = shard_paths, []
        overrides["RADAR_PRETRAIN_LOCAL_PATHS"] = _json.dumps(train_paths)
        if val_paths:
            overrides["RADAR_PRETRAIN_VAL_LOCAL_PATHS"] = _json.dumps(val_paths)

    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    # Capture harness/runner Python logging into a file so the validator
    # has a real training log to mirror as an artifact (the harness only
    # logs to stdout otherwise).
    train_log_path = logs_dir / "train.log"
    file_handler = logging.FileHandler(train_log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    capture_loggers = [
        logging.getLogger("runner"),
        logging.getLogger("runner.harness"),
        logging.getLogger("runner.timeseries_forecast"),
    ]
    for lg in capture_loggers:
        lg.addHandler(file_handler)
        # Make sure INFO-level records reach the handler even if the root
        # logger is set higher.
        if lg.level == 0 or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
    try:
        config = TrainingConfig(
            seed=seed,
            round_id=seed,
            min_flops=min_flops,
            max_flops=max_flops,
            submission_id=f"local-{seed}",
            time_budget=int(task.time_budget_seconds),
        )
        runner = TSForecastingRunner()
        result = harness_run(runner, code, config)
    except Exception as e:  # noqa: BLE001
        for lg in capture_loggers:
            lg.removeHandler(file_handler)
        file_handler.close()
        return {
            "success": False,
            "metric": None,
            "objectives": {"flops_equivalent_size": 0, "num_params": 0,
                           "train_seconds": time.time() - started},
            "loss_curve": [],
            "analysis": "ts_forecasting harness crashed",
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
            "workdir": str(workdir),
        }
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    for lg in capture_loggers:
        lg.removeHandler(file_handler)
    file_handler.close()

    # Dump the raw harness result (val cadence, peak VRAM, val history,
    # etc.) so miners can mine richer training telemetry than what fits
    # into the validator's experiment row.
    try:
        import json as _json
        (logs_dir / "harness_result.json").write_text(
            _json.dumps(result, indent=2, default=str), encoding="utf-8",
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("could not write harness_result.json: %s", e)

    # Translate harness result → local validator shape.
    status = result.get("status", "")
    success = status in ("ok", "completed", "success")
    flops_equiv = int(result.get("flops_equivalent_size") or 0)
    num_params = int(round(float(result.get("num_params_M") or 0.0) * 1_000_000))
    train_seconds = float(
        result.get("training_time_seconds")
        or result.get("train_seconds")
        or (time.time() - started)
    )
    train_hist = result.get("train_loss_history") or []
    val_hist = result.get("val_loss_history") or []
    loss_curve = [float(x.get("loss", 0.0)) for x in train_hist]
    val_losses = [float(x.get("loss", 0.0)) for x in val_hist if x.get("loss") is not None]
    best_val_loss = (
        float(result["best_val_loss"])
        if result.get("best_val_loss") is not None
        else (val_losses[-1] if val_losses else (loss_curve[-1] if loss_curve else None))
    )

    # Training compute for this run (Σ FLOPs the harness accrued), summed
    # with the lineage offset → cumulative_compute, the continuation
    # frontier's x-axis. Shard provenance lets a child exclude what its
    # ancestors trained on.
    this_compute = float(result.get("this_run_flops") or 0.0)
    cumulative_compute = float(
        result.get("cumulative_flops")
        if result.get("cumulative_flops") is not None
        else float(compute_offset) + this_compute
    )
    objectives = {
        "flops_equivalent_size": flops_equiv,
        "num_params": num_params,
        "train_seconds": train_seconds,
        "this_compute": this_compute,
        "cumulative_compute": cumulative_compute,
        "pretrain_shards": [Path(p).name for p in train_paths],
        "shard_reuse": bool(shard_reuse),
    }
    if best_val_loss is not None:
        objectives["best_val_loss"] = float(best_val_loss)

    # Phase C — GIFT-Eval CRPS/MASE on the saved checkpoint. Combined
    # leaderboard-style metric is geomean(normalized_crps, normalized_mase).
    # No fallback: if the full benchmark can't produce both values, the
    # experiment fails.
    checkpoint_path = result.get("checkpoint_path") if success else None
    if not success:
        fail = _ts_failure(
            f"training did not succeed (status={status})",
            objectives, loss_curve, workdir,
            error=str(result.get("error") or status),
        )
        # Surface a shape-incompatible warm-start so the validator can
        # retry the proposal as a fresh run instead of just failing it.
        fail["harness_status"] = status
        return fail
    if not checkpoint_path:
        return _ts_failure(
            "harness did not return a checkpoint_path",
            objectives, loss_curve, workdir,
        )
    if not Path(checkpoint_path).exists():
        return _ts_failure(
            f"checkpoint missing at {checkpoint_path}",
            objectives, loss_curve, workdir,
        )
    if not Path(cache_dir).is_dir():
        return _ts_failure(
            f"GIFT-Eval cache dir missing at {cache_dir} "
            f"(set RADAR_GIFT_EVAL_CACHE / run `python -m local.fetch_gift_eval`)",
            objectives, loss_curve, workdir,
        )
    try:
        eval_metrics = _gift_eval_score(
            code, checkpoint_path, cache_dir, seed,
        )
    except Exception as e:  # noqa: BLE001
        return _ts_failure(
            f"GIFT-Eval failed: {type(e).__name__}: {e}",
            objectives, loss_curve, workdir,
        )

    crps = eval_metrics.get("crps")
    mase = eval_metrics.get("mase")
    if (crps is None or mase is None
            or not math.isfinite(crps) or not math.isfinite(mase)):
        return _ts_failure(
            f"GIFT-Eval returned non-finite metrics (crps={crps} mase={mase})",
            objectives, loss_curve, workdir,
        )

    objectives["crps"] = float(crps)
    objectives["mase"] = float(mase)
    if "n_tasks" in eval_metrics:
        objectives["n_tasks"] = int(eval_metrics["n_tasks"])

    # Capture the checkpoint's tensor signature (names + shapes, from the
    # safetensors header — no torch) so the /signature endpoint can tell a
    # continuation miner exactly what shapes a compatible warm-start needs.
    try:
        from local.checkpoints import read_signature
        sig = read_signature(checkpoint_path)
        if sig:
            objectives["param_signature"] = sig
    except Exception as e:  # noqa: BLE001
        logger.debug("could not capture param signature: %s", e)

    # Geomean of the two normalized leaderboard aggregates. Both lower=better.
    metric = math.sqrt(max(crps, 0.0) * max(mase, 0.0))

    analysis = (
        f"task=ts_forecasting status={status} pretrain_shards={len(train_paths)} "
        f"reuse={int(bool(shard_reuse))} crps={crps:.4f} mase={mase:.4f}"
    )

    return {
        "success": True,
        "metric": float(metric),
        "objectives": objectives,
        "loss_curve": loss_curve,
        "analysis": analysis,
        "error": "",
        "workdir": str(workdir),
    }


def _ts_failure(
    reason: str,
    objectives: dict,
    loss_curve: list,
    workdir: Path,
    *,
    error: str | None = None,
) -> dict:
    """Build a failed ts_forecasting result. No metric fallback — GIFT-Eval
    is the only acceptable signal."""
    logger.warning("ts_forecasting failed: %s", reason)
    return {
        "success": False,
        "metric": None,
        "objectives": objectives,
        "loss_curve": loss_curve,
        "analysis": f"task=ts_forecasting failed: {reason}",
        "error": error if error is not None else reason,
        "workdir": str(workdir),
    }


def _gift_eval_score(
    code: str, checkpoint_path: str, cache_dir: str, seed: int,
) -> dict:
    """Load the trained model from ``checkpoint_path`` and run the full
    GIFT-Eval leaderboard (all 97 tasks).

    Returns ``{"crps": float, "mase": float, "n_tasks": int, "per_task": [...]}``
    where crps/mase are the geomean of per-task normalized values (against
    seasonal-naive) — both lower=better, comparable to the leaderboard.
    """
    if not Path(cache_dir).is_dir():
        raise FileNotFoundError(f"GIFT-Eval cache dir missing: {cache_dir}")

    import torch
    from safetensors.torch import load_file

    from prepare import (  # type: ignore[import-not-found]
        CONTEXT_LEN, NUM_VARIATES, PREDICTION_LEN, QUANTILES, validate,
    )

    build_model = _exec_submission(code)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(CONTEXT_LEN, PREDICTION_LEN, NUM_VARIATES, QUANTILES).to(device)
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict)
    if hasattr(model, "reset"):
        model.reset()
    model.eval()

    metrics = validate(model, seed=seed, data_dir=cache_dir)
    out = {"crps": float(metrics["crps"]), "mase": float(metrics["mase"])}
    if "n_tasks" in metrics:
        out["n_tasks"] = int(metrics["n_tasks"])
        out["per_task"] = metrics.get("per_task", [])
    return out
