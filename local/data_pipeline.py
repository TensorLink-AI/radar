"""Trainer dispatch for the ``ts_data_pipeline`` task.

The data-pipeline task asks the miner for a synthetic data generator /
augmentor instead of an architecture. The validator pairs every proposal
with the **current frozen architecture** (see ``local/frozen_arch.py``),
trains it from fresh weights on the miner's pipeline under a fixed compute
budget, and scores it on:

  metric = geomean(AULC_train_val, sqrt(crps * mase))

where AULC is the trapezoidal area under the in-training val-loss curve
(lower=better; rewards both "drops fast" *and* "ends low") and crps/mase
come from the standard GIFT-Eval Phase C — held out so the pipeline can't
game it by memorising leaderboard shapes.

The miner's submission must define::

    def build_pipeline(context_len, prediction_len, num_variates, quantiles):
        return iter(...)  # yields {"input": tensor, "target": tensor}

Optional knobs (all no-ops if absent):
    PIPELINE_BATCH_SIZE: int
    def configure_pipeline() -> dict
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterator, Optional

from local.frozen_arch import FrozenArch

logger = logging.getLogger(__name__)


def _ts_runner_paths_setup() -> None:
    pkg = Path(__file__).resolve().parent.parent / "runner" / "timeseries_forecast"
    p = str(pkg)
    if p not in sys.path:
        sys.path.insert(0, p)


def _exec_pipeline_submission(code: str) -> Any:
    """Run the miner's pipeline code; return ``build_pipeline``."""
    ns: dict[str, Any] = {"__name__": "data_pipeline_submission"}
    exec(code, ns)
    bp = ns.get("build_pipeline")
    if not callable(bp):
        raise ValueError(
            "data_pipeline submission must define "
            "build_pipeline(context_len, prediction_len, num_variates, quantiles)",
        )
    return bp


class _DataPipelineRunner:
    """TaskRunner that pairs a frozen arch with a miner-supplied dataloader.

    Delegates everything model/loss/val-metrics related to the standard
    ``TSForecastingRunner`` so val + GIFT-Eval semantics stay identical.
    ``get_dataloader`` is the only override — it yields from the miner's
    pipeline factory.
    """

    def __init__(self, miner_pipeline_factory):
        from runner.timeseries_forecast.train import TSForecastingRunner
        self._inner = TSForecastingRunner()
        self._mp = miner_pipeline_factory

    def build_model(self, sub: Any, device: str) -> Any:
        return self._inner.build_model(sub, device)

    def get_dataloader(self, batch_size: int) -> Iterator:
        c = self._inner._load_constants()
        try:
            it = self._mp(
                c["context_len"], c["prediction_len"],
                c["num_variates"], c["quantiles"],
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"miner build_pipeline raised: {e}") from e
        if it is None:
            raise RuntimeError("miner build_pipeline returned None")
        return _ValidatingBatchIter(it, c, batch_size)

    def get_val_dataloader(self, batch_size: int):
        return self._inner.get_val_dataloader(batch_size)

    def default_loss(self, predictions, targets):
        return self._inner.default_loss(predictions, targets)

    def wrap_loss(self, sub_loss_fn):
        return self._inner.wrap_loss(sub_loss_fn)

    def compute_val_metrics(self, predictions, targets, inputs):
        return self._inner.compute_val_metrics(predictions, targets, inputs)

    def measure_flops(self, model: Any, device: str) -> int:
        return self._inner.measure_flops(model, device)


class _ValidatingBatchIter:
    """Coerces miner output into the {input, target} torch-tensor contract.

    Accepts dicts ``{"input"|"context", "target"}`` or tuples
    ``(inputs, targets)``. Converts numpy to torch, enforces shape
    consistency. Fails the run on the first malformed batch rather than
    letting the harness raise opaque CUDA errors.
    """

    def __init__(self, inner, constants: dict, batch_size: int):
        self._inner = iter(inner)
        self._c = constants
        self._bs = int(batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        import torch
        batch = next(self._inner)
        if isinstance(batch, dict):
            inp = batch.get("input", batch.get("context"))
            tgt = batch.get("target")
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            inp, tgt = batch
        else:
            raise TypeError(
                f"pipeline batch must be dict or (inputs, targets) tuple, "
                f"got {type(batch).__name__}",
            )
        if inp is None or tgt is None:
            raise TypeError("pipeline batch missing 'input' or 'target'")
        inp = _as_tensor(inp, torch.float32)
        tgt = _as_tensor(tgt, torch.float32)
        if inp.dim() != 3:
            raise ValueError(
                f"pipeline 'input' must be (batch, context_len, num_variates), "
                f"got shape {tuple(inp.shape)}",
            )
        if tgt.dim() != 3:
            raise ValueError(
                f"pipeline 'target' must be (batch, prediction_len, num_variates), "
                f"got shape {tuple(tgt.shape)}",
            )
        # Trust the miner on context_len / prediction_len — the harness pads
        # or truncates via align_pred_target. Just sanity-check batch dim.
        if inp.shape[0] != tgt.shape[0]:
            raise ValueError(
                f"pipeline batch dim mismatch: input={inp.shape[0]} "
                f"target={tgt.shape[0]}",
            )
        return {"input": inp, "target": tgt}


def _as_tensor(x, dtype):
    import torch
    if torch.is_tensor(x):
        return x.to(dtype) if x.dtype != dtype else x
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype)
    except ImportError:
        pass
    return torch.as_tensor(x, dtype=dtype)


def _compute_aulc(val_curve: list[dict]) -> Optional[float]:
    """Per-step normalized AULC: trapezoidal integral / step range.

    Lower = better. ``val_curve`` items: ``{"step": int, "loss": float}``.
    Returns None if fewer than 2 finite points (curve too sparse to score).
    """
    pts = [
        (float(p.get("step", i)), float(p["loss"]))
        for i, p in enumerate(val_curve)
        if p.get("loss") is not None and math.isfinite(float(p["loss"]))
    ]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[0])
    x0, x1 = pts[0][0], pts[-1][0]
    rng = x1 - x0
    if rng <= 0:
        return None
    area = 0.0
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        area += 0.5 * (yb + ya) * (xb - xa)
    return area / rng


def _fail(reason: str, objectives: dict, loss_curve: list,
          workdir: Path, *, error: Optional[str] = None,
          val_curve: Optional[list] = None,
          harness_status: str = "") -> dict:
    logger.warning("ts_data_pipeline failed: %s", reason)
    out = {
        "success": False,
        "metric": None,
        "objectives": objectives,
        "loss_curve": loss_curve,
        "val_curve": val_curve or [],
        "analysis": f"task=ts_data_pipeline failed: {reason}",
        "error": error if error is not None else reason,
        "workdir": str(workdir),
    }
    if harness_status:
        out["harness_status"] = harness_status
    return out


def run_data_pipeline_training(
    code: str,
    *,
    seed: int,
    task,
    min_flops: int,
    max_flops: int,
    frozen_arch: FrozenArch,
) -> dict:
    """Train the frozen arch from fresh weights on the miner's pipeline,
    then run GIFT-Eval on the saved checkpoint.

    Returns the same dict shape ``local/validator.py`` already consumes for
    ts_forecasting, plus a ``frozen_arch_version`` key in ``objectives`` so
    the dashboard can segment runs across snapshots.
    """
    started = time.time()
    _ts_runner_paths_setup()

    try:
        from runner.harness import TrainingConfig, run_training as harness_run
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "metric": None,
            "objectives": {"flops_equivalent_size": 0, "num_params": 0,
                           "train_seconds": 0.0,
                           "frozen_arch_version": frozen_arch.version},
            "loss_curve": [],
            "analysis": "ts_data_pipeline runner unavailable",
            "error": f"{type(e).__name__}: {e} (install torch + [gift_eval])",
            "workdir": "",
        }

    workdir = Path(tempfile.mkdtemp(prefix="radar_dp_"))
    submission_path = workdir / "submission.py"
    checkpoint_dir = workdir / "checkpoints"
    logs_dir = workdir / "logs"
    for d in (checkpoint_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    cache_dir = os.environ.get("RADAR_GIFT_EVAL_CACHE", "/tmp/radar_gift_eval")
    val_cache = os.environ.get(
        "RADAR_PRETRAIN_VAL_CACHE", "/tmp/radar_pretrain_val",
    )
    val_paths: list[str] = []
    if Path(val_cache).is_dir():
        val_paths = sorted(str(p) for p in Path(val_cache).glob("*.parquet"))

    # Exec the miner pipeline once up front so syntax/import errors surface
    # before we burn the training budget.
    try:
        miner_pipeline = _exec_pipeline_submission(code)
    except Exception as e:  # noqa: BLE001
        return _fail(
            "miner pipeline failed to load", {
                "flops_equivalent_size": 0, "num_params": 0,
                "train_seconds": 0.0,
                "frozen_arch_version": frozen_arch.version,
            }, [], workdir,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
        )

    overrides = {
        "CHECKPOINT_DIR": str(checkpoint_dir),
        "SUBMISSION_PATH": str(submission_path),
        "RADAR_GIFT_EVAL_CACHE": cache_dir,
        # Force the harness to NOT pull pretrain shards — the miner's
        # pipeline is the only training source.
        "RADAR_PRETRAIN_LOCAL_PATHS": "",
        "RADAR_PRETRAIN_SHARD_URLS": "",
        "RADAR_PRETRAIN_VAL_LOCAL_PATHS": (
            json.dumps(val_paths) if val_paths else ""
        ),
        "RADAR_STEP_OFFSET": "0",
        "RADAR_FLOPS_OFFSET": "0",
    }
    # Strip any continuation env left over from prior in-process calls.
    overrides["PARENT_CHECKPOINT_PATH"] = ""
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    # Remove rather than set-to-empty for the keys the harness checks via
    # ``getenv() == ""`` -> falsey semantics. PARENT_CHECKPOINT_PATH is
    # specifically tested with ``if env:`` so empty disables warm-start.

    train_log_path = logs_dir / "train.log"
    file_handler = logging.FileHandler(
        train_log_path, mode="w", encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
    ))
    capture_loggers = [
        logging.getLogger("runner"),
        logging.getLogger("runner.harness"),
        logging.getLogger("runner.timeseries_forecast"),
        logging.getLogger("local.data_pipeline"),
    ]
    for lg in capture_loggers:
        lg.addHandler(file_handler)
        if lg.level == 0 or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)

    try:
        config = TrainingConfig(
            seed=seed,
            round_id=seed,
            min_flops=min_flops,
            max_flops=max_flops,
            submission_id=f"local-dp-{seed}",
            time_budget=int(task.time_budget_seconds),
        )
        runner = _DataPipelineRunner(miner_pipeline)
        result = harness_run(runner, frozen_arch.code, config)
    except Exception as e:  # noqa: BLE001
        for lg in capture_loggers:
            lg.removeHandler(file_handler)
        file_handler.close()
        return _fail(
            "harness crashed", {
                "flops_equivalent_size": 0, "num_params": 0,
                "train_seconds": time.time() - started,
                "frozen_arch_version": frozen_arch.version,
            }, [], workdir,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    for lg in capture_loggers:
        lg.removeHandler(file_handler)
    file_handler.close()

    try:
        (logs_dir / "harness_result.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8",
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("could not write harness_result.json: %s", e)

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
    val_curve = [
        {"step": int(x.get("step", i)), "loss": float(x["loss"])}
        for i, x in enumerate(val_hist) if x.get("loss") is not None
    ]
    aulc = _compute_aulc(val_curve)
    best_val_loss = (
        float(result["best_val_loss"])
        if result.get("best_val_loss") is not None
        else (val_curve[-1]["loss"] if val_curve else None)
    )

    this_compute = float(result.get("this_run_flops") or 0.0)
    objectives = {
        "flops_equivalent_size": flops_equiv,
        "num_params": num_params,
        "train_seconds": train_seconds,
        "this_compute": this_compute,
        "cumulative_compute": this_compute,
        "frozen_arch_version": int(frozen_arch.version),
        "frozen_arch_source_id": int(frozen_arch.source_experiment_id),
    }
    if best_val_loss is not None:
        objectives["best_val_loss"] = float(best_val_loss)
    if aulc is not None:
        objectives["aulc"] = float(aulc)
    spikes = result.get("num_spikes_skipped")
    if spikes:
        objectives["num_spikes_skipped"] = int(spikes)

    checkpoint_path = result.get("checkpoint_path") if success else None
    if not success:
        return _fail(
            f"training did not succeed (status={status})",
            objectives, loss_curve, workdir,
            error=str(result.get("error") or status),
            val_curve=val_curve, harness_status=status,
        )
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return _fail(
            f"checkpoint missing ({checkpoint_path})",
            objectives, loss_curve, workdir, val_curve=val_curve,
        )
    if not Path(cache_dir).is_dir():
        return _fail(
            f"GIFT-Eval cache dir missing at {cache_dir}",
            objectives, loss_curve, workdir, val_curve=val_curve,
        )
    if aulc is None:
        return _fail(
            "in-training val curve too sparse to score (need val shard cache)",
            objectives, loss_curve, workdir, val_curve=val_curve,
        )

    try:
        from local.trainer import _gift_eval_score
        eval_metrics = _gift_eval_score(
            frozen_arch.code, checkpoint_path, cache_dir, seed,
        )
    except Exception as e:  # noqa: BLE001
        return _fail(
            f"GIFT-Eval failed: {type(e).__name__}: {e}",
            objectives, loss_curve, workdir, val_curve=val_curve,
        )

    crps = eval_metrics.get("crps")
    mase = eval_metrics.get("mase")
    if (crps is None or mase is None
            or not math.isfinite(crps) or not math.isfinite(mase)):
        return _fail(
            f"GIFT-Eval non-finite (crps={crps} mase={mase})",
            objectives, loss_curve, workdir, val_curve=val_curve,
        )

    objectives["crps"] = float(crps)
    objectives["mase"] = float(mase)
    if "n_tasks" in eval_metrics:
        objectives["n_tasks"] = int(eval_metrics["n_tasks"])

    gift = math.sqrt(max(crps, 0.0) * max(mase, 0.0))
    # Composite trajectory + leaderboard score: geomean(AULC, GIFT). Both
    # lower=better. Stamping frozen_arch_version in objectives means a chart
    # filter can keep cross-version comparisons honest.
    metric = math.sqrt(max(aulc, 0.0) * max(gift, 0.0))
    objectives["gift_metric"] = gift

    analysis = (
        f"task=ts_data_pipeline status={status} "
        f"frozen_arch=v{frozen_arch.version} aulc={aulc:.4f} "
        f"crps={crps:.4f} mase={mase:.4f} gift={gift:.4f}"
    )
    return {
        "success": True,
        "metric": float(metric),
        "objectives": objectives,
        "loss_curve": loss_curve,
        "val_curve": val_curve,
        "analysis": analysis,
        "error": "",
        "workdir": str(workdir),
    }
