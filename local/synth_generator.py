"""Trainer dispatch for the ``synthetic_data_generator`` task.

Cousin of ``ts_data_pipeline``: the miner submits a synthetic data generator
(``build_pipeline``); the validator trains a model on it and scores the result
on GIFT-Eval. The two differences from ts_data_pipeline are deliberate:

1. **The architecture is fixed.** Every round trains the *same* miniature
   ~10M Toto-2.0-style causal patch decoder (``local/synthetic_arch.py``),
   never the round-varying ts_forecasting frontier. So a continuation round is
   literally "keep training the same model's weights on (hopefully better)
   synthetic data" — the lineage is always warm-start-compatible.

2. **Scoring is GIFT-only**, exactly like ts_forecasting:
   ``metric = sqrt(crps * mase)`` (lower=better). No AULC composite — the
   in-training val curve is kept only as diagnostics (and to pick the
   best-val checkpoint), never as score. A sparse val curve is therefore not
   fatal here.

The heavy lifting (the harness runner that pairs a fixed architecture with a
miner dataloader, batch validation, harness-result translation) is shared with
``local/data_pipeline.py`` so there is one implementation of the pipeline
contract.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional

from local.data_pipeline import (
    _DataPipelineRunner,
    _compute_aulc,
    _exec_pipeline_submission,
    _ts_runner_paths_setup,
)
from local.synthetic_arch import REFERENCE_ARCH_VERSION, reference_arch

logger = logging.getLogger(__name__)


def _fail(reason: str, objectives: dict, loss_curve: list, workdir: Path,
          *, error: Optional[str] = None, val_curve: Optional[list] = None,
          harness_status: str = "") -> dict:
    logger.warning("synthetic_data_generator failed: %s", reason)
    out = {
        "success": False,
        "metric": None,
        "objectives": objectives,
        "loss_curve": loss_curve,
        "val_curve": val_curve or [],
        "analysis": f"task=synthetic_data_generator failed: {reason}",
        "error": error if error is not None else reason,
        "workdir": str(workdir),
    }
    if harness_status:
        out["harness_status"] = harness_status
    return out


def run_synth_generator_training(
    code: str,
    *,
    seed: int,
    task,
    min_flops: int,
    max_flops: int,
    parent_checkpoint_path: str | None = None,
    compute_offset: float = 0.0,
    step_offset: int = 0,
) -> dict:
    """Train the fixed reference arch on the miner's pipeline, then GIFT-Eval.

    Mirrors ``run_data_pipeline_training`` but with the fixed reference
    architecture and GIFT-only scoring. ``parent_checkpoint_path`` warm-starts
    a continuation from a parent's weights (always shape-compatible — the arch
    never changes); ``compute_offset`` / ``step_offset`` shift the recorded
    curve coordinates into lineage-absolute space.

    Returns the dict shape ``local/validator.py`` already consumes, with
    ``synth_arch_version`` stamped in ``objectives`` so continuation lineages
    pin to a model version.
    """
    started = time.time()
    _ts_runner_paths_setup()

    arch = reference_arch()

    def _base_objectives() -> dict:
        return {
            "flops_equivalent_size": 0,
            "num_params": 0,
            "train_seconds": time.time() - started,
            "synth_arch_version": int(REFERENCE_ARCH_VERSION),
        }

    try:
        from runner.harness import TrainingConfig, run_training as harness_run
    except Exception as e:  # noqa: BLE001
        return _fail(
            "synthetic_data_generator runner unavailable",
            _base_objectives(), [], Path("."),
            error=f"{type(e).__name__}: {e} (install torch + [gift_eval])",
        )

    workdir = Path(tempfile.mkdtemp(prefix="radar_sdg_"))
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

    # Exec the miner pipeline up front so syntax/import errors surface before
    # we burn the training budget.
    try:
        miner_pipeline = _exec_pipeline_submission(code)
    except Exception as e:  # noqa: BLE001
        return _fail(
            "miner pipeline failed to load", _base_objectives(), [], workdir,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
        )

    overrides = {
        "CHECKPOINT_DIR": str(checkpoint_dir),
        "SUBMISSION_PATH": str(submission_path),
        "RADAR_GIFT_EVAL_CACHE": cache_dir,
        # The miner's pipeline is the only training source — never pull the
        # real pretrain corpus.
        "RADAR_PRETRAIN_LOCAL_PATHS": "",
        "RADAR_PRETRAIN_SHARD_URLS": "",
        "RADAR_PRETRAIN_VAL_LOCAL_PATHS": (
            json.dumps(val_paths) if val_paths else ""
        ),
        "RADAR_STEP_OFFSET": str(int(step_offset)),
        "RADAR_FLOPS_OFFSET": str(int(compute_offset)),
        "PARENT_CHECKPOINT_PATH": (
            str(parent_checkpoint_path) if parent_checkpoint_path else ""
        ),
    }
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)

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
        logging.getLogger("local.synth_generator"),
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
            submission_id=f"local-sdg-{seed}",
            time_budget=int(task.time_budget_seconds),
        )
        runner = _DataPipelineRunner(miner_pipeline)
        result = harness_run(runner, arch.code, config)
    except Exception as e:  # noqa: BLE001
        for lg in capture_loggers:
            lg.removeHandler(file_handler)
        file_handler.close()
        return _fail(
            "harness crashed", _base_objectives(), [], workdir,
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
    aulc = _compute_aulc(val_curve)  # diagnostics only — never scored
    best_val_loss = (
        float(result["best_val_loss"])
        if result.get("best_val_loss") is not None
        else (val_curve[-1]["loss"] if val_curve else None)
    )

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
        "synth_arch_version": int(REFERENCE_ARCH_VERSION),
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
            f"GIFT-Eval cache dir missing at {cache_dir} "
            f"(set RADAR_GIFT_EVAL_CACHE / run `python -m local.fetch_gift_eval`)",
            objectives, loss_curve, workdir, val_curve=val_curve,
        )

    try:
        from local.trainer import _gift_eval_score
        eval_metrics = _gift_eval_score(arch.code, checkpoint_path, cache_dir, seed)
    except Exception as e:  # noqa: BLE001
        return _fail(
            f"GIFT-Eval failed: {type(e).__name__}: {e}",
            objectives, loss_curve, workdir, val_curve=val_curve,
        )

    from local.eval_metrics import finalize_gift_eval
    final = finalize_gift_eval(eval_metrics)
    if final is None:
        return _fail(
            f"GIFT-Eval non-finite (crps={eval_metrics.get('crps')} "
            f"mase={eval_metrics.get('mase')})",
            objectives, loss_curve, workdir, val_curve=val_curve,
        )
    crps, mase = final["crps"], final["mase"]
    objectives["crps"] = crps
    objectives["mase"] = mase
    objectives.update(final["extras"])

    # Capture the trained checkpoint's tensor signature so /signature can serve
    # it to a continuation miner picking a parent.
    try:
        from local.checkpoints import read_signature
        sig = read_signature(checkpoint_path)
        if sig:
            objectives["param_signature"] = sig
    except Exception as e:  # noqa: BLE001
        logger.debug("could not capture param signature: %s", e)

    # GIFT-only score — geomean of the two normalized leaderboard aggregates
    # (scored subset when a canary is held out), identical to ts_forecasting.
    metric = final["metric"]

    aulc_repr = f" aulc={aulc:.4f}" if aulc is not None else ""
    analysis = (
        f"task=synthetic_data_generator status={status} "
        f"arch=v{REFERENCE_ARCH_VERSION}{aulc_repr} "
        f"crps={crps:.4f} mase={mase:.4f}"
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
