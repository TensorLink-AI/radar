"""Phase C evaluation — the trust anchor.

Every validator independently downloads checkpoints and computes metrics.
Cheap (seconds per model on CPU), fully deterministic, provides consensus.

Miner-submitted architecture code runs in an isolated subprocess to prevent
access to the validator's environment variables, filesystem, or memory.

Each task runner provides its own EVAL_TEMPLATE with task-specific imports
and model signatures. The evaluator loads the right template based on
the challenge's runner_dir.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Optional, TYPE_CHECKING

from shared.eval_result import build_error_result

if TYPE_CHECKING:
    import torch
    from shared.r2_audit import R2AuditLog
    from shared.task import TaskSpec

logger = logging.getLogger(__name__)

_DEFAULT_RUNNER_DIR = "runner/timeseries_forecast"


def _get_eval_template(runner_dir: str) -> str:
    """Load EVAL_TEMPLATE from the task's zero-dep `eval_template` module.

    Each task exposes EVAL_TEMPLATE at module level in
    `{runner_dir}/eval_template.py`. Falls back to ts_forecasting if the
    task module isn't importable.
    """
    mod_path = (runner_dir or _DEFAULT_RUNNER_DIR).replace("/", ".")
    try:
        mod = importlib.import_module(f"{mod_path}.eval_template")
    except ModuleNotFoundError:
        logger.warning(
            "No eval_template module for runner_dir=%s; falling back to %s",
            runner_dir, _DEFAULT_RUNNER_DIR,
        )
        fallback = _DEFAULT_RUNNER_DIR.replace("/", ".")
        mod = importlib.import_module(f"{fallback}.eval_template")
    return mod.EVAL_TEMPLATE


def evaluate_checkpoint(
    architecture_code: str,
    checkpoint_path: str,
    eval_split_seed: int = 42,
    device: str = "cpu",
    timeout: int = 120,
    runner_dir: str = "",
    task: "TaskSpec | None" = None,
) -> dict:
    """Evaluate a single checkpoint in an isolated subprocess.

    Runs miner code in a separate process so it cannot access the
    validator's memory, environment variables, or filesystem beyond
    the temp directory.

    Args:
        runner_dir: Relative path to the runner directory (e.g.
                    "runner/timeseries_forecast"). Falls back to
                    runner/timeseries_forecast if empty.
        task: TaskSpec used to shape error-path result dicts. When None,
              falls back to the legacy ts_forecasting shape so direct
              unit-test callers keep working.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy checkpoint into subprocess temp dir so it's co-located
        # and won't be affected by /tmp cleanup or concurrent access
        import shutil
        local_ckpt = os.path.join(tmpdir, "checkpoint.safetensors")
        shutil.copy2(checkpoint_path, local_ckpt)

        # Write architecture code
        arch_path = os.path.join(tmpdir, "submission.py")
        with open(arch_path, "w") as f:
            f.write(architecture_code)

        # Write eval runner script (per-task template)
        runner_path = os.path.join(tmpdir, "run_eval.py")
        template = _get_eval_template(runner_dir)
        runner_code = template.format(
            arch_path=arch_path,
            checkpoint_path=local_ckpt,
            eval_split_seed=eval_split_seed,
            device=device,
        )
        with open(runner_path, "w") as f:
            f.write(runner_code)

        # Run in subprocess with restricted environment
        # Resolve runner_dir — use task's runner_dir, fall back to ts_forecasting
        effective_runner_dir = runner_dir or "runner/timeseries_forecast"
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        runner_path_abs = os.path.join(project_root, effective_runner_dir)
        shared_path_abs = project_root
        clean_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
            "HOME": tmpdir,
            "PYTHONPATH": f"{runner_path_abs}:{shared_path_abs}",
            "RADAR_GIFT_EVAL_CACHE": os.environ.get(
                "RADAR_GIFT_EVAL_CACHE", "/tmp/radar_gift_eval"
            ),
            "RADAR_EVAL_DATA": os.environ.get("RADAR_EVAL_DATA", "gift_eval"),
        }
        # Do NOT forward R2 credentials, wallet keys, or BASILICA tokens

        try:
            result = subprocess.run(
                [sys.executable, runner_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=clean_env,
                cwd=tmpdir,
            )
            if result.returncode != 0:
                return build_error_result(
                    task, f"Eval subprocess failed: {result.stderr[:500]}",
                )
            # Parse only the last non-empty line of stdout to avoid
            # spurious output from miner code (print statements, warnings).
            return _parse_last_json_line(result.stdout, task=task)
        except subprocess.TimeoutExpired:
            return build_error_result(
                task, f"Eval subprocess timed out ({timeout}s)",
            )
        except (OSError, json.JSONDecodeError) as e:
            return build_error_result(task, f"Eval subprocess error: {e}")


def _parse_last_json_line(stdout: str, task: "TaskSpec | None" = None) -> dict:
    """Parse JSON from the last non-empty line of subprocess stdout.

    Miner code loaded via importlib may produce print output during import.
    The eval runner template always prints its JSON result as the final line.
    """
    lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
    if not lines:
        return build_error_result(task, "Eval subprocess produced no output")
    # Try last line first (expected), then scan backwards
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return build_error_result(
        task, f"Eval subprocess returned no valid JSON line: {lines[-1][:200]}",
    )


async def evaluate_all_checkpoints(
    r2: "R2AuditLog",
    round_id: int,
    training_metas: dict[int, dict],
    challenge,
    task: "TaskSpec",
    tmp_dir: str = "/tmp/eval",
) -> dict[int, dict]:
    """Evaluate all checkpoints for a round.

    For each miner with a successful training:
      1. Download architecture.py and checkpoint.safetensors from R2
      2. Call evaluate_checkpoint()
      3. Verify FLOPs matches trainer's claim
      4. Apply size gate
    """
    from shared.scoring import passes_size_gate
    from shared.artifacts import download_training_artifacts

    os.makedirs(tmp_dir, exist_ok=True)
    results: dict[int, dict] = {}
    device = os.getenv("RADAR_EVAL_DEVICE", "cpu")

    success_count = sum(1 for m in training_metas.values() if m.get("status") == "success")
    skipped = {uid: m.get("status", "?") for uid, m in training_metas.items() if m.get("status") != "success"}
    logger.info(
        "Phase C evaluation starting: %d total metas, %d successful, %d skipped (round %d)",
        len(training_metas), success_count, len(skipped), round_id,
    )
    if skipped:
        for uid, status in skipped.items():
            logger.info("  UID %d skipped (status=%s)", uid, status)

    for uid, meta in training_metas.items():
        if meta.get("status") != "success":
            continue

        miner_hotkey = meta.get("miner_hotkey", f"uid_{uid}")
        logger.info("Evaluating UID %d (miner %s...) round %d", uid, miner_hotkey[:16], round_id)

        # Download and verify all artifacts
        artifacts = download_training_artifacts(r2, round_id, miner_hotkey, tmp_dir)
        if not artifacts:
            logger.warning("UID %d: failed to download artifacts", uid)
            continue

        if not artifacts.verified:
            logger.warning("UID %d: artifact verification failed: %s", uid, artifacts.verification_error)
            results[uid] = build_error_result(
                task, f"verification failed: {artifacts.verification_error}",
            )
            continue

        if not artifacts.architecture_code:
            logger.warning("UID %d: architecture code missing from R2 — skipping eval", uid)
            results[uid] = build_error_result(task, "architecture code not found in R2")
            continue

        # Re-verify checkpoint hash immediately before eval (TOCTOU defense)
        if artifacts.meta.checkpoint_sha256:
            from shared.artifacts import sha256_file
            actual_hash = sha256_file(artifacts.checkpoint_path)
            if actual_hash != artifacts.meta.checkpoint_sha256:
                logger.warning("UID %d: checkpoint hash changed after download (TOCTOU)", uid)
                results[uid] = build_error_result(
                    task, "checkpoint hash mismatch at eval time",
                )
                continue

        # Evaluate
        if not os.path.exists(artifacts.checkpoint_path):
            logger.error(
                "UID %d: checkpoint file missing before eval: %s",
                uid, artifacts.checkpoint_path,
            )
            results[uid] = build_error_result(
                task, f"checkpoint file missing: {artifacts.checkpoint_path}",
            )
            continue

        logger.info(
            "UID %d: starting eval (checkpoint=%.2fMB arch=%d bytes)",
            uid, os.path.getsize(artifacts.checkpoint_path) / (1024 * 1024),
            len(artifacts.architecture_code),
        )
        runner_dir = challenge.task.get("runner_dir", "") if isinstance(challenge.task, dict) else ""
        metrics = evaluate_checkpoint(
            artifacts.architecture_code, artifacts.checkpoint_path,
            eval_split_seed=challenge.eval_split_seed,
            device=device,
            runner_dir=runner_dir,
            task=task,
        )

        # Verify FLOPs claim
        trainer_claimed = meta.get("flops_equivalent_size", 0)
        validator_measured = metrics.get("flops_equivalent_size", 0)
        metrics["flops_verified"] = verify_flops_claim(trainer_claimed, validator_measured)

        # Size gate
        metrics["passed_size_gate"] = passes_size_gate(metrics, challenge)

        results[uid] = metrics
        obj_summary = " ".join(
            f"{o.name}={metrics.get(o.name, -1):.6f}" for o in task.objectives
        ) if task.objectives else "(no objectives)"
        logger.info(
            "UID %d eval: %s flops=%d gate=%s verified=%s",
            uid, obj_summary, validator_measured,
            metrics["passed_size_gate"], artifacts.verified,
        )

        # Cleanup checkpoint file
        try:
            os.remove(artifacts.checkpoint_path)
        except OSError:
            pass

    return results


def verify_flops_claim(
    trainer_claimed: int,
    validator_measured: int,
    tolerance: float = 0.02,
) -> bool:
    """FLOPs-equivalent should be near-identical. Allow 2% tolerance."""
    if trainer_claimed == 0 or validator_measured == 0:
        return True  # Can't verify if either is zero
    ratio = abs(trainer_claimed - validator_measured) / max(trainer_claimed, 1)
    return ratio <= tolerance
