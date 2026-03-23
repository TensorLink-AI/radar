"""
Affinetes Actor for the official subnet training environment.

Runs inside runner/timeseries_forecast Docker container on a Basilica GPU pod.
Receives miner code via execute_code(), writes to disk, runs harness,
returns the execution trace with metrics and a verifiable manifest.
"""

import hashlib
import json
import os
import re
import subprocess
import time
import zipfile


# Env vars to strip from subprocess (secrets that miners must not access)
_SECRET_KEYS = {"R2_PRESIGNED_URL", "R2_ACCESS_KEY", "R2_SECRET_KEY"}


class Actor:
    def __init__(self):
        self.workspace = "/workspace"

    async def execute_code(
        self,
        code: str,
        target_file: str = "submission.py",
        run_command: str = "python {target}",
        time_budget: int = 300,
        kill_timeout: int = 600,
        eval_command: str = "",
        objectives: list[dict] = None,
        experiment_id: str = "",
        miner_uid: int = -1,
        validator_uid: int = -1,
        challenge_id: str = "",
        seed: int = 42,
        env_vars: dict[str, str] = None,
    ) -> dict:
        """Write miner code to disk, run harness, build manifest bundle."""
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(os.path.join(self.workspace, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace, "checkpoints"), exist_ok=True)

        # Hash miner code before writing
        code_sha256 = hashlib.sha256(code.encode()).hexdigest()

        target_path = os.path.join(self.workspace, target_file)
        with open(target_path, "w") as f:
            f.write(code)

        # Build clean env (strip secrets, add SEED, merge caller env_vars)
        clean_env = {
            k: v for k, v in os.environ.items() if k not in _SECRET_KEYS
        }
        clean_env["SEED"] = str(seed)
        clean_env["TIME_BUDGET"] = str(time_budget)
        if env_vars:
            clean_env.update(env_vars)

        cmd = run_command.format(target=target_path)
        start = time.time()

        # ── Run harness ──────────────────────────────
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=kill_timeout, cwd=self.workspace, env=clean_env,
            )
            elapsed = time.time() - start
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            return_code = proc.returncode
            success = return_code == 0
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            stdout = ""
            stderr = f"TIMEOUT after {kill_timeout}s"
            return_code = -9
            success = False
        except Exception as e:
            elapsed = time.time() - start
            stdout = ""
            stderr = f"RUNNER ERROR: {type(e).__name__}: {e}"
            return_code = -1
            success = False

        # ── Run frozen eval if harness succeeded ─────
        eval_stdout = ""
        eval_stderr = ""
        if success and eval_command:
            try:
                eval_proc = subprocess.run(
                    eval_command, shell=True, capture_output=True, text=True,
                    timeout=120, cwd=self.workspace, env=clean_env,
                )
                eval_stdout = eval_proc.stdout or ""
                eval_stderr = eval_proc.stderr or ""
            except subprocess.TimeoutExpired:
                eval_stderr = "EVAL TIMEOUT after 120s"
            except Exception as e:
                eval_stderr = f"EVAL ERROR: {type(e).__name__}: {e}"

        # ── Build trace ──────────────────────────────
        trace_parts = []
        if stdout:
            trace_parts.append(stdout)
        if stderr:
            trace_parts.append(f"=== STDERR ===\n{stderr}")
        trace_parts.append(f"\nWall clock: {elapsed:.1f}s")
        trace = "\n".join(trace_parts)

        # ── Extract metrics ──────────────────────────
        metrics = {}
        if objectives:
            for obj in objectives:
                name = obj.get("name", "")
                pattern = obj.get("pattern", "")
                if not name or not pattern:
                    continue
                # Eval output overrides training output for same metric
                for source in [eval_stdout, stdout]:
                    match = re.search(pattern, source)
                    if match:
                        try:
                            metrics[name] = float(match.group(1))
                        except (ValueError, IndexError):
                            pass
                        break

        # ── Build manifest ───────────────────────────
        stdout_sha256 = hashlib.sha256(stdout.encode()).hexdigest()
        stderr_sha256 = hashlib.sha256(stderr.encode()).hexdigest()

        manifest = {
            "experiment_id": experiment_id,
            "challenge_id": challenge_id,
            "miner_uid": miner_uid,
            "validator_uid": validator_uid,
            "seed": seed,
            "code_sha256": code_sha256,
            "stdout_sha256": stdout_sha256,
            "stderr_sha256": stderr_sha256,
            "metrics": metrics,
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "timestamp": time.time(),
        }
        # Self-hash: hash manifest content without manifest_sha256 field
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
        manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()

        # ── Write manifest ───────────────────────────
        manifest_path = os.path.join(self.workspace, "logs", "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # ── Write logs ───────────────────────────────
        stdout_path = os.path.join(self.workspace, "logs", "stdout.log")
        stderr_path = os.path.join(self.workspace, "logs", "stderr.log")
        eval_path = os.path.join(self.workspace, "logs", "eval_output.log")

        with open(stdout_path, "w") as f:
            f.write(stdout)
        with open(stderr_path, "w") as f:
            f.write(stderr)
        if eval_stdout or eval_stderr:
            with open(eval_path, "w") as f:
                f.write(eval_stdout)
                if eval_stderr:
                    f.write(f"\n=== STDERR ===\n{eval_stderr}")

        # ── Bundle zip ───────────────────────────────
        bundle_path = os.path.join(self.workspace, "logs", "experiment_bundle.zip")
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("code.py", code)
            zf.writestr("stdout.log", stdout)
            zf.writestr("stderr.log", stderr)
            if eval_stdout or eval_stderr:
                eval_content = eval_stdout
                if eval_stderr:
                    eval_content += f"\n=== STDERR ===\n{eval_stderr}"
                zf.writestr("eval_output.log", eval_content)
            # Include generated samples if present
            samples_path = os.path.join(
                self.workspace, "logs", "generated_samples.json"
            )
            if os.path.exists(samples_path):
                zf.write(samples_path, "generated_samples.json")

        return {
            "success": success,
            "return_code": return_code,
            "trace": trace,
            "exec_time": elapsed,
            "manifest": manifest,
            "bundle_path": bundle_path,
        }

    async def health(self) -> dict:
        """Health check."""
        return {"status": "ok", "workspace": self.workspace}
