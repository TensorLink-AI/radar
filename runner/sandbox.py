"""Sandbox lifecycle management for the trainer.

Run miner-supplied training code in a separate, network-isolated
subprocess.  The parent process keeps network access (so it can
prefetch presigned data shards and upload artifacts).  The child
process inherits NONE of the parent's secrets (R2 keys, wallet keys,
Basilica tokens) and runs through ``sandbox_runner.py`` which installs
an import-time block on every common HTTP/RPC client.

Two-stage pipeline:

1.  ``prefetch_*``: parent downloads presigned pretrain / GIFT-Eval data
    onto the local filesystem.
2.  ``run_sandbox``: parent spawns ``sandbox_runner.py`` (optionally
    wrapped by ``sandbox_wrap.sh`` for a hard network namespace) with a
    minimal env, points the runner at the prefetched files, and parses
    the JSON result envelope on stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from typing import Iterable

logger = logging.getLogger(__name__)

# Workspace layout — must match the Dockerfile.
# /workspace itself is read-only (chmod 555) on the hardened image so
# the bootstrap can guarantee no extra files appear in protected dirs.
# Per-job state lives under /var/radar/sandbox instead.
SANDBOX_ROOT = "/var/radar/sandbox"
SHARD_DIR = os.path.join(SANDBOX_ROOT, "shards")
VAL_SHARD_DIR = os.path.join(SANDBOX_ROOT, "val_shards")
GIFT_EVAL_DIR = os.path.join(SANDBOX_ROOT, "gift_eval")
CHECKPOINT_DIR = os.path.join(SANDBOX_ROOT, "checkpoints")

WRAPPER_SCRIPT = "/usr/local/bin/sandbox_wrap.sh"
SANDBOX_RUNNER = "/workspace/sandbox_runner.py"

# Env vars that must NEVER reach the sandbox.
_SECRET_ENV_PREFIXES = (
    "R2_",
    "AWS_",
    "BASILICA_",
    "WALLET_",
    "BT_",
    "OPENAI_",
    "ANTHROPIC_",
    "GEMINI_",
    "DESEARCH_",
    "RADAR_R2_",
)
_SECRET_ENV_KEYS = frozenset({
    "BITTENSOR_HOTKEY_PHRASE",
    "BITTENSOR_HOTKEY_SEED",
    "GHCR_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
})


# ── Data prefetch (parent process — has network) ────────────────────

def _reset_dir(path: str) -> None:
    """Remove and recreate a directory so prior-round files don't linger."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


async def _download_one(client, url: str, dest: str) -> bool:
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        logger.warning("Prefetch failed for %s: %s", dest, e)
        return False


async def prefetch_shards(
    urls: Iterable[str],
    dest_dir: str = SHARD_DIR,
    timeout: float = 120.0,
) -> list[str]:
    """Download every shard URL into ``dest_dir`` and return the local paths.

    Failures are logged and skipped — the harness already tolerates
    missing shards.  Files keep their natural ``.parquet`` extension so
    ``pandas.read_parquet`` recognises them.
    """
    import httpx

    url_list = list(urls)
    if not url_list:
        return []

    _reset_dir(dest_dir)

    paths: list[str] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for i, url in enumerate(url_list):
            dest = os.path.join(dest_dir, f"shard_{i:05d}.parquet")
            if await _download_one(client, url, dest):
                paths.append(dest)

    logger.info("Prefetched %d/%d shards into %s", len(paths), len(url_list), dest_dir)
    return paths


async def prefetch_gift_eval(
    urls: dict[str, str],
    dest_dir: str = GIFT_EVAL_DIR,
    timeout: float = 120.0,
) -> str | None:
    """Mirror GIFT-Eval validation data onto the local filesystem.

    The arrow files keep their original layout (``<dest>/<name>/data-00000-of-00001.arrow``)
    so the existing ``_discover_cached_datasets`` logic works unchanged.
    """
    import httpx
    from pathlib import Path

    if not urls:
        return None

    cache = Path(dest_dir)
    cache.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    async with httpx.AsyncClient(timeout=timeout) as client:
        for name, url in urls.items():
            local_dir = cache / name
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / "data-00000-of-00001.arrow"
            if local_path.exists() and local_path.stat().st_size > 0:
                continue
            try:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
                downloaded += 1
            except Exception as e:
                logger.warning("GIFT-Eval prefetch failed for %s: %s", name, e)

    logger.info("Prefetched %d GIFT-Eval datasets into %s", downloaded, dest_dir)
    return str(cache)


# ── Sandbox subprocess (parent process — spawns child) ──────────────

def _scrub_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a minimal env for the sandbox subprocess.

    Drops every variable that looks like a credential and only forwards
    ``PATH``, locale settings, and CUDA visibility.  Extra non-secret
    overrides can be supplied by the caller.
    """
    base: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": "/tmp",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONUNBUFFERED": "1",
    }
    for fwd in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "TZ"):
        if fwd in os.environ:
            base[fwd] = os.environ[fwd]
    if extra:
        for k, v in extra.items():
            if _is_secret(k):
                continue
            base[k] = v
    return base


def _is_secret(key: str) -> bool:
    if key in _SECRET_ENV_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in _SECRET_ENV_PREFIXES)


def _build_command(config_path: str) -> list[str]:
    """Choose the wrapper script when present so we get a netns when possible."""
    interp = "python3"
    if os.path.exists(WRAPPER_SCRIPT) and os.access(WRAPPER_SCRIPT, os.X_OK):
        return [WRAPPER_SCRIPT, interp, SANDBOX_RUNNER, config_path]
    return [interp, SANDBOX_RUNNER, config_path]


def _last_json_line(text: str) -> dict | None:
    """Return the last line of ``text`` that parses as a JSON object."""
    for raw in reversed(text.strip().splitlines()):
        line = raw.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


async def run_sandbox(
    config: dict,
    *,
    timeout_buffer: int = 180,
    stdout_cap_bytes: int = 10 * 1024 * 1024,
    stderr_cap_bytes: int = 10 * 1024 * 1024,
) -> tuple[dict, str]:
    """Spawn ``sandbox_runner.py`` and return ``(result, stderr_log)``.

    ``timeout_buffer`` is added to ``config['time_budget']`` so the
    sandbox can finish saving the checkpoint before we kill it.  The
    returned ``stderr_log`` is the last ``stderr_cap_bytes`` of the
    child's stderr (harness logs + miner ``print()``).

    Stdout and stderr are read incrementally and capped — a miner that
    spews unbounded output gets killed at the cap so it can't OOM the
    trainer or stall the parent on a stuck pipe.  The result envelope
    is written via a private fd from inside the sandbox; only the JSON
    on that channel is treated as authoritative.
    """
    os.makedirs(SANDBOX_ROOT, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    config_path = os.path.join(SANDBOX_ROOT, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    timeout = int(config.get("time_budget", 300)) + timeout_buffer
    env = _scrub_env()
    cmd = _build_command(config_path)

    logger.info(
        "Spawning sandbox: cmd=%s timeout=%ds (round=%s submission=%s)",
        cmd[0], timeout, config.get("round_id"),
        config.get("submission_id", "?")[:12],
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=SANDBOX_ROOT,
        )
    except FileNotFoundError as e:
        return (
            {"status": "failed", "error": f"sandbox spawn failed: {e}"},
            "",
        )

    overflow: dict[str, str] = {}

    async def _drain(stream, cap: int, label: str) -> bytes:
        buf = bytearray()
        while True:
            chunk = await stream.read(64 * 1024)
            if not chunk:
                return bytes(buf)
            buf.extend(chunk)
            if len(buf) > cap:
                overflow.setdefault(label, f"{label} exceeded {cap} bytes")
                # Drop the connection so the writer hits EPIPE next syscall,
                # then kill the process.  Keep ``buf`` truncated to the cap.
                if proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                # Drain remaining bytes silently to release the pipe.
                while True:
                    extra = await stream.read(64 * 1024)
                    if not extra:
                        break
                return bytes(buf[:cap])

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            asyncio.gather(
                _drain(proc.stdout, stdout_cap_bytes, "stdout"),
                _drain(proc.stderr, stderr_cap_bytes, "stderr"),
            ),
            timeout=timeout,
        )
        await proc.wait()
        timed_out = False
    except asyncio.TimeoutError:
        proc.kill()
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                asyncio.gather(
                    _drain(proc.stdout, stdout_cap_bytes, "stdout"),
                    _drain(proc.stderr, stderr_cap_bytes, "stderr"),
                ),
                timeout=10,
            )
        except Exception:
            stdout_b = stderr_b = b""
        timed_out = True

    stderr = stderr_b.decode("utf-8", "replace")[-stderr_cap_bytes:]

    if timed_out:
        return (
            {
                "round_id": config.get("round_id", 0),
                "submission_id": config.get("submission_id", "unknown"),
                "status": "failed",
                "error": f"sandbox timed out after {timeout}s",
            },
            stderr,
        )

    if overflow:
        return (
            {
                "round_id": config.get("round_id", 0),
                "submission_id": config.get("submission_id", "unknown"),
                "status": "failed",
                "error": "sandbox killed: " + ", ".join(sorted(overflow.values())),
            },
            stderr,
        )

    stdout = stdout_b.decode("utf-8", "replace")
    result = _last_json_line(stdout)
    if result is None:
        return (
            {
                "round_id": config.get("round_id", 0),
                "submission_id": config.get("submission_id", "unknown"),
                "status": "failed",
                "error": (
                    f"sandbox produced no JSON result (exit={proc.returncode}); "
                    f"stderr tail: {stderr[-500:]}"
                ),
            },
            stderr,
        )

    if proc.returncode != 0 and result.get("status") not in (
        "failed", "timeout", "build_failed", "size_violation",
    ):
        result.setdefault("status", "failed")
        result.setdefault(
            "error", f"sandbox exit {proc.returncode}: {stderr[-500:]}",
        )

    return result, stderr
