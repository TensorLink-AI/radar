"""Sandbox subprocess entry point for miner training code.

Spawned by ``runner/sandbox.py`` inside an isolated subprocess (and, when
the host kernel allows it, an empty network namespace).  Reads a JSON
config from ``argv[1]``, installs an import hook that blocks every
network-capable Python module, then runs the generic training harness
exactly like the in-process path used to.

Defense in depth (in order of strength):

1.  Subprocess + clean ``env=`` from the parent (no R2 / wallet creds).
2.  Optional network namespace via ``runner/sandbox_wrap.sh`` — when
    available the sandbox cannot open external sockets at all.
3.  ``NetworkBlocker`` import hook — even if the namespace fallback hit,
    miner code cannot ``import httpx`` / ``import boto3`` / etc.

The sandbox produces ONE machine-readable line of output: the final JSON
result on the last non-empty stdout line.  Anything else printed (miner
``print()``, harness logs) is mirrored to stderr so the parent can
capture it for the artifact log without corrupting the result envelope.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import io
import json
import os
import sys


# High-level network modules we refuse to load inside the sandbox.  We
# deliberately exclude low-level primitives (``socket``, ``ssl``) because
# torch / pandas / asyncio touch them at import time — the network
# namespace is the right tool to block raw sockets when available.  The
# import blocker is here to stop the obvious exfiltration paths a miner
# would actually use.
_BLOCKED_MODULES = frozenset({
    "aiohttp",
    "boto3",
    "botocore",
    "ftplib",
    "grpc",
    "http",
    "httpcore",
    "httpx",
    "imaplib",
    "paramiko",
    "poplib",
    "pycurl",
    "requests",
    "smtplib",
    "telnetlib",
    "urllib",
    "urllib3",
    "websocket",
    "websockets",
    "xmlrpc",
})


class _NetworkBlocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Refuse to load network-capable modules from miner code paths."""

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.partition(".")[0]
        if top in _BLOCKED_MODULES or fullname in _BLOCKED_MODULES:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        raise ImportError(
            f"sandbox: import of '{spec.name}' is blocked — network access "
            "is not permitted inside the trainer sandbox",
        )

    def exec_module(self, module):  # pragma: no cover - never reached
        raise ImportError(f"sandbox: import of '{module.__name__}' is blocked")


def _block_network_imports() -> None:
    """Install the import blocker and evict already-loaded network modules.

    Must run AFTER harness / task imports (those touch ``urllib`` etc. at
    import time) but BEFORE any miner code executes.
    """
    sys.meta_path.insert(0, _NetworkBlocker())
    for name in list(sys.modules):
        top = name.partition(".")[0]
        if top in _BLOCKED_MODULES or name in _BLOCKED_MODULES:
            sys.modules.pop(name, None)


_RESULT_FD: int | None = None


def _seal_stdout() -> None:
    """Hand the parent's stdout pipe to a private fd, then redirect fd 1 to /dev/null.

    After this runs, the only handle that still reaches the parent is
    ``_RESULT_FD``, which lives only in this module.  Miner code can
    ``print()``, hit ``sys.__stdout__``, or ``os.write(1, …)`` all it
    wants — those bytes go to /dev/null (or stderr, for ``print()`` via
    the ``_Tee`` redirect) and cannot spoof the JSON result envelope.
    """
    global _RESULT_FD
    real_stdout_fd = sys.__stdout__.fileno() if sys.__stdout__ is not None else 1
    _RESULT_FD = os.dup(real_stdout_fd)

    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, real_stdout_fd)
    os.close(devnull)

    # Drop the Python-side reference so a miner ``sys.__stdout__.write(...)``
    # can't even reach the (now redirected to /dev/null) fd object.
    sys.__stdout__ = None  # type: ignore[assignment]

    class _Tee(io.TextIOBase):
        def write(self, s):  # type: ignore[override]
            try:
                sys.__stderr__.write(s)
            except Exception:
                pass
            return len(s)

        def flush(self):  # type: ignore[override]
            try:
                sys.__stderr__.flush()
            except Exception:
                pass

    sys.stdout = _Tee()


def _apply_env(config: dict) -> None:
    """Translate the sandbox config into the env vars the harness reads."""
    os.environ["CHECKPOINT_DIR"] = config.get(
        "checkpoint_dir", "/workspace/sandbox/checkpoints",
    )
    os.environ["SEED"] = str(config.get("seed", 42))
    os.environ["TIME_BUDGET"] = str(config.get("time_budget", 300))

    gift_dir = config.get("gift_eval_dir", "")
    if gift_dir:
        os.environ["RADAR_GIFT_EVAL_CACHE"] = gift_dir

    train_paths = config.get("pretrain_shard_paths") or []
    if train_paths:
        os.environ["RADAR_PRETRAIN_LOCAL_PATHS"] = json.dumps(train_paths)
    else:
        os.environ.pop("RADAR_PRETRAIN_LOCAL_PATHS", None)
    # Strip any URLs the parent may have left in the inherited env so the
    # data loader can't accidentally fall back to a network fetch.
    os.environ.pop("RADAR_PRETRAIN_SHARD_URLS", None)

    val_paths = config.get("pretrain_val_shard_paths") or []
    if val_paths:
        os.environ["RADAR_PRETRAIN_VAL_LOCAL_PATHS"] = json.dumps(val_paths)
    else:
        os.environ.pop("RADAR_PRETRAIN_VAL_LOCAL_PATHS", None)
    os.environ.pop("RADAR_PRETRAIN_VAL_SHARD_URLS", None)


def _emit(result: dict) -> None:
    """Write the final JSON result envelope on the private result fd.

    Falls back to fd 1 only when ``_seal_stdout`` hasn't run yet (very
    early failures before the harness is loaded).  Miner code never
    reaches this path because the seal happens before any miner code
    executes.
    """
    payload = (json.dumps(result) + "\n").encode()
    fd = _RESULT_FD if _RESULT_FD is not None else 1
    try:
        os.write(fd, payload)
    except OSError:
        # Last-ditch: try the original stdout fd.
        try:
            os.write(1, payload)
        except OSError:
            pass


def main() -> int:
    if len(sys.argv) < 2:
        _emit({"status": "failed", "error": "sandbox: missing config path"})
        return 2

    try:
        with open(sys.argv[1]) as f:
            config = json.load(f)
    except Exception as e:
        _emit({"status": "failed", "error": f"sandbox: bad config: {e}"})
        return 2

    _apply_env(config)
    _seal_stdout()

    # Import the harness BEFORE installing the blocker so torch / pandas /
    # asyncio (which transitively touch ``urllib`` at import) succeed.
    sys.path.insert(0, "/workspace")
    try:
        from runner.harness import TrainingConfig, run_training
    except Exception as e:
        _emit({"status": "failed", "error": f"sandbox: harness import failed: {e}"})
        return 2

    task_name = config.get("task_name", "ts_forecasting")
    if task_name in ("ts_forecasting", "ml_training"):
        try:
            from runner.timeseries_forecast.train import _runner as task_runner
        except Exception as e:
            _emit({"status": "failed", "error": f"sandbox: task import failed: {e}"})
            return 2
    else:
        _emit({"status": "failed", "error": f"sandbox: unknown task '{task_name}'"})
        return 2

    # Now seal the sandbox: miner code cannot pull in httpx / boto3 / etc.
    _block_network_imports()

    tc = TrainingConfig.from_dict(config)
    try:
        result = run_training(task_runner, config["architecture_code"], tc)
    except Exception as e:
        _emit({
            "round_id": tc.round_id,
            "miner_hotkey": tc.miner_hotkey,
            "status": "failed",
            "error": f"sandbox: harness raised {type(e).__name__}: {e}",
        })
        return 1

    _emit(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
