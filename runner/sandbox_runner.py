"""Sandbox runner -- network-isolated subprocess for miner code execution.

Reads config JSON (argv[1]), loads miner submission, trains, saves checkpoint.
All data access is via local files (prefetch mode) or localhost proxy (proxy mode).
No external network access.
"""

from __future__ import annotations

import importlib.abc
import json
import os
import sys


# ── Blocked modules (network-capable) ──────────────────────────────

BLOCKED_MODULES = {
    "httpx", "requests", "urllib", "urllib3", "urllib.request",
    "http", "http.client", "aiohttp", "socket", "boto3", "botocore",
    "websocket", "websockets", "grpc", "paramiko", "ftplib",
    "smtplib", "imaplib", "poplib", "xmlrpc",
}


class NetworkBlocker(importlib.abc.MetaPathFinder):
    """Block network-capable module imports inside the sandbox."""

    def find_module(self, fullname: str, path=None):
        top = fullname.split(".")[0]
        if top in BLOCKED_MODULES or fullname in BLOCKED_MODULES:
            return self
        return None

    def load_module(self, fullname: str):
        raise ImportError(f"Module '{fullname}' is blocked in sandbox")


def _block_network_imports():
    """Install the import blocker and purge already-loaded network modules."""
    sys.meta_path.insert(0, NetworkBlocker())
    for name in list(sys.modules.keys()):
        if name.split(".")[0] in BLOCKED_MODULES:
            del sys.modules[name]


# ── Main ───────────────────────────────────────────────────────────

def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)

    # Block network BEFORE loading miner code
    _block_network_imports()

    # Sandbox environment
    os.environ["CHECKPOINT_DIR"] = "/workspace/sandbox/checkpoints"
    os.environ["SEED"] = str(config.get("seed", 42))
    os.environ["TIME_BUDGET"] = str(config.get("time_budget", 300))

    # Data delivery -- prefetch or proxy, transparent to the runner
    data_mode = config.get("data_mode", "prefetch")

    if data_mode == "prefetch":
        if config.get("local_data_dir"):
            os.environ["RADAR_GIFT_EVAL_CACHE"] = config["local_data_dir"]
        if config.get("local_shard_paths"):
            os.environ["RADAR_PRETRAIN_LOCAL_PATHS"] = json.dumps(
                config["local_shard_paths"]
            )
        os.environ.pop("RADAR_PRETRAIN_SHARD_URLS", None)

    elif data_mode == "proxy":
        # Future: sandbox reads from localhost proxy
        proxy_base = config.get("proxy_url", "http://127.0.0.1:9999")
        n_shards = config.get("n_shards", 0)
        shard_urls = [f"{proxy_base}/shard/{i}" for i in range(n_shards)]
        os.environ["RADAR_PRETRAIN_SHARD_URLS"] = json.dumps(shard_urls)
        if config.get("local_data_dir"):
            os.environ["RADAR_GIFT_EVAL_CACHE"] = config["local_data_dir"]

    # Import harness (root-owned, read-only)
    sys.path.insert(0, "/workspace")
    from runner.harness import TrainingConfig, run_training as generic_run
    from runner.timeseries_forecast.train import _runner

    tc = TrainingConfig.from_dict(config)
    result = generic_run(_runner, config["architecture_code"], tc)

    # Ensure checkpoint is in sandbox dir
    default_ckpt = "/workspace/checkpoints/model.safetensors"
    sandbox_ckpt = "/workspace/sandbox/checkpoints/model.safetensors"
    if os.path.exists(default_ckpt) and not os.path.exists(sandbox_ckpt):
        os.makedirs(os.path.dirname(sandbox_ckpt), exist_ok=True)
        os.rename(default_ckpt, sandbox_ckpt)
    if "checkpoint_path" in result:
        result["checkpoint_path"] = sandbox_ckpt

    print(json.dumps(result))


if __name__ == "__main__":
    main()
