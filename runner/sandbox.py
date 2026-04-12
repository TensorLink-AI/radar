"""Sandbox subprocess management for isolated miner code execution.

Handles data prefetching, sandbox subprocess spawning, and the future
data proxy. Called by server.py during training jobs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────
SANDBOX_DATA_MODE = os.getenv("RADAR_SANDBOX_DATA_MODE", "prefetch")


# ── Data prefetch ──────────────────────────────────────────────────

async def prefetch_shards(shard_urls: list[str]) -> list[str]:
    """Download all shards to local files before sandbox starts."""
    import httpx
    shard_dir = "/workspace/sandbox/shards"
    os.makedirs(shard_dir, exist_ok=True)

    # Clear stale shards from previous round
    for f in os.listdir(shard_dir):
        os.remove(os.path.join(shard_dir, f))

    paths: list[str] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for i, url in enumerate(shard_urls):
            path = os.path.join(shard_dir, f"shard_{i}.parquet")
            try:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)
                paths.append(path)
            except Exception as e:
                logger.warning("Prefetch shard %d failed: %s", i, e)

    logger.info("Prefetched %d/%d shards", len(paths), len(shard_urls))
    return paths


# ── Sandbox subprocess ─────────────────────────────────────────────

async def run_sandbox(config_path: str, training_config: dict) -> dict:
    """Spawn sandbox_runner.py in network-isolated subprocess."""
    timeout = training_config.get("time_budget", 300) + 180

    cmd = [
        "/usr/local/bin/sandbox_network.sh",
        f"python3 /workspace/sandbox_runner.py {config_path}",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {
            "round_id": training_config.get("round_id", 0),
            "miner_hotkey": training_config.get("miner_hotkey", "unknown"),
            "status": "timeout",
            "error": f"Sandbox timed out after {timeout}s",
        }

    if stderr:
        logger.info("Sandbox stderr (last 2000): %s", stderr.decode()[-2000:])

    if proc.returncode != 0:
        return {
            "round_id": training_config.get("round_id", 0),
            "miner_hotkey": training_config.get("miner_hotkey", "unknown"),
            "status": "failed",
            "error": f"Exit {proc.returncode}: {stderr.decode()[:500]}",
        }

    # Parse last JSON line from stdout
    for line in reversed(stdout.decode().strip().split("\n")):
        if line.strip().startswith("{"):
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError:
                continue

    return {
        "round_id": training_config.get("round_id", 0),
        "miner_hotkey": training_config.get("miner_hotkey", "unknown"),
        "status": "failed",
        "error": "No valid JSON in sandbox output",
    }


# ── Data proxy (future, for proxy mode) ───────────────────────────

async def run_data_proxy(allowed_urls: list[str], port: int = 9999):
    """Localhost-only HTTP proxy that serves presigned URLs to the sandbox.

    Only serves URLs in the allowed list. Sandbox accesses /shard/N
    which maps to allowed_urls[N]. Everything else is rejected.
    Not used in prefetch mode (default).
    """
    from aiohttp import web
    import httpx

    url_map = {i: url for i, url in enumerate(allowed_urls)}

    async def handle_shard(request):
        idx = int(request.match_info["idx"])
        if idx not in url_map:
            return web.Response(status=404, text="Not found")
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(url_map[idx], follow_redirects=True)
                return web.Response(
                    body=resp.content,
                    content_type="application/octet-stream",
                )
        except Exception as e:
            return web.Response(status=502, text=str(e))

    async def handle_reject(request):
        return web.Response(status=403, text="Forbidden")

    app = web.Application()
    app.router.add_get("/shard/{idx}", handle_shard)
    app.router.add_route("*", "/{path:.*}", handle_reject)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    logger.info(
        "Data proxy listening on 127.0.0.1:%d (%d shards)", port, len(url_map),
    )

    try:
        await asyncio.Event().wait()  # run until cancelled
    finally:
        await runner.cleanup()
