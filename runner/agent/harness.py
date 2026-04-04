"""Frozen agent harness — the official sandbox for miner agent code.

Mirrors the trainer pattern: subnet owner controls the image, miner only
provides .py files.  The harness:

1. Reads Challenge JSON from stdin (passed by affinetes / validator)
2. Loads the miner's agent module from /workspace/agent/agent.py
3. Injects a GatedClient that only allows requests to approved URLs
4. Calls the miner's ``design_architecture(challenge, client)`` function
5. Writes Proposal JSON to stdout, reasoning trace to stderr

The miner's agent module MUST define:
    design_architecture(challenge: dict, client: GatedClient) -> dict
        Returns {"code": str, "name": str, "motivation": str}

The GatedClient is the ONLY way to make HTTP requests.  The Docker image
has no ``requests``, ``httpx``, or ``aiohttp`` installed, and iptables
blocks all egress except to allowed hosts.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tarfile
import tempfile
import traceback

# url_gate is copied into the image at /workspace/shared/
sys.path.insert(0, "/workspace")
from shared.url_gate import GatedClient, parse_allowed_urls  # noqa: E402


def log(msg: str):
    """Write reasoning trace to stderr (captured by validator)."""
    print(msg, file=sys.stderr)


# ── Scratchpad helpers (use GatedClient, not raw urllib) ─────────────

def load_scratchpad(challenge: dict, client: GatedClient,
                    local_dir: str = "/tmp/scratchpad") -> str:
    """Download and extract the miner's persistent scratchpad."""
    os.makedirs(local_dir, exist_ok=True)
    url = challenge.get("scratchpad_get_url", "")
    if not url:
        return local_dir
    try:
        archive_path = os.path.join(local_dir, "state.tar.gz")
        data = client.get(url, timeout=30)
        with open(archive_path, "wb") as f:
            f.write(data)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(local_dir, filter="data")
        os.remove(archive_path)
        log(f"Loaded scratchpad ({len(os.listdir(local_dir))} files)")
    except Exception as e:
        log(f"Scratchpad load: {e} (starting fresh)")
    return local_dir


def save_scratchpad(challenge: dict, client: GatedClient,
                    local_dir: str = "/tmp/scratchpad") -> bool:
    """Upload the miner's scratchpad via presigned PUT."""
    url = challenge.get("scratchpad_put_url", "")
    if not url:
        return False
    try:
        fd, archive_path = tempfile.mkstemp(suffix=".tar.gz", prefix="scratchpad_")
        os.close(fd)
        with tarfile.open(archive_path, "w:gz") as tar:
            for root, _dirs, files in os.walk(local_dir):
                for f in files:
                    full = os.path.join(root, f)
                    arcname = os.path.relpath(full, local_dir)
                    tar.add(full, arcname=arcname)

        max_mb = challenge.get("scratchpad_max_mb", 10)
        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        if size_mb > max_mb:
            log(f"Scratchpad too large ({size_mb:.1f}MB > {max_mb}MB). Not saving.")
            os.remove(archive_path)
            return False

        with open(archive_path, "rb") as f:
            data = f.read()
        client.put(url, data, content_type="application/gzip")
        os.remove(archive_path)
        log(f"Saved scratchpad ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        log(f"Scratchpad save failed: {e}")
        return False


# ── Submission loader ────────────────────────────────────────────────

def _load_agent(path: str):
    """Load the miner's agent module from a .py file."""
    spec = importlib.util.spec_from_file_location("miner_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # 1. Read challenge from stdin
    challenge_raw = sys.stdin.read()
    try:
        challenge = json.loads(challenge_raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid challenge JSON: {e}"}))
        sys.exit(1)

    # 2. Build GatedClient from allowed URLs in challenge + env
    allowed_raw = challenge.get("allowed_urls", os.environ.get("AGENT_ALLOWED_URLS", ""))
    allowed_prefixes = parse_allowed_urls(allowed_raw)

    # Always allow the validator proxy and scratchpad URLs passed in the challenge
    for key in ("db_url", "desearch_url", "llm_url", "scratchpad_get_url", "scratchpad_put_url"):
        url = challenge.get(key, "")
        if url:
            # Allow the full URL as-is (presigned URLs are unique per request)
            allowed_prefixes.append(url)

    # Inject agent token as default header so all proxy requests are authenticated
    default_headers = {}
    agent_token = challenge.get("agent_token", "")
    if agent_token:
        default_headers["X-Agent-Token"] = agent_token

    client = GatedClient(allowed_prefixes, default_headers=default_headers)
    log(f"GatedClient initialised with {len(allowed_prefixes)} allowed prefixes")

    # 3. Load miner's agent module
    agent_path = os.environ.get("AGENT_MODULE", "/workspace/agent/agent.py")
    if not os.path.exists(agent_path):
        print(json.dumps({"error": f"Agent module not found: {agent_path}"}))
        sys.exit(1)

    # Add agent dir to sys.path so inter-file imports work
    agent_dir = os.path.dirname(agent_path)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    try:
        agent_mod = _load_agent(agent_path)
    except Exception as e:
        log(f"Failed to load agent module: {e}")
        log(traceback.format_exc())
        print(json.dumps({"error": f"Agent load failed: {e}"}))
        sys.exit(1)

    if not hasattr(agent_mod, "design_architecture") or not callable(agent_mod.design_architecture):
        print(json.dumps({"error": "Agent module missing design_architecture()"}))
        sys.exit(1)

    # 4. Inject scratchpad helpers into the module namespace
    agent_mod.load_scratchpad = lambda ch, local_dir="/tmp/scratchpad": load_scratchpad(ch, client, local_dir)
    agent_mod.save_scratchpad = lambda ch, local_dir="/tmp/scratchpad": save_scratchpad(ch, client, local_dir)

    # 5. Call the miner's agent
    try:
        result = agent_mod.design_architecture(challenge, client)
    except Exception as e:
        log(f"design_architecture() failed: {e}")
        log(traceback.format_exc())
        print(json.dumps({"error": f"Agent failed: {e}"}))
        sys.exit(1)

    # 6. Validate and output
    if not isinstance(result, dict) or "code" not in result:
        print(json.dumps({"error": "design_architecture() must return dict with 'code' key"}))
        sys.exit(1)

    proposal = {
        "code": result.get("code", ""),
        "name": result.get("name", ""),
        "motivation": result.get("motivation", ""),
    }
    print(json.dumps(proposal))


if __name__ == "__main__":
    main()
