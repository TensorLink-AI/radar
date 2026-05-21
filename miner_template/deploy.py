"""Deploy guide for Radar miners.

Miners have two components:
  1. Agent code: .py files in a directory. The miner neuron POSTs them to
     the DB server. Validators fetch the code and run it inside the
     official sandboxed agent image.
  2. Trainer listener: a lightweight HTTP server on the miner's neuron
     process. GPU pods deploy on-demand via Basilica when validators send
     TrainerRequests.

No Docker image needed for agents — just write .py files.
No upfront GPU cost — pods deploy on-demand.

Usage:
  # Start miner (agent code auto-submitted from --agent_dir)
  python miner/neuron.py \\
      --agent_dir miner_template/ \\
      --listener_port 8090
"""

import argparse
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def signed_request_headers(body: bytes, hotkey: str = "") -> dict:
    """Build HMAC-signed headers for a miner → validator request.

    Replaces the old chain-wallet signing flow. Uses
    ``RADAR_SHARED_SECRET`` from the env (matching what validators
    configure on their side).
    """
    from shared.auth import sign_request
    headers = sign_request(None, body)
    if hotkey:
        headers["X-Miner-Hotkey"] = hotkey
    headers.setdefault("Content-Type", "application/json")
    return headers


def post_trainer_ready(
    validator_url: str,
    round_id: int,
    hotkey: str,
    trainer_url: str,
    instance_name: str = "",
) -> Optional[int]:
    """Notify the validator's proxy that a miner trainer pod is up.

    Returns the HTTP status code, or None on transport error.
    """
    import httpx
    payload = json.dumps({
        "round_id": round_id,
        "miner_hotkey": hotkey,
        "trainer_url": trainer_url,
        "instance_name": instance_name,
    }).encode()
    headers = signed_request_headers(payload, hotkey=hotkey)
    try:
        resp = httpx.post(
            validator_url.rstrip("/") + "/trainer/ready",
            content=payload, headers=headers, timeout=15,
        )
        return resp.status_code
    except Exception as e:  # pragma: no cover — diagnostic
        logger.warning("Failed to post trainer_ready: %s", e)
        return None


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Deploy Radar miner")
    parser.add_argument("--agent_dir", type=str, default="miner_template/",
                        help="Directory containing agent .py files")
    parser.add_argument("--trainer_image", type=str,
                        default="ghcr.io/tensorlink-ai/radar/radar-runner:latest",
                        help="Trainer image (deployed on Basilica on-demand)")
    parser.add_argument("--listener_port", type=int, default=8090,
                        help="Port for warm-standby trainer listener")
    args = parser.parse_args()

    # Check agent directory
    if not os.path.isdir(args.agent_dir):
        print(f"Error: agent directory not found: {args.agent_dir}")
        return

    py_files = [f for f in os.listdir(args.agent_dir) if f.endswith(".py")]
    if not py_files:
        print(f"Error: no .py files found in {args.agent_dir}")
        return

    print(f"Agent directory: {args.agent_dir}")
    print(f"  .py files:    {sorted(py_files)}")
    print(f"Trainer image:  {args.trainer_image}")
    print(f"Listener port:  {args.listener_port}")
    print()
    print("How it works:")
    print("  1. Miner neuron reads .py files from --agent_dir")
    print("  2. POSTs them to the DB server (stored in R2 + Postgres)")
    print("  3. Validators fetch code, inject into official agent image, run it")
    print()
    print("Start miner:")
    print(f"  python miner/neuron.py \\")
    print(f"    --agent_dir {args.agent_dir} \\")
    print(f"    --listener_port {args.listener_port} \\")
    print(f"    --trainer_image {args.trainer_image}")
    print()
    print("Notes:")
    print("  - Agent code must define: design_architecture(challenge, client)")
    print("  - The client is a GatedClient — the only way to make HTTP requests")
    print("  - Available proxy endpoints in the challenge:")
    print("      challenge['db_url']       — experiment database")
    print("      challenge['desearch_url'] — arxiv search (Desearch)")
    print("      challenge['llm_url']      — LLM inference (Chutes AI)")
    print("  - No GPU needed to start — trainer pods deploy on-demand via Basilica")
    print("  - Set BASILICA_API_TOKEN env var for Basilica SDK auth")
    print("  - To update agent code, edit .py files and restart the miner")


if __name__ == "__main__":
    main()
