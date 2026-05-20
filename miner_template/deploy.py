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
import logging
import os

logger = logging.getLogger(__name__)


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
