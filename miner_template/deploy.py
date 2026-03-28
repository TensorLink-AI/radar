"""Deploy guide for Radar miners — warm-standby trainer model.

Miners have two components:
  1. Agent: a Docker image pushed to a registry. Validators pull and run it.
  2. Trainer listener: a lightweight HTTP server on the miner's neuron process.
     GPU pods deploy on-demand via Basilica when validators send TrainerRequests,
     with public_metadata=True for attestation.

No upfront GPU cost — the miner just needs the neuron process running with
an open port. GPU pods are created during Phase A and auto-teardown via TTL.

Usage:
  # 1. Build and push your agent Docker image
  docker build -t myregistry/my-agent:v1 miner_template/
  docker push myregistry/my-agent:v1

  # 2. Start miner with listener (no GPU needed until training starts)
  python miner/neuron.py \\
      --docker_image myregistry/my-agent:v1 \\
      --listener_port 8090 \\
      --trainer_image ghcr.io/tensorlink-ai/radar/ts-runner:latest
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Deploy Radar miner")
    parser.add_argument("--docker_image", type=str, required=True,
                        help="Agent Docker image (e.g. myregistry/my-agent:v1)")
    parser.add_argument("--trainer_image", type=str,
                        default="ghcr.io/tensorlink-ai/radar/ts-runner:latest",
                        help="Sanctioned trainer image (deployed on Basilica on-demand)")
    parser.add_argument("--listener_port", type=int, default=8090,
                        help="Port for warm-standby trainer listener")
    args = parser.parse_args()

    print(f"Agent image:    {args.docker_image}")
    print(f"Trainer image:  {args.trainer_image}")
    print(f"Listener port:  {args.listener_port}")
    print()
    print("Next steps:")
    print(f"  1. Push your agent image: docker push {args.docker_image}")
    print(f"  2. Start miner:")
    print(f"     python miner/neuron.py \\")
    print(f"       --docker_image {args.docker_image} \\")
    print(f"       --listener_port {args.listener_port} \\")
    print(f"       --trainer_image {args.trainer_image}")
    print()
    print("Notes:")
    print("  - No GPU needed to start — pods deploy on-demand via Basilica")
    print("  - Validators control GPU spec (gpu_count, gpu_model, memory)")
    print("  - Pods auto-teardown via TTL if release signal is missed")
    print("  - Set BASILICA_API_TOKEN env var for Basilica SDK auth")


if __name__ == "__main__":
    main()
