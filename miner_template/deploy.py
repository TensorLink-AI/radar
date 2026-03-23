"""Deploy trainer on Basilica + commit agent image to chain.

Miners have two components:
  1. Agent: a Docker image pushed to a registry. Validators pull and run it.
  2. Trainer: the sanctioned training image deployed on Basilica.

Usage:
  # 1. Build and push your agent Docker image
  docker build -t myregistry/my-agent:v1 miner_template/
  docker push myregistry/my-agent:v1

  # 2. Deploy trainer on Basilica + commit to chain
  python miner_template/deploy.py \\
      --docker_image myregistry/my-agent:v1 \\
      --trainer_image ghcr.io/tensorlink-ai/radar/ts-runner:latest
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def deploy_trainer(image: str) -> str:
    """Deploy the sanctioned trainer container on Basilica. Returns URL."""
    # TODO: Implement Basilica deployment via basilica-sdk
    # from basilica import BasilicaClient
    # client = BasilicaClient()
    # pod = client.deploy(name="radar-trainer", image=image, port=8001,
    #                     gpu_count=1, memory="16Gi")
    # return pod.url
    logger.info("Deploy trainer image: %s", image)
    return ""


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Deploy Radar miner")
    parser.add_argument("--docker_image", type=str, required=True,
                        help="Agent Docker image (e.g. myregistry/my-agent:v1)")
    parser.add_argument("--trainer_image", type=str,
                        default="ghcr.io/tensorlink-ai/radar/ts-runner:latest",
                        help="Sanctioned trainer image")
    args = parser.parse_args()

    trainer_url = deploy_trainer(args.trainer_image)

    print(f"Agent image: {args.docker_image}")
    print(f"Trainer URL: {trainer_url}")
    print()
    print("Next steps:")
    print(f"  1. Push your agent image: docker push {args.docker_image}")
    print(f"  2. Start miner:")
    print(f"     python miner/neuron.py \\")
    print(f"       --docker_image {args.docker_image} \\")
    print(f"       --trainer_url {trainer_url or '<basilica-url>'}")


if __name__ == "__main__":
    main()
