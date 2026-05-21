"""Miner template — starter agent + prompt-evolution scaffolding.

The miner CLI (``miner/neuron.py``) drives this package's two surfaces:

  * ``agent.py`` — ``design_architecture(challenge, client)`` entrypoint
    called by the validator each round.
  * ``prompts`` + ``optimizers`` + ``results_client`` — feedback loop
    that lets miners pull their own scored history and evolve their
    agent's prompts (e.g. via GEPA).

See ``docs/optimizer.md`` for the loop in full.
"""
