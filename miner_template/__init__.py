"""Minimal subset of miner_template kept for the local stack.

Surfaces:

* ``miner_template.prompts`` — atomic-write population store the local
  agent reads (``active.json``) and ``local/optimize.py`` writes.
* ``miner_template.optimizers`` — pluggable registry plus the built-in
  ``random_mutate`` and ``gepa`` adapters.

The original distributed-deployment surfaces (``agent.py``,
``deploy.py``, ``results_client.py``) were removed when the repo was
pruned to the local stack.
"""
