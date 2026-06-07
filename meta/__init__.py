"""Meta-orchestrator for radar experimental units.

A sibling to ``local/`` that knows how to spin up N validator+miner
"experimental units" on a cloud provider (RunPod first), monitor them,
budget-cap them, and aggregate results across experiments so a
higher-level Claude Code session can run a research loop.

This package does not modify ``local/`` — it launches ``local/run.py``
over SSH and reads the R2 paths the existing code already writes.
"""

from __future__ import annotations

__version__ = "0.1.0"
