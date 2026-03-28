"""Tests for cross-modality scoring fairness.

Two miners on different modalities scored purely on normalised_ce.
Secondary metrics (CRPS, perplexity, etc.) must not affect frontier.
"""

import math

from shared.database import DataElement
from shared.pareto import ParetoFront
from shared.scoring import score_round


class _MockChallenge:
    min_flops_equivalent = 100_000
    max_flops_equivalent = 500_000


def _gc_objectives():
    from shared.task import Objective
    return [
        Objective(
            name="normalised_ce",
            pattern=r"normalised_ce:\s*([\d.]+)",
            lower_is_better=True, weight=1.0, primary=True,
        ),
        Objective(
            name="crps",
            pattern=r"crps:\s*([\d.]+)",
            lower_is_better=True, weight=0.0,
        ),
        Objective(
            name="perplexity",
            pattern=r"perplexity:\s*([\d.]+)",
            lower_is_better=True, weight=0.0,
        ),
        Objective(
            name="mse",
            pattern=r"mse:\s*([\d.]+)",
            lower_is_better=True, weight=0.0,
        ),
    ]


def _pareto():
    return ParetoFront(max_size=10)


def test_cross_modality_scoring_pure_normalised_ce():
    """Two miners, different modalities, scored only on normalised_ce."""
    eval_results = {
        # Miner 0: continuous modality (has CRPS, no perplexity)
        0: {
            "normalised_ce": 0.8,
            "crps": 0.5,
            "mase": 0.6,
            "perplexity": float("inf"),
            "mse": float("inf"),
            "flops_equivalent_size": 200_000,
            "passed_size_gate": True,
        },
        # Miner 1: tokens modality (has perplexity, no CRPS)
        1: {
            "normalised_ce": 0.7,
            "crps": float("inf"),
            "mase": float("inf"),
            "perplexity": 3.5,
            "mse": float("inf"),
            "flops_equivalent_size": 300_000,
            "passed_size_gate": True,
        },
    }
    scores = score_round(
        eval_results, _MockChallenge(), _pareto(), _gc_objectives(), {},
    )
    # Miner 1 has better normalised_ce -> higher score
    assert scores[1] > scores[0]


def test_secondary_metrics_dont_affect_ranking():
    """Different secondary metrics don't change who wins — only normalised_ce matters."""
    eval_results = {
        # Miner 0: worse normalised_ce but excellent CRPS
        0: {
            "normalised_ce": 0.9,
            "crps": 0.01,  # very good CRPS (irrelevant)
            "flops_equivalent_size": 200_000,
            "passed_size_gate": True,
        },
        # Miner 1: better normalised_ce but no CRPS
        1: {
            "normalised_ce": 0.7,
            "crps": float("inf"),
            "perplexity": 2.0,
            "flops_equivalent_size": 200_000,
            "passed_size_gate": True,
        },
    }
    scores = score_round(
        eval_results, _MockChallenge(), _pareto(), _gc_objectives(), {},
    )
    # Miner 1 wins on normalised_ce despite having no CRPS
    assert scores[1] > scores[0]
