"""
Wire format between validators and miners.

Challenge: what the validator sends to miners each round.
Proposal: what the miner returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Challenge:
    """Sent by validator to every miner at the start of a round."""

    challenge_id: str = ""
    seed: int = 0
    round_id: int = 0
    min_flops_equivalent: int = 0
    max_flops_equivalent: int = 0
    eval_split_seed: int = 0
    task: dict = field(default_factory=dict)
    db_url: str = ""
    desearch_url: str = ""

    # Pareto front filtered to this round's size bucket
    # Each entry: {code, metric, objectives} for frontier points in range
    feasible_frontier: list = field(default_factory=list)

    # Agent scratchpad — presigned R2 URLs for persistent private storage
    scratchpad_get_url: str = ""  # presigned GET URL for scratchpad.tar.gz
    scratchpad_put_url: str = ""  # presigned PUT URL for scratchpad.tar.gz

    def to_json(self) -> str:
        return json.dumps({
            "challenge_id": self.challenge_id,
            "seed": self.seed,
            "round_id": self.round_id,
            "min_flops_equivalent": self.min_flops_equivalent,
            "max_flops_equivalent": self.max_flops_equivalent,
            "eval_split_seed": self.eval_split_seed,
            "task": self.task,
            "db_url": self.db_url,
            "desearch_url": self.desearch_url,
            "feasible_frontier": self.feasible_frontier,
            "scratchpad_get_url": self.scratchpad_get_url,
            "scratchpad_put_url": self.scratchpad_put_url,
        })

    @classmethod
    def from_json(cls, s: str) -> Challenge:
        d = json.loads(s)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Proposal:
    """Returned by a miner in response to a Challenge."""

    code: str = ""
    name: str = ""
    motivation: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "code": self.code,
            "name": self.name,
            "motivation": self.motivation,
        })

    @classmethod
    def from_json(cls, s: str) -> Proposal:
        d = json.loads(s)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
