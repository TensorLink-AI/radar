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
    scratchpad_get_url: str = ""   # presigned GET URL for scratchpad.tar.gz
    scratchpad_put_url: str = ""   # presigned PUT URL for scratchpad.tar.gz
    scratchpad_max_mb: int = 10    # size limit enforced by agent

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
            "scratchpad_max_mb": self.scratchpad_max_mb,
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


@dataclass
class TrainerRequest:
    """Validator → Miner: prepare your trainer pod for this round.

    Includes the GPU spec the miner must deploy on Basilica.
    The validator controls hardware requirements to ensure consistent
    training conditions across the round.
    """
    round_id: int = 0
    challenge_id: str = ""
    seed: int = 0
    min_flops_equivalent: int = 0
    max_flops_equivalent: int = 0
    time_budget: int = 300
    validator_db_url: str = ""

    # GPU spec — miner must deploy a pod matching these requirements
    gpu_count: int = 1
    gpu_model: str = "NVIDIA-RTX-A4000"
    memory: str = "16Gi"

    def to_json(self) -> str:
        return json.dumps({
            "round_id": self.round_id,
            "challenge_id": self.challenge_id,
            "seed": self.seed,
            "min_flops_equivalent": self.min_flops_equivalent,
            "max_flops_equivalent": self.max_flops_equivalent,
            "time_budget": self.time_budget,
            "validator_db_url": self.validator_db_url,
            "gpu_count": self.gpu_count,
            "gpu_model": self.gpu_model,
            "memory": self.memory,
        })

    @classmethod
    def from_json(cls, s: str) -> TrainerRequest:
        d = json.loads(s)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainerReady:
    """Miner → Validator: my trainer pod is live at this URL."""
    round_id: int = 0
    trainer_url: str = ""
    instance_name: str = ""
    miner_hotkey: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "round_id": self.round_id,
            "trainer_url": self.trainer_url,
            "instance_name": self.instance_name,
            "miner_hotkey": self.miner_hotkey,
        })

    @classmethod
    def from_json(cls, s: str) -> TrainerReady:
        d = json.loads(s)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainerRelease:
    """Validator → Miner: training done, tear down your pod."""
    round_id: int = 0
    miner_hotkey: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "round_id": self.round_id,
            "miner_hotkey": self.miner_hotkey,
        })

    @classmethod
    def from_json(cls, s: str) -> TrainerRelease:
        d = json.loads(s)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
