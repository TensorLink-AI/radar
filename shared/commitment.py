"""Miner identity record passed around in-process.

Carries the data validators need about a registered miner — agent code
hash, listener URL, hotkey/uid label, optional trainer image. Used by
collection / coordinator / dispatch as the canonical per-miner payload.
The fields are plain Python; no chain reads or writes live here anymore.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class ImageCommitment:
    """A miner's committed identity: agent code hash + listener URL.

    Agent code is served from the miner's listener (GET /agent_code).
    The code_hash lets validators verify integrity after fetching.
    """

    code_hash: str = ""
    subnet_version: str = ""

    # Warm-standby trainer (miner-hosted lightweight listener, no GPU)
    listener_url: str = ""
    trainer_image: str = ""

    # Agent pod (miner-hosted on Basilica / Targon / RunPod)
    agent_url: str = ""
    agent_attestation_id: str = ""

    # Internal fields
    miner_uid: int = -1
    hotkey: str = ""

    def to_json(self) -> str:
        data = {
            "code_hash": self.code_hash,
            "subnet_version": self.subnet_version,
            "listener_url": self.listener_url,
            "trainer_image": self.trainer_image,
            "agent_url": self.agent_url,
            "agent_attestation_id": self.agent_attestation_id,
        }
        return json.dumps({k: v for k, v in data.items() if v}, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str, miner_uid: int = -1, hotkey: str = "") -> ImageCommitment:
        d = json.loads(s)
        return cls(
            code_hash=d.get("code_hash", "") or d.get("ch", ""),
            subnet_version=d.get("subnet_version", "") or d.get("v", ""),
            listener_url=d.get("listener_url", "") or d.get("l", ""),
            trainer_image=d.get("trainer_image", "") or d.get("t", ""),
            agent_url=d.get("agent_url", "") or d.get("a", ""),
            agent_attestation_id=d.get("agent_attestation_id", "") or d.get("at", ""),
            miner_uid=miner_uid,
            hotkey=hotkey,
        )

    @property
    def is_valid(self) -> bool:
        return bool(self.code_hash) and bool(self.listener_url)
