"""Local prompt-population store for miners.

A *population* is a small list of ``Prompt`` rows kept on disk under
``prompts/`` next to the miner's working directory:

    prompts/
      active.json              # current population the agent reads
      history/
        gen_001.json           # archived populations
        gen_002.json
        ...

Atomic writes use ``tmp + os.replace`` so an interrupted optimizer
never leaves the agent with a half-written file.

The agent reads the active population each round via ``load_active()``
and picks one variant with ``pick_for_round(pop, round_id)``.  The
chosen prompt's ``id`` round-trips back via the submission so the
optimizer can correlate Phase C scores with prompt variants.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS_DIR = Path(os.getenv("MINER_PROMPTS_DIR", "prompts"))
ACTIVE_FILE = "active.json"
HISTORY_DIR = "history"


@dataclass
class Prompt:
    """A single prompt variant in the population."""

    id: str
    template: str
    generation: int = 0
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def new(cls, template: str, *, generation: int = 0,
            parent_id: Optional[str] = None,
            metadata: Optional[dict] = None) -> "Prompt":
        return cls(
            id=uuid.uuid4().hex,
            template=template,
            generation=generation,
            parent_id=parent_id,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Prompt":
        return cls(
            id=str(d.get("id") or uuid.uuid4().hex),
            template=str(d.get("template", "")),
            generation=int(d.get("generation", 0)),
            parent_id=d.get("parent_id"),
            metadata=dict(d.get("metadata") or {}),
        )


# ── Filesystem layout helpers ────────────────────────────────────────


def _root(prompts_dir: Optional[Path] = None) -> Path:
    return Path(prompts_dir or DEFAULT_PROMPTS_DIR)


def _active_path(prompts_dir: Optional[Path] = None) -> Path:
    return _root(prompts_dir) / ACTIVE_FILE


def _history_dir(prompts_dir: Optional[Path] = None) -> Path:
    return _root(prompts_dir) / HISTORY_DIR


def _atomic_write_json(path: Path, payload) -> None:
    """Write ``payload`` to ``path`` via ``.tmp`` + ``os.replace`` so the
    reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ── Active-population I/O ────────────────────────────────────────────


def load_active(prompts_dir: Optional[Path] = None) -> list[Prompt]:
    """Read the active population.  Returns ``[]`` if the file doesn't
    exist yet (fresh miner) — callers should treat this as "seed me"."""
    path = _active_path(prompts_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("prompts: active.json unreadable (%s) — empty pop", e)
        return []
    rows = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    return [Prompt.from_dict(r) for r in rows if isinstance(r, dict)]


def save_active(prompts: Iterable[Prompt],
                prompts_dir: Optional[Path] = None) -> None:
    """Atomically write the active population."""
    items = [p.to_dict() for p in prompts]
    payload = {"prompts": items}
    _atomic_write_json(_active_path(prompts_dir), payload)


def archive_current(generation: int,
                    prompts_dir: Optional[Path] = None) -> Optional[Path]:
    """Copy the current ``active.json`` into ``history/gen_NNN.json``.

    Returns the destination path, or ``None`` if there's no active file
    yet (first generation — nothing to archive).
    """
    src = _active_path(prompts_dir)
    if not src.exists():
        return None
    dst = _history_dir(prompts_dir) / f"gen_{generation:03d}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return dst


def list_history(prompts_dir: Optional[Path] = None) -> list[int]:
    """Return archived generation numbers, sorted ascending."""
    d = _history_dir(prompts_dir)
    if not d.exists():
        return []
    gens: list[int] = []
    for f in d.iterdir():
        if not f.name.startswith("gen_") or not f.name.endswith(".json"):
            continue
        try:
            gens.append(int(f.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(gens)


def load_generation(generation: int,
                    prompts_dir: Optional[Path] = None) -> list[Prompt]:
    """Read an archived generation back into memory."""
    path = _history_dir(prompts_dir) / f"gen_{generation:03d}.json"
    if not path.exists():
        raise FileNotFoundError(f"no archived generation {generation}")
    payload = json.loads(path.read_text())
    rows = payload.get("prompts") if isinstance(payload, dict) else payload
    return [Prompt.from_dict(r) for r in rows if isinstance(r, dict)]


def rollback_to(generation: int,
                prompts_dir: Optional[Path] = None) -> None:
    """Restore an archived generation as the new active population.

    The currently active population is first archived under the next
    free generation number, so rollback is reversible.
    """
    pop = load_generation(generation, prompts_dir)
    next_gen = (list_history(prompts_dir) or [0])[-1] + 1
    archive_current(next_gen, prompts_dir)
    save_active(pop, prompts_dir)


# ── Selection ────────────────────────────────────────────────────────


def pick_for_round(population: list[Prompt], round_id: int) -> Prompt:
    """Deterministic round-robin over the population.

    Same ``round_id`` always picks the same prompt — so explore/exploit
    looks the same to anyone replaying the round.  Empty population
    raises ``ValueError``; callers should ``seed_default()`` first.
    """
    if not population:
        raise ValueError("empty population — call seed_default() first")
    return population[round_id % len(population)]


# ── Seeding ──────────────────────────────────────────────────────────


DEFAULT_SEED_TEMPLATE = (
    "You are a model-architecture researcher designing a PyTorch model "
    "for time-series forecasting.\n"
    "Round budget: FLOPs in [{min_flops}, {max_flops}].\n"
    "Frontier so far (best metric per architecture):\n"
    "{frontier}\n\n"
    "Design a single architecture targeting ~60% of the FLOPs ceiling. "
    "Return the model code, a short name, and a one-line motivation."
)


def seed_default(prompts_dir: Optional[Path] = None,
                 template: str = DEFAULT_SEED_TEMPLATE) -> list[Prompt]:
    """Write a single starter prompt as the active population.  No-op
    if ``active.json`` already exists.  Returns the resulting active
    population either way.
    """
    existing = load_active(prompts_dir)
    if existing:
        return existing
    seed = Prompt.new(template=template, generation=0)
    save_active([seed], prompts_dir)
    return [seed]
