"""Declarative experiment spec.

One YAML file per ablation. The orchestrator reads it, expands ``base``
+ each ``units[*].overrides`` into a concrete per-unit launch config,
and submits to the scheduler.

Schema (one example::

    name: cont-equilibrium-sweep-2026-06
    base:
      task: ts_forecasting
      rounds: 200
      miners: 2
      agent_dir: miners/claude_style_v2
      training_seconds: 3600
      shards_per_round: 4
    units:
      - id: eq50
        overrides: { continuation_equilibrium: 0.5 }
      - id: eq70
        overrides: { continuation_equilibrium: 0.7 }
    pod:
      provider: runpod
      gpu: A100-40GB
      image: ghcr.io/tensorlink-ai/radar:latest
      disk_gb: 200
      max_hours: 24
    cost:
      max_usd_per_hour: 0.30
      budget_usd: 50

is the canonical shape.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Whitelist of flags accepted in base / overrides. Anything outside this
# set is rejected at load time so a typo doesn't silently produce a
# launch with the wrong knob.
_BASE_FLAGS: frozenset[str] = frozenset({
    "task", "tasks",
    "rounds", "miners",
    "agent_dir", "agent_module", "wiki_dir",
    "phase_a_seconds", "agent_seconds", "training_seconds", "gap_seconds",
    "shards_per_round",
    "checkpoint_dir",
    "continuation",
    "continuation_warmup_rounds", "continuation_step_pct",
    "continuation_step_every", "continuation_equilibrium",
    "frozen_arch_refresh_every", "frozen_pipeline_refresh_every",
    "frozen_pipeline_num_shards", "frozen_pipeline_batches_per_shard",
    "log_level",
})

_UNIT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")


@dataclass
class PodSpec:
    provider: str = "runpod"
    gpu: str = "A100-40GB"
    image: str = ""
    disk_gb: int = 100
    max_hours: float = 24.0
    # Provider-specific kwargs (region, network volume id, etc.) — passed
    # through to the provider create() call.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostSpec:
    max_usd_per_hour: float = 0.50
    budget_usd: float = 100.0


@dataclass
class UnitSpec:
    id: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSpec:
    name: str
    base: dict[str, Any]
    units: list[UnitSpec]
    pod: PodSpec
    cost: CostSpec
    # Raw source (for audit copy into experiments/<name>/spec.yaml).
    raw: str = ""

    def merged(self, unit: UnitSpec) -> dict[str, Any]:
        """Return base + unit.overrides as the flat config for that unit."""
        out = dict(self.base)
        out.update(unit.overrides)
        return out


def _yaml_load(text: str) -> Any:
    """Best-effort YAML load.

    We use PyYAML if available; otherwise fall back to JSON (so the same
    files can be authored as JSON for environments without the extra).
    """
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "meta.spec needs PyYAML for YAML files; install "
                "`pip install -e .[meta]` or convert the spec to JSON."
            ) from e


def load_spec(path: str | Path) -> ExperimentSpec:
    path = Path(path)
    raw = path.read_text()
    data = _yaml_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    return parse_spec(data, raw=raw)


def parse_spec(data: dict[str, Any], raw: str = "") -> ExperimentSpec:
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("spec.name must be a non-empty string")
    if not _UNIT_ID_RE.match(name):
        # Allow longer names for experiment-level (vs unit IDs).
        if not re.match(r"^[a-z0-9][a-z0-9_.\-]{0,63}$", name):
            raise ValueError(
                f"spec.name {name!r} must match [a-z0-9][a-z0-9_.\\-]{{0,63}}"
            )

    base = data.get("base") or {}
    if not isinstance(base, dict):
        raise ValueError("spec.base must be a mapping")
    _validate_flags(base, where="base")

    raw_units = data.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ValueError("spec.units must be a non-empty list")
    units: list[UnitSpec] = []
    seen_ids: set[str] = set()
    for i, u in enumerate(raw_units):
        if not isinstance(u, dict):
            raise ValueError(f"spec.units[{i}] must be a mapping")
        uid = u.get("id")
        if not isinstance(uid, str) or not _UNIT_ID_RE.match(uid):
            raise ValueError(
                f"spec.units[{i}].id must match {_UNIT_ID_RE.pattern}"
            )
        if uid in seen_ids:
            raise ValueError(f"duplicate unit id: {uid}")
        seen_ids.add(uid)
        overrides = u.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"spec.units[{i}].overrides must be a mapping")
        _validate_flags(overrides, where=f"units[{i}].overrides")
        units.append(UnitSpec(id=uid, overrides=dict(overrides)))

    pod_raw = data.get("pod") or {}
    if not isinstance(pod_raw, dict):
        raise ValueError("spec.pod must be a mapping")
    pod = PodSpec(
        provider=str(pod_raw.get("provider", "runpod")),
        gpu=str(pod_raw.get("gpu", "A100-40GB")),
        image=str(pod_raw.get("image", "")),
        disk_gb=int(pod_raw.get("disk_gb", 100)),
        max_hours=float(pod_raw.get("max_hours", 24.0)),
        extra={
            k: v for k, v in pod_raw.items()
            if k not in {"provider", "gpu", "image", "disk_gb", "max_hours"}
        },
    )

    cost_raw = data.get("cost") or {}
    if not isinstance(cost_raw, dict):
        raise ValueError("spec.cost must be a mapping")
    cost = CostSpec(
        max_usd_per_hour=float(cost_raw.get("max_usd_per_hour", 0.50)),
        budget_usd=float(cost_raw.get("budget_usd", 100.0)),
    )
    if cost.max_usd_per_hour <= 0:
        raise ValueError("cost.max_usd_per_hour must be > 0")
    if cost.budget_usd <= 0:
        raise ValueError("cost.budget_usd must be > 0")

    return ExperimentSpec(
        name=name,
        base=dict(base),
        units=units,
        pod=pod,
        cost=cost,
        raw=raw,
    )


def _validate_flags(d: dict[str, Any], *, where: str) -> None:
    bad = [k for k in d if k not in _BASE_FLAGS]
    if bad:
        raise ValueError(
            f"{where}: unknown flag(s) {bad!r}; allowed: "
            f"{sorted(_BASE_FLAGS)}"
        )


def render_run_argv(cfg: dict[str, Any], *, db_path: str) -> list[str]:
    """Render a merged unit cfg into a ``python local/run.py …`` argv.

    Used by the scheduler when SSHing into a pod. Keeps the flag mapping
    in one place so spec validation and launch stay in sync.
    """
    argv: list[str] = ["python", "local/run.py", "--db", db_path]
    for k, v in cfg.items():
        flag = "--" + k
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv.extend([flag, str(v)])
    return argv
