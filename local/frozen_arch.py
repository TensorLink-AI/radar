"""Frozen-architecture snapshots for the ts_data_pipeline task.

The data-pipeline task scores miners on how well a synthetic data generator
trains a *fixed* model. To keep the comparison fair across rounds — but still
chase improvements as the architecture frontier moves — we freeze the current
best ts_forecasting architecture as a versioned snapshot and refresh it every
N successful ts_data_pipeline rounds.

Layout::

    $RADAR_FROZEN_ARCH_DIR/        (default local/frozen_archs/)
      manifest.json                # {"current": N, "versions": [...]}
      v000001.json                 # {version, code, source_*, created_at}
      v000002.json
      ...

Versioning is monotonic; promotion picks the lowest-metric ts_forecasting
experiment with both crps + mase recorded. Promotion is gated on
``every_n`` *successful* ts_data_pipeline rounds — mirroring the
``successful_round_count`` clock continuation already uses — so a stalled
data-pipeline run can't silently swap the target out from under itself.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DIR = "local/frozen_archs"
DATA_PIPELINE_TASK = "ts_data_pipeline"
SOURCE_TASK = "ts_forecasting"


@dataclass
class FrozenArch:
    version: int
    code: str
    source_experiment_id: int
    source_metric: float
    source_crps: float
    source_mase: float
    source_flops: int
    source_name: str
    created_at: float

    def to_dict(self) -> dict:
        return asdict(self)


class FrozenArchStore:
    """Versioned on-disk store of frozen architectures."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(
            base_dir or os.environ.get("RADAR_FROZEN_ARCH_DIR") or DEFAULT_DIR,
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.base_dir / "manifest.json"

    def _read_manifest(self) -> dict:
        if not self._manifest_path.exists():
            return {"current": 0, "versions": []}
        try:
            return json.loads(self._manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {"current": 0, "versions": []}

    def _write_manifest(self, m: dict) -> None:
        tmp = self._manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(m, indent=2))
        tmp.replace(self._manifest_path)

    def _version_path(self, version: int) -> Path:
        return self.base_dir / f"v{version:06d}.json"

    def current_version(self) -> int:
        return int(self._read_manifest().get("current", 0))

    def current(self) -> Optional[FrozenArch]:
        v = self.current_version()
        if v <= 0:
            return None
        return self.load(v)

    def load(self, version: int) -> Optional[FrozenArch]:
        p = self._version_path(version)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            return FrozenArch(**d)
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("frozen arch v%d unreadable: %s", version, e)
            return None

    def all_versions(self) -> list[dict]:
        """Metadata (no code) for every persisted version, oldest first."""
        out: list[dict] = []
        for version in sorted(self._read_manifest().get("versions", [])):
            arch = self.load(int(version))
            if arch is None:
                continue
            d = arch.to_dict()
            d.pop("code", None)
            out.append(d)
        return out

    def save(self, *, code: str, source_experiment_id: int,
             source_metric: float, source_crps: float, source_mase: float,
             source_flops: int, source_name: str) -> FrozenArch:
        next_version = self.current_version() + 1
        arch = FrozenArch(
            version=next_version,
            code=code,
            source_experiment_id=int(source_experiment_id),
            source_metric=float(source_metric),
            source_crps=float(source_crps),
            source_mase=float(source_mase),
            source_flops=int(source_flops),
            source_name=source_name,
            created_at=time.time(),
        )
        self._version_path(next_version).write_text(
            json.dumps(arch.to_dict(), indent=2),
        )
        m = self._read_manifest()
        versions = list(m.get("versions", []))
        versions.append(next_version)
        self._write_manifest({"current": next_version, "versions": versions})
        logger.info(
            "frozen arch v%d snapshotted from exp=%d metric=%.6f flops=%d",
            next_version, source_experiment_id, source_metric, source_flops,
        )
        return arch


def _candidate_from_store(store) -> Optional[dict]:
    """Pick the lowest-metric ts_forecasting experiment with crps + mase."""
    exps = store.recent_experiments(n=10_000)
    cands = [
        e for e in exps
        if e.get("task") == SOURCE_TASK
        and e.get("success")
        and e.get("metric") is not None
        and e.get("objectives", {}).get("crps") is not None
        and e.get("objectives", {}).get("mase") is not None
        and e.get("code")
    ]
    if not cands:
        return None
    return min(cands, key=lambda e: e["metric"])


def maybe_refresh(
    arch_store: FrozenArchStore, db_store, *, every_n: int = 50,
) -> Optional[FrozenArch]:
    """Snapshot a new frozen arch when conditions are met.

    Snapshots when either:
      * there is no current version yet (bootstrap), or
      * ``every_n`` successful ts_data_pipeline rounds have completed since
        the last snapshot.

    Returns the newly-saved FrozenArch on snapshot, None otherwise.
    """
    cand = _candidate_from_store(db_store)
    if cand is None:
        return None
    current = arch_store.current()
    if current is None:
        return _do_save(arch_store, cand)

    # Refresh cadence: every N successful data-pipeline rounds.
    successful = db_store.successful_round_count(task=DATA_PIPELINE_TASK)
    if every_n <= 0:
        return None
    last_refresh_at = current.created_at
    # Count successful data-pipeline rounds that happened since last refresh.
    # We approximate by counting rows newer than created_at to avoid storing
    # extra state in the manifest.
    try:
        row = db_store._conn.execute(
            "SELECT COUNT(DISTINCT round_id) AS n FROM experiments "
            "WHERE success = 1 AND task = ? AND timestamp > ?",
            (DATA_PIPELINE_TASK, last_refresh_at),
        ).fetchone()
        rounds_since = int(row["n"] or 0)
    except Exception:  # noqa: BLE001
        rounds_since = successful
    if rounds_since < every_n:
        return None
    # Only refresh if the candidate is meaningfully better than the current
    # frozen source (avoids re-snapshotting the same experiment).
    if cand["id"] == current.source_experiment_id:
        return None
    return _do_save(arch_store, cand)


def _do_save(arch_store: FrozenArchStore, cand: dict) -> FrozenArch:
    objs = cand.get("objectives", {}) or {}
    return arch_store.save(
        code=cand["code"],
        source_experiment_id=int(cand["id"]),
        source_metric=float(cand["metric"]),
        source_crps=float(objs.get("crps") or 0.0),
        source_mase=float(objs.get("mase") or 0.0),
        source_flops=int(objs.get("flops_equivalent_size") or 0),
        source_name=str(cand.get("name") or ""),
    )
