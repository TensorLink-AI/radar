"""Frozen-pipeline snapshots — the mirror of ``local/frozen_arch.py``.

The ts_forecasting task can mix synthetic shards produced by the current
best ts_data_pipeline run into its pretrain stream. To keep this fair
across rounds — and to give the data-pipeline track a downstream
yardstick — we snapshot the current-best pipeline as a versioned frozen
artifact, re-render its synthetic shards once at promotion, and refresh
every N successful ts_forecasting rounds.

Refresh cadence is intentionally **staggered** from the frozen-arch
refresh (defaults: arch every 50 pipeline rounds, pipeline every 30
forecasting rounds). If both refreshed at the same time the absolute
ts_forecasting frontier would jump for two unrelated reasons at once
and attribution would be impossible.

Layout::

    $RADAR_FROZEN_PIPELINE_DIR/        (default local/frozen_pipelines/)
      manifest.json                    # {"current": N, "versions": [...]}
      v000001.json                     # {version, code, source_*, shard_paths}
      v000001/shards/*.parquet         # pre-rendered synthetic shards
      v000002.json
      v000002/shards/*.parquet
      ...
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DIR = "local/frozen_pipelines"
SOURCE_TASK = "ts_data_pipeline"
CONSUMER_TASK = "ts_forecasting"


@dataclass
class FrozenPipeline:
    version: int
    code: str
    source_experiment_id: int
    source_metric: float
    source_crps: float
    source_mase: float
    source_flops: int
    source_name: str
    frozen_arch_version: int
    created_at: float
    shard_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class FrozenPipelineStore:
    """Versioned on-disk store of frozen synthetic generators."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(
            base_dir or os.environ.get("RADAR_FROZEN_PIPELINE_DIR") or DEFAULT_DIR,
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

    def _version_meta_path(self, version: int) -> Path:
        return self.base_dir / f"v{version:06d}.json"

    def version_shard_dir(self, version: int) -> Path:
        return self.base_dir / f"v{version:06d}" / "shards"

    def current_version(self) -> int:
        return int(self._read_manifest().get("current", 0))

    def current(self) -> Optional[FrozenPipeline]:
        v = self.current_version()
        if v <= 0:
            return None
        return self.load(v)

    def load(self, version: int) -> Optional[FrozenPipeline]:
        p = self._version_meta_path(version)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            d.setdefault("shard_paths", [])
            return FrozenPipeline(**d)
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("frozen pipeline v%d unreadable: %s", version, e)
            return None

    def all_versions(self) -> list[dict]:
        """Metadata (no code) for every persisted version, oldest first."""
        out: list[dict] = []
        for version in sorted(self._read_manifest().get("versions", [])):
            pipe = self.load(int(version))
            if pipe is None:
                continue
            d = pipe.to_dict()
            d.pop("code", None)
            out.append(d)
        return out

    def save(self, *, code: str, source_experiment_id: int,
             source_metric: float, source_crps: float, source_mase: float,
             source_flops: int, source_name: str,
             frozen_arch_version: int,
             shard_paths: Optional[list[str]] = None) -> FrozenPipeline:
        next_version = self.current_version() + 1
        pipe = FrozenPipeline(
            version=next_version,
            code=code,
            source_experiment_id=int(source_experiment_id),
            source_metric=float(source_metric),
            source_crps=float(source_crps),
            source_mase=float(source_mase),
            source_flops=int(source_flops),
            source_name=source_name,
            frozen_arch_version=int(frozen_arch_version),
            created_at=time.time(),
            shard_paths=list(shard_paths or []),
        )
        self._version_meta_path(next_version).write_text(
            json.dumps(pipe.to_dict(), indent=2),
        )
        m = self._read_manifest()
        versions = list(m.get("versions", []))
        versions.append(next_version)
        self._write_manifest({"current": next_version, "versions": versions})
        logger.info(
            "frozen pipeline v%d snapshotted from exp=%d metric=%.6f shards=%d",
            next_version, source_experiment_id, source_metric, len(pipe.shard_paths),
        )
        return pipe


def _candidate_from_store(store) -> Optional[dict]:
    """Pick the lowest-metric ts_data_pipeline experiment with crps + mase."""
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
    pipe_store: FrozenPipelineStore, db_store, *,
    every_n: int = 30, num_shards: int = 16, batches_per_shard: int = 64,
) -> Optional[FrozenPipeline]:
    """Snapshot a new frozen pipeline when conditions are met.

    Triggers on either:
      * No current version (bootstrap), or
      * ``every_n`` successful ts_forecasting rounds since the last
        snapshot (staggered from the arch refresh — see module docstring).

    Returns the newly-saved FrozenPipeline on snapshot, None otherwise.
    """
    cand = _candidate_from_store(db_store)
    if cand is None:
        return None
    current = pipe_store.current()
    if current is None:
        return _do_save(pipe_store, cand, num_shards, batches_per_shard)
    if every_n <= 0:
        return None
    try:
        row = db_store._conn.execute(
            "SELECT COUNT(DISTINCT round_id) AS n FROM experiments "
            "WHERE success = 1 AND task = ? AND timestamp > ?",
            (CONSUMER_TASK, current.created_at),
        ).fetchone()
        rounds_since = int(row["n"] or 0)
    except Exception:  # noqa: BLE001
        rounds_since = 0
    if rounds_since < every_n:
        return None
    if cand["id"] == current.source_experiment_id:
        return None
    return _do_save(pipe_store, cand, num_shards, batches_per_shard)


def _do_save(pipe_store: FrozenPipelineStore, cand: dict,
             num_shards: int, batches_per_shard: int) -> FrozenPipeline:
    objs = cand.get("objectives", {}) or {}
    next_version = pipe_store.current_version() + 1
    shards_dir = pipe_store.version_shard_dir(next_version)
    shard_paths = _render_shards(
        cand["code"], shards_dir,
        num_shards=num_shards, batches_per_shard=batches_per_shard,
    )
    return pipe_store.save(
        code=cand["code"],
        source_experiment_id=int(cand["id"]),
        source_metric=float(cand["metric"]),
        source_crps=float(objs.get("crps") or 0.0),
        source_mase=float(objs.get("mase") or 0.0),
        source_flops=int(objs.get("flops_equivalent_size") or 0),
        source_name=str(cand.get("name") or ""),
        frozen_arch_version=int(objs.get("frozen_arch_version") or 0),
        shard_paths=shard_paths,
    )


def _render_shards(
    code: str, out_dir: Path, *, num_shards: int, batches_per_shard: int,
) -> list[str]:
    """Pre-render synthetic shards from the miner's pipeline.

    Best-effort: if torch / pandas aren't importable, returns ``[]`` and
    the frozen pipeline is still recorded — just without shards. A later
    refresh on a richer environment can re-render.
    """
    if num_shards <= 0 or batches_per_shard <= 0:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from local.data_pipeline import _exec_pipeline_submission
        from local.task import (
            TS_CONTEXT_LEN, TS_NUM_VARIATES, TS_PREDICTION_LEN, TS_QUANTILES,
        )
        import pandas as pd
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "frozen pipeline shard render skipped (missing deps): %s", e,
        )
        return []

    try:
        build_pipeline = _exec_pipeline_submission(code)
        iterator = iter(build_pipeline(
            TS_CONTEXT_LEN, TS_PREDICTION_LEN, TS_NUM_VARIATES,
            list(TS_QUANTILES),
        ))
    except Exception as e:  # noqa: BLE001
        logger.warning("frozen pipeline render: build_pipeline failed: %s", e)
        return []

    # Drain ``num_shards`` worth of rows from a single iterator. Calling
    # build_pipeline once per shard would produce identical shards for a
    # deterministic pipeline; consuming from one iterator lets within-stream
    # variation (or RNG advance) carry across shards.
    saved: list[str] = []
    for shard_idx in range(num_shards):
        try:
            rows = _drain_to_rows(iterator, batches_per_shard)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "frozen pipeline render: shard %d aborted (%s)", shard_idx, e,
            )
            break
        if not rows:
            break  # iterator exhausted — stop rather than write empty shards
        out_path = out_dir / f"synth-{shard_idx:05d}.parquet"
        try:
            pd.DataFrame(
                {"target": rows, "freq": ["synthetic"] * len(rows)},
            ).to_parquet(out_path, index=False)
            saved.append(str(out_path))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "frozen pipeline render: write shard %d failed (%s)",
                shard_idx, e,
            )
    logger.info(
        "rendered %d/%d synthetic shards into %s",
        len(saved), num_shards, out_dir,
    )
    return saved


def _drain_to_rows(it, batches_per_shard: int) -> list[list[float]]:
    """Pull batches from the miner's iterator; flatten input+target into
    raw series rows (each row a list[float] the pretrain loader expects)."""
    import torch
    out: list[list[float]] = []
    try:
        iterator = iter(it)
    except TypeError:
        return out
    for _ in range(batches_per_shard):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        except Exception:  # noqa: BLE001
            break
        if isinstance(batch, dict):
            inp = batch.get("input", batch.get("context"))
            tgt = batch.get("target")
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            inp, tgt = batch
        else:
            continue
        if inp is None or tgt is None:
            continue
        try:
            inp_t = inp if torch.is_tensor(inp) else torch.as_tensor(inp)
            tgt_t = tgt if torch.is_tensor(tgt) else torch.as_tensor(tgt)
        except Exception:  # noqa: BLE001
            continue
        if inp_t.dim() != 3 or tgt_t.dim() != 3:
            continue
        # Concat along time, take the first variate (the loader uses 1D series).
        series = torch.cat([inp_t, tgt_t], dim=1)[:, :, 0]
        try:
            series_list = series.detach().cpu().tolist()
        except Exception:  # noqa: BLE001
            continue
        for row in series_list:
            if not row:
                continue
            out.append([float(x) for x in row])
    return out
