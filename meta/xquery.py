"""Cross-experiment SQL — the substrate for the Ralph loop.

Opens every experiment's ``meta.db`` read-only and runs a parameterised
SQL across all of them, unioning the results. This is how the meta
agent gets a single view of the org's research history without us
having to build a data lake.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .store import Registry, experiment_dir


@dataclass
class XRow:
    experiment: str
    unit_id: str
    metric: float | None
    success: int
    task: str
    mode: str
    max_round: int
    n_rows: int
    n_success: int


def open_all(
    data_root: Path,
) -> Iterator[tuple[str, sqlite3.Connection]]:
    """Yield (experiment_name, read-only Connection) for every
    registered experiment's ``meta.db``."""
    reg = Registry(data_root / "registry.db")
    try:
        for row in reg.list_experiments():
            db = Path(row["path"]) / "meta.db"
            if not db.exists():
                continue
            uri = f"file:{db}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                yield row["name"], conn
            finally:
                conn.close()
    finally:
        reg.close()


def best_per_unit(data_root: Path) -> list[XRow]:
    """One row per (experiment, unit) summarizing best metric + counts.

    This is what ``meta xcompare`` defaults to — a compact digest the
    meta agent reasons over.
    """
    out: list[XRow] = []
    for exp, conn in open_all(data_root):
        rows = conn.execute(
            """
            SELECT unit_id,
                   MIN(metric) AS best_metric,
                   MAX(success) AS any_success,
                   COUNT(*)    AS n_rows,
                   SUM(success) AS n_success,
                   MAX(round_id) AS max_round,
                   MAX(task)   AS task,
                   MAX(mode)   AS mode
            FROM frontier
            GROUP BY unit_id
            """
        ).fetchall()
        for r in rows:
            out.append(XRow(
                experiment=exp,
                unit_id=r["unit_id"],
                metric=r["best_metric"],
                success=int(r["any_success"] or 0),
                task=r["task"] or "",
                mode=r["mode"] or "new",
                max_round=int(r["max_round"] or 0),
                n_rows=int(r["n_rows"] or 0),
                n_success=int(r["n_success"] or 0),
            ))
    return out


def query_all(
    data_root: Path,
    sql: str,
    params: tuple = (),
) -> list[dict[str, Any]]:
    """Run an arbitrary SQL across every experiment's ``meta.db`` and
    return a single flat list of rows (with an ``experiment`` column
    prepended).

    The SQL must be read-only; we open each DB in ``mode=ro`` so any
    DML/DDL will error out at execution time.
    """
    out: list[dict[str, Any]] = []
    for exp, conn in open_all(data_root):
        for row in conn.execute(sql, params).fetchall():
            d = {"experiment": exp}
            d.update({k: row[k] for k in row.keys()})
            out.append(d)
    return out


def xcompare_digest(data_root: Path) -> dict[str, Any]:
    """A compact JSON digest of the cross-experiment state.

    Pre-cooked for the Ralph loop: registered experiments + their
    status, top-N units by best metric, worst-N, and the
    slowest-progressing units. The meta agent uses this as its primary
    context.
    """
    rows = best_per_unit(data_root)
    with_metric = [r for r in rows if r.metric is not None]
    with_metric.sort(key=lambda r: r.metric)  # type: ignore[arg-type]
    reg = Registry(data_root / "registry.db")
    try:
        experiments = [
            {
                "name": r["name"],
                "status": r["status"],
                "budget_usd": float(r["budget_usd"] or 0),
                "spent_usd": float(r["spent_usd"] or 0),
            }
            for r in reg.list_experiments()
        ]
    finally:
        reg.close()
    return {
        "n_experiments": len(experiments),
        "experiments": experiments,
        "n_units": len(rows),
        "n_units_with_results": len(with_metric),
        "top_5": [_xrow_dict(r) for r in with_metric[:5]],
        "bottom_5": [_xrow_dict(r) for r in with_metric[-5:][::-1]],
        "no_progress": [
            _xrow_dict(r) for r in rows if r.metric is None
        ][:10],
    }


def _xrow_dict(r: XRow) -> dict[str, Any]:
    return {
        "experiment": r.experiment,
        "unit_id": r.unit_id,
        "metric": r.metric,
        "task": r.task,
        "mode": r.mode,
        "max_round": r.max_round,
        "n_rows": r.n_rows,
        "n_success": r.n_success,
    }


def dump_digest(data_root: Path) -> str:
    return json.dumps(xcompare_digest(data_root), indent=2, sort_keys=True)
