"""Audit + apply reviewer-staged prompt variants to live miner pools.

The reviewer writes candidate prompts into ``reviewer/proposals/<session>/
rev_*.json``. This script is the only code that crosses the boundary
into ``miners/.../prompts/active.json``. It:

* Lists pending reviewer-authored variants across sessions.
* Diffs them against each miner's live pool.
* Enforces a diversity cap (default: ``rev_*`` ids ≤ 30%% of pool).
* Retires the worst-performing ``rev_*`` variant first when the cap
  would otherwise be exceeded, using ``experiments.prompt_id`` scoring.
* Atomically rewrites ``active.json`` via tmp + rename.

By default operates on both ``claude_style_v4`` and ``claude_style_v5``;
restrict with ``--miner-dir``.

Usage::

    # interactive — list pending and prompt before applying
    python -m local.promote_prompts --db local/radar_local.db

    # non-interactive (CI / scripted promotion of a specific session)
    python -m local.promote_prompts --db local/radar_local.db \\
        --session 20260607T140000Z --apply --yes
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("local.promote_prompts")

DEFAULT_MINER_DIRS = (
    "miners/claude_style_v4",
    "miners/claude_style_v5",
)
DEFAULT_CAP_PCT = 0.30


@dataclass
class Variant:
    id: str
    template: str
    motivation: str
    source: Path

    @classmethod
    def load(cls, path: Path) -> "Variant":
        data = json.loads(path.read_text())
        return cls(
            id=str(data["id"]),
            template=str(data["template"]),
            motivation=str(data.get("motivation", "")),
            source=path,
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _proposals_root() -> Path:
    return _repo_root() / "reviewer" / "proposals"


def _read_active(miner_dir: Path) -> list[dict]:
    path = miner_dir / "prompts" / "active.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    rows = payload.get("prompts") if isinstance(payload, dict) else payload
    return rows if isinstance(rows, list) else []


def _write_active(miner_dir: Path, rows: list[dict]) -> None:
    """Atomic rewrite: write to .tmp then rename. v5/v4's reader uses
    plain open()+json.load(), so rename atomicity is the only thing
    standing between a mid-write read and a corrupt parse."""
    path = miner_dir / "prompts" / "active.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"prompts": rows}, indent=2, sort_keys=True))
    tmp.replace(path)


def _pending_variants(session: str | None) -> list[Variant]:
    root = _proposals_root()
    if not root.exists():
        return []
    sessions = (
        [root / session] if session else
        [p for p in sorted(root.iterdir()) if p.is_dir()]
    )
    out: list[Variant] = []
    for s in sessions:
        for p in sorted(s.glob("rev_*.json")):
            try:
                out.append(Variant.load(p))
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning("skip malformed variant %s: %s", p, e)
    return out


def _scores_by_prompt_id(conn: sqlite3.Connection) -> dict[str, tuple[float, int]]:
    """``prompt_id -> (avg_score, n)`` over successful experiments."""
    rows = conn.execute(
        "SELECT prompt_id, AVG(score) AS s, COUNT(*) AS n "
        "FROM experiments WHERE prompt_id != '' AND success = 1 "
        "GROUP BY prompt_id"
    ).fetchall()
    return {r["prompt_id"]: (float(r["s"]), int(r["n"])) for r in rows}


def _enforce_cap(
    new_pool: list[dict], cap_pct: float,
    scores: dict[str, tuple[float, int]],
) -> tuple[list[dict], list[str]]:
    """Drop the worst-scoring ``rev_*`` variants until they fit under
    the cap. Returns ``(kept_pool, retired_ids)``. Variants with no
    scoring data yet are treated as score=+inf so the promoter doesn't
    prematurely retire them — they get a fair shot first."""
    while True:
        rev_ids = [r["id"] for r in new_pool if r["id"].startswith("rev_")]
        if not rev_ids or len(rev_ids) <= cap_pct * len(new_pool):
            return new_pool, []
        worst_id = min(
            rev_ids,
            key=lambda i: scores.get(i, (float("inf"), 0))[0],
        )
        new_pool = [r for r in new_pool if r["id"] != worst_id]
        retired = [worst_id]
        # Continue loop in case more retirements are needed; accumulate.
        # We rebuild ``rev_ids`` from the new pool next iteration.
        pool2, more_retired = _enforce_cap(new_pool, cap_pct, scores)
        return pool2, retired + more_retired


def _promote_one(
    miner_dir: Path, variants: list[Variant], cap_pct: float,
    scores: dict[str, tuple[float, int]], dry_run: bool,
) -> dict:
    """Promote ``variants`` into ``miner_dir``. Returns a report dict."""
    active = _read_active(miner_dir)
    by_id = {r.get("id", ""): r for r in active}
    added: list[str] = []
    updated: list[str] = []
    for v in variants:
        entry = {"id": v.id, "template": v.template,
                 "motivation": v.motivation}
        if v.id in by_id:
            by_id[v.id] = entry
            updated.append(v.id)
        else:
            active.append(entry)
            by_id[v.id] = entry
            added.append(v.id)
    new_pool, retired = _enforce_cap(active, cap_pct, scores)
    if not dry_run:
        _write_active(miner_dir, new_pool)
    return {
        "miner_dir": str(miner_dir),
        "pool_size_before": len(by_id) - len(added),
        "pool_size_after": len(new_pool),
        "added": added,
        "updated": updated,
        "retired": retired,
    }


def _print_pending(variants: list[Variant]) -> None:
    if not variants:
        print("(no pending reviewer variants)")
        return
    print(f"Pending reviewer-staged variants ({len(variants)}):")
    for v in variants:
        rel = v.source.relative_to(_repo_root())
        print(f"  {v.id}")
        print(f"    source: {rel}")
        if v.motivation:
            print(f"    motivation: {v.motivation[:200]}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--session", default=None,
                        help="Promote only this session dir (UTC stamp)")
    parser.add_argument("--miner-dir", action="append", default=None,
                        help="Target miner dir; repeat for multiple. "
                             "Defaults to v4+v5.")
    parser.add_argument("--cap-pct", type=float, default=DEFAULT_CAP_PCT)
    parser.add_argument("--apply", action="store_true",
                        help="Actually write active.json (default: dry-run)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive confirmation when --apply set")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    variants = _pending_variants(args.session)
    _print_pending(variants)
    if not variants:
        return 0

    miner_dirs = [
        _repo_root() / d for d in (args.miner_dir or DEFAULT_MINER_DIRS)
    ]
    for d in miner_dirs:
        if not d.exists():
            logger.warning("miner dir does not exist: %s", d)

    conn = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        scores = _scores_by_prompt_id(conn)
    finally:
        conn.close()

    if args.apply and not args.yes:
        confirm = input(
            f"\nApply to {len(miner_dirs)} miner dir(s)? [y/N] "
        ).strip().lower()
        if confirm not in ("y", "yes"):
            print("aborted")
            return 1

    for d in miner_dirs:
        if not d.exists():
            continue
        report = _promote_one(
            d, variants, cap_pct=args.cap_pct,
            scores=scores, dry_run=not args.apply,
        )
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
