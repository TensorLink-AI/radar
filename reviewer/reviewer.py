"""Cross-round meta-reviewer for the radar-local stack.

Self-contained: lives outside ``local/`` and ``miners/``. Reads the
live SQLite DB; writes ONLY inside ``reviewer/proposals/<session>/``.
The single bridge into production is ``local/promote_prompts.py``,
which audits proposals and atomically rewrites each miner's
``prompts/active.json``. See ``reviewer/README.md`` for the full
architecture, trust boundary, and ops runbook.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from reviewer.prompt_template import build_prompt
from reviewer.queries import (
    aggregate_summary, new_experiments_since_or_none,
    recent_failures, round_window, self_scorecard,
)
from reviewer.sanitize import sample_events, to_json
from reviewer.schema_check import SchemaDriftError, ensure_schema

logger = logging.getLogger("reviewer")

DEFAULT_MAX_HOURS = 12.0
DEFAULT_MODEL = "opus"
DEFAULT_ROUNDS_BACK = 50
DEFAULT_MIN_NEW_EXPERIMENTS = 25
DEFAULT_MIN_INTERVAL_HOURS = 24.0
ALLOWED_TOOLS = ("Read", "Grep", "Glob", "Write")
HYPOTHESES_FENCE_RE = re.compile(
    r"```(?:json)?\s*HYPOTHESES\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
SESSION_DIR_RE = re.compile(r"^\d{8}T\d{6}Z$")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _open_db_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _prior_sessions(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and SESSION_DIR_RE.match(p.name)
    )


def _trigger_check(
    conn: sqlite3.Connection, proposals_root: Path,
    min_new: int, min_hours: float,
) -> tuple[bool, str]:
    if (proposals_root / "STOP").exists():
        return False, "STOP sentinel present"
    sessions = _prior_sessions(proposals_root)
    if not sessions:
        return True, "first run"
    last = sessions[-1]
    age_h = (time.time() - last.stat().st_mtime) / 3600.0
    if age_h < min_hours:
        return False, f"last session {age_h:.1f}h ago (< {min_hours}h)"
    last_round = _read_session_round_hi(last)
    n = new_experiments_since_or_none(conn, last_round)
    if n is not None and n < min_new:
        return False, (
            f"only {n} new experiments since round {last_round} (< {min_new})"
        )
    return True, "ok"


def _read_session_round_hi(session_dir: Path) -> Optional[int]:
    summary = session_dir / "summary.json"
    if not summary.exists():
        return None
    try:
        return int(json.loads(summary.read_text()).get("round_hi"))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _load_hypotheses(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skip malformed hypothesis: %r", line[:120])
    return out


def _write_hypotheses(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def _parse_hypotheses_block(text: str) -> Optional[list[dict]]:
    m = HYPOTHESES_FENCE_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1).strip())
    except json.JSONDecodeError as e:
        logger.warning("HYPOTHESES block not valid JSON: %s", e)
        return None
    if not isinstance(data, list):
        logger.warning("HYPOTHESES block not a JSON array")
        return None
    return data


def _run_claude(prompt: str, model: str, session_dir: Path,
                max_hours: float) -> tuple[int, str]:
    cmd = [
        "claude", "--print",
        "--model", model,
        "--permission-mode", "default",
        "--allowed-tools", *ALLOWED_TOOLS,
        "--add-dir", str(session_dir),
        prompt,
    ]
    logger.info("spawning claude (model=%s, wall-clock=%.1fh)", model, max_hours)
    proc = subprocess.run(
        cmd, cwd=str(session_dir), capture_output=True, text=True,
        timeout=max_hours * 3600.0,
    )
    if proc.returncode != 0:
        logger.warning("claude exit %d; stderr tail: %s",
                       proc.returncode, proc.stderr[-2000:])
    return proc.returncode, proc.stdout


def _log_reviewer_event(db_path: Path, **fields) -> None:
    sys.path.insert(0, str(_repo_root()))
    from local.store import LocalStore
    store = LocalStore(db_path)
    try:
        store.record_agent_event(**fields)
    finally:
        store.close()


def run_session(
    db_path: Path, *, rounds_back: int = DEFAULT_ROUNDS_BACK,
    max_hours: float = DEFAULT_MAX_HOURS, model: str = DEFAULT_MODEL,
    min_new: int = DEFAULT_MIN_NEW_EXPERIMENTS,
    min_hours: float = DEFAULT_MIN_INTERVAL_HOURS,
    force: bool = False, proposals_root: Optional[Path] = None,
) -> Optional[Path]:
    proposals_root = proposals_root or (_repo_root() / "reviewer" / "proposals")
    proposals_root.mkdir(parents=True, exist_ok=True)

    conn = _open_db_ro(db_path)
    try:
        ensure_schema(conn)
        if not force:
            ok, reason = _trigger_check(conn, proposals_root, min_new, min_hours)
            if not ok:
                logger.info("skipping: %s", reason)
                _log_reviewer_event(
                    db_path, kind="reviewer", miner_id="reviewer",
                    endpoint="skip", status=0, latency_ms=0.0,
                    request={"min_new": min_new, "min_hours": min_hours},
                    response={"reason": reason}, error=None,
                )
                return None

        # Load prior hypotheses BEFORE creating the new session dir so we
        # don't accidentally pick up the empty file we're about to write.
        prior_sessions = _prior_sessions(proposals_root)
        prior_hypotheses = (
            _load_hypotheses(prior_sessions[-1] / "hypotheses.jsonl")
            if prior_sessions else []
        )

        session_dir = proposals_root / _utc_stamp()
        session_dir.mkdir(parents=True, exist_ok=True)

        lo, hi = round_window(conn, rounds_back)
        summary = aggregate_summary(conn, lo, hi)
        scorecard = self_scorecard(conn)
        failures = recent_failures(conn, lo, hi)
        events_sample = sample_events(conn, lo=lo, hi=hi)
        (session_dir / "summary.json").write_text(to_json(summary))

        prompt = build_prompt(
            session_dir=session_dir, summary=summary, scorecard=scorecard,
            failures=failures, events_sample=events_sample,
            hypotheses=prior_hypotheses, max_hours=max_hours,
        )
        start = time.time()
        rc, stdout = _run_claude(prompt, model, session_dir, max_hours)
        elapsed = time.time() - start

        if stdout:
            (session_dir / "stdout.txt").write_text(stdout)
        new_h = _parse_hypotheses_block(stdout) if stdout else None
        hypotheses_path = session_dir / "hypotheses.jsonl"
        if new_h is not None:
            _write_hypotheses(hypotheses_path, new_h)
            logger.info("wrote %d hypotheses → %s", len(new_h), hypotheses_path)
        else:
            logger.warning("no HYPOTHESES block parsed; carrying prior forward")
            _write_hypotheses(hypotheses_path, prior_hypotheses)

        staged = len(list(session_dir.glob("rev_*.json")))
        _log_reviewer_event(
            db_path, kind="reviewer", miner_id="reviewer",
            endpoint="run", status=rc, latency_ms=elapsed * 1000.0,
            request={"window": [lo, hi], "rounds_back": rounds_back,
                     "max_hours": max_hours, "model": model},
            response={"session_dir": str(session_dir),
                      "hypotheses_written": len(new_h) if new_h else 0,
                      "staged_variants": staged,
                      "scorecard_rows": len(scorecard)},
            error=None if rc == 0 else f"claude exit {rc}",
        )
        return session_dir
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--rounds-back", type=int, default=DEFAULT_ROUNDS_BACK)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-new-experiments", type=int,
                        default=DEFAULT_MIN_NEW_EXPERIMENTS)
    parser.add_argument("--min-interval-hours", type=float,
                        default=DEFAULT_MIN_INTERVAL_HOURS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--proposals-root", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        path = run_session(
            args.db, rounds_back=args.rounds_back, max_hours=args.max_hours,
            model=args.model, min_new=args.min_new_experiments,
            min_hours=args.min_interval_hours, force=args.force,
            proposals_root=args.proposals_root,
        )
    except SchemaDriftError as e:
        logger.error("%s", e)
        return 2
    if path is not None:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
