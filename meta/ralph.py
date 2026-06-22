"""Ralph loop — continuous meta-optimizer.

Spawns a headless Claude Code session every ``--interval`` with the
researcher prompt + the latest ``xcompare`` digest pre-loaded as
context. The session reads the digest, proposes a new YAML under
``meta/experiments/``, calls ``meta submit``, then exits. The Ralph
wrapper enforces:

* ``--max-concurrent-experiments`` — never queue past this many running
  experiments.
* ``--total-budget-usd`` — never spend more than this in aggregate.

These are checked *before* the Claude session even fires so a runaway
researcher can't drain the budget.

Skipping logic: if ``xcompare`` returns the same digest hash as the
previous tick (no new experiments finished), we sleep through this
interval — no point re-reasoning over identical input.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

from . import xquery
from .store import Registry


logger = logging.getLogger(__name__)


def _digest_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _spent_total(data_root: Path) -> float:
    reg = Registry(data_root / "registry.db")
    try:
        return sum(float(r["spent_usd"] or 0) for r in reg.list_experiments())
    finally:
        reg.close()


def _running_count(data_root: Path) -> int:
    reg = Registry(data_root / "registry.db")
    try:
        return sum(
            1 for r in reg.list_experiments()
            if r["status"] in {"queued", "running"}
        )
    finally:
        reg.close()


def run_loop(
    *,
    data_root: Path,
    prompt_path: Path,
    interval_sec: float = 1800.0,
    max_concurrent: int = 4,
    total_budget_usd: float = 500.0,
    claude_cli: str = "claude",
    once: bool = False,
) -> None:
    """Main Ralph loop. Runs until SIGINT or budget exhausted.

    Each tick:
      1. Compute the xcompare digest.
      2. Skip if digest unchanged.
      3. Skip if running ≥ max_concurrent or spent ≥ total_budget.
      4. Materialize digest + prompt into a single context file.
      5. Shell out to ``claude --print --append-system-prompt …`` so
         the session is headless and produces a self-contained turn.
      6. Sleep ``interval_sec``.
    """
    prev_hash: str = ""
    prompt_template = prompt_path.read_text()
    scratch = data_root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    while True:
        digest = xquery.dump_digest(data_root)
        h = _digest_hash(digest)

        spent = _spent_total(data_root)
        running = _running_count(data_root)

        if spent >= total_budget_usd:
            logger.warning(
                "ralph: total budget reached (${:.2f} >= ${:.2f}); stopping",
                spent, total_budget_usd,
            )
            return
        if running >= max_concurrent:
            logger.info(
                "ralph: %d experiments queued/running ≥ max %d; skipping",
                running, max_concurrent,
            )
        elif h == prev_hash and prev_hash:
            logger.info(
                "ralph: digest unchanged (%s); skipping", h,
            )
        else:
            try:
                _fire_session(
                    prompt_template, digest, scratch,
                    claude_cli=claude_cli, data_root=data_root,
                )
                prev_hash = h
            except Exception:
                logger.exception("ralph: session crashed")

        if once:
            return
        time.sleep(interval_sec)


def _fire_session(
    prompt_template: str,
    digest: str,
    scratch: Path,
    *,
    claude_cli: str,
    data_root: Path,
) -> None:
    """Launch one headless Claude session."""
    ts = int(time.time())
    ctx = scratch / f"context-{ts}.md"
    ctx.write_text(
        prompt_template
        + "\n\n## Current xcompare digest\n\n```json\n"
        + digest
        + "\n```\n"
    )
    # Headless mode: --print emits the model's final message to stdout
    # and exits. The CWD is the radar repo root so the session can edit
    # meta/experiments/*.yaml directly.
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        claude_cli, "--print",
        "--append-system-prompt", f"@{ctx}",
    ]
    logger.info("ralph: launching session ctx=%s", ctx)
    env = dict(os.environ)
    env.setdefault("RADAR_META_DATA_ROOT", str(data_root))
    p = subprocess.run(
        cmd, cwd=str(repo_root), env=env,
        capture_output=True, text=True, timeout=60 * 60,
    )
    log_path = scratch / f"session-{ts}.log"
    log_path.write_text(
        f"$ {shlex.join(cmd)}\nrc={p.returncode}\n\n"
        f"-- stdout --\n{p.stdout}\n\n-- stderr --\n{p.stderr}\n"
    )
    if p.returncode != 0:
        logger.warning("ralph: session rc=%d; see %s", p.returncode, log_path)
