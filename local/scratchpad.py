"""Local-disk scratchpad for miner agents.

The miner-side contract (mirrors real radar):

* ``load_scratchpad(challenge) -> str`` — returns a directory the agent
  can read/write freely. Persists across rounds.
* ``save_scratchpad(challenge, scratch_dir) -> None`` — commits the
  agent's writes. No-op in local since reads/writes are in-place;
  copies contents back if the agent handed us a different directory.

The agents under ``miners/`` call these as injected globals (``# noqa:
F821 — injected``). ``local/miner.py`` injects them into the agent
module's namespace after import and before invoking
``design_architecture``.

Layout::

    local/scratchpads/<miner_id>/...   # persistent per-miner

R2-free by design — real radar backs this with presigned PUT/GET URLs;
the local stack just touches the filesystem.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("local/scratchpads")


def make_scratchpad_pair(
    miner_id: str,
    root: Path | str = DEFAULT_ROOT,
) -> tuple[Callable[[dict], str], Callable[[dict, str], None]]:
    """Return ``(load_scratchpad, save_scratchpad)`` bound to ``miner_id``.

    The persistent dir is created lazily on first ``load_scratchpad``.
    """
    root_path = Path(root)
    miner_root = (root_path / miner_id).resolve()

    def load_scratchpad(challenge: dict) -> str:
        miner_root.mkdir(parents=True, exist_ok=True)
        return str(miner_root)

    def save_scratchpad(challenge: dict, scratch_dir: str) -> None:
        # Agent wrote in place (the path we returned). Nothing to do.
        if Path(scratch_dir).resolve() == miner_root:
            return
        # Agent mutated a different directory. Mirror contents back.
        miner_root.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(scratch_dir, miner_root, dirs_exist_ok=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "scratchpad copyback failed (%s → %s): %s",
                scratch_dir, miner_root, e,
            )

    return load_scratchpad, save_scratchpad
