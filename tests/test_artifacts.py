"""Tests for local.artifacts cleanup helpers.

Covers trainer-workdir reclamation: per-result cleanup and the startup
sweep that reclaims dirs orphaned by a crashed/killed validator run.
"""

import tempfile
from pathlib import Path

from local.artifacts import cleanup_workdir, sweep_orphan_workdirs


def test_cleanup_workdir_removes_tree():
    workdir = Path(tempfile.mkdtemp(prefix="radar_ts_"))
    (workdir / "checkpoints").mkdir()
    (workdir / "checkpoints" / "model.bin").write_bytes(b"x")
    cleanup_workdir(workdir)
    assert not workdir.exists()


def test_cleanup_workdir_none_is_noop():
    # Should not raise.
    cleanup_workdir(None)


def test_sweep_orphan_workdirs_removes_stragglers():
    a = Path(tempfile.mkdtemp(prefix="radar_ts_"))
    b = Path(tempfile.mkdtemp(prefix="radar_ts_"))
    (a / "logs").mkdir()
    (a / "logs" / "train.log").write_text("hi")

    # A non-matching temp dir must survive the sweep.
    keep = Path(tempfile.mkdtemp(prefix="unrelated_"))
    try:
        removed = sweep_orphan_workdirs()
        assert removed >= 2
        assert not a.exists()
        assert not b.exists()
        assert keep.exists()
    finally:
        cleanup_workdir(keep)
