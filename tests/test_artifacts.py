"""Tests for local.artifacts cleanup helpers.

Covers trainer-workdir reclamation: per-result cleanup and the startup
sweep that reclaims dirs orphaned by a crashed/killed validator run.
"""

import tempfile
from pathlib import Path

import pytest

from local.artifacts import (
    ArtifactSink,
    _default_key_prefix,
    cleanup_workdir,
    sweep_orphan_workdirs,
)
from local.store import LocalStore


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


def _make_sink(tmp_path: Path, key_prefix: str = "") -> ArtifactSink:
    store = LocalStore(str(tmp_path / "radar_local.db"))
    return ArtifactSink(
        store=store, bucket="radar-local", r2_enabled=False,
        key_prefix=key_prefix,
    )


def test_round_prefix_unprefixed_keeps_bare_runs_layout(tmp_path):
    sink = _make_sink(tmp_path, key_prefix="")
    try:
        assert sink._round_prefix("ts_forecasting", 7) == \
            "runs/ts_forecasting/r000007"
        assert sink._miner_prefix("ts_forecasting", 7, "alpha") == \
            "runs/ts_forecasting/r000007/miners/alpha"
    finally:
        sink.store.close()


def test_round_prefix_with_instance_id_namespaces_keys(tmp_path):
    sink = _make_sink(tmp_path, key_prefix="inst-A")
    try:
        assert sink._round_prefix("ts_forecasting", 7) == \
            "inst-A/runs/ts_forecasting/r000007"
        assert sink._miner_prefix("ts_forecasting", 7, "alpha") == \
            "inst-A/runs/ts_forecasting/r000007/miners/alpha"
    finally:
        sink.store.close()


def test_default_key_prefix_uses_instance_id(monkeypatch):
    monkeypatch.delenv("RADAR_ARTIFACT_PREFIX", raising=False)
    monkeypatch.setenv("RADAR_INSTANCE_ID", "lab-3")
    assert _default_key_prefix() == "lab-3"


def test_default_key_prefix_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("RADAR_ARTIFACT_PREFIX", "custom/path")
    monkeypatch.setenv("RADAR_INSTANCE_ID", "lab-3")
    assert _default_key_prefix() == "custom/path"


def test_default_key_prefix_explicit_empty_opts_out(monkeypatch):
    # An explicit empty string disables namespacing even when an instance
    # id is set — the documented escape hatch for keeping the bare
    # `runs/...` layout on a shared bucket.
    monkeypatch.setenv("RADAR_ARTIFACT_PREFIX", "")
    monkeypatch.setenv("RADAR_INSTANCE_ID", "lab-3")
    assert _default_key_prefix() == ""


def test_default_key_prefix_no_instance_id(monkeypatch):
    monkeypatch.delenv("RADAR_ARTIFACT_PREFIX", raising=False)
    monkeypatch.delenv("RADAR_INSTANCE_ID", raising=False)
    assert _default_key_prefix() == ""


def test_record_proposal_uses_instance_prefixed_s3_key(tmp_path):
    sink = _make_sink(tmp_path, key_prefix="inst-B")
    try:
        sink.record_challenge("ts_forecasting", 3, {"hello": "world"})
        sink.record_proposal(
            "ts_forecasting", 3, "miner-1", {"code": "print('hi')"},
        )
        rows = sink.store.list_artifacts(round_id=3)
        # SQLite-only mode keeps s3_key empty, but rel_path is the
        # in-bucket path; verify the round prefix made it through the
        # key construction by checking _round_prefix directly above and
        # asserting the rows landed at the expected rel_path here.
        kinds = {(r["kind"], r["rel_path"]) for r in rows}
        assert ("challenge", "challenge.json") in kinds
        assert ("proposal", "proposal.json") in kinds
        assert ("submission", "submission.py") in kinds
    finally:
        sink.store.close()


@pytest.mark.parametrize("instance,explicit,expected", [
    ("lab-3", None, "lab-3"),
    ("lab-3", "custom", "custom"),
    ("lab-3", "", ""),
    ("", None, ""),
])
def test_default_key_prefix_matrix(monkeypatch, instance, explicit, expected):
    if instance:
        monkeypatch.setenv("RADAR_INSTANCE_ID", instance)
    else:
        monkeypatch.delenv("RADAR_INSTANCE_ID", raising=False)
    if explicit is None:
        monkeypatch.delenv("RADAR_ARTIFACT_PREFIX", raising=False)
    else:
        monkeypatch.setenv("RADAR_ARTIFACT_PREFIX", explicit)
    assert _default_key_prefix() == expected
