"""Tests for shared.peers — static peer registry."""

import json
import os
import textwrap

import pytest

from shared import peers as peers_mod


@pytest.fixture
def peers_file(tmp_path, monkeypatch):
    """Write a miners.json fixture and point MINERS_CONFIG_PATH at it."""
    path = tmp_path / "miners.json"
    path.write_text(json.dumps({
        "miners": [
            {"uid": 0, "hotkey": "miner0", "endpoint": "http://miner0:8000", "stake": 1.5},
            {"uid": 1, "hotkey": "miner1", "endpoint": "http://miner1:8000", "stake": 2.0},
            {"uid": 2, "hotkey": "miner2", "endpoint": "http://miner2:8000", "stake": 0.0},
        ],
    }))
    monkeypatch.setenv("MINERS_CONFIG_PATH", str(path))
    peers_mod.reset_cache()
    yield path
    peers_mod.reset_cache()


def test_load_peers_basic(peers_file):
    ps = peers_mod.load_peers()
    assert len(ps) == 3
    assert ps[0].uid == 0
    assert ps[0].hotkey == "miner0"
    assert ps[0].endpoint == "http://miner0:8000"
    assert ps[0].stake == 1.5


def test_get_peer_by_hotkey(peers_file):
    p = peers_mod.get_peer_by_hotkey("miner1")
    assert p is not None
    assert p.uid == 1
    assert peers_mod.get_peer_by_hotkey("missing") is None


def test_get_peer_by_uid(peers_file):
    p = peers_mod.get_peer_by_uid(2)
    assert p is not None
    assert p.hotkey == "miner2"
    assert peers_mod.get_peer_by_uid(99) is None


def test_missing_file_returns_empty(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("MINERS_CONFIG_PATH", str(tmp_path / "no.json"))
    peers_mod.reset_cache()
    with caplog.at_level("WARNING"):
        ps = peers_mod.load_peers()
    assert ps == []
    assert any("does not exist" in r.message for r in caplog.records)


def test_malformed_json_returns_empty(tmp_path, monkeypatch, caplog):
    path = tmp_path / "miners.json"
    path.write_text("{not json")
    monkeypatch.setenv("MINERS_CONFIG_PATH", str(path))
    peers_mod.reset_cache()
    with caplog.at_level("WARNING"):
        ps = peers_mod.load_peers()
    assert ps == []


def test_skips_bad_entries(tmp_path, monkeypatch):
    path = tmp_path / "miners.json"
    path.write_text(json.dumps({
        "miners": [
            {"uid": 0, "hotkey": "ok"},
            {"hotkey": "missing-uid"},        # skipped (no uid)
            "not-a-dict",                       # skipped
            {"uid": 9, "hotkey": "also-ok", "endpoint": "x"},
        ],
    }))
    monkeypatch.setenv("MINERS_CONFIG_PATH", str(path))
    peers_mod.reset_cache()
    ps = peers_mod.load_peers()
    assert {p.uid for p in ps} == {0, 9}


def test_get_hotkey_for_uid_fallback(peers_file):
    assert peers_mod.get_hotkey_for_uid(1) == "miner1"
    # Missing UID falls back to the legacy uid_{n} placeholder.
    assert peers_mod.get_hotkey_for_uid(999) == "uid_999"


def test_cache_reload_on_mtime_change(peers_file):
    ps1 = peers_mod.load_peers()
    assert len(ps1) == 3
    # Mutate the file (and bump its mtime) — cache should refresh.
    data = json.loads(peers_file.read_text())
    data["miners"].append({"uid": 7, "hotkey": "newcomer"})
    peers_file.write_text(json.dumps(data))
    new_mtime = os.path.getmtime(peers_file) + 1
    os.utime(peers_file, (new_mtime, new_mtime))
    ps2 = peers_mod.load_peers()
    assert {p.uid for p in ps2} == {0, 1, 2, 7}
