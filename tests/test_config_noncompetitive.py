"""Tests for Config.NONCOMPETITIVE + service-key wiring."""

from __future__ import annotations

import importlib

import pytest


def _reload_config(monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import config as cfg
    importlib.reload(cfg)
    return cfg.Config


def test_noncompetitive_defaults_false(monkeypatch):
    cfg = _reload_config(monkeypatch, RADAR_NONCOMPETITIVE=None)
    assert cfg.NONCOMPETITIVE is False


def test_noncompetitive_truthy_variants(monkeypatch):
    for val in ("true", "TRUE", "1", "yes"):
        cfg = _reload_config(monkeypatch, RADAR_NONCOMPETITIVE=val)
        assert cfg.NONCOMPETITIVE is True, f"failed for {val}"


def test_noncompetitive_falsy_variants(monkeypatch):
    for val in ("false", "0", "no", ""):
        cfg = _reload_config(monkeypatch, RADAR_NONCOMPETITIVE=val)
        assert cfg.NONCOMPETITIVE is False, f"failed for {val}"


def test_service_key_id_defaults_operator(monkeypatch):
    cfg = _reload_config(monkeypatch, RADAR_SERVICE_KEY_ID=None)
    assert cfg.SERVICE_KEY_ID == "operator"


def test_service_key_id_overrideable(monkeypatch):
    cfg = _reload_config(monkeypatch, RADAR_SERVICE_KEY_ID="my-op")
    assert cfg.SERVICE_KEY_ID == "my-op"


def test_service_key_passes_through(monkeypatch):
    cfg = _reload_config(monkeypatch, RADAR_SERVICE_KEY="a" * 64)
    assert cfg.SERVICE_KEY == "a" * 64
