"""Tests for runner/launcher.py — server vs RunPod handler branch.

The launcher's job is to pick the right runtime entry based on env
vars. We stub both terminal calls (uvicorn.run, runpod.serverless.start)
so the test never actually starts a server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("RP_HANDLER_NAME", "RADAR_TRAINER_MODE", "TRAINER_PORT"):
        monkeypatch.delenv(var, raising=False)


def _import_launcher():
    """Fresh import so each test sees the current env."""
    import importlib
    import runner.launcher as mod
    return importlib.reload(mod)


def test_default_starts_fastapi_server(clean_env):
    launcher = _import_launcher()
    fake_uvicorn = MagicMock()
    fake_server = MagicMock()
    fake_server.app = MagicMock()
    with patch.dict("sys.modules", {
        "uvicorn": fake_uvicorn,
        "runner.server": fake_server,
    }):
        launcher.main()
    fake_uvicorn.run.assert_called_once()
    args, kwargs = fake_uvicorn.run.call_args
    assert kwargs.get("host") == "0.0.0.0"
    assert kwargs.get("port") == 8081


def test_runpod_env_starts_handler(clean_env, monkeypatch):
    monkeypatch.setenv("RP_HANDLER_NAME", "handle")
    launcher = _import_launcher()
    fake_runpod = MagicMock()
    with patch.dict("sys.modules", {"runpod": fake_runpod}):
        launcher.main()
    fake_runpod.serverless.start.assert_called_once()
    cfg = fake_runpod.serverless.start.call_args[0][0]
    assert "handler" in cfg


def test_radar_trainer_mode_runpod_forces_handler(clean_env, monkeypatch):
    monkeypatch.setenv("RADAR_TRAINER_MODE", "runpod")
    launcher = _import_launcher()
    fake_runpod = MagicMock()
    with patch.dict("sys.modules", {"runpod": fake_runpod}):
        launcher.main()
    fake_runpod.serverless.start.assert_called_once()


def test_runpod_mode_detection():
    from runner.launcher import _is_runpod_mode
    import os
    # Save & clear
    saved = {k: os.environ.pop(k, None) for k in ("RP_HANDLER_NAME", "RADAR_TRAINER_MODE")}
    try:
        assert not _is_runpod_mode()
        os.environ["RP_HANDLER_NAME"] = "h"
        assert _is_runpod_mode()
        del os.environ["RP_HANDLER_NAME"]
        os.environ["RADAR_TRAINER_MODE"] = "runpod"
        assert _is_runpod_mode()
        os.environ["RADAR_TRAINER_MODE"] = "anything-else"
        assert not _is_runpod_mode()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
