"""Tests for validator/db_proxy.py — reverse proxy for miners."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from validator.db_proxy import (
    app, set_config, set_metagraph, rotate_agent_token, get_agent_token,
    get_ready_trainers, clear_ready_trainers, _trainer_ready,
    _check_rate_limit, _rate_window, _rate_lock,
)


def _setup_proxy():
    """Configure proxy with a fake upstream, no Epistula auth, and a valid agent token."""
    set_metagraph(None)
    rotate_agent_token()
    set_config(
        db_api_url="http://fake-db:8090",
        wallet=None,
        metagraph=None,
        rate_limits={"db": (100, 60), "desearch": (100, 60), "llm": (100, 60), "agent_code": (100, 60)},
    )


def _auth_headers() -> dict[str, str]:
    """Return headers with a valid agent token."""
    return {"X-Agent-Token": get_agent_token()}


def test_health():
    _setup_proxy()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["proxy"] is True


def test_ready_trainers_tracking():
    """Test trainer readiness tracking functions."""
    _trainer_ready.clear()
    _trainer_ready[1] = {0: "ready_msg_0", 1: "ready_msg_1"}
    assert get_ready_trainers(1) == {0: "ready_msg_0", 1: "ready_msg_1"}
    assert get_ready_trainers(2) == {}
    clear_ready_trainers(1)
    assert get_ready_trainers(1) == {}


def test_proxy_no_db_url():
    """Without a DB API URL configured, proxy returns 503."""
    set_metagraph(None)
    rotate_agent_token()
    set_config(
        db_api_url="",
        wallet=None,
        metagraph=None,
        rate_limits={"db": (100, 60), "desearch": (100, 60), "llm": (100, 60), "agent_code": (100, 60)},
    )
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/experiments/recent", headers=_auth_headers())
    # Should get 503 because db_api_url is empty
    assert r.status_code == 503


def test_per_category_rate_limits():
    """Each route category has an independent rate-limit bucket."""
    # Set tight limits: 2 for db, 2 for llm
    set_metagraph(None)
    rotate_agent_token()
    set_config(
        db_api_url="http://fake-db:8090",
        wallet=None,
        metagraph=None,
        rate_limits={"db": (2, 60), "llm": (2, 60), "desearch": (2, 60), "agent_code": (1, 3600)},
    )
    identity = "test-miner-99"
    # Clear any existing state
    with _rate_lock:
        _rate_window.clear()

    # Exhaust the "db" bucket
    assert _check_rate_limit(identity, "db") is True
    assert _check_rate_limit(identity, "db") is True
    assert _check_rate_limit(identity, "db") is False  # blocked

    # "llm" bucket should still be available
    assert _check_rate_limit(identity, "llm") is True
    assert _check_rate_limit(identity, "llm") is True
    assert _check_rate_limit(identity, "llm") is False  # blocked

    # "desearch" still independent
    assert _check_rate_limit(identity, "desearch") is True

    # "agent_code" allows 1 per hour
    assert _check_rate_limit(identity, "agent_code") is True
    assert _check_rate_limit(identity, "agent_code") is False
