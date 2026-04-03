"""Tests for validator/db_proxy.py — reverse proxy for miners."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from validator.db_proxy import (
    app, set_config, set_metagraph,
    get_ready_trainers, clear_ready_trainers, _trainer_ready,
)


def _setup_proxy():
    """Configure proxy with a fake upstream and no auth."""
    set_metagraph(None)
    set_config(
        db_api_url="http://fake-db:8090",
        wallet=None,
        metagraph=None,
        rate_limit=100,
    )


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
    set_config(
        db_api_url="",
        wallet=None,
        metagraph=None,
        rate_limit=100,
    )
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/experiments/recent")
    # Should get 503 because db_api_url is empty
    assert r.status_code == 503
