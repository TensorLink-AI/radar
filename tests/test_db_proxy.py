"""Tests for validator/db_proxy.py — reverse proxy for miners."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from validator.db_proxy import (
    app, set_config, rotate_agent_token, get_agent_token,
    get_ready_trainers, clear_ready_trainers, _trainer_ready,
    _check_rate_limit, _rate_window, _rate_lock,
)


def _setup_proxy():
    """Configure proxy with a fake upstream, no Epistula auth, and a valid agent token."""
    rotate_agent_token()
    set_config(
        db_api_url="http://fake-db:8090",
        wallet=None,
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
    rotate_agent_token()
    set_config(
        db_api_url="",
        wallet=None,
        rate_limits={"db": (100, 60), "desearch": (100, 60), "llm": (100, 60), "agent_code": (100, 60)},
    )
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/experiments/recent", headers=_auth_headers())
    # Should get 503 because db_api_url is empty
    assert r.status_code == 503


def test_per_category_rate_limits():
    """Each route category has an independent rate-limit bucket."""
    # Set tight limits: 2 for db, 2 for llm
    rotate_agent_token()
    set_config(
        db_api_url="http://fake-db:8090",
        wallet=None,
        rate_limits={"db": (2, 60), "llm": (2, 60), "desearch": (2, 60)},
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


@pytest.mark.asyncio
async def test_proxy_llm_no_retry():
    """LLM requests are NOT retried at the proxy layer — the llm_proxy
    handles its own retries and circuit breaking.  A 503 passes through."""
    _setup_proxy()

    sign_call_count = 0

    def mock_sign(wallet, body):
        nonlocal sign_call_count
        sign_call_count += 1
        return {
            "X-Epistula-Signed-By": "hk0",
            "X-Epistula-Timestamp": str(1000 + sign_call_count),
            "X-Epistula-Nonce": f"nonce-{sign_call_count}",
            "X-Epistula-Signature": "sig",
        }

    mock_wallet = MagicMock()

    resp_503 = MagicMock()
    resp_503.status_code = 503
    resp_503.headers = {"content-type": "application/json"}
    resp_503.content = b'{"error": "not ready"}'

    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_client.post = AsyncMock(return_value=resp_503)

    from validator import db_proxy

    with patch.object(db_proxy, "_wallet", mock_wallet), \
         patch.object(db_proxy, "_client", mock_client), \
         patch("validator.db_proxy.sign_request", side_effect=mock_sign):

        from httpx import ASGITransport, AsyncClient

        token = get_agent_token()
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/llm/v1/chat/completions",
                json={"model": "test", "messages": []},
                headers={"X-Agent-Token": token, "X-Miner-UID": "1"},
            )

    # 503 passes through without retry — only 1 sign + 1 upstream call
    assert resp.status_code == 503
    assert sign_call_count == 1
    assert mock_client.post.call_count == 1
