"""Miner-side services reachability probe (local/miner.py).

Complements the validator-side stale-challenge expiry: while the
validator is down, a restarting miner must not invoke the agent against
a challenge whose embedded services port is dead.
"""

from __future__ import annotations

import socket

from local.miner import _services_reachable


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_reachable_when_listening():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        ch = {"allowed_urls": f"http://127.0.0.1:{port}"}
        assert _services_reachable(ch) is True
    finally:
        server.close()


def test_unreachable_on_dead_port():
    ch = {"allowed_urls": f"http://127.0.0.1:{_free_port()}"}
    assert _services_reachable(ch, timeout=0.5) is False


def test_no_services_url_passes():
    assert _services_reachable({}) is True
    assert _services_reachable({"allowed_urls": ""}) is True


def test_falls_back_to_db_url():
    ch = {"db_url": f"http://127.0.0.1:{_free_port()}"}
    assert _services_reachable(ch, timeout=0.5) is False
