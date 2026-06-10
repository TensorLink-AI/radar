"""Stale-open-challenge expiry (local/store.py).

A challenge left 'open' by a validator that died mid-phase-A carries the
dead process's ephemeral services port in its payload; a miner that picks
it up burns the round on connection-refused. The restarted validator
expires those rows at startup via ``expire_open_challenges``.
"""

from __future__ import annotations

from local.store import LocalStore


def test_expire_open_challenges(tmp_path):
    store = LocalStore(tmp_path / "t.db")
    store.post_challenge("ch-stale", 42, {"llm_url": "http://127.0.0.1:51237/llm"})
    store.post_challenge("ch-replicate", 43, {}, status="collected")
    store.post_challenge("ch-done", 41, {}, status="done")

    assert store.open_challenge()["challenge_id"] == "ch-stale"
    assert store.expire_open_challenges() == 1
    assert store.open_challenge() is None
    # Re-running is a no-op; non-open rows are untouched.
    assert store.expire_open_challenges() == 0

    rows = {
        cid: store._conn.execute(
            "SELECT status FROM challenges WHERE challenge_id = ?", (cid,)
        ).fetchone()["status"]
        for cid in ("ch-stale", "ch-replicate", "ch-done")
    }
    assert rows == {
        "ch-stale": "expired",
        "ch-replicate": "collected",
        "ch-done": "done",
    }


def test_fresh_challenge_visible_after_expiry(tmp_path):
    store = LocalStore(tmp_path / "t.db")
    store.post_challenge("ch-old", 42, {})
    store.expire_open_challenges()
    store.post_challenge("ch-new", 42, {"llm_url": "http://127.0.0.1:60000/llm"})
    got = store.open_challenge()
    assert got["challenge_id"] == "ch-new"
    assert got["payload"]["llm_url"].endswith(":60000/llm")
