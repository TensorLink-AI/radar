"""Tests for shared.event_redact — sanitises validator events before
they're served on the public dashboard JSON API.
"""

from shared.event_redact import redact_event, redact_payload


def test_drops_sensitive_keys():
    out = redact_payload({
        "message": "ok",
        "env": {"AWS_SECRET_ACCESS_KEY": "secret"},
        "traceback": "File ...",
    })
    assert out == {"message": "ok"}


def test_redacts_presigned_url():
    payload = {
        "message": "uploaded https://r2.example.com/path?X-Amz-Signature=deadbeef&foo=bar",
    }
    out = redact_payload(payload)
    assert "deadbeef" not in out["message"]
    assert "[REDACTED]" in out["message"]


def test_collapses_absolute_paths_in_messages():
    payload = {"message": 'File "/home/user/radar/validator/neuron.py", line 42'}
    out = redact_payload(payload)
    assert "/home/user/radar" not in out["message"]
    assert "neuron.py" in out["message"]


def test_keeps_ss58_hotkeys():
    # Real-looking SS58 is 47-48 chars, starts with "5".
    hotkey = "5" + "F" * 47  # 48 chars
    payload = {"message": f"miner {hotkey} submitted"}
    out = redact_payload(payload)
    assert hotkey in out["message"]


def test_redacts_long_opaque_token():
    token = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345abcdefgh"  # 40 chars but not git sha
    # Not a git sha (contains uppercase) → should be redacted.
    payload = {"message": f"api_key={token}"}
    out = redact_payload(payload)
    assert token not in out["message"]
    assert "[REDACTED]" in out["message"]


def test_keeps_git_sha():
    sha = "deadbeef" * 5  # 40 hex chars
    payload = {"message": f"git rev: {sha}"}
    out = redact_payload(payload)
    assert sha in out["message"]


def test_redact_event_shape():
    row = {
        "id": 7,
        "hotkey": "5F" + "x" * 46,
        "ts": 100.0,
        "round_id": 42,
        "kind": "log",
        "level": "info",
        "payload": {"message": "hello", "env": {"X": "y"}},
    }
    out = redact_event(row)
    assert out["id"] == 7
    assert out["round_id"] == 42
    assert out["kind"] == "log"
    assert out["level"] == "info"
    assert "env" not in out["payload"]
    assert out["payload"]["message"] == "hello"


def test_nested_redaction():
    payload = {
        "items": [
            {"url": "https://x.com/?X-Amz-Signature=abc"},
            {"path": "/home/user/x.py"},
        ],
    }
    out = redact_payload(payload)
    assert "[REDACTED]" in out["items"][0]["url"]
    assert "/home/user" not in out["items"][1]["path"]
