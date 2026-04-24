"""Redaction for validator log events served on the public dashboard.

The /dashboard/api/validators/{hotkey}/events endpoint is world-readable
(per the public-by-design contract in CLAUDE.md). Validator stdout can
contain R2 presigned URLs, file paths inside stack traces, environment
dumps, and other artefacts that shouldn't be published.

This module produces a public-safe view of an event row. The original
raw row stays in Postgres untouched so operators can still inspect it
through the cookie-gated Jinja UI.
"""

from __future__ import annotations

import re
from typing import Any

# R2 / S3 presigned URLs carry a long ``X-Amz-Signature=`` query param.
# Match the whole URL up to the next whitespace / quote so we redact the
# full credentials envelope, not just the signature itself.
_PRESIGNED_URL_RE = re.compile(
    r"https?://[^\s\"'<>]*[?&]X-Amz-Signature=[^\s\"'<>]+",
    re.IGNORECASE,
)

# Generic AWS / R2 key patterns that occasionally appear bare in logs.
_AWS_KEY_RE = re.compile(r"X-Amz-(?:Credential|Signature|Security-Token)=[^\s&\"'<>]+")

# Absolute filesystem paths from tracebacks. We collapse the directory
# part so the file name + line number survive (useful for debugging) but
# the host's directory layout doesn't leak.
_ABS_PATH_RE = re.compile(r'(?:/[\w.\-]+){2,}/([\w.\-]+\.py)')

# Long opaque tokens (>=24 chars of base64-ish). Catches API keys / JWT
# fragments / hex secrets that operators occasionally print.
_LONG_TOKEN_RE = re.compile(r'\b[A-Za-z0-9_\-]{32,}\b')

# Whitelist common identifiers that look like long tokens but are safe:
# git SHAs, Bittensor SS58 hotkeys (which are already public on-chain).
_TOKEN_WHITELIST = {
    # Bittensor SS58 addresses always start with "5" and are 47-48 chars.
    # We let those through by checking the prefix below rather than via a
    # giant set.
}

_REDACTED = "[REDACTED]"


def _redact_string(s: str) -> str:
    if not s:
        return s
    out = _PRESIGNED_URL_RE.sub(_REDACTED, s)
    out = _AWS_KEY_RE.sub(_REDACTED, out)
    out = _ABS_PATH_RE.sub(r"<path>/\1", out)

    def _maybe_mask_token(m: re.Match) -> str:
        token = m.group(0)
        # Bittensor SS58 hotkeys: world-readable on-chain, keep them.
        if len(token) in (47, 48) and token.startswith("5"):
            return token
        # Git SHAs are 40 hex chars — fine to expose.
        if len(token) == 40 and all(c in "0123456789abcdef" for c in token.lower()):
            return token
        return _REDACTED

    out = _LONG_TOKEN_RE.sub(_maybe_mask_token, out)
    return out


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    return value


# Payload keys that always get dropped from public output regardless of
# their value type. ``env`` and ``traceback`` are the obvious risky ones;
# stack traces still leak through the ``message`` field but with paths
# collapsed and tokens masked by ``_redact_string``.
_DROP_KEYS = frozenset({
    "env",
    "environ",
    "traceback",
    "stack",
    "exc_info",
})


def redact_payload(payload: Any) -> Any:
    """Return a public-safe copy of an event payload.

    - Drops well-known sensitive keys outright.
    - Redacts presigned URLs, AWS auth fragments, opaque long tokens.
    - Collapses absolute filesystem paths to ``<path>/file.py``.
    - Leaves Bittensor SS58 hotkeys + git SHAs intact (already public).
    """
    if isinstance(payload, dict):
        return {
            k: _redact_value(v)
            for k, v in payload.items()
            if k not in _DROP_KEYS
        }
    return _redact_value(payload)


def redact_event(row: dict) -> dict:
    """Public-safe projection of a single ``validator_events`` row dict."""
    return {
        "id": row.get("id"),
        "hotkey": row.get("hotkey"),
        "ts": row.get("ts"),
        "round_id": row.get("round_id"),
        "kind": row.get("kind"),
        "level": row.get("level") or "",
        "payload": redact_payload(row.get("payload") or {}),
    }
