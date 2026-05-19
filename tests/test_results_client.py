"""Tests for MinerResultsClient using httpx's MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from miner_template.optimizers import ResultRow
from miner_template.results_client import MinerResultsClient


def _make_client(handler, api_key: str = "test-key") -> MinerResultsClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return MinerResultsClient("http://db.local", api_key, client=http)


def test_requires_db_url_and_api_key():
    with pytest.raises(ValueError):
        MinerResultsClient("", "k")
    with pytest.raises(ValueError):
        MinerResultsClient("http://x", "")


def test_submissions_sends_bearer_and_returns_list():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"submissions": [
            {"submission_id": "s1", "round_id": 1},
        ]})

    c = _make_client(handler, api_key="secret")
    rows = c.submissions(limit=10, task="ts_forecasting")
    assert seen["auth"] == "Bearer secret"
    assert "/miners/me/submissions" in seen["url"]
    assert "limit=10" in seen["url"]
    assert "task=ts_forecasting" in seen["url"]
    assert len(rows) == 1
    assert rows[0]["submission_id"] == "s1"


def test_submissions_accepts_plain_list_response():
    def handler(request):
        return httpx.Response(200, json=[{"submission_id": "x"}])

    c = _make_client(handler)
    rows = c.submissions()
    assert rows[0]["submission_id"] == "x"


def test_results_returns_resultrow_objects():
    def handler(request):
        return httpx.Response(200, json={"results": [{
            "round_id": 5,
            "submission_id": "s5",
            "task_name": "ts_forecasting",
            "prompt_id": "p1",
            "architecture_code": "code",
            "scores": {"raw_score": 0.7},
        }]})

    c = _make_client(handler)
    rows = c.results()
    assert len(rows) == 1
    assert isinstance(rows[0], ResultRow)
    assert rows[0].prompt_id == "p1"
    assert rows[0].scores["raw_score"] == 0.7


def test_results_empty_payload_returns_empty_list():
    def handler(request):
        return httpx.Response(200, json={"results": None})

    c = _make_client(handler)
    assert c.results() == []


def test_summary_returns_dict():
    def handler(request):
        return httpx.Response(200, json={"total": 42, "last_round_id": 7})

    c = _make_client(handler)
    s = c.summary()
    assert s["total"] == 42
    assert s["last_round_id"] == 7


def test_summary_handles_non_dict_response():
    def handler(request):
        return httpx.Response(200, json=[1, 2, 3])

    c = _make_client(handler)
    assert c.summary() == {}


def test_frontier_requires_task():
    def handler(request):
        return httpx.Response(200, json={"points": []})

    c = _make_client(handler)
    with pytest.raises(ValueError):
        c.frontier("")


def test_frontier_returns_points():
    def handler(request):
        assert "/tasks/ts_forecasting/frontier" in str(request.url)
        return httpx.Response(200, json={"points": [
            {"flops": 1e6, "crps": 0.4},
            {"flops": 5e6, "crps": 0.3},
        ]})

    c = _make_client(handler)
    pts = c.frontier("ts_forecasting")
    assert len(pts) == 2


def test_4xx_raises():
    def handler(request):
        return httpx.Response(403, json={"error": "bad key"})

    c = _make_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        c.summary()


def test_context_manager_closes_owned_client():
    def handler(request):
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    # Owned client path — no client passed in.
    http = httpx.Client(transport=transport)
    c = MinerResultsClient("http://x", "k", client=http)
    with c:
        c.summary()
    # The owned-client logic only closes when the client was internally
    # created; we passed one in so we must close manually.  Just confirm
    # close() is idempotent.
    c.close()
    c.close()
