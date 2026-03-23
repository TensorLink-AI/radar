"""Tests for validator.desearch_proxy — rate-limited arxiv search proxy."""

import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from validator.desearch_proxy import (
    DesearchProxy,
    SearchQuery,
    SearchResponse,
    SearchResult,
    _parse_sn22_response,
    register_routes,
    set_proxy,
)


# ── Unit tests for DesearchProxy ────────────────────────────────────────────


class TestDesearchProxy:
    def test_initial_quota(self):
        proxy = DesearchProxy(max_queries=20)
        assert proxy.remaining_queries(0) == 20
        assert proxy.remaining_queries(1) == 20

    def test_rate_limiting(self):
        proxy = DesearchProxy(max_queries=3)
        proxy._record_query(0)
        assert proxy.remaining_queries(0) == 2
        proxy._record_query(0)
        assert proxy.remaining_queries(0) == 1
        proxy._record_query(0)
        assert proxy.remaining_queries(0) == 0

    def test_rate_limit_per_miner(self):
        """Each miner has independent quota."""
        proxy = DesearchProxy(max_queries=2)
        proxy._record_query(0)
        proxy._record_query(0)
        assert proxy.remaining_queries(0) == 0
        assert proxy.remaining_queries(1) == 2  # miner 1 unaffected

    def test_reset_limits(self):
        proxy = DesearchProxy(max_queries=2)
        proxy._record_query(0)
        proxy._record_query(0)
        assert proxy.remaining_queries(0) == 0
        proxy.reset_limits()
        assert proxy.remaining_queries(0) == 2

    def test_old_queries_pruned(self):
        """Queries older than tempo_seconds are pruned."""
        proxy = DesearchProxy(max_queries=2, tempo_seconds=10)
        # Manually add an old timestamp
        proxy._query_counts[0].append(time.time() - 20)
        assert proxy.remaining_queries(0) == 2  # old query pruned

    @pytest.mark.asyncio
    async def test_search_rate_limit_exceeded(self):
        proxy = DesearchProxy(max_queries=1)
        proxy._record_query(0)
        with pytest.raises(HTTPException) as exc_info:
            await proxy.search(0, "attention mechanisms")
        assert exc_info.value.status_code == 429


# ── Unit tests for response parsing ─────────────────────────────────────────


class TestParseSN22Response:
    def test_parse_list_format(self):
        data = [
            {"title": "Paper A", "authors": ["Auth1"], "abstract": "abs",
             "arxiv_id": "2301.00001", "published": "2023-01-01",
             "url": "https://arxiv.org/abs/2301.00001"},
        ]
        results = _parse_sn22_response(data)
        assert len(results) == 1
        assert results[0].title == "Paper A"
        assert results[0].arxiv_id == "2301.00001"

    def test_parse_dict_with_results_key(self):
        data = {"results": [{"title": "Paper B", "id": "2301.00002", "link": "http://x"}]}
        results = _parse_sn22_response(data)
        assert len(results) == 1
        assert results[0].arxiv_id == "2301.00002"
        assert results[0].url == "http://x"

    def test_parse_dict_with_papers_key(self):
        data = {"papers": [{"title": "Paper C"}]}
        results = _parse_sn22_response(data)
        assert len(results) == 1
        assert results[0].title == "Paper C"

    def test_parse_empty(self):
        assert _parse_sn22_response([]) == []
        assert _parse_sn22_response({"results": []}) == []

    def test_parse_non_dict_items_skipped(self):
        data = [{"title": "OK"}, "bad", 42, None]
        results = _parse_sn22_response(data)
        assert len(results) == 1


# ── Integration tests with FastAPI ──────────────────────────────────────────


def _make_app() -> tuple[FastAPI, DesearchProxy]:
    app = FastAPI()
    proxy = DesearchProxy(max_queries=5)
    set_proxy(proxy)
    register_routes(app)
    return app, proxy


class TestDesearchRoutes:
    def test_health(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/desearch/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_quota(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/desearch/quota", headers={"X-Miner-UID": "0"})
        assert r.status_code == 200
        assert r.json()["remaining_queries"] == 5
        assert r.json()["miner_uid"] == 0

    def test_quota_with_hotkey(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/desearch/quota", headers={
            "X-Miner-UID": "5",
            "X-Miner-Hotkey": "5GrwvaEF...",
        })
        assert r.status_code == 200
        assert r.json()["miner_uid"] == 5
        assert r.json()["miner_hotkey"] == "5GrwvaEF..."

    def test_quota_hotkey_optional(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/desearch/quota", headers={"X-Miner-UID": "0"})
        assert r.status_code == 200
        assert r.json()["miner_hotkey"] == ""

    def test_missing_miner_uid(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/desearch/quota")
        assert r.status_code == 400

    def test_invalid_miner_uid(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/desearch/quota", headers={"X-Miner-UID": "abc"})
        assert r.status_code == 400

    def test_negative_miner_uid(self):
        app, _ = _make_app()
        client = TestClient(app)
        r = client.get("/desearch/quota", headers={"X-Miner-UID": "-1"})
        assert r.status_code == 400

    def test_search_validation(self):
        app, _ = _make_app()
        client = TestClient(app)
        # Empty query should fail validation
        r = client.post(
            "/desearch/search",
            json={"query": "", "max_results": 5},
            headers={"X-Miner-UID": "0"},
        )
        assert r.status_code == 422  # Pydantic validation error
