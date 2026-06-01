"""Fallback chain for local.providers.desearch.

Verifies that Desearch 429 / 5xx / network errors flow through to
arxiv, and that arxiv failures flow through to OpenAlex. We don't
actually hit any external host — every helper is monkeypatched.
"""

from __future__ import annotations

import urllib.error

import pytest

from local import providers


def _raise_429(*_a, **_kw):
    raise urllib.error.HTTPError(
        "https://api.desearch.ai/desearch/ai/search",
        429, "Too Many Requests", {}, None,
    )


def _raise_500(*_a, **_kw):
    raise urllib.error.HTTPError(
        "https://api.desearch.ai/desearch/ai/search",
        500, "Server Error", {}, None,
    )


def _raise_400(*_a, **_kw):
    raise urllib.error.HTTPError(
        "https://api.desearch.ai/desearch/ai/search",
        400, "Bad Request", {}, None,
    )


def _raise_oserror(*_a, **_kw):
    raise OSError("network down")


@pytest.fixture(autouse=True)
def _configure_desearch_key(monkeypatch):
    monkeypatch.setenv("DESEARCH_API_KEY", "test-key")
    monkeypatch.delenv("RADAR_DESEARCH_SN22_URL", raising=False)
    monkeypatch.delenv("RADAR_DESEARCH_DISABLE_FALLBACKS", raising=False)
    monkeypatch.delenv("RADAR_OPENALEX_MAILTO", raising=False)


def test_desearch_429_falls_through_to_arxiv(monkeypatch):
    monkeypatch.setattr(providers, "_desearch_remote", _raise_429)
    monkeypatch.setattr(
        providers, "_arxiv_search",
        lambda q, n: [{"title": "arxiv hit", "abstract": "", "arxiv_id": "1", "url": "u"}],
    )

    def _should_not_call(*_a, **_kw):
        raise AssertionError("openalex should not be reached when arxiv succeeds")

    monkeypatch.setattr(providers, "_openalex_search", _should_not_call)

    out = providers.desearch({"query": "time series foundation model"})
    assert out == {"results": [{"title": "arxiv hit", "abstract": "", "arxiv_id": "1", "url": "u"}]}


def test_desearch_500_falls_through_to_arxiv(monkeypatch):
    monkeypatch.setattr(providers, "_desearch_remote", _raise_500)
    monkeypatch.setattr(
        providers, "_arxiv_search",
        lambda q, n: [{"title": "ok", "abstract": "", "arxiv_id": "", "url": ""}],
    )
    out = providers.desearch({"query": "q"})
    assert out["results"] and out["results"][0]["title"] == "ok"


def test_desearch_non_retryable_does_not_fall_through(monkeypatch):
    monkeypatch.setattr(providers, "_desearch_remote", _raise_400)

    def _should_not_call(*_a, **_kw):
        raise AssertionError("fallback should not run on 4xx other than 429")

    monkeypatch.setattr(providers, "_arxiv_search", _should_not_call)
    monkeypatch.setattr(providers, "_openalex_search", _should_not_call)

    out = providers.desearch({"query": "q"})
    assert out["results"] == []
    assert "HTTPError 400" in out["error"]


def test_arxiv_failure_falls_through_to_openalex(monkeypatch):
    monkeypatch.setattr(providers, "_desearch_remote", _raise_429)
    monkeypatch.setattr(providers, "_arxiv_search", _raise_oserror)
    monkeypatch.setattr(
        providers, "_openalex_search",
        lambda q, n, date_filter: [
            {"title": "openalex hit", "abstract": "abs", "arxiv_id": "", "url": "doi"}
        ],
    )
    out = providers.desearch({"query": "q", "date_filter": "PAST_YEAR"})
    assert out == {"results": [{"title": "openalex hit", "abstract": "abs", "arxiv_id": "", "url": "doi"}]}


def test_all_providers_fail_returns_error(monkeypatch):
    monkeypatch.setattr(providers, "_desearch_remote", _raise_429)
    monkeypatch.setattr(providers, "_arxiv_search", _raise_oserror)
    monkeypatch.setattr(providers, "_openalex_search", _raise_oserror)
    out = providers.desearch({"query": "q"})
    assert out["results"] == []
    assert "desearch:" in out["error"]
    assert "arxiv:" in out["error"]
    assert "openalex:" in out["error"]


def test_disable_fallbacks_env_keeps_old_behaviour(monkeypatch):
    monkeypatch.setenv("RADAR_DESEARCH_DISABLE_FALLBACKS", "1")
    monkeypatch.setattr(providers, "_desearch_remote", _raise_429)

    def _should_not_call(*_a, **_kw):
        raise AssertionError("fallbacks disabled — must not run")

    monkeypatch.setattr(providers, "_arxiv_search", _should_not_call)
    monkeypatch.setattr(providers, "_openalex_search", _should_not_call)

    out = providers.desearch({"query": "q"})
    assert out["results"] == []
    assert "HTTPError 429" in out["error"]


def test_no_desearch_key_skips_straight_to_arxiv(monkeypatch):
    monkeypatch.delenv("DESEARCH_API_KEY", raising=False)
    monkeypatch.setattr(
        providers, "_arxiv_search",
        lambda q, n: [{"title": "t", "abstract": "", "arxiv_id": "", "url": ""}],
    )
    out = providers.desearch({"query": "q"})
    assert out["results"] and out["results"][0]["title"] == "t"


def test_openalex_abstract_reconstruction():
    inverted = {"hello": [0, 3], "world": [1], "again": [2]}
    assert providers._openalex_abstract(inverted) == "hello world again hello"
    assert providers._openalex_abstract({}) == ""
    assert providers._openalex_abstract(None) == ""  # type: ignore[arg-type]
