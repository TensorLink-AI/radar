"""Tests for prompt_id round-trip through Proposal + DataElement + INSERT."""

from __future__ import annotations

import json

from shared.database import DataElement
from shared.pg_schema import INSERT_SQL, element_to_params
from shared.protocol import Proposal


def test_proposal_defaults_empty_prompt_id():
    p = Proposal()
    assert p.prompt_id == ""


def test_proposal_to_json_includes_prompt_id():
    p = Proposal(code="x", prompt_id="abc123")
    d = json.loads(p.to_json())
    assert d["prompt_id"] == "abc123"


def test_proposal_from_json_resilient_to_missing_prompt_id():
    p = Proposal.from_json(json.dumps({"code": "x"}))
    assert p.prompt_id == ""


def test_proposal_from_json_roundtrips_prompt_id():
    p = Proposal.from_json(json.dumps({"code": "x", "prompt_id": "p1"}))
    assert p.prompt_id == "p1"


def test_data_element_defaults_empty_prompt_id():
    e = DataElement()
    assert e.prompt_id == ""


def test_data_element_carries_prompt_id():
    e = DataElement(name="x", prompt_id="p-abc")
    assert e.prompt_id == "p-abc"


def test_element_to_params_includes_prompt_id_as_last_arg():
    e = DataElement(name="x", code="c", prompt_id="p-1")
    params = element_to_params(e, next_id=42)
    # INSERT_SQL has 25 placeholders; the new prompt_id is $25.
    assert len(params) == 25
    assert params[-1] == "p-1"


def test_element_to_params_normalizes_empty_prompt_id_to_none():
    e = DataElement(name="x", code="c")  # prompt_id default ""
    params = element_to_params(e, next_id=0)
    assert params[-1] is None  # so the DB stores NULL not ''


def test_insert_sql_lists_prompt_id_column():
    assert "prompt_id" in INSERT_SQL
    # 25 positional placeholders.
    assert "$25" in INSERT_SQL
