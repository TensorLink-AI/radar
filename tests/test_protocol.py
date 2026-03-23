"""Tests for shared.protocol — round-trip serialization."""

from shared.protocol import Challenge, Proposal


def test_challenge_round_trip():
    c = Challenge(
        challenge_id="tempo_42",
        seed=12345,
        round_id=42,
        task={"name": "ml_training", "time_budget": 300},
        db_url="http://db.internal:8080",
        desearch_url="http://desearch.internal:8081",
        feasible_frontier=[
            {"code": "import torch", "metric": 0.85, "objectives": {"crps": 0.85}},
        ],
    )
    json_str = c.to_json()
    c2 = Challenge.from_json(json_str)
    assert c2.challenge_id == c.challenge_id
    assert c2.seed == c.seed
    assert c2.round_id == c.round_id
    assert c2.task == c.task
    assert c2.db_url == c.db_url
    assert c2.desearch_url == c.desearch_url
    assert len(c2.feasible_frontier) == 1
    assert c2.feasible_frontier[0]["metric"] == 0.85


def test_proposal_round_trip():
    p = Proposal(
        code="import torch\nclass Model: modified = True",
        name="exp_gated_attn_v1",
        motivation="Replaced softmax with GLA per arxiv:2312.06635",
    )
    json_str = p.to_json()
    p2 = Proposal.from_json(json_str)
    assert p2.code == p.code
    assert p2.name == p.name
    assert p2.motivation == p.motivation


def test_challenge_empty():
    c = Challenge()
    json_str = c.to_json()
    c2 = Challenge.from_json(json_str)
    assert c2.challenge_id == ""
    assert c2.seed == 0
    assert c2.feasible_frontier == []


def test_challenge_phase_split_fields():
    """Phase-split fields serialize correctly."""
    c = Challenge(
        challenge_id="round_123",
        min_flops_equivalent=100_000,
        max_flops_equivalent=500_000,
        eval_split_seed=42,
        round_id=123,
    )
    json_str = c.to_json()
    c2 = Challenge.from_json(json_str)
    assert c2.min_flops_equivalent == 100_000
    assert c2.max_flops_equivalent == 500_000
    assert c2.eval_split_seed == 42
    assert c2.round_id == 123


def test_challenge_backward_compat():
    """Old challenges without new fields still deserialize."""
    import json
    old_json = json.dumps({
        "challenge_id": "tempo_42",
        "seed": 12345,
        "task": {},
        "db_url": "",
        "desearch_url": "",
    })
    c = Challenge.from_json(old_json)
    assert c.challenge_id == "tempo_42"
    assert c.min_flops_equivalent == 0
    assert c.round_id == 0
    assert c.feasible_frontier == []


def test_challenge_empty_frontier():
    """Empty frontier (bootstrapping) serializes correctly."""
    c = Challenge(
        challenge_id="round_1",
        round_id=1,
        min_flops_equivalent=100_000,
        max_flops_equivalent=500_000,
        feasible_frontier=[],
    )
    json_str = c.to_json()
    c2 = Challenge.from_json(json_str)
    assert c2.feasible_frontier == []
