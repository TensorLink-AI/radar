"""Coverage for the new RunPod fields on TrainerReady."""

from __future__ import annotations

from shared.protocol import TrainerReady


def test_runpod_fields_default_empty():
    """Backwards-compat: existing Basilica/Targon callers must keep working."""
    r = TrainerReady(round_id=1, trainer_url="http://x")
    assert r.runpod_endpoint_id == ""
    assert r.runpod_template_id == ""


def test_to_json_includes_runpod_fields():
    r = TrainerReady(
        round_id=1,
        trainer_url="http://miner-listener:8091",
        miner_hotkey="hk",
        runpod_endpoint_id="ep_x",
        runpod_template_id="tpl_y",
        gpu_class="H100",
        deployed_image_digest="sha256:abc",
    )
    import json
    blob = json.loads(r.to_json())
    assert blob["runpod_endpoint_id"] == "ep_x"
    assert blob["runpod_template_id"] == "tpl_y"
    assert blob["gpu_class"] == "H100"
    assert blob["deployed_image_digest"] == "sha256:abc"


def test_from_json_round_trips_runpod_fields():
    original = TrainerReady(
        round_id=2,
        trainer_url="http://x",
        runpod_endpoint_id="ep",
        runpod_template_id="tpl",
    )
    restored = TrainerReady.from_json(original.to_json())
    assert restored == original


def test_from_json_ignores_unknown_fields():
    """Older miners sending extra fields shouldn't break newer validators."""
    blob = '{"round_id": 1, "trainer_url": "x", "future_field": "ignored"}'
    r = TrainerReady.from_json(blob)
    assert r.round_id == 1
    assert r.trainer_url == "x"


def test_targon_only_fields_stay_empty_on_runpod_envelope():
    r = TrainerReady(
        round_id=1, trainer_url="x", runpod_endpoint_id="ep",
    )
    assert r.targon_workload_uid == ""
    assert r.cvm_ip == ""
