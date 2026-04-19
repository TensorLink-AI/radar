"""Tests for shared.provenance — pure-Python helpers (detect_components, compute_similarity).

DB-backed provenance tests are in tests/test_pg_provenance.py.
"""

from shared.provenance import detect_components, compute_similarity


# ── detect_components ────────────────────────────────


def test_detect_rmsnorm_and_gelu():
    comps = detect_components("self.norm = RMSNorm(dim)\nself.act = GELU()")
    assert "RMSNorm" in comps
    assert "GELU" in comps


def test_detect_no_components():
    assert detect_components("x = 1 + 2") == []


def test_detect_adamw():
    comps = detect_components("optimizer = AdamW(model.parameters())")
    assert "AdamW" in comps


def test_detect_custom_patterns():
    comps = detect_components("MyCustomLayer()", {"MyCustomLayer": r"\bMyCustomLayer\b"})
    assert comps == ["MyCustomLayer"]


# ── compute_similarity ───────────────────────────────


def test_compute_similarity_identical():
    code = "class Model:\n    def forward(self, input_tensor): return self.linear(input_tensor)"
    sim = compute_similarity(code, code)
    assert sim["jaccard"] == 1.0
    assert sim["diff_ratio"] == 1.0
    assert sim["diff_size"] == 0.0


def test_compute_similarity_different():
    sim = compute_similarity(
        "class EncoderModel:\n    def forward(self): return self.encoder()",
        "class DecoderModel:\n    def backward(self): return self.decoder()",
    )
    assert sim["jaccard"] < 1.0
    assert sim["diff_ratio"] < 1.0
    assert sim["diff_size"] > 0.0


def test_compute_similarity_returns_all_signals():
    sim = compute_similarity("class Foo: pass", "class Bar: pass")
    assert "jaccard" in sim
    assert "diff_ratio" in sim
    assert "diff_size" in sim
