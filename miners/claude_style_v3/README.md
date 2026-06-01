# claude_style_v3 — analyst-augmented, divergence-stimulated miner

A fork of `miners/claude_style_v2` (the non-GEPA variant) with three
creativity-stimulating mechanisms added: a landscape-analyst pre-pass,
a two-phase brainstorm-then-prune researcher, and per-round
architectural primitive injection. The single-slot `_operator_prompt`
that v2 passes to the designer is preserved unchanged — v3 is **not**
a multi-slot GEPA fork. If you want per-slot prompt evolution, see
`miners/claude_style_v2_gepa`.

## What's new vs. `claude_style_v2`

| Concern | `v2` | `v3` |
|---|---|---|
| Pipeline | `researcher → designer ↔ critic` | `analyst → (phase A) → researcher → designer ↔ critic` |
| Landscape reflection | researcher calls `query_db` ad hoc | dedicated **analyst** subagent emits an *axes-of-variation* digest before researcher runs (technique #1) |
| Researcher ideation | single tool-driven brief pass | **two-phase**: Phase A high-temperature brainstorm with hard quotas → Phase B prunes to brief (technique #3) |
| Creativity steering | seed prompt nudges only | deterministic per-round **primitive injection** — analyst + researcher must address each one (technique #5) |
| GEPA prompt evolution | designer-slot only (single `_operator_prompt`) | same (unchanged) — no analyst/researcher/critic slot evolution in v3 |

## The three mechanisms

### 1. Analyst subagent (axes-of-variation digest)

Runs once per round before the researcher, with a narrow read-only
tool surface (`query_db`, `list_frontier`, `analyze_task`,
`time_remaining`). Output is a structured digest:

```json
{
  "axes_explored":            [...],
  "axes_fixed_across_frontier": [...],
  "saturated_families":       [...],
  "absent_families":          [...],
  "failure_patterns":         [...],
  "injected_primitives_eval": [
    {"primitive": "...", "status": "absent|present|saturated|unknown",
     "recommendation": "explore|skip|hybridize", "note": "..."}
  ],
  "recommend_explore":        [...],
  "recommend_avoid":          [...]
}
```

The researcher embeds the digest verbatim. The point isn't to
enumerate every model on the DB — it's to surface the architectural
**axes the frontier doesn't vary on**. Those are the highest-signal
divergence directions: an idea that moves on a fixed axis is more
likely to dominate than an idea that varies an already-explored axis.

Budget: `ANALYST_BUDGET_FRACTION=0.10` of total, capped at 120s.

### 2. Two-phase researcher (brainstorm → brief)

**Phase A** is a single high-temperature (0.95) `chat()` call with
no tools. It generates 10 candidate ideas under hard quotas:

- ≥3 ideas marked `[risky]` (deliberately unconventional)
- ≥2 ideas marked `[cross-domain]` (importing a primitive from
  vision / audio / signal processing / RL / graph learning)
- Every injected primitive must appear in at least one idea
- No two ideas may share their primary sequence operator AND
  tokenization scheme

**Phase B** is the existing tool-driven Subagent loop, seeded with
the Phase A output and the analyst digest. It prunes the brainstorm
to 2 strong `ideas_to_try` and emits the JSON brief (now with an
optional `primitive_rejections` list).

The quotas matter: they convert "be creative" from a vibe into a
schema constraint the model has to fill, which broadens the
candidate distribution more reliably than a temperature bump alone.

### 3. Primitive injection

`core/primitives.py` curates 15 architectural primitives drawn from
families that are systematically under-represented on typical miner
frontiers (state-space, depthwise-sep conv, MLP-mixer, linear-
recurrent, Hyena, Fourier mixing, Hopfield, structured sparsity,
graph-conv, wavelet, iterated refinement, …).

Each round samples 2 primitives deterministically on `(round_id,
task_name)`. The analyst classifies each one against the DB; the
researcher must either use each primitive in an `ideas_to_try` entry
or explicitly reject it with reason. Determinism means parallel
miners on the same round see the same injection, and the same
round-seed reproduces identical primitives across reruns.

## Tunables

| Constant (in `agent.py`) | Default | Effect |
|---|---|---|
| `ANALYST_BUDGET_FRACTION` | 0.10 | Fraction of round budget for the analyst. |
| `ANALYST_BUDGET_CAP` | 120 | Absolute cap on analyst wall-clock seconds. |
| `RESEARCHER_BUDGET_FRACTION` | 0.12 | Fraction for researcher (Phase A + Phase B). Lower than v2's 0.15 to fund the analyst. |
| `DESIGNER_BUDGET_FRACTION` | 0.78 | Fraction for the designer (down from v2's 0.80). |
| `PRIMITIVES_PER_ROUND` | 2 | How many primitives are injected per round. |
| `PHASE_A_TEMPERATURE` (in `subagents/researcher.py`) | 0.95 | Brainstorm temperature. |

## What to watch to know if v3 beats v2

- **Frontier turnover rate** on the same task — higher means the
  analyst is successfully redirecting toward absent families.
- **Researcher `query_db` calls per round** — should drop sharply
  (the digest replaces ad-hoc DB browsing).
- **`recommend_explore` → shipped design** — causal check that the
  digest actually steers the designer, not just the researcher.
- **Injected-primitive usage rate** in `ideas_to_try` vs.
  `primitive_rejections` — calibrates the injection pool. If
  primitives are rejected >80% of the time the pool is too out-of-
  distribution; if accepted ~100% the pool isn't pushing hard enough.

## Compatibility notes

- Inherits v2's `_operator_prompt` flow unchanged: the designer
  reads it from the challenge dict; the orchestrator stashes the
  active variant ID as `prompt_id` on the submission. Run
  `local/optimize.py --optimizer gepa` against this agent and you'll
  evolve the designer slot the same way v2 does.
- No `prompts/active.json` is shipped. Drop one in if you want
  designer-slot variants; otherwise the designer falls back to its
  hardcoded principles.
- The three new pieces (analyst, Phase A, primitives) have no
  operator-prompt hooks. They're hardcoded — if you want to evolve
  them, fork to `claude_style_v2_gepa` and add `metadata.slot:
  "analyst"` rows to its `active.json`.
