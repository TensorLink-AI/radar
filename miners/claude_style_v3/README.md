# claude_style_v3 — analyst-augmented, divergence-stimulated miner

A fork of `miners/claude_style_v2_gepa` that adds three creativity-
stimulating mechanisms on top of the GEPA wiring. The base wiring
(multi-slot body-injected prompts, freeze-except-one varied slot,
`min_samples` gate) carries over unchanged.

## What's new vs. `claude_style_v2_gepa`

| Concern | `v2_gepa` | `v3` |
|---|---|---|
| Pipeline | `researcher → designer ↔ critic` | `analyst → (phase A) → researcher → designer ↔ critic` |
| GEPA slots | `{researcher, designer, critic}` | `{analyst, researcher, designer, critic}` |
| `prompt_id` | `r:<rid>\|d:<did>\|c:<cid>` | `a:<aid>\|r:<rid>\|d:<did>\|c:<cid>` |
| Landscape reflection | researcher calls `query_db` ad hoc | dedicated **analyst** subagent emits an *axes-of-variation* digest before researcher runs (technique #1) |
| Researcher ideation | single tool-driven brief pass | **two-phase**: Phase A high-temperature brainstorm with hard quotas → Phase B prunes to brief (technique #3) |
| Creativity steering | seed prompt nudges only | deterministic per-round **primitive injection** — analyst + researcher must address each one (technique #5) |

## The three mechanisms

### 1. Analyst subagent (axes-of-variation digest)

Runs once per round before the researcher, with a narrow read-only
tool surface (`query_db`, `list_frontier`, `analyze_task`,
`time_remaining`). Output is a structured digest:

```json
{
  "axes_explored":            [...],
  "axes_fixed_across_frontier": [...],  // highest-signal field
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
miners on the same round see the same injection — GEPA can attribute
outcomes to specific injections cleanly.

## GEPA workflow (extended for the analyst slot)

Same as `v2_gepa`, plus the analyst as a fourth slot:

```bash
# Vary the analyst slot, freeze the other three to their elites.
RADAR_GEPA_VARIED_SLOT=analyst python local/run.py \
    --agent_dir miners/claude_style_v3 \
    --task synth_regression --rounds 100

python -m local.optimize --optimizer gepa --slot analyst \
    --min_samples 3 --task synth_regression --watch
```

`RADAR_GEPA_VARIED_SLOT=rotate` cycles `analyst → researcher →
designer → critic` across rounds. The default remains `designer`
(highest-leverage slot).

## Seed prompts

`prompts/active.json` ships **eight** seeds — two per slot now that
the analyst is in the rotation:

- `seed-analyst-saturation` — focus on saturated vs. absent families.
- `seed-analyst-trend` — focus on frontier turnover + failure clusters.
- `seed-researcher-frontier-gaps` / `seed-researcher-non-attention`
  (unchanged from v2_gepa).
- `seed-designer-two-sketches` / `seed-designer-late-window`
  (unchanged).
- `seed-critic-structural-change` / `seed-critic-flops-direction`
  (unchanged).

## Tunables

| Env var | Default | Effect |
|---|---|---|
| `RADAR_GEPA_VARIED_SLOT` | `designer` | Which slot varies per round; v3 accepts `analyst` in addition to the v2_gepa values. |
| `ANALYST_BUDGET_FRACTION` | 0.10 (constant) | Fraction of round budget for the analyst. |
| `PRIMITIVES_PER_ROUND` | 2 (constant) | How many primitives are injected per round. |
| `PHASE_A_TEMPERATURE` | 0.95 (constant) | Temperature for the Phase A brainstorm. |

## What to watch to know if v3 beats v2_gepa

- Frontier-turnover rate on the same task — higher means the analyst
  is successfully redirecting toward absent families.
- Researcher `query_db` call count per round — should drop sharply
  (the digest replaces ad-hoc DB browsing).
- Whether `recommend_explore` families actually appear in submitted
  designer code (causal check that the digest steers).
- Injected-primitive usage rate in `ideas_to_try` vs.
  `primitive_rejections` — calibrates whether the injection pool is
  set right or needs broader / narrower primitives.
- GEPA on the analyst slot — does an evolved analyst principle beat
  the seeds within ~20 rounds?
