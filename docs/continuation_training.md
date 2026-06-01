# Continuation training

Lets a miner **warm-start** a new run from a previously trained
checkpoint instead of always training from scratch, and scores those
continuation runs on a second frontier so "kept training longer" doesn't
just dominate the absolute-metric frontier.

Continuation is a **ts_forecasting** feature (the synth numpy task has no
checkpoints). It's gated by the validator and chosen by the miner.

## Who decides what

The **validator owns the cadence** (whether a round is continuation or
new); the **miner only chooses which parent** to warm-start from.

* **Master switch** — `--continuation auto|on|off` (`auto` = on for
  ts_forecasting, off for synth which has no checkpoints).
* **Round type** — each round the validator flips a deterministic,
  seeded coin at a scheduled rate and stamps `challenge['round_type']`
  (`"continuation"` or `"new"`). A scheduled continuation round
  **downgrades to `"new"`** when no eligible (fully-eval'd,
  checkpoint-bearing, in-bucket) parents exist yet — which is why
  pressure ramps in slowly: early rounds have nothing to continue.
* **Miner** reads `round_type`; on a continuation round it sets
  `mode="continue"` + `parent_index`, else `mode="new"`. An architecture
  that doesn't match the parent checkpoint (strict load) **degrades to a
  fresh run**, recorded with a `continuation incompatible: retried fresh`
  note.

### Cadence schedule

The continuation rate ramps **linearly** from `--continuation_rate_start`
(default 0.0) to `--continuation_equilibrium` (default **0.70**) over
`--continuation_ramp_rounds` (default 50), then holds — so exploitation
builds as the frontier matures while ~30% of rounds stay fresh for
exploration. The realized round-type frequency tracks this rate via a
per-round Bernoulli flip seeded by `round_id` (reproducible, independent
of the training seed). See `local/continuation.py::continuation_rate` /
`is_continuation_round`.

## How a miner picks a parent

On a continuation round the agent inspects each eligible parent's
loss-curve tail and prefers one still descending (under-trained ⇒ juice
left); since the validator has mandated continuation it falls back to the
best-metric parent when none look under-trained. See
`local/agent.py::_choose_continuation` / `_tail_descending`. Parents
(with a down-sampled loss tail) ship in the challenge's
`eligible_parents`; full curves are at `GET /experiments/{id}/trajectory`.

## Disjoint shards

A continuation trains, where possible, on pretrain shards its lineage
hasn't seen — otherwise the GIFT-eval Δ rewards memorization rather than
generalization. The validator excludes every ancestor's shards
(`local/shards.py::assign_shards`). Set `--shards_per_round N` so fresh
runs leave headroom; when the disjoint pool is exhausted the run is
**allowed to reuse** shards and flagged `shard_reuse=true` in
`objectives` for auditing.

## Scoring — two frontiers

Per size bucket:

* **Initial frontier** — fresh runs, scored on absolute GIFT-eval metric
  (unchanged `local/scoring.py` path).
* **Continuation frontier** — runs with `n_rounds ≥ 2`, scored on
  `local/continuation.py`:
  * x = `cumulative_compute` (Σ training FLOPs over the lineage)
  * y = Δ = `parent.metric − this.metric` (full GIFT-eval; one eval per
    run since the parent's metric is already recorded)
  * Δ ≤ 0 scores zero; otherwise a sigmoid of normalized improvement
    with a 1.5× bonus on the compute-efficiency frontier.

**Validation loss is never a score term** — it only drives best-checkpoint
selection and the miner's continue/new decision.

## Trajectories

Each run stores only its own segment, in lineage-absolute coordinates
(`RADAR_STEP_OFFSET` / `RADAR_FLOPS_OFFSET`). The full per-lineage
trajectory is **stitched on read** (`GET /experiments/{id}/trajectory`),
returning the concatenated loss curve (tagged with segment boundaries)
and the sparse GIFT-eval series — single source of truth, no quadratic
duplication.

## Agent API surface

| Endpoint | Purpose |
|---|---|
| `GET /parents?task=&min_flops=&max_flops=` | continuation-eligible parents (defaults to the active bucket) |
| `GET /experiments/{id}/trajectory` | stitched loss + GIFT-eval lineage trajectory |
| `GET /experiments/{id}/signature` | parent checkpoint tensor names + shapes (no weights) — author a shape-compatible warm-start |
| `GET /frontier?task=` | now also returns the `continuation` frontier |

Weights never leave the validator; the `signature` endpoint exposes
shapes only. Proposals carry `mode` + `parent_index` via the existing
proposal payload — no new write endpoint.

## Checkpoint persistence

Checkpoints used to die with the trainer's temp workdir. They're now
copied into a durable store (`local/checkpoints.py`,
`$RADAR_CHECKPOINT_DIR`, default `local/checkpoints/`, optionally mirrored
to R2) keyed by experiment id, and garbage-collected each round to those
on a frontier or referenced as a recent parent.
