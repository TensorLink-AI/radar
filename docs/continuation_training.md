# Continuation training

Lets a miner **warm-start** a new run from a previously trained
checkpoint instead of always training from scratch, and scores those
continuation runs on a second frontier so "kept training longer" doesn't
just dominate the absolute-metric frontier.

Continuation is a **ts_forecasting** feature (the synth numpy task has no
checkpoints). It's gated by the validator and chosen by the miner.

## The toggle

* **Validator** decides whether continuation is allowed this round
  (`--continuation auto|on|off`; `auto` = on for ts_forecasting) and
  advertises the **eligible parents** (fully-eval'd, checkpoint-bearing,
  in-bucket) in the challenge.
* **Miner** chooses per proposal by setting `mode: "new" | "continue"`
  and `parent_index` in the returned dict. The validator validates the
  choice; an ineligible parent or an architecture that doesn't match the
  parent checkpoint (strict load) **degrades to a fresh run** and is
  recorded with a `continuation incompatible: retried fresh` note.

## How a miner picks a parent

The agent inspects the parent's loss-curve tail: a curve still descending
at the end of its budget is under-trained → a good continuation
candidate; a plateaued curve → prefer a fresh design. See
`local/agent.py::_choose_continuation` / `_tail_descending` for the
reference implementation. Parents (with a down-sampled loss tail) ship in
the challenge's `eligible_parents`; full curves are at
`GET /experiments/{id}/trajectory`.

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
