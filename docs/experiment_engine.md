# Experiment engine: round types, noise floor, paired scoring, lab reports

The validator owns a vocabulary of round types beyond `new` /
`continuation`, plus the measurement machinery that makes frontier
movements interpretable. Everything follows the pattern continuation
established: **seeded validator-owned coins decide the round type; the
miner only fills in the blank; results are stamped so they stay
attributable forever.**

## Round types

Scheduled by one seeded draw per round
(`local/round_types.py::special_round_type`, namespace
`special-round-{round_id}`), which pre-empts the continuation flip.
Each type downgrades to a normal round when its prerequisite is missing
(`downgrade_reason` is stamped on the challenge, mirroring continuation
downgrades).

| type | flag (default) | what happens |
|---|---|---|
| `replicate` | `--replicate_pct` (0.05) | Validator-only: re-train a seeded frontier member's exact code with this round's seed. No challenge is opened to miners (the bookkeeping row posts as `collected`). Recorded with `mode="replicate"`, `objectives.replicate_of`, score 0, excluded from every frontier. Feeds the noise floor. |
| `ablate` | `--ablate_pct` (0.05) | Challenge carries `ablation_target` (a frontier member's card incl. code); the miner is asked for a **minimal one-component diff**. Scored normally; `objectives.ablation_of` pins the comparison anchor for lab reports. |
| `recipe_only` | `--recipe_pct` (0.05) | Challenge carries `recipe_base`; the architecture surface (`build_model` + top-level classes) must be AST-identical to the base — only optimizer/scheduler/training-config hooks may change. A violating submission is treated as a normal new round (noted, not failed). |
| `transfer` | `--transfer_pct` (0.05) | A smaller-bucket frontier winner ships as `transfer_source`; the miner scales the design into this round's bucket. `objectives.transfer_of` pins provenance. `recipe_only`/`transfer` only fire on bucketed (non-pipeline) tasks. |

### Agent support

`miners/claude_style_v4` and `_v5` understand all of the above
(`core/round_context.py`): ablate/transfer rounds replace the
analyst+researcher phases with a focused brief + target-code preamble,
recipe_only rides the agents' existing in-round recipe machinery, the
screening tier ships validated alternates as `candidates`, and the
analyst reads `/lab_reports` + `/experiments/noise`. Other agents
degrade gracefully — an unaware agent just treats special rounds as
normal ones (recipe_only then downgrades validator-side).

## Noise floor

`local/noise.py` pools every (original, replicate) metric pair:
`σ = sqrt(mean(d²)/2)`. `GET /experiments/noise` serves
`{n_pairs, sigma_metric, sigma_rel, threshold}` where `threshold =
1.645·σ` is the one-sided 95% band. Until replicates accumulate, the
floor is unknown — which is itself the honest answer.

## Per-task eval persistence + canary

`prepare.validate` always computed a per-dataset breakdown; it now
persists. `local/eval_metrics.py::finalize_gift_eval` (used by all three
torch dispatchers) writes `objectives.per_task` =
`[{name, ncrps, nmase}, ...]` and recomputes the scored aggregates from
it.

Set `RADAR_EVAL_CANARY_FRAC=0.15` to hold a deterministic (name-hashed,
salt `gift-eval-canary-v1`) 15% of GIFT tasks **out of the scored
aggregate**. Canary aggregates land in `objectives.canary_*` and are
never selected on — divergence between scored and canary trends is the
benchmark-overfitting alarm. The fraction is pinned into the
continuation epoch (`eval_canary_frac`), since changing it changes what
`metric` means. Default off (0).

## Paired continuation scoring

Continuation Δ is a difference of two noisy geomeans, but per-dataset
noise is correlated between parent and child. When both carry
`per_task`, the validator computes a paired sign test
(`eval_metrics.paired_per_task_delta`, stamped to `objectives.paired`)
and `score_continuation` zeroes any "improvement" that isn't
significant (one-sided 95%, ≥10 joined tasks). Scalar Δ ≤ 0 still
zeroes as before; with no paired data the scalar path is unchanged.
The realized Δ is now also persisted to `objectives.delta`, which fixes
a latent bug where the continuation frontier never accumulated points.

## Lab reports

`local/lab_reports.py` writes a structured post-mortem per experiment at
round end (table `lab_reports`): hypothesis (miner motivation), the
validator-declared baseline (parent / ablation target / recipe base /
transfer source / replicate original), scalar Δ, paired stats, top
improved/regressed datasets, diff stats, and a noise-aware verdict.
With a provider key set, a 2-3-sentence LLM narrative is appended
(`RADAR_LAB_REPORT_LLM=0` opts out).

Agent surface: `GET /lab_reports?n=&task=` and
`GET /experiments/{id}/report` (builds on demand for old rows).

## Screening tier

`--screen_candidates K` (ts_forecasting, fresh rounds) advertises
`challenge.screening`; the agent may include
`candidates=[{name, code}, ...]` in its proposal. Each candidate gets
`--screen_seconds` (600) of training and a truncated eval
(`--screen_eval_tasks`, 15, via `RADAR_GIFT_EVAL_MAX_TASKS`); only the
winner runs at full budget. Screen runs never become experiment rows
(subset metrics aren't comparable) — summaries land in the final
experiment's `objectives.screening`.

## Training isolation

`local/train_subprocess.py` runs Phase B+C in a spawned subprocess with
a hard timeout (budget·1.5 + eval allowance, override
`RADAR_TRAIN_TIMEOUT_SEC`); a segfault or hang in submitted code becomes
a failed experiment instead of killing the validator.
`RADAR_TRAIN_ISOLATION=auto|always|never` (auto = subprocess for torch
tasks, in-process for the numpy task).

## Frozen-pipeline control gate + per-round render

Promotion of a frozen pipeline is now gated on **measured downstream
value**: `--frozen_pipeline_control_seconds` (600; 0 disables) trains
the best ts_forecasting arch twice with the same seed — with and
without the candidate's synthetic shards — and promotes only when the
mixed run wins (`local/pipeline_control.py`). Rejected candidates are
remembered in the manifest and not retried.
`--frozen_pipeline_render_per_round` re-renders the generator's shards
each round (round-seeded stream window) instead of reusing one tiny
fixed corpus.
