# radar-local

A single-laptop stack for autonomous-ML-research rounds: validator
designs a challenge, a miner agent picks an architecture for the
bucket, a frozen training harness trains and evaluates it on a
held-out split, the validator scores against a size-gated Pareto
frontier. SQLite is the only state store.

```bash
pip install numpy
python local/run.py --rounds 5
```

That spawns two OS processes — `local/validator.py` and
`local/miner.py` — talking through `local/radar_local.db`. Multiple
miners with `--miners 3`; point at your own agent dir with
`--agent_dir`; expose markdown notes to the agent with
`--wiki_dir`.

For LLM access, arxiv search, and the prompt-evolution loop see
[`local/README.md`](local/README.md).

## What's in here

| Path | What it is |
|---|---|
| `local/` | The whole stack — validator, miner, trainer, SQLite store, agent-facing services |
| `miners/` | Self-contained miner agents (`autonomous`, `claude_style[_v2]`, `openai_sdk[_v2]`, `patch_decoder`). Pass any subdir to `--agent_dir`. |
| `miner_template/prompts.py` | Atomic-write prompt-population store (`active.json` + history) |
| `miner_template/optimizers/` | Pluggable optimizer registry; built-ins `random_mutate` and `gepa` |
| `shared/url_gate.py` | `GatedClient` the miner uses to reach validator services |
| `shared/gift_eval.py` | GIFT-Eval manifest, R2 download, Arrow loader, rolling-origin windowing |
| `shared/r2_audit.py` | `HippiusStorage` / `R2AuditLog` S3 client (Hippius primary, R2 fallback) |
| `local/fetch_gift_eval.py` | CLI to pre-cache Arrow datasets from R2/Hippius |
| `local/fetch_pretrain.py` | CLI to pre-cache parquet pretrain shards from the (separate) pretrain bucket |
| `runner/harness.py` + `runner/timeseries_forecast/` | Generic torch training harness + ts_forecasting TaskRunner (pretrain + GIFT-Eval eval) |
| `shared/pretrain_data.py` | `PretrainBenchmark` (R2 manifest → shard keys → presigned URLs) and `ShuffleBuffer` |
| `pyproject.toml` | One dependency: `numpy`. GEPA needs `dspy` (extra); GIFT-Eval needs `boto3 + pyarrow` (extra); the real torch task needs `[ts_forecasting]`. |

## Fetching GIFT-Eval data

```bash
pip install -e .[gift_eval]
export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=gift-eval-benchmark
# or HIPPIUS_ACCESS_KEY_ID / HIPPIUS_SECRET_ACCESS_KEY / HIPPIUS_BUCKET

python -m local.fetch_gift_eval --datasets m4_hourly electricity/H
python -m local.fetch_gift_eval --short        # leaderboard short subset
python -m local.fetch_gift_eval --all          # everything (~55 datasets)
```

Cached Arrow files land under `$RADAR_GIFT_EVAL_CACHE` (default `/tmp/radar_gift_eval`) and are read by `shared.gift_eval.load_dataset_for_task`.

## Running the real ts_forecasting task

```bash
pip install -e .[ts_forecasting]                # torch + boto3 + pyarrow + pandas + safetensors + httpx
export RADAR_PRETRAIN_BUCKET=gift-eval-pretrain  # separate from the eval bucket
python -m local.fetch_gift_eval --short          # eval data
python -m local.fetch_pretrain --n 8             # pretrain shards (last one reserved as val)

python local/run.py --task ts_forecasting --agent_dir miners/autonomous
```

Under `--task ts_forecasting` the validator publishes `{context_len, prediction_len, num_variates, quantiles}` task params, agents return `build_model(context_len, prediction_len, num_variates, quantiles)`, and `local/trainer.py` dispatches the submission through `runner.harness.run_training`. The harness pretrains on any parquet shards found under `$RADAR_PRETRAIN_CACHE` (default `/tmp/radar_pretrain`) and evals on Arrow files under `$RADAR_GIFT_EVAL_CACHE`. With no shards present, it trains on random data — useful for smoke-testing the wiring before bucket creds arrive.

## Running a real miner agent

```bash
python local/run.py --rounds 5 --agent_dir miners/autonomous
python local/run.py --rounds 5 --agent_dir miners/patch_decoder
```

This repo started as a pruned-down fork of the distributed radar
subnet (Postgres + Targon/RunPod hosting + image hardening + chain
commitments). All of that machinery was removed when we shrank to the
laptop scope — see PR #175 for the diff if you need the history.
