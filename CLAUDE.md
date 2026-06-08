# CLAUDE.md — radar-local

## What this is

A single-laptop validator + miner pair that runs phase A → B → C of
the (formerly distributed) radar workflow against a SQLite broker.
Two OS processes, no Docker, no Postgres, no chain, no HMAC. See
`README.md` for the user-facing intro and `local/README.md` for the
full surface.

## Layout

```
local/           The whole stack (one file per responsibility).
miners/          Self-contained miner agents (autonomous,
                 claude_style[_v2], openai_sdk[_v2], patch_decoder).
                 Each subdir is a valid --agent_dir target.
miner_template/  Trimmed to prompts.py + optimizers/ for the
                 prompt-evolution loop.
shared/          url_gate.py (GatedClient) + gift_eval.py (manifest,
                 R2 download, Arrow loader) + r2_audit.py
                 (HippiusStorage / R2AuditLog S3 client).
tests/           pytest suite for the shared/ modules.
```

## GIFT-Eval / R2 access

`shared.r2_audit.HippiusStorage` (alias `R2AuditLog`) is the
S3-compatible client. It reads creds from env in this order:
`HIPPIUS_*` first, then legacy `R2_*` (`R2_ACCOUNT_ID` derives the
per-account endpoint). The default Hippius endpoint is
`https://s3.hippius.com`; pointing at `*.r2.cloudflarestorage.com`
forces region `auto` instead of `decentralized`.

`shared.gift_eval` owns the GIFT-Eval manifest (key → R2 subpath),
the SHORT/MED_LONG leaderboard lists, deterministic per-round
dataset selection, Arrow IPC parsing, and rolling-origin window
construction. Within the `gift-eval-benchmark` bucket the object key
is `gift-eval-full/{subpath}/data-00000-of-00001.arrow` (the bucket
name is **not** part of the key — `r2_prefix` defaults to
`gift-eval-full`).

`python -m local.fetch_gift_eval` is the CLI wrapper over
`ensure_datasets_cached()` — use it to prefetch Arrow files into
`$RADAR_GIFT_EVAL_CACHE` (default `/tmp/radar_gift_eval`). Requires
`pip install -e .[gift_eval]` (pulls boto3 + pyarrow).

## Real ts_forecasting task

`local/run.py --task ts_forecasting` (and `local/validator.py --task
ts_forecasting`) flips from the numpy MLP regression task to the
torch pretrain + GIFT-Eval pipeline. The dispatch lives in
`local/trainer.py::_run_ts_forecasting`, which:

1. Sources training shards and a fixed in-training val split:
   - **Training shards stream by default.** The pretrain corpus is
     multi-TB, so rather than download it the dispatcher presigns a
     deterministic per-seed subset (`PretrainBenchmark.select_shards`,
     which auto-excludes the manifest's `val_shard_keys`) and sets
     `RADAR_PRETRAIN_SHARD_URLS`; the loader fetches one shard at a time.
     `RADAR_PRETRAIN_STREAM_N` caps the subset (0 = all). Prefetched
     local shards in `$RADAR_PRETRAIN_CACHE` win when present (offline
     runs keep working). Streaming needs creds + `boto3`/`httpx`; with
     none, training falls back to GIFT-Eval data.
   - **The val split is a small, fixed download.** It lives in its own
     dir `$RADAR_PRETRAIN_VAL_CACHE` (default `/tmp/radar_pretrain_val`,
     fetched via `fetch_pretrain --val`) so it never overlaps the
     round-varying training shards — val loss curves stay comparable
     across runs. If the val dir resolves to the same path as the train
     dir it falls back to reserving the deterministic last shard; if no
     val shards are found, in-training val is disabled.
2. Sets `CHECKPOINT_DIR`, `SUBMISSION_PATH`, `RADAR_*_LOCAL_PATHS`
   in the env and calls `runner.harness.run_training` with a
   `TSForecastingRunner`.
3. Reloads the saved checkpoint and runs
   `runner.timeseries_forecast.prepare.validate` on the **full**
   GIFT-Eval leaderboard — always all 97 tasks (SHORT_DATASETS
   expanded with MED_LONG terms). No subset knob exists in the local
   stack. Per-task CRPS/MASE are normalized against seasonal-naive
   and geomean'd; the SQLite `metric` is `sqrt(crps * mase)`
   (lower=better) and both raw values land in `objectives` alongside
   `best_val_loss`. **No fallback** — if the checkpoint is missing,
   the cache dir is missing, or GIFT-Eval raises / returns
   non-finite values, the experiment is recorded as a failure
   (`success=False`, `metric=None`). `best_val_loss` stays in
   `objectives` for diagnostics but is never used as the score.

The frozen runner uses sibling-style imports (`from prepare import
...`) inherited from the sandboxed-pod era — the dispatcher adds
`runner/timeseries_forecast/` to `sys.path` so they resolve when
driven from outside a pod.

The pretrain bucket is **separate** from the eval bucket; both share
credentials. Resolution order: `RADAR_PRETRAIN_BUCKET` →
`HIPPIUS_PRETRAIN_BUCKET` → `R2_PRETRAIN_BUCKET` →
`gift-eval-pretrain`. See `local/fetch_pretrain.py::_pretrain_bucket`.

Heavy deps live in the `[ts_forecasting]` extra (torch, safetensors,
pandas, httpx) so the synthetic stack stays numpy-only.

## Continuation training

`--continuation auto|on|off` (auto = on for ts_forecasting) lets a miner
warm-start a run from a prior checkpoint instead of training from scratch.
**The validator owns the cadence**: each round it flips a seeded coin at a
scheduled rate and stamps `challenge['round_type']` (`continuation`/`new`).
The rate is keyed on **attempted rounds** (any experiment row — failed
rounds count too) — 0% until `--continuation_warmup_rounds` (20), then
+`--continuation_step_pct` (2%) every `--continuation_step_every` (3)
up to `--continuation_equilibrium` (0.70): 0→70% over ~105 rounds
post-warmup. A scheduled continuation round downgrades to `new` when no
eligible parents exist; on scheduled-continuation rounds the validator
also biases the bucket pick toward buckets that *have* eligible parents
(rather than round-robin) so the schedule actually fires instead of
landing on empty buckets. The downgrade reason
(`no_eligible_parents` / `no_in_bucket_parents`) is stamped onto the
challenge payload alongside `scheduled_round_type` so the dashboard can
distinguish "always-novel" from "downgraded continuation". The **miner
only picks which parent** (`mode`/`parent_index`) on continuation rounds. The
validator gates eligible parents, runs lineage-disjoint shard assignment
(`--shards_per_round`), and scores continuations on a **second frontier**
— `cumulative_compute` vs GIFT-eval Δ (`parent.metric − this.metric`) —
separate from the absolute initial frontier. Val loss is never scored.
Checkpoints persist via `local/checkpoints.py`. Agent surface:
`/parents`, `/experiments/{id}/trajectory`, `/experiments/{id}/signature`,
and a `continuation` block on `/frontier`. Full write-up in
`docs/continuation_training.md`. Code: `local/continuation.py`,
`local/shards.py`, `local/checkpoints.py`.

## synthetic_data_generator task

`local/run.py --task synthetic_data_generator` (and `--task` on the
validator) is a sibling of `ts_data_pipeline`: miners design a synthetic
data generator (`build_pipeline(context_len, prediction_len,
num_variates, quantiles)` yielding `{input, target}` batches), the
validator trains a model on it and scores on GIFT-Eval. Two deliberate
differences from `ts_data_pipeline`:

1. **The architecture is fixed.** Every round trains the *same*
   miniature (~10M-param) Toto-2.0-style **causal patch decoder** —
   contiguous channel-independent patching with an arcsinh robust scaler
   and missing-value masking, residual-MLP patch projections, learned
   patch+positional embeddings, causal (decoder) transformer blocks with
   PerDimScale attention, and learned forecast-query tokens that emit the
   horizon as contiguous output patches in a single pass (no last-token
   bottleneck). The source text lives in `local/synthetic_arch_code.py`
   (`REFERENCE_ARCH_CODE`); `local/synthetic_arch.py` is the version/card
   wrapper. **Bumping `REFERENCE_ARCH_VERSION` to 2 is required before
   the upgraded arch is merged** (it is parameter-incompatible with the
   original v1 model). It stays source *text* so the validator stays
   numpy-only until a round actually runs. There
   is no `FrozenArchStore` for this task — the arch never moves, so a
   continuation round is literally "keep training the same model's
   weights on (hopefully better) data" and warm-starts are always
   shape-compatible. The fixed arch rides on the challenge as the same
   `frozen_arch` card `ts_data_pipeline` uses, so the miner-facing
   pipeline tools work unchanged.
2. **Scoring is GIFT-only**, exactly like `ts_forecasting`:
   `metric = sqrt(crps * mase)` (lower=better). No AULC composite — the
   in-training val curve is diagnostics only (and best-val checkpoint
   selection), so a sparse val curve is not fatal here. **No fallback**:
   missing checkpoint / GIFT cache / non-finite metrics ⇒ failure.

`objectives` stamps `synth_arch_version` (gates continuation lineages
against a future model swap). Continuation uses the standard second
frontier (`cumulative_compute` vs GIFT Δ). Needs the same caches as
`ts_forecasting` (`RADAR_GIFT_EVAL_CACHE` + pretrain val shard) and the
`[ts_forecasting]` extra. Code: `local/synth_generator.py`
(`run_synth_generator_training`), `local/synthetic_arch.py`; dispatch in
`local/trainer.py`; wiring in `local/task.py` + `local/validator.py`.

## Key files

| File | Purpose |
|------|---------|
| `local/store.py` | SQLite (WAL) broker — challenges, proposals, experiments, artifacts, agent_events. |
| `local/services.py` | Threaded stdlib HTTP server on 127.0.0.1 — `/experiments`, `/llm/chat`, `/desearch/search`, `/wiki`. Logs every miner call (identified by `X-Miner-Id`) to `agent_events`. |
| `local/export_events.py` | Dump `agent_events` to JSONL (`--out`), optionally upload to R2/Hippius (`--r2-bucket --r2-key`). Disable capture with `RADAR_DISABLE_EVENT_LOG=1`. Per-row payload cap defaults to 4MB (override with `RADAR_AGENT_EVENT_MAX_BYTES`) so long transcripts + reasoning traces aren't clipped. Also exposes `flush_round_to_r2()` — the validator calls it at round-end when `RADAR_EVENT_LOG_R2_BUCKET` is set (prefix overridable via `RADAR_EVENT_LOG_R2_PREFIX`, default `agent-events`); successful uploads prune the round's local rows so SQLite stays bounded. Every miner-facing endpoint logs a row (LLM/desearch/wiki *and* `/frontier`, `/experiments/*`, `/artifacts/*`, …) so the trace covers behavior reads, not just provider calls. |
| `local/providers.py` | LLM (Chutes → OpenAI → stub), arxiv via `export.arxiv.org`, file-backed wiki. |
| `local/validator.py` | Round loop: picks bucket → publishes challenge → drains proposals → trains + evaluates → scores. |
| `local/miner.py` | Polls for challenges, loads the agent from `--agent_dir`, calls `design_architecture(challenge, client?)`. |
| `local/agent.py` | Default miner agent. Reads the active prompt via `miner_template.prompts`. |
| `local/trainer.py` | Frozen Phase B+C: numpy MLP forward/backward, MSE on held-out test split. |
| `local/scoring.py` | Size gate + sigmoid-improvement + Pareto bonus. |
| `local/task.py` | Synthetic 8-dim regression + FLOPs-equivalent size buckets. |
| `local/optimize.py` | Prompt-population CLI (`gepa` / `random_mutate`). |
| `local/run.py` | Launches validator + N miners as subprocesses. |
| `local/backup.py` | R2/Hippius backup of `radar_local.db`. Restores `<prefix>/latest.db.gz` at validator start if the local DB is missing, then a daemon thread snapshots (sqlite online backup → gzip) every `RADAR_BACKUP_INTERVAL_SEC` to `<prefix>/snapshots/<UTC>.db.gz` + overwrites `latest.db.gz`. **Default-on whenever `HIPPIUS_*`/`R2_*` creds are set** — defaults bucket=`radar-backups`, prefix=`radar-backups` (or `radar-backups/<RADAR_INSTANCE_ID>` when set, which also namespaces `agent-events`, the checkpoint prefix, and the artifact-mirror prefix), interval=3600s. Override with `RADAR_BACKUP_BUCKET`/`RADAR_BACKUP_PREFIX`; set `RADAR_BACKUP_DISABLE=1` to opt out. |
| `local/artifacts.py` | Per-round artifact mirror — SQLite always, R2/Hippius optional. Writes challenge/proposal/submission/result/logs/checkpoints under `[<RADAR_INSTANCE_ID>/]runs/{task}/r{round_id:06d}/...` in `RADAR_ARTIFACT_BUCKET` (default `radar-local`). The instance-ID prefix keeps co-tenant instances on a shared bucket from clobbering each other; override with `RADAR_ARTIFACT_PREFIX` (empty string = bare `runs/...`). |
| `local/dashboard.py` + `local/dashboard.html` | Read-only stdlib HTTP dashboard over `radar_local.db`. `python -m local.dashboard --db ... --port 8765`, open in Chrome. |
| `miner_template/prompts.py` | `active.json` + `history/gen_NNN.json` atomic-write population store. |
| `miner_template/optimizers/` | Pluggable optimizer registry (`gepa`, `random_mutate`, `pkg.mod:func`). |
| `shared/url_gate.py` | `GatedClient` enforcing the per-challenge `allowed_urls`. |

## Commands

```bash
# Install
pip install -e .

# Single-laptop run
python local/run.py --rounds 5
python local/run.py --agent_dir /path/to/agent --wiki_dir /path/to/notes --miners 3

# Dashboard (read-only, runs beside the validator)
python -m local.dashboard --db local/radar_local.db --port 8765
# → open http://127.0.0.1:8765/ in Chrome

# Prompt evolution
python -m local.optimize --agent_dir /path/to/agent --optimizer random_mutate
CHUTES_API_KEY=cpk_... python -m local.optimize \
    --agent_dir /path/to/agent --optimizer gepa --watch
```

## Code style

- No file over 300 lines.
- Type hints on public functions.
- `@dataclass` for plain data.
- `logger = logging.getLogger(__name__)`.
- The stack must run with only `numpy` installed; everything else is
  optional (DSPy for GEPA, a provider key for the real LLM proxy).
