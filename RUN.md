# Run radar-local under pm2

Quickstart for launching the validator, a miner, and the dashboard as
managed processes with [pm2](https://pm2.keymetrics.io/).

## 1. Load env vars

Put `HIPPIUS_*` / `R2_*` / `CHUTES_API_KEY` / `LLM_URL` etc. in a
`.env` file (see `.env.example`), then export the whole file into
the current shell:

```bash
set -a
source .env
set +a
```

`set -a` marks every variable assigned afterwards for export, so
`source .env` puts the values into the environment pm2 inherits when
you launch processes from this shell. `set +a` turns it back off.

## 2. Install (cached)

```bash
pip install -e ".[ts_forecasting,gift_eval,miners]"
```

Re-running is cheap — pip skips already-satisfied wheels, so this
step is idempotent and uses pip's wheel cache between containers.

## 3. Launch the three processes

```bash
# Validator — drives rounds, runs the services HTTP server, hosts /llm proxy.
# Pin the services port so the miner can use a stable LLM_URL.
pm2 start --name radar-validator --interpreter python3 \
  --no-autorestart -- local/run.py \
    --task ts_forecasting \
    --miners 0 \
    --services_port 8000 \
    --rounds 0

# Miner — points at the validator's /llm proxy. Adjust --agent_dir to taste.
LLM_URL=http://127.0.0.1:8000/llm \
pm2 start --name radar-miner --interpreter python3 \
  --no-autorestart -- local/miner.py \
    --db local/radar_local.db \
    --agent_dir miners/claude_style_v2

# Dashboard — read-only sqlite viewer on :8765.
pm2 start --name radar-dashboard --interpreter python3 \
  --no-autorestart -- -m local.dashboard \
    --db local/radar_local.db \
    --port 8765
```

Open the dashboard at <http://127.0.0.1:8765/>.

## 4. Manage

```bash
pm2 ls                       # status of all three
pm2 logs radar-validator     # tail validator log
pm2 logs radar-miner --lines 200
pm2 restart radar-miner
pm2 stop all                 # stop everything
pm2 delete all               # remove from pm2's process table
```

## Notes

- The validator auto-prefetches the pretrain val shard and the full
  GIFT-Eval benchmark on startup when `--task ts_forecasting` is set
  (`local/validator.py:_ensure_ts_caches`). First boot is slow; later
  boots reuse `$RADAR_GIFT_EVAL_CACHE` and `$RADAR_PRETRAIN_VAL_CACHE`.
- `--services_port 8000` matters: if you leave it at `0`, the validator
  picks an ephemeral port and any `LLM_URL` exported in the miner's env
  goes stale on restart.
- pm2 inherits the env of the shell that ran `pm2 start`, so re-run
  `set -a; source .env; set +a` before adding new processes if you
  change `.env`.
