# reviewer/

Cross-round meta-reviewer for radar-local. A periodic Claude Code session
that reads the live SQLite DB + sanitized `agent_events` sample, surfaces
systemic patterns no single miner can see, and stages prompt variants
that compete on `experiments.score` like any other mutation.

**Self-contained.** Lives outside `local/` and `miners/`. Reads from the
DB, writes only into `reviewer/proposals/<UTC>/`. Deleting this directory
leaves the live stack byte-for-byte unchanged.

## Trust boundary

```
                READ                             WRITE
reviewer/  ───────────► local/radar_local.db    reviewer/proposals/<session>/
                                                + 1 agent_events row (log)

                                  ┌──────────────────────────────────┐
                                  │ local/promote_prompts.py         │
                                  │ THE TRUST GATE                   │
                                  │ - audits staged variants         │
                                  │ - enforces ≤30% rev_* cap        │
                                  │ - atomic-rewrites active.json    │
                                  └──────────────────────────────────┘
                                          │
                                          ▼
                            miners/claude_style_v{4,5}/prompts/active.json
                                          │
                                          ▼ next round
                                  miners → validator → scoring → loop
```

The reviewer never touches `miners/` or `local/`. The promoter is the
only bridge, and it lives in `local/` so it's owned and reviewed with
the main codebase.

## Subprocess restrictions

```
claude --print
       --model opus
       --permission-mode default          # NOT acceptEdits
       --allowed-tools Read Grep Glob Write    # no Bash, no Edit
       --add-dir reviewer/proposals/<session>  # only writable path
```

No shell. No `sqlite3` CLI. All DB queries run in the parent Python
process and are handed to the subprocess as pre-sanitized JSON.

## Untrusted content handling

`agent_events.request_json` / `response_json` may contain LLM provider
output or web content from outside our trust boundary. Before any of it
reaches the reviewer:

- `sanitize.py` wraps each blob in `<untrusted_external_data>` tags,
- strips NUL bytes,
- caps each row at 64KB (vs the DB's 4MB cap).

The prompt explicitly instructs the reviewer to treat tagged content
as data, not instructions.

## Self-scoring loop

Each session opens with the reviewer's own track record:

```sql
SELECT prompt_id, AVG(score), AVG(metric), COUNT(*), MAX(round_id)
FROM experiments
WHERE prompt_id LIKE 'rev_%' AND success = 1
GROUP BY prompt_id;
```

The reviewer must reconcile prior hypotheses and retire losing variants
**before** opening new threads. This closes the loop the validator
opened: bad reviewer suggestions get out-scored, the reviewer sees the
scores, retires them.

## Trigger gate

`reviewer.py` short-circuits unless:

- `≥ --min-new-experiments` (default 25) new rows since last session, AND
- `≥ --min-interval-hours` (default 24) since last session, AND
- `reviewer/proposals/STOP` sentinel does not exist (kill switch).

Skips are logged to `agent_events` with `kind="reviewer"`, `endpoint="skip"`.

Bypass with `--force` for ad-hoc runs.

## Usage

### Run a session

```bash
python -m reviewer.reviewer --db local/radar_local.db
# or, ad-hoc:
python -m reviewer.reviewer --db local/radar_local.db --force --max-hours 6
```

### List pending variants

```bash
python -m local.promote_prompts --db local/radar_local.db
```

### Promote a session into live miners

```bash
# v4 + v5 are the defaults
python -m local.promote_prompts \
    --db local/radar_local.db \
    --session 20260607T140000Z \
    --apply
```

### Kill switch

```bash
touch reviewer/proposals/STOP
```

Reviewer skips immediately, logs the skip, exits clean. Remove the file
to re-enable.

## Layout

```
reviewer/
  reviewer.py            CLI orchestrator
  prompt_template.py     the system/user prompt the model sees
  sanitize.py            untrusted-payload wrapping + sampling
  schema_check.py        fail-fast on radar_local.db schema drift
  proposals/             session outputs (created at runtime)
    <UTC>/
      summary.json       structured snapshot we fed the model
      review.md          markdown report (written by claude)
      hypotheses.jsonl   open/closed/refuted threads
      rev_<id>.json      staged prompt variants (0 to 3 per session)
      stdout.txt         raw subprocess output (audit trail)
    STOP                 sentinel to disable next run
```

## Why this design

- **One delivery surface** — staged prompt variants. No shared notes
  files, no direct writes to `miners/`. Everything that affects
  production must go through the promoter, which is small, pure, and
  code-reviewed alongside miner changes.
- **Reviewer's output competes on score.** It doesn't bypass the
  validator's judgment; it just generates more-informed candidates
  than random mutation would.
- **Diversity cap.** Reviewer-authored variants can never exceed 30% of
  any miner's pool. At least 70% of the population is always non-`rev_*`.
- **Isolation.** Subprocess `--add-dir` restricted to one session dir.
  No `Bash`, no `Edit`. A confused or compromised reviewer can write
  arbitrary garbage into one session dir and that's the blast radius.

## Phasing

Phase 1 (this commit): reviewer + promoter + atomic-read patches.
Promotion is manual (`--apply --yes`).

Phase 2: dashboard panel listing reviewer runs + pending variants.

Phase 3: auto-promotion of variants with ≥N rounds of scoring above
the active median.

Phase 4 (optional): run reviewer on a different model family from the
miners to mitigate compounding bias.
