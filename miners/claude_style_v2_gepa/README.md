# claude_style_v2_gepa — GEPA-tuned multi-subagent miner

A fork of `miners/claude_style_v2` rewired so the prompt-population
surface is **multi-slot**: each of the three subagents (researcher,
designer, critic) has its own optimisable system-prompt directive,
co-evolved by `local/optimize.py --optimizer gepa`.

## What's different from `claude_style_v2`

| Concern | `claude_style_v2` | `claude_style_v2_gepa` |
|---|---|---|
| Active prompts file | `prompts/active.json`, untyped rows, round-robin by `round_id`. | `prompts/active.json`, each row tagged via `metadata.slot` ∈ {`researcher`, `designer`, `critic`}. Round-robin **within** each slot. |
| Surface optimised | Designer system-prompt tail only. | All three subagent system-prompt tails, independently. |
| `prompt_id` round-trip | One bare UUID → only the designer slot is attributable. | Composite `r:<rid>\|d:<did>\|c:<cid>` → optimizer can attribute scores to per-slot variants via `--slot`. |
| Default prompts dir | `./prompts` (relative to cwd). | `<this agent>/prompts` (absolute, ships pre-seeded). |
| Seed content | Generic `DEFAULT_SEED_TEMPLATE` with unsubstituted `{frontier}` tokens. | Six claude_v2-aware directives (2 per slot) hand-written for the subagent contracts. |

## Running it

```bash
# Validator + miner using this agent:
python local/run.py --agent_dir miners/claude_style_v2_gepa --rounds 5

# GEPA optimisation pass (designer slot, the highest-impact one):
CHUTES_API_KEY=cpk_... python -m local.optimize \
    --optimizer gepa --slot designer --watch

# Separate GEPA passes for the other slots:
python -m local.optimize --optimizer gepa --slot researcher
python -m local.optimize --optimizer gepa --slot critic
```

`local/optimize.py` defaults `--agent_dir` to this directory when
`--optimizer gepa` is given without an explicit agent, so the
shipped seed population is used out of the box.

## How the wiring works

1. `agent.py:_load_active_prompts(round_id)` reads `prompts/active.json`,
   groups rows by `metadata.slot`, and picks one variant per slot by
   `round_id % len(slot_bucket)`. Returns `{researcher, designer, critic}
   → {id, template}`.

2. The orchestrator stashes three keys on the challenge dict:
   `_operator_prompt_<slot>` (template) and `_operator_prompt_<slot>_id`
   (variant UUID).

3. Each subagent runner reads its own slot's directive and appends it
   to its hardcoded system prompt:
   - `subagents/researcher.py::_build_subagent`
   - `subagents/designer.py::run_designer`
   - `subagents/critic.py::run_critic` (threaded in via the designer's
     `_make_critic_callback`)

4. The submission's `prompt_id` is the composite
   `r:<rid>|d:<did>|c:<cid>`. `local/optimize.py --slot <name>`
   parses it, keeps the rows for that slot, and rewrites each
   `ResultRow.prompt_id` to the bare slot ID before handing it to
   GEPA — so GEPA's group-by-prompt_id logic attributes scores to
   the correct variant.

5. `miner_template/optimizers/gepa.py` and `random_mutate.py`
   preserve the parent `metadata.slot` when minting children, so
   mutated variants stay routable across generations.

## Seed prompts

`prompts/active.json` ships with six hand-written seeds — two per
slot — that target the actual decision points of each subagent (see
the file for the exact text). These replace the generic
`DEFAULT_SEED_TEMPLATE` that `miner_template.prompts.seed_default()`
would otherwise write (which was authored for an older
single-shot agent and contains unsubstituted `{frontier}` tokens).

Edit the file directly to add more seed variants — anything tagged
with a valid `metadata.slot` will be picked up by the loader.
