# claude_style_v2_gepa — GEPA-tuned multi-subagent miner

A fork of `miners/claude_style_v2` wired so the prompt-population
surface is **multi-slot, body-injected, and freeze-except-one** so
GEPA actually accumulates signal as experiments compound.

## What's different from `claude_style_v2`

| Concern | `claude_style_v2` | `claude_style_v2_gepa` |
|---|---|---|
| Active prompts file | `prompts/active.json`, untyped rows, round-robin by `round_id`. | `prompts/active.json`, each row tagged via `metadata.slot` ∈ {`researcher`, `designer`, `critic`}. |
| Surface optimised | Designer system-prompt tail only. | All three subagent system prompts' **`## Principles` body section** — directives replace the section in-place rather than tail-appending. |
| Per-round variation | Always varies the single active variant. | `RADAR_GEPA_VARIED_SLOT` controls which slot varies per round; others pin to their elite. Default `designer` — single-variable so credit assignment is clean. |
| `prompt_id` round-trip | One bare UUID → only the designer slot is attributable. | Composite `r:<rid>\|d:<did>\|c:<cid>` → optimizer attributes scores per-slot via `--slot`. |
| Mutation gate | None — mutates after 1 sample. | `--min_samples N` holds undersampled variants over until they accumulate trials, cutting variance noise. |
| Default prompts dir | `./prompts` (relative to cwd). | `<this agent>/prompts` (absolute, ships pre-seeded). |
| Seed content | Generic `DEFAULT_SEED_TEMPLATE` with unsubstituted `{frontier}` tokens. | Six claude_v2-aware **drop-in replacements** for the `## Principles` body section (2 per slot). |

## Recommended workflow (compound signal across experiments)

### Phase 1: Designer-only evolution on the cheap task

```bash
# Run the miner (default RADAR_GEPA_VARIED_SLOT=designer freezes
# researcher + critic at their elite; only the designer slot varies).
python local/run.py --agent_dir miners/claude_style_v2_gepa \
    --task synth_regression --rounds 100

# Evolve only the designer slot. min_samples=3 holds variants over
# until each has 3 trials; GEPA only mutates variants past the gate.
CHUTES_API_KEY=cpk_... python -m local.optimize \
    --optimizer gepa --slot designer --min_samples 3 \
    --task synth_regression --watch
```

`synth_regression` rounds are cheap (numpy MLP, ~seconds each), so
the noise budget for prompt evolution is much better than spending
expensive `ts_forecasting` rounds. The designer slot has the most
direct effect on the scored code, so this is where the cleanest
signal lives.

### Phase 2: Rotate to critic, then researcher

```bash
# Switch the agent's varied slot:
RADAR_GEPA_VARIED_SLOT=critic python local/run.py \
    --agent_dir miners/claude_style_v2_gepa \
    --task synth_regression --rounds 100

# Evolve the critic slot:
python -m local.optimize --optimizer gepa --slot critic \
    --min_samples 3 --task synth_regression --watch
```

Repeat for `researcher` (lowest-signal slot — likely needs more
samples to move).

### Phase 3: Promote winners to `ts_forecasting`

```bash
# Lock in the evolved prompts (active.json now reflects them) and
# run on the real task. Keep RADAR_GEPA_VARIED_SLOT=designer
# (or `all` if you trust the other slots) for the final tuning.
python local/run.py --agent_dir miners/claude_style_v2_gepa \
    --task ts_forecasting --rounds 20
python -m local.optimize --optimizer gepa --slot designer \
    --min_samples 5 --task ts_forecasting --watch
```

`min_samples` should be larger on the expensive task — each run
costs more, so each variant earns its mutation budget by surviving
more trials.

## Why these design choices

**Freeze-except-one** (`RADAR_GEPA_VARIED_SLOT`). When all three
slots vary every round, the score under combo `(r₁, d₁, c₁)` is
confounded — you can't attribute it to any single slot. Holding two
slots at their elite means each round is a clean A/B on the third.
Three single-variable experiments compose into useful signal; one
three-variable experiment doesn't.

**Body injection, not tail appending.** The original wiring stuck
the directive at the end of an 8k-char system prompt, where earlier
longer sections dominated by virtue of mass and specificity.
Replacing the `## Principles` block in-place gives the directive
the same leverage as the hardcoded principles had.

**Min-samples gate.** With LLM temperature 0.7 and a step-function
size gate, a single trial per variant is mostly noise. Letting
each variant accumulate 3–5 trials before GEPA reflects on it
cuts the chance of locking in lucky lineages.

**Body-replacement seeds.** The shipped seeds are drop-in
replacements for the principles section, written at the same
register and length as the hardcoded text. This means generation 0
is already a sensible system prompt, not a generic placeholder.

## How the wiring works

1. `agent.py:_load_active_prompts(round_id)` reads
   `prompts/active.json`, groups by `metadata.slot`, and applies the
   `RADAR_GEPA_VARIED_SLOT` policy:
   - Varied slot: round-robin within its bucket by `round_id`.
   - Pinned slots: `_pick_elite(bucket)` = highest generation, then
     lex-largest id for determinism (so a freshly-mutated child wins
     over its parent at the same generation count).

2. The orchestrator stashes three keys per slot on the challenge:
   `_operator_prompt_<slot>` (template), `_operator_prompt_<slot>_id`
   (variant UUID). Logs which slot varied vs. which were elite.

3. Each subagent builder accepts `operator_directive` and uses it
   to **replace** the `## Principles` (or critic Rules) body
   section. Empty directive → hardcoded fallback.

4. The submission's `prompt_id` is `r:<rid>|d:<did>|c:<cid>`.
   `local/optimize.py --slot <name>` filters by prefix and rewrites
   each `ResultRow.prompt_id` to the bare slot ID before handing it
   to GEPA, so the group-by-prompt_id logic attributes correctly.

5. `local/optimize.py --min_samples N` splits the population into
   tested (`≥ N` samples) and undersampled. Only tested variants
   are mutated; undersampled survive unchanged.

6. `miner_template/optimizers/gepa.py` and `random_mutate.py`
   preserve parent `metadata.slot` when minting children — tags
   survive across generations.

## Seed prompts

`prompts/active.json` ships six hand-written seeds — two per slot —
sized as drop-in replacements for the `## Principles` body section.
Add more by editing the file; anything tagged with a valid
`metadata.slot` is picked up by the loader. After the first GEPA
pass, generations 1+ accumulate alongside; archived previous
generations land in `prompts/history/gen_NNN.json`.
