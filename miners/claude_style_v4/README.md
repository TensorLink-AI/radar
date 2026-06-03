# claude_style_v4 — training-recipe-aware two-pass miner

A fork of `miners/claude_style_v3` that closes v3's biggest blind spot:
**the agent only ever thought about `build_model`**. The training surface
the harness exposes — `build_optimizer`, `build_scheduler`,
`training_config`, `compute_loss`, `configure_amp`, `transform_batch`,
`on_step_*` — was demoted to a one-line "Optional hooks" footnote in
the designer prompt, and every fallback template shipped
`Adam(lr=1e-3)` with no scheduler / no AMP / no grad clip.

v4 doesn't ask the designer to do more in one shot. It splits the
decision **sequentially**:

1. Architecture pass (the v3 designer, unchanged).
2. Recipe-tuner pass — same designer, same tools, but the prompt
   freezes the architecture from pass 1 and asks for a strong
   training recipe only.

The framing is the same one the continuation flow already uses
successfully on continuation rounds; v4 just reuses it inside a
fresh round.

## What's new vs. `claude_style_v3`

| Concern | `v3` | `v4` |
|---|---|---|
| Designer focus per pass | architecture + recipe in one prompt | pass 1: architecture only, pass 2: recipe only |
| Recipe knobs in prompt | one-line "Optional hooks" footnote | dedicated pass with a recipe-only preamble + a `training_recipe_norm` divergence target |
| Analyst output | architectural axes only | architectural axes + `training_recipe_norm` (what frontier members share on the training side) |
| Analyst tool surface | `query_db`, `list_frontier`, `analyze_task`, `time_remaining` | + `get_frontier_member` (so it can actually read recipe hooks off frontier code) |
| Researcher brief | architecture-only | unchanged (intentionally) — the recipe field rides on the analyst digest, not the brief |
| Fallback recipe | `Adam(lr=1e-3)` | AdamW + weight decay + warmup-cosine + bf16 AMP + grad clip + `training_config` |
| Continuation rounds | designer = recipe pass | unchanged — already the right behaviour |

## The three mechanisms

### 1. Recipe-tuner second pass

After the v3 architecture pass produces a validated candidate
(`submit` stashes it as `best_so_far` because pass-1's deadline
fires with > `EARLY_SUBMIT_GATE_SECONDS` remaining), the orchestrator:

1. Extracts the architecture portion (everything but the recipe
   hooks) from the validated code via
   `core.continuation.extract_arch_source`.
2. Synthesizes an `_inround_recipe_context` dict containing
   `frozen_arch` (the extracted block) and `training_recipe_norm`
   (passed through from the analyst).
3. Calls `run_designer` a second time with the same tools but a
   recipe-only system prompt — the new `_inround_recipe_preamble`
   in `prompts.py`. The designer sees: "Here is your already-
   validated architecture, copy it verbatim; spend this slice on
   the training recipe."
4. After pass 2 returns, an architecture-divergence guard
   (`continuation.arch_matches`) discards pass-2's submission if
   it changed the architecture, restoring pass-1's stashed best.

The agent never sees architecture + recipe at the same time. Each
prompt is single-focus.

### 2. Analyst `training_recipe_norm`

The analyst gains `get_frontier_member` and one new digest field.
It reads 1–2 frontier members' source and reports what's identical
on the training side:

```json
{
  "training_recipe_norm": {
    "optimizer":     "Adam (every member, no AdamW or Lion)",
    "learning_rate": "1e-3 (no schedule, no warmup)",
    "lr_schedule":   "none — every member uses a constant LR",
    "weight_decay":  "0 (omitted)",
    "loss":          "task default (no member overrides compute_loss)",
    "amp_dtype":     "bfloat16 default",
    "batch_size":    "task default",
    "grad_clip":     "none",
    "notes":         "training-recipe surface is unexplored — divergence here is cheap"
  }
}
```

The recipe-tuner pass inlines this verbatim into its preamble: "Here's
what everyone does on the recipe surface — beat it." The researcher's
brief is **not** extended with this field — keeping the brief
architecture-only protects the researcher from having to balance two
surfaces (and protects the designer's pass 1 from being pushed toward
the recipe early).

### 3. Fallback templates with a real recipe

Every `core/fallback_templates.py` archetype now ships
`build_optimizer` / `build_scheduler` / `training_config` /
`configure_amp` / `on_step_end` instead of just bare
`Adam(lr=1e-3)`. The shared block (`RECIPE_DEFAULTS`):

- **`build_optimizer`** — AdamW with weight_decay = 0.01 on
  non-bias / non-norm tensors, 0.0 on biases & norms. Betas
  `(0.9, 0.95)`, lr `3e-4`.
- **`build_scheduler`** — `LambdaLR` with 5% linear warmup → cosine
  decay to 10% of base.
- **`training_config`** — `batch_size=32`, `grad_clip=1.0`,
  log-cadence val schedule with `val_base_step=100`, `val_growth=1.6`.
- **`configure_amp`** — `{enabled: True, dtype: "bfloat16"}`.

This raises the floor for any submission that doesn't override the
recipe AND gives the LLM a stronger starting prior — improving on a
decent recipe is easier than inventing one.

Templates are versioned (`FALLBACK_VERSION = "v3"`) so the experiment
DB can distinguish v4-fallback submissions from v3-fallback ones in
post-hoc analysis.

## Inherited from v3 (unchanged)

- **Analyst subagent** runs first and emits an axes-of-variation
  digest the researcher embeds verbatim.
- **Two-phase researcher** — Phase A high-temperature brainstorm with
  hard quotas → Phase B prunes to a JSON brief.
- **Primitive injection** — `core/primitives.PRIMITIVE_POOL` sampled
  deterministically per round and threaded through analyst + researcher.
- **Single-slot `_operator_prompt`** to the designer for GEPA /
  random_mutate prompt evolution.
- **Continuation flow** in `core/continuation.py` — on validator-
  scheduled continuation rounds, the parent's `build_model` /
  `init_weights` / `COMPILE` are lifted verbatim, analyst + researcher
  are skipped, and the designer handles the recipe directly. v4 leaves
  this path untouched and skips the in-round recipe-tuner pass when
  `_continuation_context` is present (continuation already IS a recipe
  pass — running another one would just split the budget).

## Budget split

| Phase | v3 | v4 |
|---|---|---|
| Analyst | 10% / cap 120s | 10% / cap 120s |
| Researcher | 12% / cap 300s | 12% / cap 300s |
| Designer pass 1 (architecture) | 78% (rest) | rest, minus `RECIPE_TUNER_BUDGET_CAP` |
| Recipe-tuner pass 2 | — | up to 330s (fits inside `deadline - now`) |
| Packaging / fallback reserve | 30s | 30s |

`RECIPE_TUNER_BUDGET_CAP = 330` is chosen as
`EARLY_SUBMIT_GATE_SECONDS + 30s buffer` so:

- Pass 1's `submit` at its deadline always **stashes** (because
  `remaining = 330 > 300 = EARLY_SUBMIT_GATE_SECONDS`), never ships.
- Pass 2 starts with 330s remaining; its `submit` fires inside the
  ship-window naturally.

For rounds shorter than `RECIPE_TUNER_MIN_TOTAL_BUDGET = 660s` the
recipe-tuner pass is disabled and the designer reverts to its v3
full-slice behaviour. Continuation rounds also skip the recipe pass.

If pass-1 produces no validated architecture, the recipe-tuner skips
and the orchestrator falls through to the fallback template (which
now ships a decent recipe by construction).

## Tunables

| Constant (in `agent.py`) | Default | Effect |
|---|---|---|
| `ANALYST_BUDGET_FRACTION` | 0.10 | Fraction of round budget for the analyst. |
| `ANALYST_BUDGET_CAP` | 120 | Absolute cap on analyst wall-clock seconds. |
| `RESEARCHER_BUDGET_FRACTION` | 0.12 | Fraction for researcher (Phase A + Phase B). |
| `DESIGNER_BUDGET_FRACTION` | 0.78 | Fraction for the designer pass 1. |
| `RECIPE_TUNER_BUDGET_CAP` | 330 | Reserve for the in-round recipe pass. |
| `RECIPE_TUNER_MIN_SECONDS` | 90 | Below this many seconds at pass-2 start, skip pass 2. |
| `RECIPE_TUNER_MIN_TOTAL_BUDGET` | 660 | Below this much round budget, skip pass 2 entirely. |
| `PRIMITIVES_PER_ROUND` | 2 | How many primitives are injected per round. |
| `PHASE_A_TEMPERATURE` (in `subagents/researcher.py`) | 0.95 | Brainstorm temperature. |

## What to watch to know if v4 beats v3

- **Pass-1 vs pass-2 ship rate**: how often does the recipe-tuner
  actually produce a candidate the orchestrator ships? If <30%, the
  recipe-tuner is wasting budget; tighten its time slice or improve
  the preamble.
- **Architecture-divergence rate**: how often does
  `arch_matches(pass2_code, frozen_arch)` fail? If high, the
  in-round preamble isn't communicating "frozen architecture" hard
  enough — needs strengthening.
- **Frontier members' recipe diversity** — `training_recipe_norm`
  should shrink over rounds as the analyst sees more recipe
  variants on the frontier. If it stays generic, the analyst isn't
  reading frontier code.
- **Validator metric trend on continuation-eligible rounds** — pass-2
  on fresh rounds is essentially "treat your own pass-1 as a parent
  and tune the recipe." If v4's pass-2-shipped runs land closer to
  v3's continuation-round runs than to v3's fresh-round runs, the
  mechanism is working.

## Compatibility notes

- **`_operator_prompt`** (GEPA) is preserved on the designer for
  pass 1. Pass 2 sees the same operator directive (it's still a
  designer call); if you want a separate slot for pass-2 prompt
  evolution, add an `_operator_prompt_pass2` field and a second
  active.json row.
- **`prompts/active.json`** isn't shipped here either — same as v3.
- **`_inround_recipe_context`** lives only inside one orchestrator
  call (set before pass 2, popped after). It never leaks into the
  submission payload — the orchestrator's `_cont_fields` helper still
  only emits `mode="continue"` for validator-scheduled continuation
  rounds; in-round recipe-tuned submissions ship as fresh designs.
- **Continuation rounds** (`core/continuation.py`, shared verbatim
  with v2 / v3): on `challenge["round_type"] == "continuation"` the
  orchestrator lifts the chosen parent's `build_model` /
  `init_weights` / `COMPILE` source verbatim, skips the analyst + Phase
  A + researcher, and the designer gets a frozen-architecture preamble.
  v4 leaves this flow intact; the in-round recipe pass is suppressed
  on continuation rounds.
