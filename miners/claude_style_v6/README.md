# claude_style_v6 — data-generator quality miner

A fork of `miners/claude_style_v5`. v6 inherits the entire v5 stack for
the architecture tasks (recipe-sanity inspector, divergence target,
DIVERGE critic) and concentrates its additions on the **data-generator
tasks** — `ts_data_pipeline` and `synthetic_data_generator` — where v5
left both a correctness gap and a missing feedback signal.

## What's new vs. `claude_style_v5`

| Concern | `v5` | `v6` |
|---|---|---|
| `synthetic_data_generator` validation | required `build_model`/`build_optimizer` (wrong contract) → valid `build_pipeline` self-rejected | recognised via centralised `core.is_pipeline_task` → requires `build_pipeline`, like `ts_data_pipeline` |
| `synthetic_data_generator` fallback | emitted a model (wrong contract) → guaranteed failed round | emits the procedural pipeline generator |
| Smoke test | shapes only (2 batches) | shapes + distribution-quality stats over 4 batches (variance, constant fraction, lag-1 autocorr, range, cross-batch variety) with degeneracy warnings |
| Data-design prompt | 4-bullet family menu | concrete GIFT-Eval-aware recipes (trend, multi-scale seasonality, regime switching, realistic noise, scale variety, multivariate coupling) with a diversity-first mandate |
| Fallback tag | `FALLBACK_VERSION = "v4"` | `FALLBACK_VERSION = "v5"` |

The three pipeline-task mechanisms are:

1. **Task parity.** `core/__init__.py` defines `PIPELINE_TASKS` /
   `is_pipeline_task`; `core.validation`, `core.fallback_templates`, and
   `tools.py` all gate on it so routing, validation, fallback, and tool
   descriptions agree on both pipeline tasks.
2. **Quality probe.** `core/pipeline_quality.py` computes cheap
   descriptive stats + degeneracy warnings over the batches a generator
   yields; `pipeline_smoke_test` leads with any warnings so the designer
   fixes a collapsed/zero-variance/identical-batch generator in-round
   instead of failing the held-out GIFT gate a round later.
3. **Design prompt.** `build_pipeline_designer_system_prompt` reframes the
   design space around structural coverage and points the loop at the new
   quality stats.

---

## Inherited from v5 — divergence-target + recipe-sanity

A fork of `miners/claude_style_v4` that targets v4's three remaining
failure modes:

1. **Recipe-side silent under-performance.** v4's `validate_code` smoke
   test catches recipes that *crash*. It doesn't catch recipes that
   *run fine but produce weak models* — `lr=1e-2` on an AdamW-driven
   transformer, `AdamW(weight_decay=0)` (i.e. just Adam), missing grad
   clipping on Adam-family optimizers, `COMPILE=True` without
   bf16 AMP. These are the recipes that ship, train, and end up
   mid-pack.
2. **Frontier-tying candidates.** v4's researcher names "frontier
   gaps" and "ideas to try", but doesn't require the lead idea to
   actually *break* with the frontier on any axis. The designer can
   ship a validated candidate that matches the frontier consensus on
   every fixed axis — and tie. A tie loses the Pareto bonus, so
   matching the consensus is silently worse than diverging on a
   single axis.
3. **Critic is purely diagnostic.** v4's critic emits KEEP/CHANGE/DROP
   on a per-iteration basis, but it grades on validation outcome, not
   on novelty. A passing validation produces a "ship it" critique
   whether the design is novel or a near-clone of a frontier member.

v5's additions are deliberately small — the v4 pipeline is otherwise
working — and live in three places.

## What's new vs. `claude_style_v4`

| Concern | `v4` | `v5` |
|---|---|---|
| Static recipe inspection | none — only the runtime smoke test in `core.validation` | new `core.recipe_sanity` AST inspector flags lr/weight-decay/grad-clip/scheduler/AMP/batch anti-patterns |
| Designer feedback loop | critic-only after each `validate_code` | critic **+** recipe-sanity findings appended in one message |
| Researcher brief schema | `relevant_prior_work`, `frontier_gaps`, `ideas_to_try`, `plan` | `relevant_prior_work`, `ideas_to_try`, `divergence_axes` (required, non-empty), `plan` — `frontier_gaps` collapsed into `divergence_axes` (an axis everyone fixes the same way IS the gap, naming the break is more actionable) |
| Designer "divergence" framing | implicit ("beat the frontier") | explicit ("honor the brief's `divergence_axes` or you tie") |
| Critic template | 3 lines (KEEP / CHANGE / DROP) covering architecture **and** recipe | 4 lines (+ DIVERGE) scoped to **architecture only** — recipe coverage handed off to the static inspector |
| Critic `max_tokens` | 512 | 256 (the four-line template fits comfortably and the cap stops prose padding) |
| Continuation parent friendliness | mentioned in `docs/continuation_training.md` only | explicit designer-prompt principle to keep arch / recipe cleanly separable |
| Fallback tag | `FALLBACK_VERSION = "v3"` | `FALLBACK_VERSION = "v4"` (DB analysis can tell v5 fallback rounds apart) |

## The three mechanisms

### 1. Static recipe-sanity inspector

`core/recipe_sanity.py` walks the candidate's AST (no torch import, no
exec, <1ms per call) and emits a list of `Warning` records with
severity `info` / `warn` / `critical`. The checks are conservative —
each one is gated on a known-bad signature so false positives are rare:

| Check | Trigger | Severity |
|---|---|---|
| `lr_non_positive` | optimizer call with `lr ≤ 0` or NaN | critical |
| `lr_too_large` | lr > 5e-2 | critical |
| `lr_too_small` | lr < 1e-5 | warn |
| `adamw_no_weight_decay` | `AdamW(weight_decay=0)` | warn |
| `no_scheduler` | lr ≥ 5e-4 and no `build_scheduler` | warn |
| `grad_clip_missing` / `_zero` / `_null` | `training_config` lacks a non-trivial `grad_clip` | warn |
| `no_training_config` | no `training_config` at all | info |
| `compile_no_amp` | `COMPILE = True` with no `configure_amp` opting into bf16 | info |
| `batch_too_large` | `batch_size ≥ 256` on a long-context task | warn |

The designer's post-validate callback runs the inspector right after
the LLM critic, formats the findings as a `Recipe sanity inspection:`
block (`!!!` critical, `!!` warn, `.` info), and appends it below the
critic's KEEP/CHANGE/DROP/DIVERGE message. The LLM sees the
combined feedback on its next turn — no extra LLM call, no new tool.

### 2. Divergence target in the brief

`BRIEF_SCHEMA_EXAMPLE` gains a `divergence_axes` field. The researcher
system prompt now requires it to be non-empty:

```json
{
  "divergence_axes": [
    "sequence operator: replace full attention with selective scan",
    "tokenization: drop patching, operate on raw samples"
  ]
}
```

Each entry must (a) name one axis from the analyst's
`axes_fixed_across_frontier` and (b) state the direction of the break.
The researcher prompt warns: a brief with no divergence is a request
to ship a slight variation that ties the frontier and loses the
Pareto bonus.

The designer system prompt gains a matching principle: "Honor the
divergence axis. A validated candidate that matches the frontier on
every fixed axis will tie, not dominate — go back and pick a different
idea." The designer is free to override the brief (same as v4) but
must own the trade-off.

### 3. DIVERGE line in the critic

The critic template grows from 3 lines to 4:

```
KEEP: <one sentence>
CHANGE: <one sentence>
DROP: <one sentence>
DIVERGE: <name the axis where this candidate diverges from the
          frontier consensus, or say "matches frontier on all axes">
```

When DIVERGE is "matches frontier on all axes", the critic's CHANGE
line is required to propose a structural pivot (not a hyperparam
tweak). This makes the novelty/safety trade-off visible on every
iteration rather than only after the round ships.

### 4. (Smaller) Continuation-readiness principle

The designer system prompt gains one principle:

> Design for continuation. A future round may warm-start from your
> checkpoint and tune only the training recipe. Keep `build_model` /
> `init_weights` / `COMPILE` cleanly separable from `build_optimizer`
> / `build_scheduler` / `training_config` / `compute_loss` — no shared
> closures, no hyperparams baked into the architecture.

This costs nothing if the v5 candidate never becomes a continuation
parent. If it does, the resulting checkpoint is more likely to be
liftable by the continuation flow (which fails silently on
non-extractable architectures and downgrades to a fresh design).

## Inherited from v4 (unchanged)

- **Analyst + `training_recipe_norm`** — analyst still emits the
  axes-of-variation digest and the recipe-norm block; recipe-norm
  still rides through to the recipe-tuner pass without leaking into
  the researcher's brief.
- **In-round recipe-tuner second pass** — `_run_recipe_tuner_pass`
  is unchanged. The frozen-arch / recipe-only preamble still fires on
  fresh rounds when budget permits.
- **Fallback templates with strong defaults** — `RECIPE_DEFAULTS`
  (AdamW + weight decay + warmup-cosine + bf16 AMP + grad clip + log-
  cadence val schedule) is unchanged. `FALLBACK_VERSION` bumps to
  `"v4"` so DB analysis can separate v5 fallback rounds from v4 ones.
- **Two-phase researcher** (Phase A brainstorm with quotas → Phase B
  prunes to the brief), **primitive injection**, **single-slot
  `_operator_prompt`** for GEPA / random_mutate prompt evolution.
- **Continuation flow** in `core/continuation.py` — on validator-
  scheduled continuation rounds, the parent's `build_model` /
  `init_weights` / `COMPILE` are lifted verbatim, analyst + researcher
  are skipped, and the designer handles the recipe directly. v5
  leaves this path untouched.

## Budget split

Same as v4 — the inspector adds no LLM calls and runs in <1ms, so the
budget split is unchanged.

| Phase | Budget |
|---|---|
| Analyst | 10% / cap 120s |
| Researcher | 12% / cap 300s |
| Designer pass 1 (architecture) | rest, minus `RECIPE_TUNER_BUDGET_CAP` |
| Recipe-tuner pass 2 | up to 330s |
| Packaging / fallback reserve | 30s |

## What to watch to know if v5 beats v4

- **Recipe-warning ship rate.** How often does the orchestrator ship
  a candidate that had at least one `warn` or `critical` inspector
  finding? Lower is better. Below 20% suggests the LLM is acting on
  the warnings; above 50% suggests the warning copy isn't connecting
  and the prompt needs strengthening.
- **`divergence_axes` populated rate.** How often does the researcher
  emit a non-empty `divergence_axes` list? Should be >90% — empty is
  the fallback path and indicates the researcher couldn't find any
  axis to break on.
- **Frontier-tie rate.** v4's failure mode was validated candidates
  that matched the frontier on every axis and tied for Pareto. Track
  the rate of shipped submissions whose metric is within 1% of an
  existing frontier member; v5 should reduce it.
- **Continuation-eligibility on v5 parents.** When v5's pass-1 ships
  a candidate that becomes a continuation parent for a later round,
  does the parent's architecture extract cleanly via
  `extract_arch_source`? The continuation-readiness principle is
  designed to make this happen more often.

## Compatibility notes

- **Tool surface is unchanged.** No new tools. The inspector runs as
  a callback in `subagents/designer.py`, not as an LLM-callable tool.
  If you want to expose it as a tool later (`inspect_recipe(code) ->
  json`) it's one wiring change in `tools.py`'s `build_handlers` plus
  an entry in `ROLE_TOOLS["designer"]`.
- **Critic format change is a breaking expectation for downstream
  parsers.** Anything that grepped for exactly 3 lines in critic
  output will see 4. Nothing in the v4 codebase did this — the
  designer Subagent loop appends the critic message as opaque text —
  but worth a heads-up.
- **`_operator_prompt`** still applies to the designer's pass-1 and
  pass-2 (no change from v4).
- **Continuation rounds** skip the recipe-tuner pass and the
  divergence-axes requirement (the architecture is frozen — there's
  nothing to diverge on). The critic's DIVERGE line still fires on
  continuation rounds; the prompt explicitly allows it to defer
  ("deferred — fix validation first" / "architecture frozen, recipe
  only").
- **Fallback path** — when the designer fails, the orchestrator falls
  through to `generate_fallback`. The fallback shipped is the same
  template family as v4 (`RECIPE_DEFAULTS` unchanged); only the
  version tag bumps to `"v4"`.

## Validator experiment-engine support

The orchestrator handles the validator-owned special round types
(`docs/experiment_engine.md`): **ablate** (focused one-diff brief +
target-code preamble; recipe-tuner pass skipped so the diff stays
single; also wired into the pipeline-task flow), **recipe_only**
(rides the in-round recipe machinery — base arch frozen via
`_inround_recipe_context`), **transfer** (scale-the-source brief;
recipe pass kept). When the challenge advertises `screening`, the
proposal ships extra validated designs from `state.candidates` as
`candidates=[{name, code}, ...]` for short-budget validator triage.
The analyst is pointed at `/lab_reports` and `/experiments/noise` so
recommendations weigh paired-significant evidence over sub-noise
metric gaps. Agent-side logic in `core/round_context.py`.
