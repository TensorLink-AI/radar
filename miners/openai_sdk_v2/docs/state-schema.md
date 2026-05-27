# openai_sdk Agent — Extended State Schema

Living schema doc for the four-feature substrate that shares the
scratchpad-persisted state object managed by `core/history.py`:

1. Code lineage / candidate IDs (shipped)
2. `read_my_submissions` tool (shipped)
3. Structured scratchpad readback (hypothesis ↔ candidate ↔ outcome) (shipped)
4. Tool macros (shipped)

§§1–4 describe the as-shipped schema and tool surface. §5 records the
decisions made during implementation, including places where the
original plan changed. §6 tracks delivery.

---

## 1. Current state schema

Everything below lives in `state.json` inside the scratchpad directory.
Loaded by `history.load_state` and saved by `history.save_state`. The
in-memory wrapper `state_holder` (in `tools.py`, inside `build_handlers`)
holds the dict plus two round-scoped flags; only the `state` sub-dict
is persisted.

### 1.1 Persisted keys

| Key            | Type             | Owner                       | Notes |
|----------------|------------------|-----------------------------|-------|
| `history`      | `list[dict]`     | `add_entry`, `merge_results_into_state` | Submission records. Capped at 50 (drops oldest). |
| `notes`        | `dict[str, list[str]]` | `add_note`, `format_notes` | Three sub-lists: `open_hypotheses`, `dead_ends`, `task_observations`. Each capped at `NOTES_MAX_ENTRIES = 20`. |
| `agent_notes`  | `str` (optional) | deprecated `write_scratchpad(notes=...)` | Free-form string; only present on legacy scratchpads. |

#### History entry shape (current)

Written by `add_entry` in `core/history.py:180`:

```python
{
    "name": str,
    "motivation": str,
    "code_hash": int,            # hash(code) & 0xFFFFFFFF — join key for previous_results
    "code_length": int,
    "bucket": str,               # tiny | small | medium_small | medium | large
    "flops_target": int,
    "strategy": str,
    "timestamp": float,
    "metadata": dict,            # optional, free-form
    # ── added by merge_results_into_state when previous_results arrives ──
    "score": float,              # optional
    "rank": int,                 # optional
    "rank_total": int,           # optional
    "error": str,                # optional
    "scored_round_id": str,      # optional
}
```

Note: the **code itself is not stored** — only `code_hash`. That's a
deliberate choice for scratchpad size. Today this means we cannot
reconstruct what was actually submitted from history alone.

### 1.2 Round-scoped (`state_holder`, not persisted)

| Key                    | Type   | Set by                         | Read by |
|------------------------|--------|--------------------------------|---------|
| `state`                | `dict` | `build_handlers`               | every state-touching handler |
| `wrote_this_round`     | `bool` | `_write_scratchpad`            | `_submit` (warns if false) |
| `last_validated_code`  | `str`  | `_validate_code` on success    | `agent.py` auto-submit recovery |

---

## 2. Extended schema (as shipped)

Three new top-level persisted keys: `candidates`, `submissions`,
`macros`. Hypotheses live inside the existing `notes.open_hypotheses`
section, upgraded from `list[str]` to `list[HypothesisRecord]` in place
(see §2.4). The existing `history` and `agent_notes` keys are unchanged.

### 2.1 New top-level keys

| Key            | Type                                                | Used by features |
|----------------|------------------------------------------------------|------------------|
| `candidates`   | `dict[str, CandidateRecord]` keyed by `cand_<8 hex>` | 1, 3 (link target) |
| `submissions`  | `list[SubmissionRecord]`                             | 2, 3 (score source) |
| `macros`       | `dict[str, MacroRecord]` keyed by name               | 4 |

> **Diverged from plan:** §2.1 originally proposed `hypotheses` as a
> separate top-level list. Shipped instead by upgrading
> `notes.open_hypotheses` in place — keeps the cap/section semantics
> the rest of the notes already use, and avoids a parallel rendering
> path. See §5.0(6) for the decision.

### 2.2 `CandidateRecord` — feature 1 (as shipped)

A candidate is a piece of code the agent has worked on. The
``candidates`` state key is a **dict keyed by id**, where the id is a
stable hash of the code itself (``cand_<8 hex>``). Identical code
yields the same id, so re-sketching or re-validating the same source
naturally deduplicates instead of producing parallel records.

```python
state["candidates"] = {
    "cand_a3f24c1d": {
        "code": str,                # the full source
        "flops": int | None,        # measured by sketch_architecture
        "trace": str | None,        # rendered per-layer trace, or None
        "validated": bool,          # set True by validate_code on success
        "submitted": bool,          # set True by submit when this id ships
        "created_at": float,        # time.time() at first insert
    },
    ...
}
```

**Writes:**
- `_sketch_architecture` calls `upsert_candidate` with the code, FLOPs,
  and trace. Returns the id and surfaces it as the last line of the
  tool output (`candidate_id: cand_a3f24c1d`).
- `_validate_code` either looks up an existing candidate by id (and
  ignores any `code` arg) or, if no id was passed and validation
  passes, calls `upsert_candidate` to create one. On success it sets
  `validated=True`.
- `_submit` looks up the candidate by id and ships its stored code,
  setting `submitted=True`. Without an id, submit behaves as it did
  before this feature (uses the `code` arg, no state mutation).

**Reads:** the LLM carries ids forward through tool-result text. Lineage
read tools (`list_candidates`, `get_candidate`) are deferred — the
in-conversation id is enough for feature 1's needs.

**Cap:** none for now. Hash-keyed dedup keeps growth proportional to
unique designs, not call count.

### 2.3 `SubmissionRecord` — feature 2 (as shipped)

A new top-level `submissions` list, separate from `history`. Each entry
retains the **full code blob** so the agent can re-read what it shipped
in a later round (history stayed hash-only). Capped at 50 entries
(`MAX_SUBMISSIONS`); join key with `previous_results` is `code_hash`.

```python
state["submissions"] = [
    {
        "code": str,                # full source the agent shipped
        "code_hash": int,           # join key for previous_results
        "name": str,
        "motivation": str,
        "candidate_id": str | None, # the cand_<8 hex> this came from
        "round_id": str,
        "score": float | None,      # filled by merge_results_into_state
        "rank": int | None,
        "rank_total": int | None,
        "submitted_at": float,
        # added when previous_results is merged:
        "scored_round_id": str,
    },
    ...
]
```

**Writes:**
- `_submit` calls `add_submission` before raising `SubmitSignal` so a
  later round's `previous_results` has a `code_hash` join target.
- `merge_results_into_state` (called at `agent.py` round entry)
  attaches `score`/`rank`/`rank_total`/`error`/`scored_round_id` to
  matching entries by `code_hash`.

**Reads:** `read_my_submissions(n=3)` renders newest-first with
truncation (`n>1` truncates code per entry to ~40 lines; `n=1` returns
full code).

> **Diverged from plan:** §2.3 originally said "no new top-level key —
> reuse `state['history']`". That didn't work because the openai_sdk
> agent never populates `history` (it's an autonomous-agent helper);
> code blobs would have nowhere to live. Decided in §5.0(5) to add a
> dedicated `submissions` list.

### 2.4 `HypothesisRecord` — feature 3 (as shipped)

`notes.open_hypotheses` is upgraded in place from `list[str]` to
`list[dict]`. Each entry can carry zero or more candidate ids (one
hypothesis can spawn multiple candidates) and an optional manual
verdict. Score is **not** stored on the hypothesis — it's resolved at
render time by joining `candidate_ids` against `state["submissions"]`.

```python
{
    "text": str,                     # the hypothesis itself; lookup key
    "candidate_ids": list[str],      # set by write_scratchpad / link_hypothesis
    "created_at": float | None,      # None for legacy upgraded entries
    # only present after link_hypothesis(verdict=…):
    "outcome": {
        "score": None,               # left None — score lives on submissions
        "rank": None,
        "verdict": "supported" | "refuted" | "inconclusive",
    },
}
```

**Lookup key is `text`** (no separate `id` field). `add_hypothesis`
deduplicates by exact stripped text; `link_hypothesis` looks the entry
up the same way.

**Auto-upgrade:** `_upgrade_hypotheses` walks any bare-string entries
on first access via `_notes(state)` and wraps them as
`{"text": s, "candidate_ids": [], "created_at": None}`. Idempotent.

**Writes:**
- `write_scratchpad(hypothesis=..., candidate_id=...)` appends a record
  (or appends the id to an existing record with the same text).
- `link_hypothesis(hypothesis=..., candidate_id?=, verdict?=)` —
  late-bind: append a candidate id and/or set `outcome.verdict`. Errors
  out (string return, not exception) for unknown text or invalid
  verdict.

**Reads:** `read_scratchpad` renders hypotheses with one nested status
line per linked candidate id, resolving live state from
`state["candidates"]` (validated/submitted flags) and
`state["submissions"]` (score/rank). Verdict, when present, is shown
on the hypothesis line itself:

```
- patches help [verdict: supported]
    cand_a3f24c1d: validated, submitted, scored 0.81 (rank 3/9)
```

**Cap:** 20 (the existing `NOTES_MAX_ENTRIES`).

> **Diverged from plan:** the original §2.4 proposed a top-level
> `hypotheses` list with `id` / `round_id` / `outcome.score` etc. and
> an auto-link pass. Shipped without ids (text is the key), without an
> auto-link pass (score is joined at render time, verdict is manual via
> `link_hypothesis`), and without a parallel rendering path (we
> upgraded the existing notes section in place). See §5.0(6) and
> §5.1(4).

### 2.5 `MacroRecord` — feature 4 (as shipped)

Reusable tool sequences. Stored in a dict keyed by name for O(1) lookup.

```python
{
    "name": str,
    "sequence": [
        {"tool": str, "args": dict, "output_to": str | None},
        ...
    ],
    "description": str,                # optional; rendered by list_macros
    "created_at": float,
}
```

**Substitution:** `${args.foo}` in step args resolves to the
corresponding `run_macro(args=...)` value; `${var}` resolves to a
prior step's output captured via `output_to`. Whole-string refs
preserve type (so `idx: "${args.idx}"` resolves to an int); embedded
refs stringify; missing refs leave the literal in place.

**Writes:** `define_macro(name, sequence, description="")` inserts or
overwrites. Validation happens at define time: tool names must exist,
no `submit` / `define_macro` / `run_macro` in a sequence.

**Reads:** `list_macros`, `run_macro`. `run_macro` halts on the first
step whose handler raises or whose result starts with
`error:`/`errors:` and returns the partial output.

**Caps:**
- `MAX_MACROS = 20`
- `MAX_MACRO_STEPS = 10` per sequence

> **Diverged from plan:** §2.5 originally listed `param_names`,
> `run_count`, `last_run_at`, `last_run_round_id`, and an arg-bytes
> cap. None of those shipped — we deferred to YAGNI; a usage counter
> can be added later if we need it. `param_names` was redundant with
> the implicit set of `${args.X}` references.

### 2.6 Schema-at-a-glance (as shipped)

```python
state = {
    # pre-feature
    "history":          list[HistoryEntry],            # cap 50
    "notes": {
        "open_hypotheses":  list[HypothesisRecord],    # cap 20 (upgraded in place)
        "dead_ends":        list[str],                 # cap 20
        "task_observations": list[str],                # cap 20
    },
    "agent_notes":      str,                           # legacy, optional

    # added by features 1-4
    "candidates":       dict[str, CandidateRecord],    # uncapped; sha1-keyed dedup
    "submissions":      list[SubmissionRecord],        # cap 50
    "macros":           dict[str, MacroRecord],        # cap 20
}
```

---

## 3. Migration

### 3.1 Strategy: lazy default-on-read, no version field

Existing scratchpads on disk lack the new keys. We do **not** introduce
a `schema_version` field. Instead, every accessor uses `.get(key,
default)` or `.setdefault(key, default)` — the same pattern history.py
already follows for `history` and `notes`.

`load_state` itself stays unchanged — it returns whatever JSON is on
disk. Defaults are applied at the helper level:

```python
def get_candidates(state: dict) -> dict:
    return state.get("candidates", {})

def _candidates(state: dict) -> dict:
    return state.setdefault("candidates", {})
```

Same pattern for `submissions` (default `[]`) and `macros` (default
`{}`).

### 3.2 Hypothesis auto-upgrade

`notes.open_hypotheses` is the one section that **does** auto-upgrade:
`_upgrade_hypotheses(notes)` walks the list on first access via
`_notes(state)` and wraps any bare string `s` as
`{"text": s, "candidate_ids": [], "created_at": None}`. Idempotent —
already-dict entries pass through. This was the only place auto-upgrade
made sense: the data is in the same section, the new shape is a strict
superset, and the LLM expects to keep referring to old hypotheses by
their text.

Other migrations are skipped:
- A `history` entry from before feature 1 has no `candidate_id`. We
  could fabricate one, but the candidate record wouldn't have any code
  or flops — it'd be a stub. Leave `candidate_id` absent.
- `submissions` starts empty; old rounds have no full code blob to
  back-fill, so they don't appear in `read_my_submissions`. Acceptable
  loss given the alternative (storing full code in `history` going
  back) is just as expensive.

### 3.3 Failure mode coverage

If `state.json` predates these changes:
- `read_scratchpad` — sees empty `candidates` / `submissions` /
  `macros`; the hypothesis upgrade fires lazily on first `_notes()`
  call.
- `read_my_submissions` — returns `"no submissions yet"`.
- `list_macros` — returns `"no macros defined yet"`.
- `link_hypothesis` against a legacy bare-string hypothesis works
  because the upgrade ran when `_notes` was accessed.
- First `define_macro` / sketched candidate / submit creates the
  corresponding key on demand; the file picks up the new shape on the
  next `save_state`.

### 3.4 Forward compatibility

If a user manually downgrades the agent to a pre-feature version after
new keys have been written, `save_state` will preserve them as long as
the older code calls `json.dump(state, ...)` over the whole dict, which
it does (`history.py:269`). No data loss on downgrade.

---

## 4. Tool surface (as shipped)

### 4.1 Feature 1 — candidate IDs

**Modified tools:**

| Tool | Change |
|---|---|
| `sketch_architecture(code)` | Side effect: `upsert_candidate` keyed by sha1 hash of code. Result string ends with `candidate_id: cand_<8 hex>`. |
| `validate_code(code?, candidate_id?)` | Adds optional `candidate_id`. When set, looks up the source from `state["candidates"]` and ignores the `code` arg. On success, sets `validated=True` (creating the candidate if no id was passed). |
| `submit(code?, name, motivation, candidate_id?)` | Adds optional `candidate_id`. When set, ships the stored code from state and flips `submitted=True`. Without an id, behaves as before. |

> **Diverged from plan:** `parent_candidate_id`, `list_candidates`,
> `get_candidate`, the `status` enum, and the auto-write of a history
> entry on submit were all dropped in the feature-1 rework (see
> §5.0(3) and §5.0(5)).

### 4.2 Feature 2 — `read_my_submissions`

**New tool:**

```
read_my_submissions(n: int = 3) -> str
```

Returns the n most recent entries from `state["submissions"]` (newest
first), each rendered with `round_id`, `name`, `score` (or `pending`),
`rank`, `candidate_id`, `motivation`, and code. Code is truncated to
~40 lines per entry when `n > 1`; full code when `n = 1`.

`_submit` populates `state["submissions"]` on ship;
`merge_results_into_state` attaches `score`/`rank` by `code_hash` when
the next round's `previous_results` arrives.

> **Diverged from plan:** original signature had `limit`, `scored_only`,
> `include_code` flags and pulled from `state["history"]`. Shipped
> simpler — single `n` arg, dedicated `submissions` list, code always
> available (truncated when n>1).

### 4.3 Feature 3 — structured scratchpad

**Modified tool — `write_scratchpad`:**

Existing kwargs (`hypothesis`, `dead_end`, `reason`, `observation`,
`notes`) all still work. New kwarg:

| Kwarg | Type | Effect |
|---|---|---|
| `candidate_id` | `str` | Only meaningful with `hypothesis`. Links the new (or existing-by-text) hypothesis record to this candidate. |

Setting a manual verdict via `write_scratchpad` was deferred — use
`link_hypothesis(verdict=...)` instead.

**New tool:**

```
link_hypothesis(hypothesis: str,
                candidate_id?: str,
                verdict?: "supported" | "refuted" | "inconclusive") -> str
```

Late-bind a candidate id and/or a verdict to an existing hypothesis.
Looks up by exact text. Returns an error string (not exception) for
unknown text or invalid verdict. Setting `verdict` on a hypothesis
with no `outcome` yet creates the outcome dict with `score=None`,
`rank=None`, and only `verdict` populated.

**Modified tool — `read_scratchpad`:**

Output gains a one-line summary at the top:

```
2 hypotheses, 1 candidates generated, 1 submitted, top score: 0.18
```

The `Open Hypotheses` section's renderer now resolves linked candidate
ids against `state["candidates"]` and `state["submissions"]`:

```
## Open Hypotheses
- patches help [verdict: supported]
    cand_a3f24c1d: validated, submitted, scored 0.81 (rank 3/9)
- bigger context → no candidate state yet
```

> **Diverged from plan:** §4.3 originally proposed
> `hypothesis_candidate_id` and `hypothesis_outcome` kwargs on
> `write_scratchpad`, and a `## Hypotheses` section with `hyp_NNNN`
> ids. Shipped flatter: a single `candidate_id` kwarg, lookup by text,
> verdict via `link_hypothesis`, and rendering inside the existing
> `Open Hypotheses` section.

### 4.4 Feature 4 — macros

**New tools:**

```
define_macro(name: str,
             sequence: list[{tool: str, args: dict, output_to?: str}],
             description?: str) -> str

run_macro(name: str, args?: dict) -> str

list_macros() -> str
```

`run_macro` substitutes `${args.foo}` and `${var}` placeholders inside
step args (whole-string refs preserve type; embedded refs stringify;
missing refs leave the literal in place). Steps run through the same
wrapped handler dict the LLM uses, so circuit breakers and call counts
apply uniformly. Halts on the first step whose handler raises or whose
result starts with `error:` / `errors:`, returns the partial output
with all preceding step labels.

`submit`, `define_macro`, and `run_macro` are forbidden inside macro
sequences — macros never ship and never recurse.

> **Diverged from plan:** §4.4 proposed `param_names` on
> `define_macro` and `{{var}}`-style substitution. Shipped without
> `param_names` (the implicit set of `${args.X}` refs is enough) and
> with `${var}` syntax. See §5.0(7).

---

## 5. Decisions and open questions

Items below are kept as the original review-time entries with one-line
**decided:** annotations. Where the eventual implementation differs
from the original recommendation, the annotation says so explicitly —
the doc is a living record of what we picked and why, not a
retroactively clean spec.

### 5.0 Resolved before feature 1 shipped

1. **Code storage in candidates.** Originally: store code only for
   `status in {"validated", "submitted"}`; sketched candidates carry
   `code = None`.
   **decided:** changed during the rework — every candidate stores its
   code. Hash-keyed dedup keeps growth proportional to unique designs,
   not call counts, so the savings of dropping sketch-only code
   weren't worth the asymmetry.

2. **Cap-when-referenced.** Originally: cap at `MAX_CANDIDATES = 100`
   and drop oldest first.
   **decided:** dropped the cap entirely. Hash-keyed dedup means the
   dict only grows with unique designs; bounding it would just make
   stale-id resolution probabilistic.

3. **Candidate ID format.** Originally: monotonic `cand_0042` via
   `next_candidate_seq`.
   **decided:** changed to sha1-derived `cand_<8 hex>`. Identical code
   then yields the same id, so sketch-then-validate of the same source
   is one record, not two. The counter wasn't pulling its weight.

4. **Implicit candidate creation in `validate_code`.** Auto-create a
   candidate ONLY when validation passes. Failed validates do not
   create a record.
   **decided:** shipped as-recommended.

5. **Submission also writes a history entry.** Originally: `_submit`
   adds a history entry so `merge_results_into_state` has a join
   target.
   **decided:** changed in feature 2 — score/rank flow through a new
   top-level `submissions` list instead. `history` stays unused by the
   openai_sdk agent (it's an autonomous-agent helper). `merge_results`
   was extended to update both lists.

6. **Hypothesis storage location.** Originally: a new top-level
   `hypotheses` list with `id` / `round_id` / `outcome.score` etc., in
   parallel with the legacy `notes.open_hypotheses` strings.
   **decided (during feature 3):** upgrade `notes.open_hypotheses` in
   place (list[str] → list[dict]) instead of running two parallel
   stores. Lookup key is `text` (no `id` field). Score is **not**
   stored on the hypothesis — joined at render time from
   `state["submissions"]`. Verdict is the only field on `outcome` and
   is set manually via `link_hypothesis`.

7. **Macro substitution syntax.** Originally: `{{var}}`.
   **decided (during feature 4):** `${args.foo}` and `${var}`. The
   `${args.X}` namespace makes the source of each ref obvious; the
   `${...}` form survives JSON quoting more cleanly than `{{...}}`.

### 5.1 Still open at planning time

1. **Cross-round candidate visibility.** Should `list_candidates`
   default to "this round only" or "all-time"?
   **decided:** moot — `list_candidates` / `get_candidate` were
   dropped from the surface. The id flows through tool-result text
   in-conversation; that turned out to be enough.

2. **Macro failure semantics.** Step fails → abort vs continue?
   **decided:** always-abort (no `continue_on_error` flag). Halt-on-
   first-error and return the partial output with step labels.

3. **Macro nesting.** Can a macro call `run_macro`?
   **decided:** forbidden at define time. `submit`, `define_macro`,
   and `run_macro` all reject in a sequence.

4. **`previous_results` → hypothesis outcome auto-link.** Originally:
   auto-fill `outcome.score`/`rank` after merge.
   **decided:** no separate auto-link pass. Score is resolved at
   `read_scratchpad` render time by walking
   `hypothesis.candidate_ids → state.submissions[*].score`. Verdict
   stays manual (`link_hypothesis(verdict=...)`).

5. **Notes vs hypotheses double-write.** Originally: write to both
   the legacy list[str] section and a new structured store during a
   migration window.
   **decided:** moot — we upgraded the existing section in place. No
   double-write needed.

6. **Token budget impact on `read_scratchpad`.** Adding new sections
   inflates the read result.
   **decided:** partially addressed. We added a one-line summary at
   the top, but per-section visible-item caps weren't introduced.
   Still open if read_scratchpad output grows past comfortable.

---

## 6. Delivery

| Feature | Status | Commit landed |
|---|---|---|
| 1 — candidate IDs / lineage | shipped | `72f2b75` |
| 2 — `read_my_submissions` | shipped | `846cea3` |
| 3 — hypothesis ↔ candidate ↔ outcome linkage + summary line | shipped | `8400810` |
| 3.5 — `link_hypothesis` (late-bind) | shipped | follow-up |
| 4 — tool macros | shipped | `ae2a963` |
