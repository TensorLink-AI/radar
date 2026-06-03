"""Persistent experiment history across rounds — stored in scratchpad state."""

import hashlib
import json
import os
import time


SIZE_BUCKETS = {
    "tiny":         (100_000,       500_000),
    "small":        (500_000,     2_000_000),
    "medium_small": (2_000_000,  10_000_000),
    "medium":       (10_000_000, 50_000_000),
    "large":        (50_000_000, 125_000_000),
}


def identify_bucket(flops_min: int, flops_max: int) -> str:
    """Identify which size bucket matches the given FLOPs range."""
    for name, (bmin, bmax) in SIZE_BUCKETS.items():
        if bmin == flops_min and bmax == flops_max:
            return name
    # Fuzzy match: find closest bucket by midpoint
    mid = (flops_min + flops_max) / 2
    best = "medium"
    best_dist = float("inf")
    for name, (bmin, bmax) in SIZE_BUCKETS.items():
        bmid = (bmin + bmax) / 2
        dist = abs(mid - bmid)
        if dist < best_dist:
            best_dist = dist
            best = name
    return best


def extract_flops_budget(challenge: dict) -> tuple[int, int]:
    """Extract FLOPs min/max from challenge, supporting both field formats."""
    fb = challenge.get("flops_budget", {})
    if isinstance(fb, dict) and (fb.get("min") or fb.get("max")):
        return int(fb.get("min", 0)), int(fb.get("max", 0))
    fmin = challenge.get("min_flops_equivalent", 0)
    fmax = challenge.get("max_flops_equivalent", 0)
    if fmin or fmax:
        return int(fmin), int(fmax)
    return 0, 0


def get_history(state: dict) -> list[dict]:
    """Get experiment history from scratchpad state."""
    return state.get("history", [])


def get_bucket_history(state: dict, bucket: str) -> list[dict]:
    """Get history entries for a specific size bucket."""
    return [e for e in get_history(state) if e.get("bucket") == bucket]


def format_history(entries: list[dict], max_entries: int = 10,
                   score_direction: str = "minimize") -> str:
    """Format history entries for inclusion in prompts.

    Scored entries come first (best per ``score_direction``), unscored
    entries follow (newest first), marked ``(pending)``. Output is capped
    at ``max_entries`` lines.
    """
    if not entries:
        return "No previous submissions."

    def _has_score(e: dict) -> bool:
        return isinstance(e.get("score"), (int, float))

    scored = [e for e in entries if _has_score(e)]
    pending = [e for e in entries if not _has_score(e)]
    reverse = score_direction == "maximize"
    scored.sort(key=lambda e: e["score"], reverse=reverse)
    pending.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    ordered = scored + pending

    lines: list[str] = []
    for e in ordered[:max_entries]:
        bits = [
            f"bucket={e.get('bucket', '?')}",
            f"strategy={e.get('strategy', '?')}",
        ]
        if _has_score(e):
            score_bit = f"score={e['score']:.4g}"
            rank = e.get("rank")
            total = e.get("rank_total")
            if isinstance(rank, int):
                score_bit += (
                    f", rank={rank}/{total}" if isinstance(total, int)
                    else f", rank={rank}"
                )
            bits.append(score_bit)
            err = e.get("error")
            if err:
                bits.append(f"error={str(err)[:40]!r}")
        else:
            bits.append("(pending)")
        lines.append(
            f"- {e.get('name', '?')} ({', '.join(bits)}): "
            f"{e.get('motivation', '')}"
        )
    return "\n".join(lines)


def merge_results_into_state(state: dict,
                             previous_results: list[dict] | None) -> dict:
    """Merge validator-supplied round results onto matching entries.

    Expected shape (harness-provided, optional)::

        challenge["previous_results"] = [
            {"round_id": str, "code_hash": int, "score": float | None,
             "rank": int | None, "error": str | None},
            ...
        ]

    Join key is ``code_hash``. Any entries in ``state["history"]`` AND
    ``state["submissions"]`` whose ``code_hash`` matches get ``score``,
    ``rank``, ``rank_total``, ``error``, and ``scored_round_id`` written
    onto them. Duplicate hashes all receive the update. Unknown hashes
    are ignored. Returns the mutated state for chaining.
    """
    if not previous_results:
        return state
    history = state.setdefault("history", [])
    submissions = state.setdefault("submissions", [])
    index: dict = {}
    for entry in history:
        h = entry.get("code_hash")
        if h is not None:
            index.setdefault(h, []).append(entry)
    for entry in submissions:
        h = entry.get("code_hash")
        if h is not None:
            index.setdefault(h, []).append(entry)
    for r in previous_results:
        if not isinstance(r, dict):
            continue
        h = r.get("code_hash")
        if h is None:
            continue
        targets = index.get(h) or []
        if not targets:
            continue
        for entry in targets:
            score = r.get("score")
            if isinstance(score, (int, float)):
                entry["score"] = float(score)
            rank = r.get("rank")
            if isinstance(rank, int):
                entry["rank"] = rank
            total = r.get("rank_total")
            if isinstance(total, int):
                entry["rank_total"] = total
            err = r.get("error")
            if err:
                entry["error"] = str(err)
            rid = r.get("round_id")
            if rid is not None:
                entry["scored_round_id"] = rid
    return state


def best_own_submission(state: dict,
                        score_direction: str = "minimize") -> dict | None:
    """Return the best-scored history entry, or ``None`` if none are scored.

    ``score_direction`` is ``"minimize"`` (lower is better) or ``"maximize"``.
    Ties break by most-recent timestamp.
    """
    scored = [
        e for e in state.get("history", [])
        if isinstance(e.get("score"), (int, float))
    ]
    if not scored:
        return None
    if score_direction == "maximize":
        scored.sort(key=lambda e: (-e["score"], -e.get("timestamp", 0)))
    else:
        scored.sort(key=lambda e: (e["score"], -e.get("timestamp", 0)))
    return scored[0]


def add_entry(state: dict, *, name: str, code: str, motivation: str,
              bucket: str = "", flops: int = 0,
              strategy: str = "", metadata: dict | None = None) -> dict:
    """Add a history entry to the scratchpad state. Returns updated state."""
    if "history" not in state:
        state["history"] = []
    entry = {
        "name": name,
        "motivation": motivation,
        "code_hash": hash(code) & 0xFFFFFFFF,
        "code_length": len(code),
        "bucket": bucket,
        "flops_target": flops,
        "strategy": strategy,
        "timestamp": time.time(),
    }
    if metadata:
        entry["metadata"] = metadata
    state["history"].append(entry)

    # Keep last 50 entries to avoid scratchpad bloat
    if len(state["history"]) > 50:
        state["history"] = state["history"][-50:]

    return state


# ── Candidate lineage ────────────────────────────────────────────────
#
# A candidate is a piece of code the agent has worked on. The
# ``candidates`` state key is a dict keyed by a stable ``cand_<8 hex>``
# id derived from the code itself, so identical code always yields the
# same id (natural deduplication across sketch/validate/submit).
#
# Each record holds: code, flops, trace (optional), validated (bool),
# submitted (bool), created_at.

CANDIDATE_ID_PREFIX = "cand_"
CANDIDATE_ID_HEX_LEN = 8


def make_candidate_id(code: str) -> str:
    """Stable id for a chunk of code: ``cand_`` + first 8 sha1 hex chars."""
    digest = hashlib.sha1(code.encode("utf-8")).hexdigest()
    return CANDIDATE_ID_PREFIX + digest[:CANDIDATE_ID_HEX_LEN]


def get_candidates(state: dict) -> dict:
    """Return the candidate dict (read-only view; defaults to empty)."""
    return state.get("candidates", {})


def _candidates(state: dict) -> dict:
    """Return the mutable candidate dict, creating it on demand."""
    return state.setdefault("candidates", {})


def find_candidate(state: dict, candidate_id: str) -> dict | None:
    """Look up a candidate by id. Returns None if not found."""
    return get_candidates(state).get(candidate_id)


def upsert_candidate(state: dict, *, code: str,
                     flops: int | None = None,
                     trace: str | list | None = None) -> str:
    """Insert (or update) a candidate keyed by ``make_candidate_id(code)``.

    Returns the candidate id. If a record already exists, ``flops`` and
    ``trace`` are filled in only when newly available — the existing
    ``validated``/``submitted`` flags and ``created_at`` are preserved.
    """
    cid = make_candidate_id(code)
    cands = _candidates(state)
    record = cands.get(cid)
    if record is None:
        cands[cid] = {
            "code": code,
            "flops": flops,
            "trace": trace,
            "validated": False,
            "submitted": False,
            "created_at": time.time(),
        }
    else:
        if flops is not None and record.get("flops") is None:
            record["flops"] = flops
        if trace is not None and record.get("trace") is None:
            record["trace"] = trace
    return cid


def mark_candidate_validated(state: dict, candidate_id: str) -> dict | None:
    """Set ``validated=True`` on an existing candidate. Returns the record
    or None if not found.
    """
    record = find_candidate(state, candidate_id)
    if record is None:
        return None
    record["validated"] = True
    return record


def mark_candidate_submitted(state: dict, candidate_id: str) -> dict | None:
    """Set ``submitted=True`` on an existing candidate. Returns the record
    or None if not found.
    """
    record = find_candidate(state, candidate_id)
    if record is None:
        return None
    record["submitted"] = True
    return record


# ── Submissions ──────────────────────────────────────────────────────
#
# A list, not a dict — chronological order is the dominant access
# pattern (read_my_submissions wants the most recent N), the cap-and-
# drop-oldest semantics work naturally on a list, and we don't need
# O(1) lookups by id (the join key with previous_results is
# ``code_hash`` and that's a small linear scan in practice).
#
# Each entry retains the full ``code`` blob so the agent can re-read
# what it actually shipped in a later round; today's history-only
# storage drops the source after submit and only the hash survives.

MAX_SUBMISSIONS = 50


def get_submissions(state: dict) -> list[dict]:
    """Return the submissions list (read-only view; defaults to empty)."""
    return state.get("submissions", [])


def _submissions(state: dict) -> list[dict]:
    """Return the mutable submissions list, creating it on demand."""
    return state.setdefault("submissions", [])


def add_submission(state: dict, *, code: str, name: str, motivation: str,
                   candidate_id: str | None = None,
                   round_id: str = "") -> dict:
    """Append a submission record to ``state['submissions']`` and return
    it. Caps at ``MAX_SUBMISSIONS`` by dropping the oldest entry.

    The ``code_hash`` field is computed on insert so
    ``merge_results_into_state`` can later attach score/rank from the
    validator's ``previous_results`` payload.
    """
    record = {
        "code": code,
        "code_hash": hash(code) & 0xFFFFFFFF,
        "name": name,
        "motivation": motivation,
        "candidate_id": candidate_id,
        "round_id": round_id,
        "score": None,
        "rank": None,
        "rank_total": None,
        "submitted_at": time.time(),
    }
    subs = _submissions(state)
    subs.append(record)
    if len(subs) > MAX_SUBMISSIONS:
        del subs[: len(subs) - MAX_SUBMISSIONS]
    return record


NOTES_MAX_ENTRIES = 20
NOTES_SECTIONS = ("open_hypotheses", "dead_ends", "task_observations")


def _upgrade_hypotheses(notes: dict) -> None:
    """Wrap any bare-string entries in ``open_hypotheses`` as the
    structured dict shape introduced for hypothesis→candidate linkage.

    Idempotent — already-dict entries pass through untouched. Mutates
    the list in place.
    """
    bucket = notes.get("open_hypotheses")
    if not isinstance(bucket, list):
        return
    for i, entry in enumerate(bucket):
        if isinstance(entry, str):
            bucket[i] = {
                "text": entry,
                "candidate_ids": [],
                "created_at": None,
            }


def _notes(state: dict) -> dict:
    """Return the ``notes`` sub-dict, creating it (and sections) on demand."""
    notes = state.setdefault("notes", {})
    for section in NOTES_SECTIONS:
        notes.setdefault(section, [])
    _upgrade_hypotheses(notes)
    return notes


def add_hypothesis(state: dict, *, text: str,
                   candidate_id: str | None = None) -> dict | None:
    """Add (or update) a hypothesis entry in ``open_hypotheses``.

    Returns the affected record, or None if ``text`` is empty/whitespace.
    If an existing entry has the same ``text`` (after stripping), the
    given ``candidate_id`` is appended to its ``candidate_ids`` list
    instead of creating a duplicate (and duplicate ids within one entry
    are deduped). New entries cap at ``NOTES_MAX_ENTRIES`` like the
    other sections.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    notes = _notes(state)
    bucket = notes["open_hypotheses"]
    # Find existing record by exact stripped text.
    existing = None
    for entry in bucket:
        if isinstance(entry, dict) and entry.get("text") == stripped:
            existing = entry
            break
    if existing is not None:
        if candidate_id and candidate_id not in existing.setdefault(
            "candidate_ids", [],
        ):
            existing["candidate_ids"].append(candidate_id)
        return existing
    record = {
        "text": stripped,
        "candidate_ids": [candidate_id] if candidate_id else [],
        "created_at": time.time(),
    }
    bucket.append(record)
    if len(bucket) > NOTES_MAX_ENTRIES:
        del bucket[: len(bucket) - NOTES_MAX_ENTRIES]
    return record


HYPOTHESIS_VERDICTS = ("supported", "refuted", "inconclusive")


def link_hypothesis(state: dict, *, text: str,
                    candidate_id: str | None = None,
                    verdict: str | None = None) -> dict | None:
    """Late-bind a candidate id and/or a verdict to an existing
    hypothesis, looked up by exact stripped ``text``.

    Returns the updated record, or None if no hypothesis matches.
    Raises ``ValueError`` for an invalid verdict (callers should
    pre-validate with HYPOTHESIS_VERDICTS to surface a clean error
    string instead of an exception).

    When ``verdict`` is set on a hypothesis with no ``outcome`` yet, an
    outcome dict is created with ``score=None``, ``rank=None``, and
    only ``verdict`` populated — score/rank land later via
    ``merge_results_into_state``-style propagation.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if verdict is not None and verdict not in HYPOTHESIS_VERDICTS:
        raise ValueError(
            f"unknown verdict {verdict!r}; expected one of "
            f"{HYPOTHESIS_VERDICTS}"
        )
    notes = _notes(state)
    bucket = notes["open_hypotheses"]
    record = None
    for entry in bucket:
        if isinstance(entry, dict) and entry.get("text") == stripped:
            record = entry
            break
    if record is None:
        return None
    if candidate_id:
        cids = record.setdefault("candidate_ids", [])
        if candidate_id not in cids:
            cids.append(candidate_id)
    if verdict is not None:
        outcome = record.get("outcome")
        if outcome is None:
            record["outcome"] = {
                "score": None,
                "rank": None,
                "verdict": verdict,
            }
        else:
            outcome["verdict"] = verdict
    return record


def add_note(state: dict, section: str, entry: str) -> dict:
    """Append ``entry`` to ``state['notes'][section]``, capping at
    ``NOTES_MAX_ENTRIES`` by dropping the oldest.

    For ``open_hypotheses`` the entry is stored as a structured dict via
    ``add_hypothesis``; the other two sections retain the legacy
    list-of-strings shape.
    """
    if section not in NOTES_SECTIONS:
        raise ValueError(
            f"unknown notes section {section!r}; "
            f"expected one of {NOTES_SECTIONS}"
        )
    if not isinstance(entry, str) or not entry.strip():
        # Even when the entry is rejected we still want the canonical
        # ``notes`` skeleton in place so callers can index it safely.
        _notes(state)
        return state
    if section == "open_hypotheses":
        add_hypothesis(state, text=entry)
        return state
    notes = _notes(state)
    bucket = notes[section]
    bucket.append(entry.strip())
    if len(bucket) > NOTES_MAX_ENTRIES:
        del bucket[: len(bucket) - NOTES_MAX_ENTRIES]
    return state


def format_notes(state: dict) -> str:
    """Render the three notes sections as a plain-text summary.

    For ``open_hypotheses`` each linked candidate id is rendered as a
    nested line with its current state (validated/submitted) and score
    (when previous_results has merged a score onto a matching submission).
    """
    notes = state.get("notes") or {}
    candidates = state.get("candidates") or {}
    submissions_by_cid: dict = {}
    for s in state.get("submissions") or []:
        cid = s.get("candidate_id")
        if cid:
            submissions_by_cid[cid] = s

    titles = {
        "open_hypotheses": "Open Hypotheses",
        "dead_ends": "Dead Ends",
        "task_observations": "Task Observations",
    }
    parts: list[str] = []
    for section in NOTES_SECTIONS:
        items = notes.get(section) or []
        if not items:
            continue
        if section == "open_hypotheses":
            rendered = _render_hypotheses(
                items, candidates, submissions_by_cid,
            )
        else:
            rendered = "\n".join(f"- {item}" for item in items)
        parts.append(f"## {titles[section]}\n{rendered}")
    return "\n\n".join(parts)


def _render_hypotheses(items: list, candidates: dict,
                       submissions_by_cid: dict) -> str:
    """Render the hypothesis list with one nested status line per
    linked candidate id.
    """
    lines: list[str] = []
    for item in items:
        # Defensive — auto-upgrade should have run, but a bare string
        # here is still renderable.
        if isinstance(item, str):
            lines.append(f"- {item}")
            continue
        text = item.get("text", "")
        cids = item.get("candidate_ids") or []
        outcome = item.get("outcome") or {}
        verdict = outcome.get("verdict") if isinstance(outcome, dict) else None
        header = f"- {text}"
        if verdict:
            header += f" [verdict: {verdict}]"
        lines.append(header)
        for cid in cids:
            cand = candidates.get(cid) or {}
            states: list[str] = []
            if cand.get("validated"):
                states.append("validated")
            if cand.get("submitted"):
                states.append("submitted")
            sub = submissions_by_cid.get(cid)
            if sub and isinstance(sub.get("score"), (int, float)):
                rank = sub.get("rank")
                rank_total = sub.get("rank_total")
                if isinstance(rank, int):
                    rank_text = (
                        f" (rank {rank}/{rank_total})"
                        if isinstance(rank_total, int)
                        else f" (rank {rank})"
                    )
                else:
                    rank_text = ""
                states.append(f"scored {sub['score']:.4g}{rank_text}")
            if not cand and not sub:
                states.append("(no candidate state yet)")
            elif not states:
                states.append("(pending)")
            lines.append(f"    {cid}: {', '.join(states)}")
    return "\n".join(lines)


def best_score_in_submissions(
    state: dict, score_direction: str = "minimize",
) -> float | None:
    """Return the best submission score, or None if nothing is scored.

    ``score_direction`` is ``"minimize"`` (lower is better) or
    ``"maximize"`` (higher is better).
    """
    scored = [
        s["score"] for s in get_submissions(state)
        if isinstance(s.get("score"), (int, float))
    ]
    if not scored:
        return None
    return max(scored) if score_direction == "maximize" else min(scored)


def format_scratchpad_summary(
    state: dict, score_direction: str = "minimize",
) -> str:
    """One-line situational summary for the top of read_scratchpad."""
    notes = state.get("notes") or {}
    n_hyp = len(notes.get("open_hypotheses") or [])
    candidates = state.get("candidates") or {}
    n_cand = len(candidates)
    submissions = state.get("submissions") or []
    n_sub = len(submissions)
    top = best_score_in_submissions(state, score_direction)
    top_text = f"top score: {top:.4g}" if top is not None else "no scores yet"
    return (
        f"{n_hyp} hypotheses, {n_cand} candidates generated, "
        f"{n_sub} submitted, {top_text}"
    )


# ── Macros ───────────────────────────────────────────────────────────
#
# A macro is a named sequence of tool calls the agent can replay as
# one action. Stored per-miner in scratchpad state; not shared across
# miners. Each step is ``{tool, args, output_to}``; ``args`` may use
# ``${args.foo}`` and ``${step_var}`` placeholders that ``run_macro``
# substitutes at execution time.
#
# Tool-existence and "no submit / nested macros" validation lives in
# the handler layer (it needs to know what tools exist); these helpers
# are structural only.

MAX_MACROS = 20
MAX_MACRO_STEPS = 10


def get_macros(state: dict) -> dict:
    """Return the macros dict (read-only view; defaults to empty)."""
    return state.get("macros", {})


def _macros(state: dict) -> dict:
    """Return the mutable macros dict, creating it on demand."""
    return state.setdefault("macros", {})


def find_macro(state: dict, name: str) -> dict | None:
    """Look up a macro by name. Returns None if not found."""
    return get_macros(state).get(name)


def add_macro(state: dict, *, name: str, sequence: list,
              description: str = "") -> dict:
    """Insert (or overwrite) a macro record. Caller is responsible for
    sequence validation — this stores whatever it gets and bumps the
    cap to ``MAX_MACROS`` by dropping the oldest macro by ``created_at``
    when the dict overflows.
    """
    macros = _macros(state)
    record = {
        "name": name,
        "sequence": sequence,
        "description": description,
        "created_at": time.time(),
    }
    macros[name] = record
    if len(macros) > MAX_MACROS:
        oldest = sorted(
            macros.items(),
            key=lambda kv: kv[1].get("created_at") or 0.0,
        )
        for n, _ in oldest[: len(macros) - MAX_MACROS]:
            macros.pop(n, None)
    return record


def load_state(scratch_dir: str) -> dict:
    """Load state dict from scratch_dir/state.json."""
    path = os.path.join(scratch_dir, "state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(scratch_dir: str, state: dict) -> None:
    """Save state dict to scratch_dir/state.json."""
    os.makedirs(scratch_dir, exist_ok=True)
    path = os.path.join(scratch_dir, "state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
