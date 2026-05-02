"""JSON endpoints the dashboard UI fetches from the browser.

These routes are PUBLIC. The SPA on radarnet.io/dashboard/ calls them with
``credentials: 'omit'`` and no login — every field returned here is
world-readable. Jinja pages rendered server-side also fetch these endpoints
while holding a cookie; dropping the auth dependency lets both audiences
hit the same router without divergent behavior.

Do not add fields to any response here that shouldn't be visible to the
entire internet.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from config import Config
from database.dashboard import logs as log_helpers
from database.dashboard.app import get_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard/api")

# Max points sent to the browser for a loss curve — Chart.js starts
# struggling past a few thousand points.
_MAX_LOSS_POINTS = 500

# In-process TTL cache for /heartbeat.json. Polled every 2–5s by the SPA;
# capping at 1.5s means each browser tab triggers at most one DB read per
# poll regardless of how many tabs are open across the world.
_HEARTBEAT_TTL_S = 1.5
_heartbeat_cache: Optional[tuple[float, dict]] = None


def _downsample(points: list[float], limit: int = _MAX_LOSS_POINTS) -> list[float]:
    if not points:
        return []
    if len(points) <= limit:
        return [float(p) for p in points if p is not None]
    # Stride-based downsample, always keep first and last
    step = max(1, len(points) // limit)
    sampled = list(points[::step])
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return [float(p) for p in sampled if p is not None]


def _downsample_pair(
    train: list, val: list, limit: int = _MAX_LOSS_POINTS,
) -> tuple[list, list]:
    """Downsample two index-aligned arrays with a shared stride.

    Preserves ``None`` placeholders so train and val stay aligned on a shared
    x-axis after downsampling — callers that need filtered arrays should use
    ``_downsample`` instead.
    """
    n = max(len(train), len(val))
    if n == 0:
        return [], []

    def _cast(v):
        return None if v is None else float(v)

    train = list(train) + [None] * (n - len(train))
    val = list(val) + [None] * (n - len(val))

    if n <= limit:
        return [_cast(v) for v in train], [_cast(v) for v in val]

    step = max(1, n // limit)
    train_s = [_cast(v) for v in train[::step]]
    val_s = [_cast(v) for v in val[::step]]
    # Always keep the last index so the chart endpoints stay anchored.
    if _cast(train[-1]) != (train_s[-1] if train_s else None) or \
       _cast(val[-1]) != (val_s[-1] if val_s else None):
        train_s.append(_cast(train[-1]))
        val_s.append(_cast(val[-1]))
    return train_s, val_s


def _step_loss_pairs(hist) -> list[tuple[int, float]]:
    """Coerce a ``[{step, loss}, …]`` history to ``[(step, loss), …]`` sorted by step.

    Drops malformed entries so a single bad row can't break the chart.
    """
    if not isinstance(hist, list):
        return []
    pairs: list[tuple[int, float]] = []
    for entry in hist:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        loss = entry.get("loss")
        if isinstance(step, bool) or not isinstance(step, int):
            continue
        if not isinstance(loss, (int, float)) or isinstance(loss, bool):
            continue
        pairs.append((step, float(loss)))
    pairs.sort(key=lambda p: p[0])
    return pairs


def _aligned_train_val(meta: dict) -> tuple[list, list]:
    """Align ``train_loss_history`` and ``val_loss_history`` on their step union.

    Returns ``(train, val)`` where each is a list of floats with ``None`` at
    the slots where that series didn't sample at the corresponding step.
    Returns ``([], [])`` when neither history is present, so callers can fall
    back to the legacy bare-array ``loss_curve``.
    """
    if not isinstance(meta, dict):
        return [], []
    train_pts = _step_loss_pairs(meta.get("train_loss_history"))
    val_pts = _step_loss_pairs(meta.get("val_loss_history"))
    if not train_pts and not val_pts:
        return [], []
    train_map = dict(train_pts)
    val_map = dict(val_pts)
    steps = sorted(set(train_map) | set(val_map))
    return (
        [train_map.get(s) for s in steps],
        [val_map.get(s) for s in steps],
    )


def _extract_loss_series(meta: dict) -> list[float]:
    """Pull a ``list[float]`` loss series out of a training_meta blob.

    Validators write ``train_loss_history`` and ``val_loss_history`` as
    ``[{step, loss}, …]``. Legacy blobs just carry a bare ``loss_curve``.
    """
    if not isinstance(meta, dict):
        return []
    for key in ("train_loss_history", "val_loss_history"):
        hist = meta.get(key)
        pairs = _step_loss_pairs(hist)
        if pairs:
            return [loss for _, loss in pairs]
    legacy = meta.get("loss_curve")
    if isinstance(legacy, list):
        return [float(v) for v in legacy if isinstance(v, (int, float))]
    return []


def _meta_for_public(meta: dict) -> dict:
    """Reshape a TrainingMeta blob into the form the public SPA renders.

    The dataclass writes ``train_loss_history`` / ``val_loss_history`` as
    ``[{step, loss}, …]`` (preserved for the cookie-gated Jinja UI), but the
    radarnet.io chart wants index-aligned bare-number arrays plus scalar
    ``train_loss_final`` / ``val_loss_final`` cells. Translate at the API
    boundary so old R2 blobs stay valid without a backfill.

    Train / val are aligned on the union of their step indices — slots where
    val didn't run carry ``None`` so the renderer skips them but the train
    line stays continuous.
    """
    if not isinstance(meta, dict):
        return meta

    out = dict(meta)
    train, val = _aligned_train_val(meta)

    if train or val:
        out["train_loss_history"] = train
        out["val_loss_history"] = val
        train_finals = [v for v in train if v is not None]
        val_finals = [v for v in val if v is not None]
        if train_finals:
            out["train_loss_final"] = train_finals[-1]
        if val_finals:
            out["val_loss_final"] = val_finals[-1]

    return out


@router.get("/loss_curve/{index}.json")
async def loss_curve(index: int) -> dict:
    state = get_state()
    row = await state.pool.fetchrow(
        """
        SELECT e.loss_curve, e.round_id, e.miner_hotkey, tm.meta
        FROM experiments e
        LEFT JOIN training_metas tm
               ON tm.round_id = e.round_id
              AND tm.hotkey   = e.miner_hotkey
        WHERE e.id = $1
        """,
        index,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="experiment not found")

    from shared.pg_schema import _decode_jsonb

    # Prefer training_metas (and the R2 fallback) over the bare
    # experiments.loss_curve column because only the meta carries val. The
    # column is a train-only series surviving from pre-training_metas rounds.
    train, val = [], []
    meta = _decode_jsonb(row["meta"], {}) or {}
    train, val = _aligned_train_val(meta)
    if not train and not val and row["round_id"] is not None and row["miner_hotkey"]:
        r2_meta = log_helpers.fetch_meta(
            state.r2, int(row["round_id"]), row["miner_hotkey"],
        )
        if r2_meta:
            train, val = _aligned_train_val(r2_meta)
    if not train and not val:
        curve = _decode_jsonb(row["loss_curve"], [])
        if isinstance(curve, list):
            train = [float(v) for v in curve if isinstance(v, (int, float))]

    has_val = any(v is not None for v in val)
    if has_val:
        points, val_points = _downsample_pair(train, val)
    else:
        # Train-only path: drop None placeholders so legacy SPA charts that
        # don't know how to skip nulls keep working.
        points = _downsample([v for v in train if v is not None])
        val_points = []

    # ``loss_curve`` matches the radarnet.io SPA contract; ``points`` is kept
    # for the internal Jinja dashboard.js which reads data.points.
    # ``val_points`` is the validation-loss series aligned to ``points`` on a
    # shared step axis — slots where val didn't run carry ``None``.
    return {
        "index": index,
        "loss_curve": points,
        "points": points,
        "val_points": val_points,
    }


def _bucket_label(min_flops: int, max_flops: int) -> str:
    """Human-friendly label for a (min, max) FLOPs bucket — e.g. "500K – 2M"."""

    def fmt(n: int) -> str:
        if n >= 1_000_000_000_000:
            v = n / 1_000_000_000_000
            return f"{v:g}T"
        if n >= 1_000_000_000:
            v = n / 1_000_000_000
            return f"{v:g}B"
        if n >= 1_000_000:
            v = n / 1_000_000
            return f"{v:g}M"
        if n >= 1_000:
            v = n / 1_000
            return f"{v:g}K"
        return str(int(n))

    return f"{fmt(min_flops)} – {fmt(max_flops)}"


def _resolve_buckets(task: str) -> list[dict]:
    """Resolve the bucket list for ``task`` into the API's serializable shape."""
    state = get_state()
    raw = state.get_task_buckets(task) or []
    out: list[dict] = []
    for i, entry in enumerate(raw):
        if not entry or len(entry) != 2:
            continue
        lo, hi = int(entry[0]), int(entry[1])
        if lo <= 0 or hi <= lo:
            continue
        out.append({
            "index": i,
            "min_flops": lo,
            "max_flops": hi,
            "label": _bucket_label(lo, hi),
        })
    return out


def _bucket_index_for_flops(flops: int, buckets: list[dict]) -> int:
    """Return the index of the smallest bucket containing ``flops`` (-1 if none)."""
    for b in buckets:
        if b["min_flops"] <= flops <= b["max_flops"]:
            return int(b["index"])
    return -1


@router.get("/buckets.json")
async def buckets_json(task: str = "") -> dict:
    """FLOPs-equivalent size buckets defined for ``task``.

    Tasks may override the global ``SIZE_BUCKETS`` in their YAML, so the
    dashboard SPA fetches this list per task to render a bucket selector
    that matches what the validator scorer actually enforces.
    """
    return {"task": task, "buckets": _resolve_buckets(task)}


@router.get("/pareto.json")
async def pareto_json(
    task: str = "",
    bucket: str = "",
    min_flops: int | None = None,
    max_flops: int | None = None,
) -> dict:
    """Scatter data tagged per FLOPs bucket.

    Phase C scores each experiment only against the frontier members feasible
    for its round's bucket (see ``shared.scoring.passes_size_gate``). This
    endpoint mirrors that: every point carries a ``bucket_index`` and an
    ``on_frontier`` flag computed against that bucket's frontier — never the
    global one — so the dashboard never claims a point dominates models in a
    different size class.

    Query params:
      task        – filter to one task name. Buckets vary by task.
      bucket      – integer index into the task's bucket list, or ``""``/
                    ``"all"`` for every bucket. ``"active"`` resolves to the
                    current challenge's bucket (when a challenge is live).
      min_flops / max_flops – optional manual override that bypasses the
                    bucket selection. Useful for ad-hoc inspection.
    """
    state = get_state()
    kw = {"task": task} if task else {}
    elements = await state.store.get_pareto_elements(**kw)

    def _flops(e):
        return int(e.objectives.get("flops_equivalent_size") or 0)

    buckets = _resolve_buckets(task)

    # Resolve bucket selection. ``"active"`` only works when both a challenge
    # is live and that challenge's bounds line up with one of the task's
    # buckets — otherwise it degrades to "all" rather than silently filtering
    # against the global default.
    selected_idx: Optional[int] = None
    if bucket and bucket.lower() not in ("all", ""):
        if bucket.lower() == "active":
            ch = state.get_challenge() or {}
            ch_lo = int(ch.get("min_flops_equivalent") or 0)
            ch_hi = int(ch.get("max_flops_equivalent") or 0)
            for b in buckets:
                if b["min_flops"] == ch_lo and b["max_flops"] == ch_hi:
                    selected_idx = int(b["index"])
                    break
        else:
            try:
                idx = int(bucket)
            except ValueError:
                idx = -1
            if any(b["index"] == idx for b in buckets):
                selected_idx = idx

    if selected_idx is not None:
        sel = next(b for b in buckets if b["index"] == selected_idx)
        elements = [e for e in elements if sel["min_flops"] <= _flops(e) <= sel["max_flops"]]

    # Manual override stacks on top — keeps the historical query shape working.
    if min_flops is not None:
        elements = [e for e in elements if _flops(e) >= min_flops]
    if max_flops is not None:
        elements = [e for e in elements if _flops(e) <= max_flops]

    # Bin elements per bucket so we can build one ParetoFront per bucket. Points
    # whose FLOPs fall outside every bucket land in bin ``-1`` and never join a
    # frontier — they shouldn't have passed scoring's size gate anyway.
    from shared.pareto import ParetoFront

    bins: dict[int, list] = {}
    for e in elements:
        f = _flops(e)
        if not f or e.metric is None:
            continue
        b_idx = _bucket_index_for_flops(f, buckets)
        bins.setdefault(b_idx, []).append(e)

    def _objective(elem):
        metric = elem.metric if elem.metric is not None else float("inf")
        flops = int(elem.objectives.get("flops_equivalent_size") or 0) or float("inf")
        return (metric, flops)

    frontier_ids: set = set()
    for b_idx, items in bins.items():
        if b_idx < 0 or not items:
            continue
        pf = ParetoFront(max_size=len(items) + 1, objective_fn=_objective)
        for e in items:
            pf.update(e)
        frontier_ids.update(c.element.index for c in pf.candidates)

    points = []
    for b_idx, items in bins.items():
        for e in items:
            points.append({
                "id": e.index,
                "name": e.name,
                "task": e.task,
                "flops": _flops(e),
                "metric": float(e.metric),
                "miner_hotkey": e.miner_hotkey,
                "bucket_index": b_idx,
                "on_frontier": e.index in frontier_ids,
            })
    return {
        "task": task,
        "buckets": buckets,
        "selected_bucket": selected_idx,
        "points": points,
    }


@router.get("/stats.json")
async def stats_json(task: str = "") -> dict:
    state = get_state()
    kw = {"task": task} if task else {}
    return await state.store.stats(**kw)


@router.get("/provenance/miner_rounds.json")
async def provenance_miner_rounds(rounds: int = 30) -> dict:
    """Miner × round heatmap — unique experiments queried per round."""
    from database.dashboard import queries as q

    state = get_state()
    rounds = max(1, min(int(rounds), 100))
    return await q.miner_round_activity(state.pool, rounds=rounds)


@router.get("/provenance/top_experiments.json")
async def provenance_top_experiments(top_k: int = 20) -> dict:
    """Miner × top-K experiments heatmap — raw query counts per pair."""
    from database.dashboard import queries as q

    state = get_state()
    top_k = max(1, min(int(top_k), 100))
    return await q.top_experiments_activity(state.pool, top_k=top_k)


def _require_provenance():
    prov = getattr(get_state().store, "provenance", None)
    if prov is None:
        raise HTTPException(
            status_code=503, detail="Provenance index not ready",
        )
    return prov


@router.get("/provenance/{experiment_id}/influences.json")
async def provenance_influences(experiment_id: int) -> list[dict]:
    """What this experiment was influenced by.

    Each entry is ``{source_id, evidence_type, detail}``. ``evidence_type``
    is one of ``accessed`` (the miner queried that experiment that round),
    ``frontier`` (it was on the round's frontier), or ``shared_component``
    (overlapping code component).
    """
    return await _require_provenance().get_influences(experiment_id)


@router.get("/provenance/{experiment_id}/impact.json")
async def provenance_impact(experiment_id: int) -> list[dict]:
    """Which later experiments were exposed to this one.

    Each entry is ``{target_id, evidence_type, detail}``.
    """
    return await _require_provenance().get_impact(experiment_id)


@router.get("/provenance/{experiment_id}/similar.json")
async def provenance_similar(experiment_id: int, top_k: int = 10) -> list[dict]:
    """Code-similarity ranking against recent experiments.

    Each entry is ``{target_id, jaccard, ...}``. Capped at 50.
    """
    top_k = max(1, min(int(top_k), 50))
    return await _require_provenance().get_similar(experiment_id, top_k=top_k)


@router.get("/provenance/{experiment_id}/graph.json")
async def provenance_graph(experiment_id: int) -> dict:
    """Bundled influences + impact + components for one experiment."""
    return await _require_provenance().get_experiment_graph(experiment_id)


# ── SPA summary endpoints ─────────────────────────────────────

@router.get("/tasks_stats.json")
async def tasks_stats_json() -> dict:
    """Per-task stats, keyed by task name."""
    state = get_state()
    return await state.store.stats_by_task()


@router.get("/recent.json")
async def recent_json(n: int = 20, task: str = "") -> list[dict]:
    """Most recent experiments (newest first)."""
    state = get_state()
    n = max(1, min(int(n), 200))
    kw = {"task": task} if task else {}
    elements = await state.store.get_recent(n=n, **kw)
    return [e.to_api_dict() for e in elements]


@router.get("/tasks.json")
async def tasks_json() -> list[str]:
    """Distinct task names present in the DB."""
    state = get_state()
    return await state.store.get_tasks()


@router.get("/miners.json")
async def miners_json(limit: int = 200) -> list[dict]:
    """Per-hotkey aggregates for the miners page."""
    from database.dashboard import queries as q

    state = get_state()
    limit = max(1, min(int(limit), 500))
    return await q.miner_stats(state.pool, limit=limit)


@router.get("/heartbeat.json")
async def heartbeat_json() -> dict:
    """Cheap liveness ping — returns the latest activity timestamps + round.

    Designed for 2–5s SPA polling so the UI can render "last event Ns ago"
    instead of just an "alive" dot. Result is cached in-process for ~1.5s
    so tight polling never floods Postgres.
    """
    global _heartbeat_cache
    now = time.time()
    if _heartbeat_cache is not None:
        cached_at, cached = _heartbeat_cache
        if now - cached_at < _HEARTBEAT_TTL_S:
            return {**cached, "now": now}

    state = get_state()
    last_round_id: Optional[int] = None
    last_submission_at: Optional[float] = None
    try:
        row = await state.pool.fetchrow(
            "SELECT MAX(round_id) AS last_round, MAX(timestamp) AS last_at "
            "FROM experiments",
        )
        if row is not None:
            last_round_id = (
                int(row["last_round"]) if row["last_round"] is not None else None
            )
            last_submission_at = (
                float(row["last_at"]) if row["last_at"] is not None else None
            )
    except Exception:
        logger.exception("heartbeat query failed")

    payload = {
        "last_round_id": last_round_id,
        "last_submission_at": last_submission_at,
    }
    _heartbeat_cache = (now, payload)
    return {**payload, "now": now}


@router.get("/challenge.json")
async def challenge_json() -> dict:
    """Active challenge + current frontier size (SPA reads both at once)."""
    state = get_state()
    challenge = state.get_challenge() or {}
    frontier = state.get_frontier() or []
    frontier_size = len(frontier) if isinstance(frontier, list) else 0
    return {**challenge, "frontier_size": frontier_size}


@router.get("/rounds.json")
async def rounds_json(limit: int = 30) -> list[int]:
    """Distinct round IDs seen in experiments (most recent first)."""
    from database.dashboard import queries as q

    state = get_state()
    limit = max(1, min(int(limit), 200))
    return await q.distinct_rounds(state.pool, limit=limit)


@router.get("/benchmark.json")
async def benchmark_json(task: str = "", limit: int = 100) -> dict:
    """Per-round eval-score summary for a task — newest round first.

    Mirrors the data the Jinja ``/dashboard/benchmark`` view renders so the
    public SPA can chart eval scores over time. Defaults to the first known
    task when ``task`` is omitted, matching the Jinja behaviour.
    """
    from database.dashboard import queries as q

    state = get_state()
    tasks = await state.store.get_tasks()
    if not task and tasks:
        task = tasks[0]
    limit = max(1, min(int(limit), 500))
    rows = await q.benchmark_by_round(state.pool, task=task, limit=limit) if task else []
    return {"task": task, "tasks": tasks, "rows": rows}


def _parse_bool(s: str) -> Optional[bool]:
    if s in ("", None):
        return None
    if s.lower() in ("1", "true", "yes", "success"):
        return True
    if s.lower() in ("0", "false", "no", "failed"):
        return False
    return None


def _parse_int(s: str) -> Optional[int]:
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


@router.get("/browse.json")
async def browse_json(
    request: Request,
    task: str = "",
    round_id: str = "",
    miner_hotkey: str = "",
    success: str = "",
    min_flops: str = "",
    max_flops: str = "",
    q_: str = "",
    page: int = 0,
    page_size: int = 0,
) -> dict:
    """Paginated browse — mirrors the filters on the Jinja /experiments page."""
    from database.dashboard import queries as q

    state = get_state()
    text = q_ or request.query_params.get("q", "")
    filters = q.BrowseFilters(
        task=task,
        round_id=_parse_int(round_id),
        miner_hotkey=miner_hotkey,
        success=_parse_bool(success),
        min_flops=_parse_int(min_flops),
        max_flops=_parse_int(max_flops),
        q=text,
    )
    if page_size <= 0:
        page_size = Config.DASHBOARD_PAGE_SIZE
    page_size = max(1, min(int(page_size), 200))
    result = await q.browse(state.pool, filters, page=max(0, int(page)), page_size=page_size)
    return {
        "items": [e.to_api_dict() for e in result["items"]],
        "total": int(result["total"]),
        "page": int(result["page"]),
        "page_size": int(result["page_size"]),
    }


@router.get("/experiments/{index}.json")
async def experiment_json(index: int) -> dict:
    """Full experiment detail by index."""
    state = get_state()
    elem = await state.store.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail=f"Experiment {index} not found")
    return elem.to_api_dict()


@router.get("/miners/{hotkey}/submissions.json")
async def miner_submissions_json(hotkey: str, limit: int = 100) -> dict:
    """Per-hotkey submission list + agent code history."""
    from database.dashboard import queries as q

    state = get_state()
    limit = max(1, min(int(limit), 500))
    submissions = await q.miner_submissions(state.pool, hotkey, limit=limit)
    agent_history = await q.miner_agent_history(state.pool, hotkey, limit=50)
    if not submissions and not agent_history:
        raise HTTPException(status_code=404, detail="No submissions for this hotkey")
    return {
        "hotkey": hotkey,
        "submissions": [e.to_api_dict() for e in submissions],
        "agent_history": agent_history,
    }


@router.get("/experiments/{index}/trace.txt")
async def experiment_trace_txt(index: int):
    """Phase-A agent stdout/stderr for an experiment, served as text/plain.

    Kept off ``to_api_dict`` so list endpoints stay small — traces can run to
    hundreds of KB. Logs are write-once, so the response is marked immutable
    and aggressively cacheable.
    """
    state = get_state()
    row = await state.pool.fetchrow(
        "SELECT trace FROM experiments WHERE id = $1",
        index,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    trace = row["trace"] or ""
    if not trace:
        raise HTTPException(status_code=404, detail="no trace recorded")
    return Response(
        content=trace,
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@router.get("/experiments/{index}/lineage.json")
async def experiment_lineage_json(index: int) -> dict:
    """Lineage chain with per-step diffs."""
    state = get_state()
    elem = await state.store.get(index)
    if elem is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    diffs = await state.store.get_lineage_diffs(index)
    return {
        "root": elem.to_api_dict(),
        "diffs": diffs,
    }


# ── Training logs (public — same payload shape as cookie-gated Jinja route) ──

@router.get("/logs/{round_id}/{hotkey}/meta.json")
async def logs_meta_json(round_id: int, hotkey: str):
    state = get_state()
    meta = await log_helpers.fetch_meta_cached_or_r2(
        state.pool, state.r2, round_id, hotkey,
    )
    if meta is None:
        raise HTTPException(status_code=404, detail="training_meta not found")
    return JSONResponse(_meta_for_public(meta))


@router.get("/logs/{round_id}/{hotkey}/stdout.json")
async def logs_stdout_json(round_id: int, hotkey: str, direct: int = 0):
    state = get_state()
    if direct:
        url = log_helpers.presigned_stdout_url(state.r2, round_id, hotkey)
        if not url:
            raise HTTPException(status_code=503, detail="R2 presign unavailable")
        return RedirectResponse(url=url, status_code=302)
    payload = log_helpers.fetch_stdout(state.r2, round_id, hotkey)
    if payload is None:
        raise HTTPException(status_code=404, detail="No stdout.log in R2")
    return JSONResponse(payload)


@router.get("/logs/{round_id}/{hotkey}/architecture.json")
async def logs_architecture_json(round_id: int, hotkey: str, direct: int = 0):
    state = get_state()
    if direct:
        url = log_helpers.presigned_architecture_url(state.r2, round_id, hotkey)
        if not url:
            raise HTTPException(status_code=503, detail="R2 presign unavailable")
        return RedirectResponse(url=url, status_code=302)
    payload = log_helpers.fetch_architecture(state.r2, round_id, hotkey)
    if payload is None:
        raise HTTPException(status_code=404, detail="No architecture.py in R2")
    return JSONResponse(payload)


# ── Agent code bundles (public JSON mirror of the cookie-gated Jinja view) ──

@router.get("/agent_code/{code_hash}.json")
async def agent_code_json(code_hash: str) -> dict:
    """Full agent bundle (files + entry point) plus related history.

    Matches the data the Jinja ``/dashboard/agent_code/{hash}`` view renders,
    but returns JSON so the public SPA can display bundles without cookies.

    Bundle bytes come from the ``agent_bundles`` cache in Postgres (populated
    on submission). R2 is only consulted as a fallback for rows submitted
    before the cache existed, so dashboard-mode deploys without R2
    credentials still serve the bundle.
    """
    from database.dashboard import queries as q

    state = get_state()
    record = await q.agent_bundle_record(state.pool, code_hash)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown code_hash")

    bundle = await q.agent_bundle_blob(state.pool, code_hash)
    if bundle is None and state.r2 is not None:
        try:
            bundle = state.r2.download_json(record["r2_key"])
        except Exception:
            bundle = None
    if not bundle:
        raise HTTPException(status_code=404, detail="Bundle not found")
    history = await q.miner_agent_history(state.pool, record["hotkey"], limit=50)
    return {
        "record": record,
        "entry_point": bundle.get("entry_point", record["entry_point"]),
        "files": bundle.get("files") or {},
        "history": [h for h in history if h["code_hash"] != code_hash],
    }
