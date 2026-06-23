"""Local validator process.

One Python process that drives Phase A → B → C against a SQLite-backed
store. No HMAC, no HTTP, no chain — see ``local/README.md`` for how this
maps to the real radar architecture.

Loop (per round):
  1. Pick a size bucket deterministically from ``round_id``.
  2. Read the current Pareto frontier from past experiments.
  3. Post a Challenge to SQLite (Phase A starts).
  4. Wait for proposals up to ``--phase_a_seconds``.
  5. For each proposal, run training + held-out eval in-process.
  6. Score the round, write one experiment row per miner.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path

# Allow ``python local/validator.py`` from repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local.artifacts import ArtifactSink, cleanup_workdir, sweep_orphan_workdirs
from local.backup import from_env as backup_from_env
from local.checkpoints import CheckpointStore
from local.continuation import (
    continuation_frontier,
    continuation_rate,
    is_continuation_round,
    is_extend_round,
    prepare_continuation,
)
from local.eval_metrics import paired_per_task_delta
from local.experiments_api import _parent_summary
from local.frozen_arch import FrozenArchStore, maybe_refresh as maybe_refresh_frozen
from local.frozen_pipeline import (
    FrozenPipelineStore,
    maybe_refresh as maybe_refresh_pipeline,
)
from local.round_types import special_round_type
from local.scoring import (
    compute_pareto, compute_pareto_by_bucket, passes_size_gate, score_round,
)
from local.screening import screen_candidates, screening_card
from local.services import ServicesServer
from local.special_rounds import (
    annotate_special, run_replicate_round, stamp_special_objectives,
)
from local.store import LocalStore
from local.synthetic_arch import REFERENCE_ARCH_VERSION, reference_arch
from local.task import (
    SIZE_BUCKETS, SyntheticDataGeneratorSpec, TaskSpec, TSDataPipelineSpec,
    TSForecastingSpec, buckets_for, make_spec,
)
from local.train_subprocess import run_training_isolated


logger = logging.getLogger("local.validator")


def _pick_bucket(round_id: int, task=None,
                 prefer_buckets: set[str] | None = None) -> tuple[str, int, int]:
    """Pick the bucket for this round.

    Default is round-robin keyed on ``round_id``. When ``prefer_buckets`` is
    non-empty, rotate forward from the round-robin pick until we land on a
    preferred bucket — used on scheduled-continuation rounds to bias toward
    buckets that actually have eligible parents.
    """
    buckets = buckets_for(task) if task is not None else SIZE_BUCKETS
    names = list(buckets.keys())
    start = round_id % len(names)
    name = names[start]
    if prefer_buckets and name not in prefer_buckets:
        for i in range(1, len(names)):
            cand = names[(start + i) % len(names)]
            if cand in prefer_buckets:
                name = cand
                break
    lo, hi = buckets[name]
    return name, lo, hi


def _parse_task_mixture(spec: str) -> list[tuple[str, float]]:
    """Parse ``--tasks "ts_forecasting:0.6,ts_data_pipeline:0.4"``.

    Bare names default to weight 1.0 so ``--tasks ts_forecasting`` is the
    single-task shorthand. Raises ``argparse``-style ``ValueError`` on
    malformed entries or non-positive weight sum.
    """
    out: list[tuple[str, float]] = []
    for piece in spec.split(","):
        s = piece.strip()
        if not s:
            continue
        if ":" in s:
            name, w_str = s.split(":", 1)
            try:
                weight = float(w_str)
            except ValueError as e:
                raise ValueError(f"bad weight in --tasks: {piece!r}") from e
        else:
            name, weight = s, 1.0
        if weight < 0:
            raise ValueError(f"negative weight in --tasks: {piece!r}")
        out.append((name.strip(), weight))
    if not out:
        raise ValueError("--tasks resolved to an empty mixture")
    if sum(w for _, w in out) <= 0:
        raise ValueError("--tasks weights must sum to a positive number")
    return out


def _pick_task_name(round_id: int, mixture: list[tuple[str, float]]) -> str:
    """Seeded per-round task pick. Identity on a single-task mixture."""
    if len(mixture) == 1:
        return mixture[0][0]
    total = sum(w for _, w in mixture)
    r = random.Random(f"task-mix-{round_id}").random() * total
    acc = 0.0
    for name, w in mixture:
        acc += w
        if r < acc:
            return name
    return mixture[-1][0]


def _task_dict(task) -> dict:
    """Serialise a TaskSpec / TSForecastingSpec for the challenge payload.

    Miners read this verbatim — see ``miners/*/agent.py`` and the
    ``challenge['task']`` description in radar-miner-examples/README.md.
    """
    if isinstance(task, SyntheticDataGeneratorSpec):
        return {
            "name": task.name,
            "task_params": {
                "context_len": task.context_len,
                "prediction_len": task.prediction_len,
                "num_variates": task.num_variates,
                "quantiles": list(task.quantiles),
            },
            "constraints": [
                "torch + stdlib only",
                "build_pipeline(context_len, prediction_len, num_variates, "
                "quantiles) returns an iterator of {'input', 'target'} batches",
            ],
            # Scored GIFT-only (sqrt(crps*mase)) against the fixed reference
            # arch — same surface as ts_data_pipeline, so the miner runner is
            # shared.
            "objectives": [
                {"name": "crps", "primary": True, "minimize": True},
                {"name": "mase", "primary": True, "minimize": True},
            ],
            "time_budget": task.time_budget_seconds,
            "runner_dir": "ts_data_pipeline",
        }
    if isinstance(task, TSDataPipelineSpec):
        return {
            "name": task.name,
            "task_params": {
                "context_len": task.context_len,
                "prediction_len": task.prediction_len,
                "num_variates": task.num_variates,
                "quantiles": list(task.quantiles),
            },
            "constraints": [
                "torch + stdlib only",
                "build_pipeline(context_len, prediction_len, num_variates, "
                "quantiles) returns an iterator of {'input', 'target'} batches",
            ],
            "objectives": [
                {"name": "aulc", "primary": False, "minimize": True},
                {"name": "gift_metric", "primary": False, "minimize": True},
            ],
            "time_budget": task.time_budget_seconds,
            "runner_dir": "ts_data_pipeline",
        }
    if isinstance(task, TSForecastingSpec):
        return {
            "name": task.name,
            "task_params": {
                "context_len": task.context_len,
                "prediction_len": task.prediction_len,
                "num_variates": task.num_variates,
                "quantiles": list(task.quantiles),
            },
            "constraints": ["torch + stdlib only", "build_model(context_len, "
                            "prediction_len, num_variates, quantiles) returns nn.Module"],
            "objectives": [{"name": "val_loss", "primary": True, "minimize": True}],
            "time_budget": task.time_budget_seconds,
            "runner_dir": "ts_forecasting",
        }
    return {
        "name": task.name,
        "input_dim": task.input_dim,
        "output_dim": task.output_dim,
    }


def _current_epoch(task, frozen_arch=None, frozen_pipeline=None) -> dict:
    """Continuation-pinning context for this round.

    A parent's saved checkpoint is only a valid warm-start when it was
    trained in the same context as this round. The epoch keys are
    stamped on every successful experiment's ``objectives``:

    * ts_data_pipeline pins ``frozen_arch_version`` — Δ isn't comparable
      after the yardstick arch refreshes.
    * ts_forecasting pins ``frozen_pipeline_version`` *only when one is
      currently in use*. A pure-real run (no synthetic mixed in) leaves
      the key out so it pairs freely with other pure-real ancestors —
      that's the deliberate control track.
    """
    epoch: dict = {}
    if isinstance(task, TSDataPipelineSpec) and frozen_arch is not None:
        epoch["frozen_arch_version"] = int(frozen_arch.version)
    if isinstance(task, TSForecastingSpec) and frozen_pipeline is not None:
        epoch["frozen_pipeline_version"] = int(frozen_pipeline.version)
    # The synthetic_data_generator arch is fixed, so this is always 1 today —
    # but pinning it future-proofs continuation against a model swap.
    if isinstance(task, SyntheticDataGeneratorSpec):
        epoch["synth_arch_version"] = int(REFERENCE_ARCH_VERSION)
    # The scored metric changes meaning when a canary or proxy fraction is
    # held out of the eval aggregate — pin lineages to the same splits so Δ
    # stays honest across rounds.
    from local.eval_metrics import canary_frac, proxy_frac
    frac = canary_frac()
    if frac > 0:
        epoch["eval_canary_frac"] = frac
    pfrac = proxy_frac()
    if pfrac > 0:
        epoch["eval_proxy_frac"] = pfrac
    return epoch


def _parent_in_epoch(parent: dict, epoch: dict) -> bool:
    if not epoch:
        return True
    objs = parent.get("objectives", {}) or {}
    return all(objs.get(k) == v for k, v in epoch.items())


def _build_challenge(round_id: int, store: LocalStore, task,
                     services_url: str, agent_seconds: int = 180,
                     continuation_enabled: bool = False,
                     scheduled_continuation: bool = False,
                     scheduled_extend: bool = False,
                     frozen_arch=None, frozen_pipeline=None) -> dict:
    # On scheduled-continuation rounds, bias the bucket pick toward buckets
    # that actually have eligible parents — otherwise round-robin wastes
    # most scheduled continuations on empty buckets and they all downgrade.
    prefer_buckets: set[str] | None = None
    parents_any: list[dict] = []
    if (continuation_enabled and scheduled_continuation
            and not isinstance(
                task, (TSDataPipelineSpec, SyntheticDataGeneratorSpec))):
        parents_any = store.eligible_parents(
            task=task.name, min_flops=0, max_flops=10**18,
        )
        if parents_any:
            bucket_defs = buckets_for(task)
            prefer_buckets = set()
            for e in parents_any:
                f = (e.get("objectives", {}) or {}).get("flops_equivalent_size", 0)
                for bname, (blo, bhi) in bucket_defs.items():
                    if int(blo * 0.9) <= int(f) <= int(bhi * 1.1):
                        prefer_buckets.add(bname)
                        break
    name, lo, hi = _pick_bucket(round_id, task=task, prefer_buckets=prefer_buckets)
    # Pipeline-style tasks (ts_data_pipeline, synthetic_data_generator) train a
    # fixed architecture, so FLOPs are identical across miners in a round —
    # size buckets are meaningless. Zero the bounds so the harness skips the
    # gate.
    if isinstance(task, (TSDataPipelineSpec, SyntheticDataGeneratorSpec)):
        name, lo, hi = "frozen", 0, 0
    all_exps = store.recent_experiments(n=10_000)
    # Mixed-task DBs (e.g. ts_forecasting + ts_data_pipeline against the
    # same SQLite file) mean we can't compare metrics across tasks — filter
    # to this task before computing the per-round feasible frontier.
    same_task = [e for e in all_exps if e.get("task") == task.name]
    # Per-bucket Pareto: compute the non-dominated set *within* this round's
    # size bucket, not globally-then-filter. Otherwise a cheap, lower-metric
    # model in a smaller bucket would dominate (lower metric AND lower flops)
    # the best models here and knock them off this bucket's frontier.
    in_bucket = [
        e for e in same_task if passes_size_gate(e["objectives"], lo, hi)
    ]
    feasible = [
        {
            "code": e["code"],
            "metric": e["metric"],
            "objectives": e["objectives"],
            "name": e["name"],
        }
        for e in compute_pareto(in_bucket)
    ]
    # The validator owns the round type: a scheduled continuation round only
    # becomes one if eligible (fully-eval'd, checkpoint-bearing, in-bucket)
    # parents actually exist — otherwise it degrades to a fresh round.
    eligible_parents = []
    epoch = _current_epoch(task, frozen_arch, frozen_pipeline)
    if continuation_enabled:
        eligible_parents = [
            _parent_summary(e)
            for e in store.eligible_parents(
                task=task.name, min_flops=lo, max_flops=hi,
            )
            if _parent_in_epoch(e, epoch)
        ]
    round_type = (
        "continuation"
        if (scheduled_continuation and eligible_parents) else "new"
    )
    continuation_allowed = round_type == "continuation"
    # On a synthetic_data_generator continuation round, the validator also owns
    # the extend-vs-modify split: ``extend`` re-trains the parent's own
    # generator (more compute on identical data), ``modify`` trains the miner's
    # freshly submitted one. The split is meaningless for arch-evolution tasks,
    # so the kind stays "" off the synth task.
    continuation_kind = (
        ("extend" if scheduled_extend else "modify")
        if (continuation_allowed
            and isinstance(task, SyntheticDataGeneratorSpec)) else ""
    )
    # Persist scheduled-vs-actual so the dashboard can show "scheduled
    # continuation downgraded because no parents" instead of silently
    # bucketing it as a novel round.
    scheduled_round_type = "continuation" if scheduled_continuation else "new"
    downgrade_reason = ""
    if scheduled_continuation and round_type == "new":
        downgrade_reason = (
            "no_eligible_parents" if not parents_any
            else "no_in_bucket_parents"
        )
    # ``allowed_urls`` is what the miner-side GatedClient enforces. Every
    # endpoint the agent can reach lives under ``services_url`` so a
    # single prefix is enough.
    payload = {
        "challenge_id": f"r{round_id:06d}-{uuid.uuid4().hex[:8]}",
        "round_id": round_id,
        "seed": round_id * 31 + 7,
        "min_flops_equivalent": lo,
        "max_flops_equivalent": hi,
        "bucket": name,
        "task": _task_dict(task),
        "feasible_frontier": feasible,
        "round_type": round_type,
        "scheduled_round_type": scheduled_round_type,
        "continuation_kind": continuation_kind,
        "downgrade_reason": downgrade_reason,
        "continuation_allowed": bool(continuation_allowed),
        "eligible_parents": eligible_parents if continuation_allowed else [],
        "agent_seconds": int(agent_seconds),
        # Dummy agent_token — local services.py doesn't enforce auth, but
        # the real agents' startup checks require a non-empty value.
        "agent_token": "local-dev",
        # URL surface the agent uses via GatedClient.
        "db_url": services_url,
        "llm_url": f"{services_url}/llm",
        "desearch_url": f"{services_url}/desearch",
        "cognition_wiki_url": f"{services_url}/wiki",
        "allowed_urls": services_url,
    }
    # Score-correlated feedback: when a proxy slice is held out, tell the
    # miner what ``proxy_metric`` is (a held-out, disjoint GIFT slice scored
    # with the same sqrt(crps*mase) formula) so it has a gradient to climb
    # between rounds. Each feasible-frontier member already carries its
    # ``objectives.proxy_metric``; this block is the explanation + the best
    # proxy seen so far.
    from local.eval_metrics import proxy_frac as _proxy_frac
    pfrac = _proxy_frac()
    if pfrac > 0:
        proxies = [
            f["objectives"].get("proxy_metric") for f in feasible
            if isinstance(f.get("objectives"), dict)
            and f["objectives"].get("proxy_metric") is not None
        ]
        payload["proxy_feedback"] = {
            "metric_key": "proxy_metric",
            "eval_proxy_frac": pfrac,
            "lower_is_better": True,
            "best_seen": min(proxies) if proxies else None,
            "note": (
                "proxy_metric is sqrt(crps*mase) on a FIXED held-out GIFT "
                "slice, disjoint from the scored set — a denoised, "
                "score-correlated readout of generalization. Optimize it; "
                "it tracks the true (hidden) score far better than the "
                "in-training val loss. It is NOT the scored set, so it can't "
                "be gamed directly, and a secret canary slice flags "
                "overfitting to it."
            ),
        }

    # synthetic_data_generator ships the FIXED reference arch as the same
    # ``frozen_arch`` card ts_data_pipeline uses, so the miner-facing tools
    # (frozen-arch card, pipeline probe) work unchanged.
    if isinstance(task, SyntheticDataGeneratorSpec):
        payload["frozen_arch"] = reference_arch().to_card()
    if frozen_arch is not None:
        payload["frozen_arch"] = {
            "version": frozen_arch.version,
            "source_experiment_id": frozen_arch.source_experiment_id,
            "source_metric": frozen_arch.source_metric,
            "source_crps": frozen_arch.source_crps,
            "source_mase": frozen_arch.source_mase,
            "source_flops": frozen_arch.source_flops,
            "source_name": frozen_arch.source_name,
            "code": frozen_arch.code,
        }
    if frozen_pipeline is not None:
        payload["frozen_pipeline"] = {
            "version": frozen_pipeline.version,
            "source_experiment_id": frozen_pipeline.source_experiment_id,
            "source_metric": frozen_pipeline.source_metric,
            "source_crps": frozen_pipeline.source_crps,
            "source_mase": frozen_pipeline.source_mase,
            "source_name": frozen_pipeline.source_name,
            "frozen_arch_version": frozen_pipeline.frozen_arch_version,
            "num_shards": len(frozen_pipeline.shard_paths or []),
        }
    return payload


def _ensure_ts_caches() -> bool:
    """Ensure the GIFT-Eval benchmark cache and the held-out pretrain val
    shard are populated before the validator starts a ts_forecasting round.

    Returns True on success (caches ready), False on hard failure (no creds /
    download failed) — caller exits early so we don't blow ``--training_seconds``
    on every round only to fail Phase C.
    """
    gift_cache = os.environ.get("RADAR_GIFT_EVAL_CACHE", "/tmp/radar_gift_eval")
    val_cache = os.environ.get("RADAR_PRETRAIN_VAL_CACHE", "/tmp/radar_pretrain_val")

    # 1. GIFT-Eval benchmark — every leaderboard dataset must be on disk so
    # Phase C can iterate the full 97-task list. Skip the import + R2 dance
    # when everything is already cached.
    try:
        from shared.gift_eval import (
            MED_LONG_DATASETS, SHORT_DATASETS, ensure_datasets_cached,
        )
    except ImportError as e:
        logger.error("ts_forecasting requires shared.gift_eval: %s", e)
        return False

    leaderboard = sorted({*SHORT_DATASETS, *MED_LONG_DATASETS})
    Path(gift_cache).mkdir(parents=True, exist_ok=True)
    logger.info(
        "checking GIFT-Eval cache (%d datasets) at %s", len(leaderboard), gift_cache,
    )
    status = ensure_datasets_cached(leaderboard, cache_dir=gift_cache)
    missing = [k for k, ok in status.items() if not ok]
    if missing or not status:
        logger.error(
            "GIFT-Eval cache incomplete (%d missing). First few: %s. "
            "Check HIPPIUS_*/R2_* creds and re-run.",
            len(missing) or len(leaderboard), missing[:5],
        )
        return False
    logger.info("GIFT-Eval cache OK (%d datasets ready)", len(status))

    # 2. Pretrain val shard — small fixed held-out split. If the dir already
    # has parquet files we trust them; otherwise pull from the pretrain bucket.
    val_dir = Path(val_cache)
    existing_val = sorted(val_dir.glob("*.parquet")) if val_dir.is_dir() else []
    if existing_val:
        logger.info(
            "pretrain val cache OK (%d shard(s) at %s)", len(existing_val), val_cache,
        )
        return True

    logger.info("pretrain val cache empty at %s — fetching", val_cache)
    try:
        from local.fetch_pretrain import make_pretrain_client
        from shared.pretrain_data import PretrainBenchmark
    except ImportError as e:
        logger.error("cannot fetch val shard (missing deps): %s", e)
        return False

    r2 = make_pretrain_client()
    if r2 is None:
        logger.error(
            "no pretrain S3 client (install boto3 + set HIPPIUS_*/R2_* env)",
        )
        return False
    bench = PretrainBenchmark(r2=r2)
    val_keys = bench.get_val_shard_keys()
    if not val_keys:
        logger.warning(
            "pretrain manifest declares no val shards — in-training val "
            "will be disabled by trainer fallback",
        )
        return True

    val_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for key in val_keys:
        dest = val_dir / Path(key).name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        if r2.download_file_to_disk(key, str(dest)):
            ok += 1
            logger.info("  fetched %s (%d bytes)", dest.name, dest.stat().st_size)
        else:
            fail += 1
            logger.error("  failed to download %s", key)
    if fail:
        logger.error("pretrain val fetch failed (%d/%d shards)", fail, ok + fail)
        return False
    logger.info("pretrain val cache ready (%d shard(s))", ok)
    return True


def _next_round_id(store: LocalStore) -> int:
    """Continue from the highest round_id seen so far so consecutive
    runs of the validator don't repeat round 0."""
    rows = store._conn.execute(
        "SELECT COALESCE(MAX(round_id), -1) AS r FROM challenges"
    ).fetchone()
    return int(rows["r"]) + 1


def _pretrain_pool() -> list[str]:
    """Sorted parquet shard paths in the pretrain cache (empty if absent)."""
    import os
    pdir = Path(os.environ.get("RADAR_PRETRAIN_CACHE", "/tmp/radar_pretrain"))
    if not pdir.is_dir():
        return []
    return sorted(str(p) for p in pdir.glob("*.parquet"))


def _train_proposal(payload: dict, task, round_id: int,
                    challenge: dict, prep: dict,
                    frozen_arch=None, frozen_pipeline=None) -> dict:
    """Run one proposal, retrying as a fresh run if a continuation
    warm-start turns out to be architecture-incompatible (strict load).

    Training runs through ``run_training_isolated`` — a crash or hang in
    submitted code becomes a failed experiment instead of taking down
    the validator process."""
    def _run() -> dict:
        # ``train_code`` overrides the miner submission on an extend round
        # (re-train the parent's own generator); otherwise the miner's code.
        return run_training_isolated(
            prep.get("train_code") or payload.get("code", ""),
            seed=round_id,
            task=task,
            min_flops=challenge["min_flops_equivalent"],
            max_flops=challenge["max_flops_equivalent"],
            parent_checkpoint_path=prep["parent_checkpoint_path"],
            compute_offset=prep["compute_offset"],
            step_offset=prep["step_offset"],
            shard_paths=prep["shard_paths"],
            shard_reuse=prep["shard_reuse"],
            frozen_arch=frozen_arch,
            frozen_pipeline=frozen_pipeline,
        )

    result = _run()
    if (not result.get("success")
            and result.get("harness_status") == "checkpoint_incompatible"
            and prep["mode"] == "continue"):
        logger.warning("    warm-start incompatible; retrying as a fresh run")
        prefix = (prep["note"] + "; ") if prep["note"] else ""
        prep["note"] = prefix + "continuation incompatible: retried fresh"
        prep.update(
            mode="new", parent_index=None, parent_metric=None,
            parent_per_task=None, parent_checkpoint_path=None,
            compute_offset=0.0, step_offset=0, n_rounds=1,
            continuation_kind="", train_code=None,
        )
        result = _run()
    return result


def _gc_checkpoints(store: LocalStore, ckpt_store: CheckpointStore,
                    round_id: int, keep_rounds: int = 5) -> None:
    """Drop checkpoints that aren't on a frontier or a recent parent."""
    all_exps = store.recent_experiments(n=10_000)
    keep = {e["id"] for e in compute_pareto_by_bucket(all_exps)}
    keep |= {e["id"] for e in continuation_frontier(all_exps)}
    keep |= {e["id"] for e in all_exps if e["round_id"] > round_id - keep_rounds}
    removed = ckpt_store.gc(keep)
    if removed:
        logger.info("  gc: removed %d stale checkpoint(s)", removed)


def run_round(store: LocalStore, task, round_id: int,
              phase_a_seconds: float, services_url: str,
              agent_seconds: int = 180,
              sink: ArtifactSink | None = None,
              ckpt_store: CheckpointStore | None = None,
              continuation_enabled: bool = False,
              continuation_equilibrium: float = 0.7,
              continuation_warmup_rounds: int = 20,
              continuation_step_pct: float = 2.0,
              continuation_step_every: int = 3,
              continuation_extend_pct: float = 0.5,
              shards_per_round: int = 0,
              frozen_arch_store: FrozenArchStore | None = None,
              frozen_arch_refresh_every: int = 50,
              frozen_pipeline_store: FrozenPipelineStore | None = None,
              frozen_pipeline_refresh_every: int = 30,
              frozen_pipeline_num_shards: int = 16,
              frozen_pipeline_batches_per_shard: int = 64,
              frozen_pipeline_control_seconds: int = 0,
              frozen_pipeline_render_per_round: bool = False,
              replicate_pct: float = 0.0,
              ablate_pct: float = 0.0,
              recipe_pct: float = 0.0,
              transfer_pct: float = 0.0,
              screen_candidates_n: int = 0,
              screen_seconds: int = 600,
              screen_eval_tasks: int = 15) -> None:
    # Special round types (replicate / ablate / recipe_only / transfer) are
    # scheduled by an independent seeded coin and pre-empt the continuation
    # flip for the round. Each downgrades to a normal round when its
    # prerequisite (a usable source experiment) is missing.
    special = special_round_type(
        round_id, replicate_pct=replicate_pct, ablate_pct=ablate_pct,
        recipe_pct=recipe_pct, transfer_pct=transfer_pct,
    )
    # The validator owns the cadence: a scheduled coin flip decides whether
    # this is a continuation round. The rate stays at 0 until
    # ``warmup_rounds`` attempted rounds, then climbs as a staircase to the
    # equilibrium; _build_challenge further downgrades to "new" when no
    # eligible parents exist yet.
    attempted_rounds = store.attempted_round_count(task=task.name)
    rate = continuation_rate(
        attempted_rounds,
        equilibrium=continuation_equilibrium,
        warmup_rounds=continuation_warmup_rounds,
        step_pct=continuation_step_pct,
        step_every=continuation_step_every,
    ) if continuation_enabled else 0.0
    scheduled = continuation_enabled and is_continuation_round(
        round_id, attempted_rounds,
        equilibrium=continuation_equilibrium,
        warmup_rounds=continuation_warmup_rounds,
        step_pct=continuation_step_pct,
        step_every=continuation_step_every,
    )
    # On a scheduled continuation round, a second seeded coin splits it into
    # extend (re-train the parent's own generator) vs modify. Scoped to the
    # synthetic_data_generator task, where the miner deliverable is the data
    # generator and the architecture is fixed, so warm-starts are always
    # shape-compatible regardless of which code trains.
    scheduled_extend = (
        scheduled
        and isinstance(task, SyntheticDataGeneratorSpec)
        and is_extend_round(round_id, continuation_extend_pct)
    )
    if special:
        scheduled = False
        scheduled_extend = False
    # Frozen-arch refresh (ts_data_pipeline only). Snapshots at every_n
    # successful data-pipeline rounds; the first one bootstraps from the
    # ts_forecasting frontier.
    frozen_arch = None
    if isinstance(task, TSDataPipelineSpec) and frozen_arch_store is not None:
        maybe_refresh_frozen(
            frozen_arch_store, store, every_n=frozen_arch_refresh_every,
        )
        frozen_arch = frozen_arch_store.current()
        if frozen_arch is None:
            logger.warning(
                "  no frozen architecture available yet — round drops. "
                "Run ts_forecasting first so there's a frontier to snapshot.",
            )
            return
        # Anchor the AULC half of the data-pipeline metric. Computed
        # once per snapshot on a fixed reference pipeline so miners are
        # scored on improvement-over-baseline rather than raw curve
        # area (which is dominated by val-window difficulty). Best-effort
        # — a None return means absolute-AULC fallback with a warning.
        if frozen_arch.baseline_aulc is None:
            from local.data_pipeline import ensure_baseline_aulc
            ensure_baseline_aulc(frozen_arch_store, frozen_arch, task=task)

    # Frozen-pipeline refresh (ts_forecasting only). Snapshots at every_n
    # successful forecasting rounds — staggered from arch refresh so both
    # yardsticks don't move at once. Absent pipeline = pure-real run, the
    # control track that stays comparable across versions.
    frozen_pipeline = None
    if isinstance(task, TSForecastingSpec) and frozen_pipeline_store is not None:
        maybe_refresh_pipeline(
            frozen_pipeline_store, store,
            every_n=frozen_pipeline_refresh_every,
            num_shards=frozen_pipeline_num_shards,
            batches_per_shard=frozen_pipeline_batches_per_shard,
            control_seconds=frozen_pipeline_control_seconds,
        )
        frozen_pipeline = frozen_pipeline_store.current()
        # Skip injection if the snapshot has no rendered shards (the
        # promotion was lean — torch/pandas unavailable, or the miner's
        # pipeline raised). The round still runs as pure-real.
        if frozen_pipeline is not None and not frozen_pipeline.shard_paths:
            frozen_pipeline = None
        # Per-round re-render approximates streaming from the generator:
        # each round draws a fresh window of the generator's stream
        # instead of reusing one tiny fixed corpus forever.
        if frozen_pipeline is not None and frozen_pipeline_render_per_round:
            from local.pipeline_control import rerender_for_round
            frozen_pipeline = rerender_for_round(
                frozen_pipeline_store, frozen_pipeline, round_id,
                num_shards=frozen_pipeline_num_shards,
                batches_per_shard=frozen_pipeline_batches_per_shard,
            )

    # Replicate rounds are validator-only — no challenge is opened to
    # miners; the round's whole job is re-measuring a frontier member.
    if special == "replicate":
        def _replicate_train(code: str, **kw) -> dict:
            return run_training_isolated(
                code, frozen_arch=frozen_arch,
                frozen_pipeline=frozen_pipeline, **kw,
            )
        consumed = run_replicate_round(
            store, task, round_id,
            epoch=_current_epoch(task, frozen_arch, frozen_pipeline),
            train_fn=_replicate_train, sink=sink,
        )
        if consumed:
            return
        special = ""  # no source to replicate — run a normal round

    challenge = _build_challenge(
        round_id, store, task, services_url, agent_seconds=agent_seconds,
        continuation_enabled=continuation_enabled,
        scheduled_continuation=scheduled,
        scheduled_extend=scheduled_extend,
        frozen_arch=frozen_arch,
        frozen_pipeline=frozen_pipeline,
    )
    # Ablate / recipe_only / transfer mutate the freshly built challenge
    # (round_type + target card) or record a downgrade reason.
    if special:
        annotate_special(
            challenge, special, store, task, round_id=round_id,
            epoch=_current_epoch(task, frozen_arch, frozen_pipeline),
        )
    # Screening tier: only meaningful on fresh ts_forecasting design
    # rounds — the agent may submit extra candidates for cheap triage.
    if (screen_candidates_n > 0 and isinstance(task, TSForecastingSpec)
            and challenge.get("round_type") == "new"):
        challenge["screening"] = screening_card(
            max_candidates=screen_candidates_n,
            budget_seconds=screen_seconds,
            eval_max_tasks=screen_eval_tasks,
        )
    challenge_id = challenge["challenge_id"]
    bucket = challenge["bucket"]
    continuation_allowed = challenge["continuation_allowed"]
    type_field = challenge["round_type"]
    if challenge.get("continuation_kind"):
        type_field = f"{type_field}:{challenge['continuation_kind']}"
    if challenge.get("downgrade_reason"):
        type_field = f"new<-continuation({challenge['downgrade_reason']})"
    logger.info(
        "round=%d bucket=%s flops=[%d, %d] frontier=%d type=%s "
        "(cont_rate=%.2f attempted=%d parents=%d)",
        round_id, bucket, challenge["min_flops_equivalent"],
        challenge["max_flops_equivalent"], len(challenge["feasible_frontier"]),
        type_field, rate, attempted_rounds,
        len(challenge["eligible_parents"]),
    )

    if sink is not None:
        sink.record_challenge(task.name, round_id, challenge)

    # ── Phase A: publish challenge, wait for proposals ──────
    store.post_challenge(challenge_id, round_id, challenge)
    deadline = time.time() + phase_a_seconds
    last_count = -1
    while time.time() < deadline:
        proposals = store.proposals_for(challenge_id)
        if len(proposals) != last_count:
            logger.info("  phase A: %d proposal(s) so far", len(proposals))
            last_count = len(proposals)
        # Early exit if at least one proposal in and 3 seconds have passed
        # without new arrivals — keeps single-miner laptop runs snappy.
        if proposals and time.time() > deadline - phase_a_seconds + 3:
            break
        time.sleep(0.5)
    store.mark_challenge(challenge_id, "collected")

    proposals = store.proposals_for(challenge_id)
    if not proposals:
        logger.warning("  no proposals received; round drops")
        store.mark_challenge(challenge_id, "done")
        return

    # Pretrain shard pool (ts_forecasting only) — assigned per-run so
    # continuations can avoid lineage-seen shards. ts_data_pipeline's
    # data source is the miner's pipeline; shard assignment is N/A.
    pool = (
        _pretrain_pool()
        if continuation_allowed and isinstance(task, TSForecastingSpec)
        else []
    )

    # ── Phase B + Phase C: train and evaluate every proposal ──
    results: list[dict] = []
    for p in proposals:
        payload = p["payload"]
        miner_id = p["miner_id"]
        name = payload.get("name", "unnamed")
        if sink is not None:
            sink.record_proposal(task.name, round_id, miner_id, payload)

        prep = prepare_continuation(
            store, ckpt_store,
            payload=payload, task_name=task.name,
            min_flops=challenge["min_flops_equivalent"],
            max_flops=challenge["max_flops_equivalent"],
            pool=pool, shards_per_round=shards_per_round, seed=round_id,
            current_epoch=_current_epoch(task, frozen_arch, frozen_pipeline),
            force_continuation=True,
            extend=challenge.get("continuation_kind") == "extend",
            eligible_parent_ids=[
                int(p["id"]) for p in challenge.get("eligible_parents", [])
                if isinstance(p.get("id"), int)
            ],
            miner_id=miner_id,
        ) if continuation_allowed else {
            # Continuation disabled — still preserve any payload parent_index
            # so the existing lineage/diff tracking keeps working.
            "mode": "new",
            "parent_index": (
                payload.get("parent_index")
                if isinstance(payload.get("parent_index"), int) else None
            ),
            "parent_metric": None, "parent_per_task": None,
            "parent_checkpoint_path": None, "compute_offset": 0.0,
            "step_offset": 0, "n_rounds": 1, "shard_paths": None,
            "shard_reuse": False, "continuation_kind": "", "train_code": None,
            "note": "",
        }
        if prep["note"]:
            logger.info("    %s", prep["note"])

        # Screening tier: cheap triage of extra candidates before the one
        # full-budget run. Fresh runs only — a warm-start's parent already
        # fixed the architecture.
        screening_summaries: list[dict] | None = None
        scr = challenge.get("screening")
        if (scr and prep["mode"] == "new" and not prep.get("train_code")
                and payload.get("candidates")):
            def _screen_train(code: str, **kw) -> dict:
                return run_training_isolated(
                    code, frozen_arch=frozen_arch,
                    frozen_pipeline=frozen_pipeline, **kw,
                )
            winner, screening_summaries = screen_candidates(
                payload.get("candidates"), task=task, seed=round_id,
                min_flops=challenge["min_flops_equivalent"],
                max_flops=challenge["max_flops_equivalent"],
                budget_seconds=int(scr["budget_seconds"]),
                eval_max_tasks=int(scr["eval_max_tasks"]),
                train_fn=_screen_train,
            )
            if winner is not None:
                payload["code"] = winner["code"]
                payload["name"] = winner["name"]
                name = winner["name"]

        logger.info(
            "  phase B/C: training '%s' from miner=%s mode=%s",
            name, miner_id, prep["mode"],
        )
        result = _train_proposal(
            payload, task, round_id, challenge, prep,
            frozen_arch=frozen_arch, frozen_pipeline=frozen_pipeline,
        )
        # Pin the validator-declared comparison anchor (ablate / recipe /
        # transfer) and the paired per-dataset parent comparison onto the
        # result before scoring reads them.
        stamp_special_objectives(result, payload, challenge)
        if (prep["mode"] == "continue" and result.get("success")
                and isinstance(result.get("objectives"), dict)):
            paired = paired_per_task_delta(
                prep.get("parent_per_task"),
                result["objectives"].get("per_task"),
            )
            if paired:
                result["objectives"]["paired"] = paired
        if screening_summaries and isinstance(result.get("objectives"), dict):
            result["objectives"]["screening"] = screening_summaries

        result["miner_id"] = miner_id
        result["name"] = name
        # On an extend round the parent's generator is what actually trained,
        # so record that as this experiment's code (the diff vs parent is then
        # empty by construction — extend == "more compute, same data").
        result["code"] = prep.get("train_code") or payload.get("code", "")
        result["motivation"] = payload.get("motivation", "")
        result["reasoning"] = payload.get("reasoning", "")
        result["tool_calls"] = payload.get("tool_calls", [])
        result["prompt_id"] = payload.get("prompt_id", "")
        result["mode"] = prep["mode"]
        # The extend/modify label is only meaningful on the synth task; stamp
        # the per-proposal resolved kind into objectives there so extend and
        # modify lineages stay separable on the shared second frontier.
        kind = prep.get("continuation_kind") or ""
        result["continuation_kind"] = kind
        if (kind and isinstance(task, SyntheticDataGeneratorSpec)
                and isinstance(result.get("objectives"), dict)):
            result["objectives"]["continuation_kind"] = kind
        result["parent_index"] = prep["parent_index"]
        result["parent_metric"] = prep["parent_metric"]
        result["n_rounds"] = prep["n_rounds"]
        result["continuation_note"] = prep["note"]
        if result["success"]:
            objs = result["objectives"]
            extra = ""
            if "crps" in objs and "mase" in objs:
                extra = f" crps={objs['crps']:.4f} mase={objs['mase']:.4f}"
            logger.info(
                "    metric=%.6f params=%d flops_eq=%d%s",
                result["metric"], objs["num_params"],
                objs["flops_equivalent_size"], extra,
            )
        else:
            logger.warning("    failed: %s", result.get("error", "?"))
        results.append(result)

    # ── Scoring ─────────────────────────────────────────────
    cont_frontier = continuation_frontier(store.recent_experiments(n=10_000))
    score_round(
        results,
        min_flops=challenge["min_flops_equivalent"],
        max_flops=challenge["max_flops_equivalent"],
        frontier=challenge["feasible_frontier"],
        continuation_frontier=cont_frontier,
    )

    # Write experiments + persist checkpoints + mirror artifacts
    for p, r in zip(proposals, results):
        # On failure, fold the trainer's error trace into analysis so
        # /experiments/failures returns something actionable to other
        # miners (the analysis label alone is just "training crashed").
        analysis = r["analysis"]
        if not r["success"]:
            err = (r.get("error") or "").strip()
            if err and err not in analysis:
                analysis = f"{analysis}\n{err}" if analysis else err
        note = r.get("continuation_note") or ""
        if note:
            analysis = f"{analysis} [{note}]"
        exp_id = store.add_experiment(
            round_id=round_id,
            miner_id=r["miner_id"],
            name=r["name"],
            code=r["code"],
            motivation=r["motivation"],
            reasoning=r["reasoning"],
            tool_calls=r["tool_calls"],
            metric=r["metric"],
            success=r["success"],
            objectives=r["objectives"],
            score=r["score"],
            loss_curve=r["loss_curve"],
            val_curve=r.get("val_curve") or [],
            analysis=analysis,
            parent_index=r.get("parent_index"),
            prompt_id=r["prompt_id"],
            task=task.name,
            n_rounds=int(r.get("n_rounds", 1) or 1),
            cumulative_compute=float(
                r["objectives"].get("cumulative_compute", 0.0)
            ),
            mode=r.get("mode", "new"),
        )
        workdir_str = r.get("workdir") or ""
        workdir = Path(workdir_str) if workdir_str else None
        # Persist the trained checkpoint so this experiment can later be a
        # warm-start parent, BEFORE the workdir is cleaned up.
        if r["success"] and ckpt_store is not None and workdir is not None:
            ckpt_src = workdir / "checkpoints" / "model.safetensors"
            ref = ckpt_store.save(exp_id, ckpt_src)
            if ref is not None:
                store.set_checkpoint_ref(exp_id, ref)
        try:
            if sink is not None:
                sink.record_result(task.name, round_id, r["miner_id"], r, workdir)
        finally:
            # Always reclaim the trainer workdir, even if mirroring raised —
            # otherwise the /tmp/radar_ts_* dir leaks for the run's lifetime.
            cleanup_workdir(workdir)

    # Keep only checkpoints worth warm-starting from (frontier + recent
    # lineage parents); drop the rest to bound disk use.
    if ckpt_store is not None:
        _gc_checkpoints(store, ckpt_store, round_id)

    # Structured post-mortem per experiment — the round's findings outlive
    # its scalar metric. Best-effort: a report failure must not break the
    # round loop.
    try:
        from local.lab_reports import generate_round_reports
        n_reports = generate_round_reports(store, round_id, task.name)
        if n_reports:
            logger.info("  lab reports: %d written", n_reports)
    except Exception as e:  # noqa: BLE001
        logger.warning("lab report generation failed: %s", e)

    store.mark_challenge(challenge_id, "done")

    # ── Round summary ───────────────────────────────────────
    winner = max(results, key=lambda r: r.get("score", 0.0))
    w_objs = winner.get("objectives", {}) or {}
    if "crps" in w_objs and "mase" in w_objs:
        win_extra = f" crps={w_objs['crps']:.4f} mase={w_objs['mase']:.4f}"
    else:
        win_extra = ""
    logger.info(
        "  round done: winner=%s metric=%s score=%.3f%s",
        winner["name"],
        f"{winner['metric']:.6f}" if winner.get("metric") is not None else "FAIL",
        winner.get("score", 0.0),
        win_extra,
    )
    stats = store.stats()
    logger.info(
        "  store: total=%d successful=%d best=%s",
        stats["total"], stats["successful"],
        f"{stats['best_metric']:.6f}" if stats["best_metric"] is not None else "—",
    )

    # Flush this round's agent events to R2 if configured. On success the
    # local rows are pruned so the SQLite file stays bounded on long runs;
    # on failure they're kept and the next round (or a manual export) can
    # retry. ``RADAR_EVENT_LOG_LOCAL_RETAIN`` keeps the most recent N rounds
    # in SQLite (default 20) so the dashboard's service-log tab isn't empty
    # while events still ship to R2 for durability; 0 prunes immediately.
    # Wrapped in try/except because the round is already complete by this
    # point and the flush must not break the loop.
    bucket = os.getenv("RADAR_EVENT_LOG_R2_BUCKET", "").strip()
    if bucket:
        try:
            from local.export_events import flush_round_to_r2
            explicit_prefix = os.getenv("RADAR_EVENT_LOG_R2_PREFIX", "").strip()
            if explicit_prefix:
                prefix = explicit_prefix
            else:
                instance = os.getenv("RADAR_INSTANCE_ID", "").strip()
                prefix = f"agent-events/{instance}" if instance else "agent-events"
            try:
                retain = int(os.getenv("RADAR_EVENT_LOG_LOCAL_RETAIN", "20"))
            except ValueError:
                retain = 20
            flush_round_to_r2(store, round_id, bucket, prefix=prefix,
                              retain_rounds=max(0, retain))
        except Exception as e:  # noqa: BLE001
            logger.warning("agent-events flush failed for round=%d: %s",
                           round_id, e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local radar validator")
    parser.add_argument("--db", default="local/radar_local.db",
                        help="SQLite path (default: local/radar_local.db)")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Number of rounds to run; 0 = forever")
    parser.add_argument("--phase_a_seconds", type=float, default=1900.0,
                        help="Max wait for miner proposals each round. "
                             "Should be ≥ --agent_seconds. Default 1900s "
                             "leaves 100s of slack on top of a 30-min agent.")
    parser.add_argument("--agent_seconds", type=int, default=1800,
                        help="Time budget published to the agent in "
                             "challenge['agent_seconds'] — LLM design loop. "
                             "Agents reserve the last 30s for a fallback, "
                             "so anything < 60 leaves no room for the LLM. "
                             "Default 1800s (30 min).")
    parser.add_argument("--training_seconds", type=int, default=3600,
                        help="Phase B training budget for the ts_forecasting "
                             "task (passed to runner.harness as time_budget). "
                             "Default 3600s (1 hour). Ignored for "
                             "synth_regression.")
    parser.add_argument("--gap_seconds", type=float, default=2.0,
                        help="Sleep between rounds")
    parser.add_argument("--services_port", type=int, default=0,
                        help="HTTP port for the agent-facing services "
                             "server (db/llm/desearch/wiki). 0 = pick free.")
    parser.add_argument("--services_url_file", default="",
                        help="If set, write the chosen services URL here "
                             "after binding (used by run.py to forward it "
                             "to the miner process).")
    parser.add_argument("--wiki_dir", default="",
                        help="Local directory of markdown files exposed to "
                             "the agent at GET /wiki. Empty = no wiki.")
    parser.add_argument(
        "--task", default="synth_regression",
        choices=["synth_regression", "ts_forecasting", "ts_data_pipeline",
                 "synthetic_data_generator"],
        help="Which task this validator drives (single-task shorthand). "
             "Mutually exclusive with --tasks.",
    )
    parser.add_argument(
        "--tasks", default="",
        help="Weighted mixture for interwoven dispatch, e.g. "
             "'ts_forecasting:0.6,ts_data_pipeline:0.4'. Each round "
             "picks one task with a seeded coin so families alternate. "
             "Empty = fall back to --task.",
    )
    parser.add_argument(
        "--frozen_arch_refresh_every", type=int, default=50,
        help="ts_data_pipeline only: refresh the frozen architecture every "
             "N successful data-pipeline rounds (default 50).",
    )
    parser.add_argument(
        "--frozen_arch_dir", default="",
        help="ts_data_pipeline only: where versioned frozen archs live. "
             "Empty = $RADAR_FROZEN_ARCH_DIR or local/frozen_archs.",
    )
    parser.add_argument(
        "--frozen_pipeline_refresh_every", type=int, default=30,
        help="ts_forecasting only: refresh the frozen synthetic pipeline "
             "every N successful forecasting rounds (default 30 — staggered "
             "from the arch refresh so both yardsticks don't move at once). "
             "0 disables the consumer-side feedback loop.",
    )
    parser.add_argument(
        "--frozen_pipeline_dir", default="",
        help="ts_forecasting only: where versioned frozen pipelines live. "
             "Empty = $RADAR_FROZEN_PIPELINE_DIR or local/frozen_pipelines.",
    )
    parser.add_argument(
        "--frozen_pipeline_num_shards", type=int, default=16,
        help="ts_forecasting only: how many synthetic shards to render at "
             "each frozen-pipeline promotion (default 16).",
    )
    parser.add_argument(
        "--frozen_pipeline_batches_per_shard", type=int, default=64,
        help="ts_forecasting only: batches drawn per rendered synthetic "
             "shard (default 64).",
    )
    parser.add_argument("--continuation", default="auto",
                        choices=["auto", "on", "off"],
                        help="Allow continuation (warm-start) proposals. "
                             "'auto' = on for ts_forecasting, ts_data_pipeline "
                             "and synthetic_data_generator (all persist "
                             "checkpoints), off for synth_regression.")
    parser.add_argument("--shards_per_round", type=int, default=0,
                        help="Pretrain shards assigned per run. 0 = all "
                             "(legacy; continuations then reuse shards). "
                             "Set >0 to leave disjoint headroom so "
                             "continuations train on unseen shards.")
    parser.add_argument("--checkpoint_dir", default="",
                        help="Durable checkpoint store dir for continuation "
                             "warm-starts. Empty = $RADAR_CHECKPOINT_DIR or "
                             "local/checkpoints.")
    parser.add_argument("--continuation_equilibrium", type=float, default=0.7,
                        help="Steady-state fraction of rounds the validator "
                             "schedules as continuations (default 0.70).")
    parser.add_argument("--continuation_warmup_rounds", type=int, default=20,
                        help="Attempted rounds with 0%% continuation before "
                             "the ramp starts (default 20).")
    parser.add_argument("--continuation_step_pct", type=float, default=2.0,
                        help="Percentage points the continuation rate climbs "
                             "per step after warmup (default 2.0).")
    parser.add_argument("--continuation_step_every", type=int, default=3,
                        help="Attempted rounds per ramp step (default 3). "
                             "Defaults give 0→70%% over ~105 rounds post-warmup.")
    parser.add_argument("--continuation_extend_pct", type=float, default=0.5,
                        help="Fraction of synthetic_data_generator continuation "
                             "rounds run as 'extend' (re-train the parent's own "
                             "generator — more compute on identical data) vs "
                             "'modify' (train the miner's new generator). "
                             "Default 0.5.")
    parser.add_argument("--replicate_pct", type=float, default=0.05,
                        help="Fraction of rounds run as validator-only "
                             "replicates (re-train a frontier member's exact "
                             "code, new seed) to measure the eval noise floor. "
                             "Default 0.05; 0 disables.")
    parser.add_argument("--ablate_pct", type=float, default=0.05,
                        help="Fraction of rounds where the miner is asked for "
                             "a minimal one-component diff of a frontier "
                             "member. Default 0.05; 0 disables.")
    parser.add_argument("--recipe_pct", type=float, default=0.05,
                        help="Fraction of rounds where the architecture is "
                             "frozen (AST-enforced) and only the training "
                             "recipe may change. Default 0.05; 0 disables.")
    parser.add_argument("--transfer_pct", type=float, default=0.05,
                        help="Fraction of rounds where a smaller-bucket "
                             "frontier winner is handed to the miner to scale "
                             "into this bucket. Default 0.05; 0 disables.")
    parser.add_argument("--screen_candidates", type=int, default=0,
                        help="Screening tier: max extra candidates an agent "
                             "may submit for cheap triage before the full "
                             "run (ts_forecasting fresh rounds only). "
                             "0 = disabled (default).")
    parser.add_argument("--screen_seconds", type=int, default=600,
                        help="Training budget per screening candidate "
                             "(default 600).")
    parser.add_argument("--screen_eval_tasks", type=int, default=15,
                        help="GIFT-Eval tasks used for screening eval "
                             "(default 15).")
    parser.add_argument("--frozen_pipeline_control_seconds", type=int,
                        default=600,
                        help="ts_forecasting only: paired-control budget for "
                             "frozen-pipeline promotion — the candidate's "
                             "synthetic shards must beat a pure-real run of "
                             "the best arch under this training budget before "
                             "promotion. 0 disables the gate. Default 600.")
    parser.add_argument("--frozen_pipeline_render_per_round",
                        action="store_true",
                        help="Re-render the frozen pipeline's synthetic "
                             "shards each round (fresh generator window) "
                             "instead of reusing the promotion-time corpus.")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [validator] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Reclaim trainer workdirs orphaned by a prior crashed/killed run before
    # we start minting new ones.
    sweep_orphan_workdirs()

    # Optional R2/Hippius backup: restore latest snapshot if the local DB
    # is missing, then run a periodic upload daemon for the lifetime of
    # the validator. No-op unless RADAR_BACKUP_BUCKET is set.
    backup = backup_from_env(args.db)
    if backup is not None:
        backup.restore_if_missing()

    store = LocalStore(args.db)
    if backup is not None:
        backup.start()

    # Resolve the task mixture: --tasks (weighted mix) wins; otherwise fall
    # back to the single --task shorthand. Each mixture entry gets a
    # cached TaskSpec; the round loop picks one per round.
    try:
        mixture = (
            _parse_task_mixture(args.tasks) if args.tasks else
            [(args.task, 1.0)]
        )
    except ValueError as e:
        logger.error("invalid --tasks: %s", e)
        return 2
    specs: dict[str, object] = {}
    for name, _ in mixture:
        spec = make_spec(name)
        if isinstance(spec, (TSForecastingSpec, TSDataPipelineSpec)):
            spec.time_budget_seconds = args.training_seconds
        specs[name] = spec
    needs_ts_caches = any(
        isinstance(specs[name], (TSForecastingSpec, TSDataPipelineSpec,
                                 SyntheticDataGeneratorSpec))
        for name, _ in mixture
    )
    if needs_ts_caches and not _ensure_ts_caches():
        logger.error("ts caches not ready — aborting.")
        return 2

    frozen_arch_store: FrozenArchStore | None = None
    if any(isinstance(specs[name], TSDataPipelineSpec) for name, _ in mixture):
        frozen_arch_store = FrozenArchStore(
            base_dir=args.frozen_arch_dir or None,
        )
        maybe_refresh_frozen(
            frozen_arch_store, store, every_n=args.frozen_arch_refresh_every,
        )
        current = frozen_arch_store.current()
        if current is None:
            logger.warning(
                "ts_data_pipeline: no frozen arch available yet — "
                "early rounds of that task will drop until a ts_forecasting "
                "experiment with both crps + mase lands in the same db.",
            )
        else:
            logger.info(
                "ts_data_pipeline: frozen arch v%d (exp=%d metric=%.6f)",
                current.version, current.source_experiment_id,
                current.source_metric,
            )

    frozen_pipeline_store: FrozenPipelineStore | None = None
    if (any(isinstance(specs[name], TSForecastingSpec) for name, _ in mixture)
            and args.frozen_pipeline_refresh_every > 0):
        frozen_pipeline_store = FrozenPipelineStore(
            base_dir=args.frozen_pipeline_dir or None,
        )
        maybe_refresh_pipeline(
            frozen_pipeline_store, store,
            every_n=args.frozen_pipeline_refresh_every,
            num_shards=args.frozen_pipeline_num_shards,
            batches_per_shard=args.frozen_pipeline_batches_per_shard,
            control_seconds=args.frozen_pipeline_control_seconds,
        )
        current_p = frozen_pipeline_store.current()
        if current_p is None:
            logger.info(
                "ts_forecasting: no frozen pipeline yet — early rounds "
                "run on real shards only until a ts_data_pipeline "
                "experiment with crps+mase lands in this db.",
            )
        else:
            logger.info(
                "ts_forecasting: frozen pipeline v%d (exp=%d shards=%d)",
                current_p.version, current_p.source_experiment_id,
                len(current_p.shard_paths or []),
            )

    if args.continuation == "on":
        continuation_enabled = True
    elif args.continuation == "off":
        continuation_enabled = False
    else:  # auto
        # ts_forecasting, ts_data_pipeline and synthetic_data_generator all
        # persist checkpoints and support continuation; synth_regression doesn't.
        continuation_enabled = any(
            isinstance(specs[name], (TSForecastingSpec, TSDataPipelineSpec,
                                     SyntheticDataGeneratorSpec))
            for name, _ in mixture
        )

    mixture_str = ", ".join(f"{n}:{w:g}" for n, w in mixture)
    logger.info(
        "starting; db=%s tasks=[%s] agent_seconds=%d training_seconds=%d "
        "continuation=%s (eq=%.2f warmup=%d +%.1f%%/%d extend=%.0f%%) "
        "shards_per_round=%d special=[repl=%.0f%% abl=%.0f%% rec=%.0f%% "
        "xfer=%.0f%%] screening=%d",
        args.db, mixture_str, args.agent_seconds, args.training_seconds,
        continuation_enabled, args.continuation_equilibrium,
        args.continuation_warmup_rounds, args.continuation_step_pct,
        args.continuation_step_every, args.continuation_extend_pct * 100,
        args.shards_per_round,
        args.replicate_pct * 100, args.ablate_pct * 100,
        args.recipe_pct * 100, args.transfer_pct * 100,
        args.screen_candidates,
    )

    sink = ArtifactSink.from_env(store)
    ckpt_store = CheckpointStore(
        base_dir=args.checkpoint_dir or None, sink=sink,
    ) if continuation_enabled else None

    wiki_dir = args.wiki_dir or None
    if not wiki_dir:
        # Use the first mixture entry's wiki as the served snapshot. Mixed
        # runs share the wiki dir; per-task swapping isn't worth the
        # complexity for the laptop stack.
        first_task_name = mixture[0][0]
        try:
            from shared.cognition_wiki import ensure_wiki_cached
            cached = ensure_wiki_cached(first_task_name)
        except Exception as e:
            logger.warning("cognition-wiki fetch raised %s — continuing without", e)
            cached = None
        if cached is not None:
            wiki_dir = str(cached)
            logger.info("cognition-wiki: serving %s at /wiki", wiki_dir)

    services = ServicesServer(
        store=store,
        wiki_dir=wiki_dir,
        port=args.services_port,
        sink=sink,
    )
    services_url = services.start()
    logger.info(
        "services: db=%s llm=%s/llm desearch=%s/desearch wiki=%s/wiki",
        services_url, services_url, services_url, services_url,
    )
    # A challenge left 'open' by a dead validator carries that process's
    # ephemeral services port; expire it so no miner burns a round on
    # connection-refused against the stale URL.
    expired = store.expire_open_challenges()
    if expired:
        logger.info("expired %d stale open challenge(s) from a prior run",
                    expired)
    # Self-heal: fill lab reports for any prior experiments that don't have
    # one (e.g. rounds that ran before report generation was wired in), so
    # the read-only dashboard's lab-reports tab isn't permanently empty.
    try:
        from local.lab_reports import backfill_reports
        n_backfilled = backfill_reports(store, narrate=False)
        if n_backfilled:
            logger.info("backfilled %d lab report(s) for prior experiments",
                        n_backfilled)
    except Exception as e:  # noqa: BLE001
        logger.warning("lab report backfill skipped: %s", e)
    if args.services_url_file:
        Path(args.services_url_file).write_text(services_url)

    round_id = _next_round_id(store)
    completed = 0
    try:
        while args.rounds == 0 or completed < args.rounds:
            picked_name = _pick_task_name(round_id, mixture)
            task = specs[picked_name]
            if len(mixture) > 1:
                logger.info("round=%d task=%s (mixture pick)", round_id, picked_name)
            run_round(store, task, round_id,
                      phase_a_seconds=args.phase_a_seconds,
                      services_url=services_url,
                      agent_seconds=args.agent_seconds,
                      sink=sink,
                      ckpt_store=ckpt_store,
                      continuation_enabled=continuation_enabled,
                      continuation_equilibrium=args.continuation_equilibrium,
                      continuation_warmup_rounds=args.continuation_warmup_rounds,
                      continuation_step_pct=args.continuation_step_pct,
                      continuation_step_every=args.continuation_step_every,
                      continuation_extend_pct=args.continuation_extend_pct,
                      shards_per_round=args.shards_per_round,
                      frozen_arch_store=frozen_arch_store,
                      frozen_arch_refresh_every=args.frozen_arch_refresh_every,
                      frozen_pipeline_store=frozen_pipeline_store,
                      frozen_pipeline_refresh_every=args.frozen_pipeline_refresh_every,
                      frozen_pipeline_num_shards=args.frozen_pipeline_num_shards,
                      frozen_pipeline_batches_per_shard=args.frozen_pipeline_batches_per_shard,
                      frozen_pipeline_control_seconds=args.frozen_pipeline_control_seconds,
                      frozen_pipeline_render_per_round=args.frozen_pipeline_render_per_round,
                      replicate_pct=args.replicate_pct,
                      ablate_pct=args.ablate_pct,
                      recipe_pct=args.recipe_pct,
                      transfer_pct=args.transfer_pct,
                      screen_candidates_n=args.screen_candidates,
                      screen_seconds=args.screen_seconds,
                      screen_eval_tasks=args.screen_eval_tasks)
            round_id += 1
            completed += 1
            if args.rounds == 0 or completed < args.rounds:
                time.sleep(args.gap_seconds)
    except KeyboardInterrupt:
        logger.info("interrupted; bye")
    finally:
        services.stop()
        store.close()
        if backup is not None:
            backup.stop(final=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
