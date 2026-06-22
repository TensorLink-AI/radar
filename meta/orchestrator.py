"""Orchestrator daemon: scheduler, poller, ingester, cost watcher.

Long-running process on the dedicated orchestrator pod. Owns
``meta/data/registry.db`` + one ``meta.db`` per experiment, four
threads, no shared lock — each thread serializes its own writes
through SQLite WAL.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from . import ingest
from .providers import Pod, get_provider
from .providers.base import PodSpecRT, ProviderError
from .spec import ExperimentSpec, UnitSpec, render_run_argv
from .ssh import SSHTarget, http_get_local, put_file, start_tmux, tail_tmux
from .store import ExperimentStore, Registry, experiment_dir


logger = logging.getLogger(__name__)


# Tunables. These are intentionally constants — at this scale the cost
# of making them per-experiment configurable outweighs the benefit.
POLL_INTERVAL_SEC = 60.0
INGEST_INTERVAL_SEC = 300.0
SCHEDULE_INTERVAL_SEC = 10.0
DEAD_AFTER_SEC = 5 * 60.0  # no heartbeat for 5 min → mark failed


def _instance_id(orch_id: str, exp: str, unit: str) -> str:
    """Derive the RADAR_INSTANCE_ID we'll mint for a unit.

    Per the plan: ``orch-<orch_id>-<exp>-<unit>``. The ``orch-`` prefix
    guarantees we never collide with externally-chosen instance IDs
    that someone else might use on the same R2 bucket.
    """
    return f"orch-{orch_id}-{exp}-{unit}"


class Orchestrator:
    def __init__(
        self,
        *,
        data_root: Path,
        orch_id: str = "default",
        ssh_pubkey: str = "",
        ssh_key_path: str = "",
        env_inject: dict[str, str] | None = None,
        backup_bucket: str = "radar-backups",
        backup_prefix: str = "radar-backups",
        repo_url: str = "https://github.com/tensorlink-ai/radar.git",
        repo_branch: str = "main",
    ) -> None:
        self.data_root = Path(data_root)
        self.orch_id = orch_id
        self.ssh_pubkey = ssh_pubkey
        self.ssh_key_path = ssh_key_path
        self.env_inject = dict(env_inject or {})
        self.backup_bucket = backup_bucket
        self.backup_prefix = backup_prefix
        self.repo_url = repo_url
        self.repo_branch = repo_branch

        self.data_root.mkdir(parents=True, exist_ok=True)
        self.registry = Registry(self.data_root / "registry.db")
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # In-memory cache of open ExperimentStores so we don't reopen
        # the connection per tick.
        self._stores: dict[str, ExperimentStore] = {}

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        targets = (
            ("scheduler", self._scheduler_loop),
            ("poller", self._poller_loop),
            ("ingester", self._ingester_loop),
        )
        for name, fn in targets:
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        logger.info("orchestrator started; threads: %s",
                    [t.name for t in self._threads])

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=10)
        for s in self._stores.values():
            s.close()
        self.registry.close()

    # ── submission ─────────────────────────────────────────────────────

    def submit(self, spec: ExperimentSpec) -> Path:
        """Register an experiment + its queued units. Returns the
        experiment dir path."""
        exp_dir = experiment_dir(self.data_root, spec.name)
        (exp_dir / "units").mkdir(parents=True, exist_ok=True)
        (exp_dir / "spec.yaml").write_text(spec.raw or "")
        self.registry.upsert_experiment(
            name=spec.name,
            path=str(exp_dir),
            spec_yaml=spec.raw or "",
            budget_usd=spec.cost.budget_usd,
        )
        store = self._store(spec.name)
        for u in spec.units:
            cfg = spec.merged(u)
            iid = _instance_id(self.orch_id, spec.name, u.id)
            store.insert_unit(u.id, instance_id=iid, cfg=cfg)
            (exp_dir / "units" / u.id).mkdir(parents=True, exist_ok=True)
        store.log_event(
            "submitted", payload={"units": [u.id for u in spec.units]},
        )
        # Stash the spec on the registry row so the scheduler can recover
        # provider/cost details without re-parsing the YAML each tick.
        with self.registry.tx() as c:
            c.execute(
                "UPDATE experiments SET spec_yaml = ? WHERE name = ?",
                (spec.raw, spec.name),
            )
        return exp_dir

    # ── threads ───────────────────────────────────────────────────────

    def _scheduler_loop(self) -> None:
        while not self._stop.wait(SCHEDULE_INTERVAL_SEC):
            try:
                self._scheduler_tick()
            except Exception:
                logger.exception("scheduler tick crashed")

    def _scheduler_tick(self) -> None:
        for exp_row in self.registry.list_experiments():
            if exp_row["status"] not in {"queued", "running"}:
                continue
            spec = self._load_spec_or_none(exp_row["name"])
            if spec is None:
                continue
            store = self._store(exp_row["name"])
            queued = store.list_units(status="queued")
            for u_row in queued:
                self._launch_unit(spec, store, u_row)
            # Promote experiment to 'running' once any unit is up.
            running = store.list_units(status="running")
            if running and exp_row["status"] == "queued":
                self.registry.set_status(exp_row["name"], "running")

    def _launch_unit(
        self, spec: ExperimentSpec, store: ExperimentStore, u_row: Any
    ) -> None:
        unit_id = u_row["id"]
        instance_id = u_row["instance_id"]
        try:
            provider = get_provider(spec.pod.provider)
        except ProviderError as e:
            store.log_event(
                "schedule_error", unit_id=unit_id,
                payload={"error": str(e)},
            )
            store.update_unit(unit_id, status="failed", kill_reason=str(e))
            return
        match = provider.cheapest_matching(
            gpu=spec.pod.gpu, max_usd_per_hour=spec.cost.max_usd_per_hour,
        )
        if match is None:
            msg = (
                f"no {spec.pod.gpu} pod ≤ ${spec.cost.max_usd_per_hour:.2f}/hr"
            )
            store.log_event(
                "schedule_rejected", unit_id=unit_id, payload={"reason": msg},
            )
            store.update_unit(unit_id, status="failed", kill_reason=msg)
            return
        pod_type_id, hourly = match
        env = self._env_for(unit_id, instance_id, spec)
        pod_spec = PodSpecRT(
            gpu=spec.pod.gpu, image=spec.pod.image, disk_gb=spec.pod.disk_gb,
            max_hours=spec.pod.max_hours, pod_type_id=pod_type_id,
            hourly_usd=hourly, extra=dict(spec.pod.extra),
            ssh_pubkey=self.ssh_pubkey, env=env,
        )
        try:
            pod = provider.create(pod_spec)
        except ProviderError as e:
            store.log_event(
                "create_failed", unit_id=unit_id, payload={"error": str(e)},
            )
            store.update_unit(unit_id, status="failed", kill_reason=str(e))
            return
        store.update_unit(
            unit_id,
            status="provisioning",
            provider=spec.pod.provider,
            pod_id=pod.pod_id,
            ssh_host=pod.ssh_host,
            ssh_port=pod.ssh_port,
            ssh_user=pod.ssh_user,
            hourly_usd=hourly,
            started_at=time.time(),
        )
        store.log_event(
            "pod_created", unit_id=unit_id,
            payload={"pod_id": pod.pod_id, "hourly_usd": hourly,
                     "host": f"{pod.ssh_host}:{pod.ssh_port}"},
        )
        # Hand off to the bootstrap; runs in this thread (sync) — we
        # assume create() already blocked on SSH-ready.
        try:
            self._bootstrap_and_launch(spec, store, u_row, pod)
            store.update_unit(unit_id, status="running")
        except Exception as e:
            logger.exception("bootstrap failed for %s", unit_id)
            store.log_event(
                "bootstrap_failed", unit_id=unit_id,
                payload={"error": str(e)[:500]},
            )
            store.update_unit(unit_id, status="failed", kill_reason=str(e))

    def _bootstrap_and_launch(
        self,
        spec: ExperimentSpec,
        store: ExperimentStore,
        u_row: Any,
        pod: Pod,
    ) -> None:
        target = SSHTarget(
            host=pod.ssh_host, port=pod.ssh_port, user=pod.ssh_user,
            key_path=self.ssh_key_path,
        )
        cfg = json.loads(u_row["cfg_json"])
        env = self._env_for(u_row["id"], u_row["instance_id"], spec)
        # Drop the env into a .env file that the validator will source.
        env_text = "\n".join(f"{k}={v}" for k, v in env.items()) + "\n"
        put_file(target, "/root/radar.env", env_text)
        # Repo prep is image-dependent; the canonical orchestrator image
        # ships radar pre-installed at /opt/radar. As a fallback we
        # clone it. Both paths leave /opt/radar as cwd.
        bootstrap = (
            "set -e; "
            "if [ ! -d /opt/radar/local ]; then "
            f"  git clone --depth=1 -b {self.repo_branch} {self.repo_url} /opt/radar; "
            "  cd /opt/radar && pip install -e .[ts_forecasting,miners,meta] || pip install -e .; "
            "fi; "
            "mkdir -p /opt/radar/local"
        )
        store.log_event("bootstrap_start", unit_id=u_row["id"])
        from .ssh import run  # local import keeps the module import graph thin
        run(target, bootstrap, timeout=900)
        argv = render_run_argv(cfg, db_path="local/radar_local.db")
        argv = ["bash", "-lc",
                "set -a; source /root/radar.env; set +a; " + " ".join(argv)]
        start_tmux(
            target, session="radar", cwd="/opt/radar", argv=argv,
        )
        store.log_event(
            "validator_started", unit_id=u_row["id"],
            payload={"argv": argv[-1][:500]},
        )

    def _env_for(
        self, unit_id: str, instance_id: str, spec: ExperimentSpec
    ) -> dict[str, str]:
        env = dict(self.env_inject)
        env.update({
            "RADAR_INSTANCE_ID": instance_id,
            "RADAR_BACKUP_BUCKET": self.backup_bucket,
            "RADAR_BACKUP_PREFIX": self.backup_prefix,
            "PYTHONUNBUFFERED": "1",
        })
        return env

    # ── poller ────────────────────────────────────────────────────────

    def _poller_loop(self) -> None:
        while not self._stop.wait(POLL_INTERVAL_SEC):
            try:
                self._poller_tick()
            except Exception:
                logger.exception("poller tick crashed")

    def _poller_tick(self) -> None:
        now = time.time()
        for exp_row in self.registry.list_experiments():
            if exp_row["status"] != "running":
                continue
            store = self._store(exp_row["name"])
            for u in store.list_units(status="running"):
                self._poll_unit(store, u, exp_row, now)

    def _poll_unit(
        self, store: ExperimentStore, u: Any, exp_row: Any, now: float
    ) -> None:
        target = SSHTarget(
            host=u["ssh_host"], port=int(u["ssh_port"]),
            user=u["ssh_user"], key_path=self.ssh_key_path,
        )
        res = http_get_local(target, path="/api/stats", timeout=10)
        if res.rc != 0:
            # No heartbeat. Mark failed only if we've been quiet too long.
            last = u["last_heartbeat_ts"] or u["started_at"] or now
            if now - last > DEAD_AFTER_SEC:
                self._kill_unit(store, u, reason="no_heartbeat")
            return
        try:
            stats = json.loads(res.stdout)
        except json.JSONDecodeError:
            return
        store.update_unit(
            u["id"],
            last_heartbeat_ts=now,
            last_round_seen=int(stats.get("max_round", -1) or -1),
        )
        # Cost watch: bump pod_hours and check budget.
        prev_hb = u["last_heartbeat_ts"] or u["started_at"] or now
        elapsed_hr = max(0.0, (now - prev_hb) / 3600.0)
        delta_cost = elapsed_hr * float(u["hourly_usd"] or 0.0)
        if delta_cost > 0:
            new_hours = float(u["pod_hours"] or 0.0) + elapsed_hr
            new_cost = float(u["cost_usd"] or 0.0) + delta_cost
            store.update_unit(
                u["id"], pod_hours=new_hours, cost_usd=new_cost,
            )
            spent = self.registry.add_spend(exp_row["name"], delta_cost)
            if spent > float(exp_row["budget_usd"] or 0.0) > 0:
                store.log_event(
                    "budget_exceeded", unit_id=u["id"],
                    payload={"spent_usd": spent,
                             "budget_usd": exp_row["budget_usd"]},
                )
                self._kill_unit(store, u, reason="budget_exceeded")

    def _kill_unit(
        self, store: ExperimentStore, u: Any, *, reason: str
    ) -> None:
        try:
            provider = get_provider(u["provider"])
            provider.terminate(u["pod_id"])
        except Exception as e:
            logger.warning("terminate failed for %s: %s", u["pod_id"], e)
        store.update_unit(
            u["id"], status="failed" if "fail" in reason else "done",
            finished_at=time.time(), kill_reason=reason,
        )
        store.log_event("killed", unit_id=u["id"], payload={"reason": reason})

    # ── ingester ───────────────────────────────────────────────────────

    def _ingester_loop(self) -> None:
        while not self._stop.wait(INGEST_INTERVAL_SEC):
            try:
                self._ingester_tick()
            except Exception:
                logger.exception("ingester tick crashed")

    def _ingester_tick(self) -> None:
        for exp_row in self.registry.list_experiments():
            if exp_row["status"] not in {"running", "done"}:
                continue
            store = self._store(exp_row["name"])
            exp_dir = Path(exp_row["path"])
            for u in store.list_units():
                if u["status"] in {"queued", "failed"}:
                    continue
                snap_dir = exp_dir / "units" / u["id"]
                snap_dir.mkdir(parents=True, exist_ok=True)
                try:
                    ingest.ingest_unit(
                        store, unit_id=u["id"],
                        instance_id=u["instance_id"],
                        snapshot_dir=snap_dir,
                        prefix=self.backup_prefix,
                        bucket=self.backup_bucket,
                    )
                except Exception as e:
                    logger.warning(
                        "ingest %s/%s failed: %s",
                        exp_row["name"], u["id"], e,
                    )

    # ── helpers ───────────────────────────────────────────────────────

    def _store(self, exp_name: str) -> ExperimentStore:
        if exp_name not in self._stores:
            path = experiment_dir(self.data_root, exp_name) / "meta.db"
            self._stores[exp_name] = ExperimentStore(path)
        return self._stores[exp_name]

    def _load_spec_or_none(self, name: str) -> ExperimentSpec | None:
        row = self.registry.get_experiment(name)
        if row is None or not row["spec_yaml"]:
            return None
        from .spec import parse_spec, _yaml_load
        try:
            data = _yaml_load(row["spec_yaml"])
            return parse_spec(data, raw=row["spec_yaml"])
        except Exception:
            logger.exception("cannot reload spec for %s", name)
            return None

    # ── one-off: tail a unit's log on demand ──────────────────────────

    def tail_unit(self, exp: str, unit_id: str, lines: int = 200) -> str:
        store = self._store(exp)
        u = store.get_unit(unit_id)
        if u is None:
            return f"<no such unit {unit_id}>"
        target = SSHTarget(
            host=u["ssh_host"], port=int(u["ssh_port"]),
            user=u["ssh_user"], key_path=self.ssh_key_path,
        )
        res = tail_tmux(target, session="radar", lines=lines)
        return res.stdout or res.stderr

    def kill_unit(self, exp: str, unit_id: str, *, reason: str) -> bool:
        store = self._store(exp)
        u = store.get_unit(unit_id)
        if u is None or u["status"] not in {"running", "provisioning"}:
            return False
        self._kill_unit(store, u, reason=reason)
        return True
