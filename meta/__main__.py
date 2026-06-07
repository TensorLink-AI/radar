"""Meta-CLI — ``python -m meta <verb>``.

Verbs:
    submit         queue an experiment from a YAML spec
    status         show registry + per-experiment unit status
    logs           tail a unit's validator log over SSH
    kill           terminate a unit (requires --reason)
    compare        per-unit frontier diff within one experiment
    xcompare       cross-experiment digest (JSON by default)
    export         render report.md for an experiment
    scaffold-miner fork an existing miner agent into miners/ralph_<name>/
    ralph          run the meta-optimizer loop
    daemon         start the orchestrator (scheduler/poller/ingester threads)

JSON-by-default for everything machine-readable; ``--text`` switches to
a human-friendly table where it makes sense. Designed so the Ralph loop
can pipe ``meta xcompare --json`` straight into its context window.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from . import xquery
from .orchestrator import Orchestrator
from .spec import load_spec
from .store import ExperimentStore, Registry, experiment_dir


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("RADAR_META_DATA_ROOT", str(REPO_ROOT / "meta" / "data"))
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="meta")
    p.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="orchestrator data dir (default: meta/data)",
    )
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="verb", required=True)

    sp = sub.add_parser("submit", help="queue an experiment YAML")
    sp.add_argument("spec", type=Path)

    sp = sub.add_parser("status", help="experiment+unit status")
    sp.add_argument("--experiment", default="")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("logs", help="tail a unit's validator log")
    sp.add_argument("--experiment", required=True)
    sp.add_argument("--unit", required=True)
    sp.add_argument("--tail", type=int, default=200)

    sp = sub.add_parser("kill", help="terminate a unit")
    sp.add_argument("--experiment", required=True)
    sp.add_argument("--unit", required=True)
    sp.add_argument("--reason", required=True)

    sp = sub.add_parser("compare", help="per-unit frontier diff")
    sp.add_argument("--experiment", required=True)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("xcompare", help="cross-experiment digest")
    sp.add_argument("--json", action="store_true", default=True)

    sp = sub.add_parser("export", help="render report.md")
    sp.add_argument("--experiment", required=True)
    sp.add_argument("--out", type=Path, default=None)

    sp = sub.add_parser("scaffold-miner", help="fork a miner into miners/ralph_<name>")
    sp.add_argument("--from", dest="src", required=True)
    sp.add_argument("--to", dest="dst", required=True)
    sp.add_argument("--reason", required=True)

    sp = sub.add_parser("ralph", help="run the meta-optimizer loop")
    sp.add_argument("--prompt", type=Path, required=True)
    sp.add_argument("--interval", type=float, default=1800.0)
    sp.add_argument("--max-concurrent-experiments", type=int, default=4)
    sp.add_argument("--total-budget-usd", type=float, default=500.0)
    sp.add_argument("--claude-cli", default="claude")
    sp.add_argument("--once", action="store_true")

    sp = sub.add_parser("daemon", help="run the orchestrator threads")
    sp.add_argument("--orch-id", default="default")
    sp.add_argument("--ssh-key-path", default="")
    sp.add_argument(
        "--backup-bucket",
        default=os.environ.get("RADAR_BACKUP_BUCKET", "radar-backups"),
    )
    sp.add_argument(
        "--backup-prefix",
        default=os.environ.get("RADAR_BACKUP_PREFIX", "radar-backups"),
    )

    args = p.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="[meta] %(asctime)s %(name)s %(levelname)s %(message)s",
    )

    verb = args.verb
    return _DISPATCH[verb](args)


# ── verbs ─────────────────────────────────────────────────────────────────


def _cmd_submit(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    orch = Orchestrator(data_root=args.data_root)
    path = orch.submit(spec)
    orch.stop()
    print(json.dumps({"experiment": spec.name, "path": str(path),
                      "n_units": len(spec.units)}, indent=2))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    reg = Registry(args.data_root / "registry.db")
    try:
        if args.experiment:
            row = reg.get_experiment(args.experiment)
            if row is None:
                print(f"no such experiment: {args.experiment}",
                      file=sys.stderr)
                return 1
            store = ExperimentStore(Path(row["path"]) / "meta.db")
            try:
                units = [dict(u) for u in store.list_units()]
            finally:
                store.close()
            out = {"experiment": dict(row), "units": units}
        else:
            out = {"experiments": [dict(r) for r in reg.list_experiments()]}
    finally:
        reg.close()
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    _print_status_text(out)
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    orch = Orchestrator(data_root=args.data_root)
    try:
        text = orch.tail_unit(args.experiment, args.unit, lines=args.tail)
    finally:
        orch.stop()
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def _cmd_kill(args: argparse.Namespace) -> int:
    orch = Orchestrator(data_root=args.data_root)
    try:
        ok = orch.kill_unit(args.experiment, args.unit, reason=args.reason)
    finally:
        orch.stop()
    print(json.dumps({"killed": ok, "reason": args.reason}))
    return 0 if ok else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    path = experiment_dir(args.data_root, args.experiment) / "meta.db"
    if not path.exists():
        print(f"no meta.db at {path}", file=sys.stderr)
        return 1
    store = ExperimentStore(path)
    try:
        rows = [dict(r) for r in store.best_per_unit()]
    finally:
        store.close()
    if args.json:
        print(json.dumps({"experiment": args.experiment, "rows": rows},
                         indent=2, default=str))
        return 0
    for r in rows:
        print(
            f"  {r['unit_id']:<24} best={r['best_metric']!r:<12} "
            f"n_rows={r['n_rows']:<5} n_success={r['n_success']:<5} "
            f"max_round={r['max_round']}"
        )
    return 0


def _cmd_xcompare(args: argparse.Namespace) -> int:
    print(xquery.dump_digest(args.data_root))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    path = experiment_dir(args.data_root, args.experiment)
    store = ExperimentStore(path / "meta.db")
    try:
        rows = [dict(r) for r in store.best_per_unit()]
        events = [dict(r) for r in store.recent_events(50)]
    finally:
        store.close()
    out = args.out or path / "report.md"
    body = _render_report(args.experiment, rows, events)
    out.write_text(body)
    print(str(out))
    return 0


def _cmd_scaffold_miner(args: argparse.Namespace) -> int:
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not dst.name.startswith("ralph_"):
        print(f"--to must start with 'ralph_': {dst.name}", file=sys.stderr)
        return 2
    if dst.parent != (REPO_ROOT / "miners"):
        print(f"--to must live under miners/: {dst}", file=sys.stderr)
        return 2
    if not src.exists() or not src.is_dir():
        print(f"--from is not a directory: {src}", file=sys.stderr)
        return 2
    if dst.exists():
        print(f"--to already exists, refusing to overwrite: {dst}",
              file=sys.stderr)
        return 2
    shutil.copytree(src, dst)
    # Log to the orchestrator audit trail by writing a top-level event;
    # we don't tie this to a specific experiment, so we use the registry
    # as the audit anchor.
    reg = Registry(args.data_root / "registry.db")
    try:
        with reg.tx() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS scaffold_log("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  ts REAL NOT NULL, src TEXT, dst TEXT, reason TEXT)"
            )
            c.execute(
                "INSERT INTO scaffold_log(ts, src, dst, reason) "
                "VALUES (?, ?, ?, ?)",
                (time.time(), str(src), str(dst), args.reason),
            )
    finally:
        reg.close()
    print(json.dumps({"created": str(dst), "from": str(src)}, indent=2))
    return 0


def _cmd_ralph(args: argparse.Namespace) -> int:
    from . import ralph
    ralph.run_loop(
        data_root=args.data_root,
        prompt_path=args.prompt,
        interval_sec=args.interval,
        max_concurrent=args.max_concurrent_experiments,
        total_budget_usd=args.total_budget_usd,
        claude_cli=args.claude_cli,
        once=args.once,
    )
    return 0


def _cmd_daemon(args: argparse.Namespace) -> int:
    orch = Orchestrator(
        data_root=args.data_root,
        orch_id=args.orch_id,
        ssh_key_path=args.ssh_key_path,
        backup_bucket=args.backup_bucket,
        backup_prefix=args.backup_prefix,
    )
    orch.start()

    def _shutdown(*_):
        print("[meta] shutting down…", flush=True)
        orch.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    print("[meta] orchestrator daemon running; Ctrl-C to stop", flush=True)
    while True:
        time.sleep(60)


_DISPATCH = {
    "submit": _cmd_submit,
    "status": _cmd_status,
    "logs": _cmd_logs,
    "kill": _cmd_kill,
    "compare": _cmd_compare,
    "xcompare": _cmd_xcompare,
    "export": _cmd_export,
    "scaffold-miner": _cmd_scaffold_miner,
    "ralph": _cmd_ralph,
    "daemon": _cmd_daemon,
}


# ── helpers ───────────────────────────────────────────────────────────────


def _print_status_text(out: dict) -> None:
    if "experiments" in out:
        for r in out["experiments"]:
            print(
                f"  {r['name']:<40} {r['status']:<10} "
                f"spent=${float(r['spent_usd'] or 0):.2f} / "
                f"${float(r['budget_usd'] or 0):.2f}"
            )
        return
    e = out["experiment"]
    print(f"experiment: {e['name']}  status={e['status']}  "
          f"spent=${float(e['spent_usd'] or 0):.2f}")
    for u in out["units"]:
        print(
            f"  {u['id']:<24} {u['status']:<12} "
            f"pod={u['pod_id'] or '-':<16} "
            f"${float(u['hourly_usd'] or 0):.2f}/hr  "
            f"round={u['last_round_seen']}"
        )


def _render_report(
    name: str, rows: list[dict], events: list[dict]
) -> str:
    lines = [f"# {name}", "", "## Per-unit best metric", ""]
    if not rows:
        lines.append("_(no frontier rows yet)_")
    for r in rows:
        lines.append(
            f"- **{r['unit_id']}** — best={r['best_metric']!r}, "
            f"success={r['n_success']}/{r['n_rows']}, "
            f"max_round={r['max_round']}"
        )
    lines += ["", "## Recent orchestrator events", ""]
    for e in events[:20]:
        lines.append(
            f"- `{e['kind']}` unit={e['unit_id'] or '-'} "
            f"actor={e['actor']}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
