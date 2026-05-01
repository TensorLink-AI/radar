"""Standalone verifier for RADAR experiments.

Fetch one experiment from a RADAR API, then for each substrate CID it claims
to be backed by, download the Hippius bundle, sr25519-verify the signature,
and compare the bundle's metric record with what the API returned. Self-
contained on purpose: depends only on ``shared.substrate`` and
``shared.hippius_client`` so anyone can fork it to verify a RADAR experiment
without RADAR repo access beyond the public API and an IPFS gateway.

    python -m tools.verify_radar_experiment \\
        --api-url https://radar.example.com --experiment-id 12345
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any, Optional

import httpx

# shared.substrate / shared.hippius_client are imported lazily inside the
# functions that use them: both transitively import ``bittensor``, which
# constructs an ArgumentParser at module-load time and consumes sys.argv —
# loading them here would short-circuit our own --help handling.


_GREEN, _RED, _YELLOW, _BOLD, _RESET = (
    "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[1m", "\x1b[0m",
)


def _c(s: str, c: str) -> str:
    return f"{c}{s}{_RESET}" if sys.stdout.isatty() else s


async def _fetch_experiment(api_url: str, experiment_id: int) -> dict:
    """GET /experiments/{id} and return the parsed JSON object."""
    url = f"{api_url.rstrip('/')}/experiments/{experiment_id}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        raise ValueError(f"unexpected response shape: {type(body).__name__}")
    return body


def _scalars(d: Any) -> dict:
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items()
            if isinstance(v, (bool, int, float, str)) or v is None}


def _diff(record, api_results: dict, api_uid: int, api_hk: str) -> list[str]:
    """Human-readable strings describing record/API disagreements."""
    out: list[str] = []
    rec, db = _scalars(record.metrics), _scalars(api_results)
    for k, v in rec.items():
        if k not in db:
            out.append(f"metric {k!r} present in record, missing in API")
        elif db[k] != v:
            out.append(f"metric {k!r}: record={v!r} api={db[k]!r}")
    if record.miner_uid != api_uid:
        out.append(f"miner_uid: record={record.miner_uid} api={api_uid}")
    if api_hk and record.miner_hotkey and record.miner_hotkey != api_hk:
        out.append(f"miner_hotkey: record={record.miner_hotkey!r} api={api_hk!r}")
    return out


async def _verify_one(
    hippius, cid_entry: dict, api_round_id: int,
    api_uid: int, api_hk: str, api_results: dict,
) -> dict:
    """Verify one (cid, validator_hotkey) entry against the API claim."""
    from shared.substrate import records_from_bundle, verify_record
    cid = cid_entry.get("cid", "")
    expected_hotkey = cid_entry.get("validator_hotkey", "") or None
    out: dict[str, Any] = {
        "cid": cid, "validator_hotkey": expected_hotkey or "",
        "fetchable": False, "signature_valid": False,
        "matches": False, "discrepancies": [],
    }
    if not cid:
        out["discrepancies"].append("cid is empty")
        return out
    try:
        data = await hippius.download_bundle(cid)
        out["fetchable"] = True
    except Exception as e:
        out["discrepancies"].append(f"fetch failed: {e}")
        return out
    try:
        records = records_from_bundle(data)
    except Exception as e:
        out["discrepancies"].append(f"bundle parse failed: {e}")
        return out
    record = next(
        (r for r in records
         if r.round_id == api_round_id and r.miner_uid == api_uid),
        None,
    )
    if record is None:
        out["discrepancies"].append(
            f"no record matching round_id={api_round_id} miner_uid={api_uid}"
        )
        return out
    ok, err = verify_record(record, expected_hotkey=expected_hotkey)
    out["signature_valid"] = ok
    if not ok:
        out["discrepancies"].append(f"signature: {err}")
    diffs = _diff(record, api_results, api_uid, api_hk)
    out["discrepancies"].extend(diffs)
    out["matches"] = ok and not diffs
    return out


def _print_report(experiment_id: int, results: list[dict]) -> int:
    """Render results as a report; return 0 on full pass, 1 on any failure."""
    print(_c(f"\n=== RADAR experiment #{experiment_id} verification ===", _BOLD))
    all_pass = True
    for v in results:
        tag = _c("PASS", _GREEN) if v["matches"] else _c("FAIL", _RED)
        print(f"\n[{tag}] CID: {v['cid'] or '(missing)'}")
        print(f"  validator_hotkey: {v['validator_hotkey'] or '(unset)'}")
        print(f"  fetchable:        {v['fetchable']}")
        print(f"  signature_valid:  {v['signature_valid']}")
        for d in v["discrepancies"]:
            print(_c(f"  - {d}", _YELLOW))
        if not v["matches"]:
            all_pass = False
    if all_pass:
        print(_c("\nAll substrate records verified.", _GREEN))
        return 0
    print(_c("\nVerification failed. See details above.", _RED))
    return 1


async def _run(args: argparse.Namespace) -> int:
    try:
        experiment = await _fetch_experiment(args.api_url, args.experiment_id)
    except Exception as e:
        print(_c(f"API unreachable: {e}", _RED), file=sys.stderr)
        return 3
    cids = experiment.get("substrate_cids", [])
    if not isinstance(cids, list) or not cids:
        print(_c(
            f"Experiment #{args.experiment_id} has no substrate_cids "
            "— nothing to verify.", _YELLOW,
        ))
        return 2
    api_round = int(experiment.get("round_id", -1))
    api_uid = int(experiment.get("miner_uid", -1))
    api_hk = str(experiment.get("miner_hotkey", "") or "")
    api_res = experiment.get("results") or {}
    from shared.hippius_client import HippiusClient
    hippius = HippiusClient(ipfs_api_url=args.hippius_ipfs)
    try:
        results = await asyncio.gather(*[
            _verify_one(hippius, c, api_round, api_uid, api_hk, api_res)
            for c in cids if isinstance(c, dict)
        ])
    finally:
        await hippius.close()
    return _print_report(args.experiment_id, results)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="verify_radar_experiment",
        description="Independently verify a RADAR experiment against Hippius.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0  every substrate record verified and matched the API\n"
            "  1  verification failed (signature invalid or metric mismatch)\n"
            "  2  experiment has no substrate CIDs (nothing to verify)\n"
            "  3  API unreachable"
        ),
    )
    p.add_argument("--api-url", required=True,
                   help="RADAR API base URL, e.g. https://radar.example.com")
    p.add_argument("--experiment-id", type=int, required=True,
                   help="Experiment ID to verify")
    p.add_argument("--hippius-ipfs", default="",
                   help="Hippius IPFS gateway URL. Default: public gateway.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    return asyncio.run(_run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
