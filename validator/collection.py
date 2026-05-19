"""Phase A: run miner agents and collect architecture proposals.

Validators fetch miner agent code from the DB server, inject it into
the OFFICIAL agent image, send challenges, collect proposals.
Work-split across validators — each runs a subset.
Proposals uploaded to R2 so all validators see all submissions.

Each agent run produces three trust tiers of metadata, all collected here:

* Self-reported (stderr ``trace``, ``reasoning``, ``tool_calls``):
  written by the miner's agent code; trivially fakeable.
* Observed (``agent_behavior``): pod wall-clock + per-category proxy
  call counts measured by the validator outside the pod.
* Verified: produced later in Phase C (results, score).

The combined ``agent_meta`` per UID is uploaded to R2 alongside the
Proposal so other validators see the same picture.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Optional

from config import Config
from shared.agent_code import compute_code_hash, validate_bundle
from shared.artifacts import generate_scratchpad_urls
from shared.commitment import ImageCommitment
from shared.db_client import DatabaseClient
from shared.protocol import Proposal
from shared.url_gate import parse_allowed_urls
from validator.pod_manager import (
    cleanup_agent_env, launch_agent_pod, pre_validate_agent_code,
    run_agent_on_pod,
)

logger = logging.getLogger(__name__)

# Polling parameters for R2 proposal sync
_PROPOSAL_SYNC_MAX_WAIT = 120  # max seconds to wait for other validators
_PROPOSAL_SYNC_POLL_INTERVAL = 15  # seconds between polls

# Storage hygiene: agents that emit megabytes of stderr or reasoning would
# bloat the experiments table without producing extra signal. These caps
# are intentionally generous (a typical reasoning trace is < 64 KB).
_TRACE_MAX_BYTES = 256 * 1024
_REASONING_MAX_BYTES = 256 * 1024
_TOOL_CALLS_MAX = 256


def _truncate_text(value: str, limit: int) -> str:
    """Cap a string at ``limit`` UTF-8 bytes, preserving valid UTF-8."""
    if not value:
        return ""
    raw = value.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return value
    truncated = raw[:limit].decode("utf-8", errors="ignore")
    suffix = f"\n... [truncated {len(raw) - limit} bytes]"
    return truncated + suffix


def _normalise_tool_calls(value) -> list:
    """Coerce a self-reported tool_calls payload to a bounded list of dicts."""
    if not isinstance(value, list):
        return []
    out: list = []
    for entry in value[:_TOOL_CALLS_MAX]:
        if isinstance(entry, dict):
            out.append(entry)
        else:
            # Stringify scalars so consumers always see a uniform shape.
            out.append({"value": str(entry)})
    return out


def _empty_meta() -> dict:
    """Per-UID metadata bag used when no real run produced anything."""
    return {
        "agent_log": "",
        "reasoning": "",
        "tool_calls": [],
        "agent_behavior": {},
    }


def _snapshot_agent_behavior(
    uid: int, started_at: float, exit_status: str,
) -> dict:
    """Build the validator-observed ``agent_behavior`` dict.

    Pulls per-agent proxy counters from ``validator.db_proxy`` (None if
    the proxy module isn't initialised, e.g. in unit tests). Adds the
    pod wall-clock and exit status the validator measured itself.
    """
    proxy_metrics: dict = {}
    try:
        from validator import db_proxy as _db_proxy
        proxy_metrics = _db_proxy.get_agent_behavior(uid)
    except Exception:
        proxy_metrics = {}
    return {
        "wall_clock_s": round(max(0.0, time.monotonic() - started_at), 3),
        "exit_status": exit_status,
        "proxy": proxy_metrics,
    }


def _attach_scratchpad_urls(challenge_json: str, r2, hotkey: str) -> str:
    """Inject per-miner scratchpad presigned URLs into a challenge JSON string."""
    if not Config.SCRATCHPAD_ENABLED:
        return challenge_json
    try:
        sp_get, sp_put = generate_scratchpad_urls(
            r2, hotkey, ttl=Config.SCRATCHPAD_TTL,
        )
        data = json.loads(challenge_json)
        data["scratchpad_get_url"] = sp_get
        data["scratchpad_put_url"] = sp_put
        data["scratchpad_max_mb"] = Config.SCRATCHPAD_MAX_MB
        return json.dumps(data)
    except Exception as e:
        logger.warning("Scratchpad URL generation failed for %s: %s", hotkey, e)
        return challenge_json


def _build_allowed_urls(challenge_json: str) -> str:
    """Build the allowed URL prefix string for agent pods.

    Combines the static config allowlist with dynamic URLs derived
    from the challenge (validator proxy, desearch proxy, cognition wiki).
    The wiki URL is a per-object presigned URL, so the full URL is added
    rather than a base prefix; harness.py uses the same trick for
    scratchpad URLs.
    """
    prefixes: list[str] = parse_allowed_urls(Config.AGENT_ALLOWED_URLS)
    try:
        data = json.loads(challenge_json)
        for key in ("db_url", "desearch_url", "llm_url"):
            url = data.get(key, "")
            if url:
                base = url.rstrip("/") + "/"
                if base not in prefixes:
                    prefixes.append(base)
        wiki_url = data.get("cognition_wiki_url", "")
        if wiki_url and not any(wiki_url.startswith(p) for p in prefixes):
            prefixes.append(wiki_url)
    except (json.JSONDecodeError, TypeError):
        pass
    return ",".join(prefixes)


async def _fetch_agent_bundle(
    db_client: DatabaseClient,
    commitment: ImageCommitment,
) -> Optional[dict]:
    """Fetch a miner's agent code bundle from the DB server.

    Verifies the code_hash against the on-chain commitment.
    Returns the validated bundle dict or None.
    """
    if not commitment.hotkey:
        return None

    bundle = await db_client.get_agent_code(commitment.hotkey)
    if not bundle or "files" not in bundle:
        logger.warning(
            "No agent code found for %s (uid=%d)",
            commitment.hotkey[:16], commitment.miner_uid,
        )
        return None

    # Validate bundle structure
    ok, err = validate_bundle(bundle)
    if not ok:
        logger.warning(
            "UID %d agent bundle validation failed: %s",
            commitment.miner_uid, err,
        )
        return None

    # Verify code hash matches on-chain commitment
    if commitment.code_hash:
        actual_hash = compute_code_hash(bundle["files"])
        if actual_hash != commitment.code_hash:
            logger.warning(
                "UID %d code hash mismatch: on-chain=%s actual=%s",
                commitment.miner_uid,
                commitment.code_hash[:24],
                actual_hash[:24],
            )
            return None

    return bundle


async def _run_single_agent(
    uid: int,
    commitment: ImageCommitment,
    challenge_json: str,
    r2,
    round_id: int,
    allowed_urls: str,
    bundle: dict,
) -> tuple[Optional[Proposal], dict]:
    """Run a single miner agent.

    Returns ``(proposal, meta)``. ``meta`` is a dict with keys
    ``agent_log``, ``reasoning``, ``tool_calls``, ``agent_behavior``;
    populated even when the proposal is None so the caller can record
    why the run failed.
    """
    miner_challenge_json = _attach_scratchpad_urls(
        challenge_json, r2, commitment.hotkey,
    )
    # Inject miner identity so agent pod requests get per-miner rate buckets
    try:
        _data = json.loads(miner_challenge_json)
        _data["miner_uid"] = uid
        _data["miner_hotkey"] = commitment.hotkey
        miner_challenge_json = json.dumps(_data)
    except (json.JSONDecodeError, TypeError):
        pass

    # Per-round agent budget comes from the challenge (populated from the
    # task YAML's agent_seconds, falling back to Config.AGENT_TIMEOUT).
    try:
        agent_timeout = int(json.loads(miner_challenge_json).get(
            "agent_seconds", Config.AGENT_TIMEOUT,
        ))
    except (json.JSONDecodeError, TypeError, ValueError):
        agent_timeout = Config.AGENT_TIMEOUT

    # Wrap launch + run in a single try/finally so a failure inside
    # launch_agent_pod (or anywhere before cleanup) still triggers
    # teardown. Each Basilica retry inside run_agent_on_pod can spawn a
    # NEW deployment; cleanup_agent_env force-deletes every name we
    # recorded, not just the latest, which prevents orphan accumulation
    # on HTTP-error paths.
    agent_env = None
    started_at = time.monotonic()
    try:
        agent_env = await launch_agent_pod(
            image_url=Config.OFFICIAL_AGENT_IMAGE,
            mem_limit="8192Mi",
            agent_code=bundle,
            allowed_urls=allowed_urls,
        )
        result = await run_agent_on_pod(
            agent_env, miner_challenge_json,
            timeout=agent_timeout,
            allowed_urls=allowed_urls,
            task_id=uid,
        )
        if result and isinstance(result, dict) and "code" in result:
            proposal = Proposal(
                code=result.get("code", ""),
                name=result.get("name", ""),
                motivation=result.get("motivation", ""),
                reasoning=_truncate_text(
                    result.get("reasoning", "") or "", _REASONING_MAX_BYTES,
                ),
                tool_calls=_normalise_tool_calls(result.get("tool_calls", [])),
                prompt_id=str(result.get("prompt_id", "") or "")[:64],
            )
            agent_log = _truncate_text(
                result.get("agent_log", "") or "", _TRACE_MAX_BYTES,
            )
            agent_behavior = _snapshot_agent_behavior(uid, started_at, "ok")
            meta = {
                "agent_log": agent_log,
                "reasoning": proposal.reasoning,
                "tool_calls": proposal.tool_calls,
                "agent_behavior": agent_behavior,
                "prompt_id": proposal.prompt_id,
            }
            r2.upload_json(
                f"round_{round_id}/proposals/{uid}.json",
                {
                    "code": proposal.code,
                    "name": proposal.name,
                    "motivation": proposal.motivation,
                    "reasoning": proposal.reasoning,
                    "tool_calls": proposal.tool_calls,
                    "prompt_id": proposal.prompt_id,
                    "agent_log": agent_log,
                    "agent_behavior": agent_behavior,
                },
            )
            return proposal, meta
        else:
            error_msg = ""
            result_keys = ""
            if isinstance(result, dict):
                error_msg = result.get("error", "")
                stderr = result.get("stderr", "")
                if stderr:
                    error_msg = f"{error_msg} | stderr: {stderr[:500]}"
                result_keys = ", ".join(result.keys())
            logger.warning(
                "UID %d proposal rejected (no code returned): %s | "
                "result_type=%s result_keys=[%s]",
                uid, error_msg or repr(result)[:200],
                type(result).__name__, result_keys,
            )
    finally:
        await cleanup_agent_env(agent_env)
    return None, _empty_meta()


async def _run_agents_concurrently(
    *,
    uids: list[int],
    commitments: dict[int, ImageCommitment],
    bundles: dict[int, dict],
    challenge_json: str,
    r2,
    round_id: int,
    allowed_urls: str,
    concurrency: int,
) -> dict[int, tuple[Optional[Proposal], dict]]:
    """Run multiple agents concurrently with a bounded semaphore.

    Returns {uid: (proposal_or_None, meta)}. Per-agent exceptions are
    logged and produce ``(None, _empty_meta())`` so one bad agent never
    aborts the batch.
    """
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _bounded(uid: int) -> tuple[int, Optional[Proposal], dict]:
        async with sem:
            try:
                proposal, meta = await _run_single_agent(
                    uid, commitments[uid], challenge_json, r2, round_id,
                    allowed_urls, bundles[uid],
                )
                return uid, proposal, meta
            except Exception as e:
                logger.error("UID %d agent failed: %s", uid, e)
                return uid, None, _empty_meta()

    logger.info(
        "Running %d agents concurrently (cap=%d)",
        len(uids), max(1, int(concurrency)),
    )
    raw = await asyncio.gather(*[_bounded(uid) for uid in uids])
    return {uid: (proposal, meta) for uid, proposal, meta in raw}


async def run_and_collect_agents(
    wallet,
    metagraph,
    challenge_json: str,
    round_id: int,
    seed: int,
    r2,
    my_uid: int,
    validator_uids: list[int],
    commitments: dict[int, ImageCommitment],
    get_my_assignments_fn,
    db_client: Optional[DatabaseClient] = None,
) -> tuple[dict[int, Proposal], dict[int, dict]]:
    """Phase A: run assigned miner agents, upload proposals to R2, read all.

    1. Work-split: get_my_assignments() for which agents this validator runs
    2. For each assigned miner:
       a. Fetch agent code bundle from DB server
       b. Verify code_hash against on-chain commitment
       c. launch_agent_pod() with official image + code injection
       d. run_agent_on_pod() with challenge JSON + URL allowlist
       e. Upload proposal + per-agent metadata to R2:
          round_{id}/proposals/{uid}.json
    3. Wait for other validators to upload
    4. Read all proposals from R2
    5. Dedup by code hash
    6. Return ``({uid: Proposal}, {uid: meta})`` where ``meta`` is
       ``{"agent_log", "reasoning", "tool_calls", "agent_behavior"}``.
    """
    all_uids = list(commitments.keys())
    my_agent_uids = get_my_assignments_fn(
        all_uids, validator_uids, my_uid, seed,
    )

    # Build URL allowlist once for all agents this round
    allowed_urls = _build_allowed_urls(challenge_json)

    # Build DB client if not provided
    if db_client is None:
        db_client = DatabaseClient(Config.DB_API_URL, wallet)

    # Pre-fetch all agent bundles for my assigned miners (concurrently).
    async def _prefetch(uid: int) -> tuple[int, Optional[dict]]:
        try:
            b = await _fetch_agent_bundle(db_client, commitments[uid])
            return uid, b
        except Exception as e:
            logger.error("UID %d failed to fetch agent code: %s", uid, e)
            return uid, None

    fetch_uids = [uid for uid in my_agent_uids if uid in commitments]
    fetched = await asyncio.gather(*[_prefetch(uid) for uid in fetch_uids])
    bundles: dict[int, dict] = {uid: b for uid, b in fetched if b}

    # Run my assigned agents concurrently (one task per agent, capped by a
    # semaphore). Each agent has its own pod on a separate Basilica node,
    # so no local resource contention; the cap throttles orchestration /
    # R2 fan-out and prevents one slow agent from serialising the round.
    proposals: dict[int, Proposal] = {}
    agent_meta: dict[int, dict] = {}

    runnable_uids = [uid for uid in my_agent_uids if uid in bundles]
    if runnable_uids:
        results = await _run_agents_concurrently(
            uids=runnable_uids,
            commitments=commitments,
            bundles=bundles,
            challenge_json=challenge_json,
            r2=r2,
            round_id=round_id,
            allowed_urls=allowed_urls,
            concurrency=Config.AGENT_CONCURRENCY,
        )
        for uid, (proposal, meta) in results.items():
            if proposal:
                proposals[uid] = proposal
                agent_meta[uid] = meta

    logger.info(
        "Ran %d agents, got %d proposals", len(my_agent_uids), len(proposals),
    )

    # Poll R2 for other validators' proposals
    expected_from_others = {
        uid for uid in all_uids
        if uid not in proposals and uid in commitments
    }

    if expected_from_others and len(validator_uids) > 1:
        import time as _time
        deadline = _time.monotonic() + _PROPOSAL_SYNC_MAX_WAIT
        while expected_from_others and _time.monotonic() < deadline:
            await asyncio.sleep(_PROPOSAL_SYNC_POLL_INTERVAL)
            still_missing = set()
            for uid in expected_from_others:
                try:
                    data = r2.download_json(
                        f"round_{round_id}/proposals/{uid}.json",
                    )
                    if data and "code" in data:
                        proposals[uid] = Proposal(
                            code=data["code"],
                            name=data.get("name", ""),
                            motivation=data.get("motivation", ""),
                            reasoning=_truncate_text(
                                data.get("reasoning", "") or "",
                                _REASONING_MAX_BYTES,
                            ),
                            tool_calls=_normalise_tool_calls(
                                data.get("tool_calls", []),
                            ),
                            prompt_id=str(
                                data.get("prompt_id", "") or ""
                            )[:64],
                        )
                        agent_meta[uid] = {
                            "agent_log": _truncate_text(
                                data.get("agent_log", "") or "",
                                _TRACE_MAX_BYTES,
                            ),
                            "reasoning": proposals[uid].reasoning,
                            "tool_calls": proposals[uid].tool_calls,
                            "prompt_id": proposals[uid].prompt_id,
                            "agent_behavior": data.get("agent_behavior", {})
                                if isinstance(data.get("agent_behavior"), dict)
                                else {},
                        }
                    else:
                        still_missing.add(uid)
                except Exception:
                    still_missing.add(uid)
            expected_from_others = still_missing
            if not expected_from_others:
                break
            logger.info(
                "R2 sync: still waiting for %d proposals (%ds remaining)",
                len(expected_from_others),
                int(deadline - _time.monotonic()),
            )
    else:
        for uid in all_uids:
            if uid in proposals:
                continue
            try:
                data = r2.download_json(
                    f"round_{round_id}/proposals/{uid}.json",
                )
                if data and "code" in data:
                    proposals[uid] = Proposal(
                        code=data["code"],
                        name=data.get("name", ""),
                        motivation=data.get("motivation", ""),
                        reasoning=_truncate_text(
                            data.get("reasoning", "") or "",
                            _REASONING_MAX_BYTES,
                        ),
                        tool_calls=_normalise_tool_calls(
                            data.get("tool_calls", []),
                        ),
                    )
                    agent_meta[uid] = {
                        "agent_log": _truncate_text(
                            data.get("agent_log", "") or "",
                            _TRACE_MAX_BYTES,
                        ),
                        "reasoning": proposals[uid].reasoning,
                        "tool_calls": proposals[uid].tool_calls,
                        "agent_behavior": data.get("agent_behavior", {})
                            if isinstance(data.get("agent_behavior"), dict)
                            else {},
                    }
            except Exception:
                pass

    # Fallback: run agents locally for UIDs still missing after polling
    missing_uids = [
        uid for uid in all_uids
        if uid not in proposals and uid not in my_agent_uids
        and uid in commitments
    ]
    if missing_uids:
        logger.info(
            "R2 sync missed %d proposals — fetching code + running locally: %s",
            len(missing_uids), missing_uids,
        )
        # Fetch all fallback bundles concurrently
        async def _fetch(uid: int) -> tuple[int, Optional[dict]]:
            try:
                b = await _fetch_agent_bundle(db_client, commitments[uid])
                return uid, b
            except Exception as e:
                logger.error("UID %d failed to fetch agent code (fallback): %s", uid, e)
                return uid, None

        fetched = await asyncio.gather(*[_fetch(uid) for uid in missing_uids])
        fallback_bundles = {uid: b for uid, b in fetched if b}

        if fallback_bundles:
            results = await _run_agents_concurrently(
                uids=list(fallback_bundles.keys()),
                commitments=commitments,
                bundles=fallback_bundles,
                challenge_json=challenge_json,
                r2=r2,
                round_id=round_id,
                allowed_urls=allowed_urls,
                concurrency=Config.AGENT_CONCURRENCY,
            )
            for uid, (proposal, meta) in results.items():
                if proposal:
                    proposals[uid] = proposal
                    agent_meta[uid] = meta

    # Dedup by code hash
    seen_hashes: dict[str, int] = {}  # hash -> first uid that claimed it
    deduped: dict[int, Proposal] = {}
    deduped_meta: dict[int, dict] = {}
    for uid, p in sorted(proposals.items()):
        h = hashlib.sha256(p.code.strip().encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes[h] = uid
            deduped[uid] = p
            deduped_meta[uid] = agent_meta.get(uid, _empty_meta())
        else:
            logger.warning(
                "UID %d proposal rejected (duplicate): code hash %s "
                "already submitted by UID %d | name=%r code_len=%d",
                uid, h[:16], seen_hashes[h], p.name[:80], len(p.code),
            )

    dup_count = len(proposals) - len(deduped)
    logger.info(
        "Phase A complete: %d total proposals, %d after dedup (%d duplicates removed)",
        len(proposals), len(deduped), dup_count,
    )
    return deduped, deduped_meta
