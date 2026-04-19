"""Phase A: run miner agents and collect architecture proposals.

Validators fetch miner agent code from the DB server, inject it into
the OFFICIAL agent image, send challenges, collect proposals.
Work-split across validators — each runs a subset.
Proposals uploaded to R2 so all validators see all submissions.

Agent stderr (reasoning trace) is captured and stored alongside proposals.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Optional

from config import Config
from shared.agent_code import compute_code_hash, validate_bundle
from shared.artifacts import generate_scratchpad_urls
from shared.commitment import ImageCommitment
from shared.db_client import DatabaseClient
from shared.protocol import Proposal
from shared.url_gate import parse_allowed_urls
from validator.pod_manager import (
    launch_agent_pod, pre_validate_agent_code, run_agent_on_pod,
)

logger = logging.getLogger(__name__)

# Polling parameters for R2 proposal sync
_PROPOSAL_SYNC_MAX_WAIT = 120  # max seconds to wait for other validators
_PROPOSAL_SYNC_POLL_INTERVAL = 15  # seconds between polls


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
    from the challenge (validator proxy, desearch proxy).
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
) -> tuple[Optional[Proposal], str]:
    """Run a single miner agent and return (proposal, agent_log) or (None, "")."""
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

    agent_env = await launch_agent_pod(
        image_url=Config.OFFICIAL_AGENT_IMAGE,
        mem_limit="8192Mi",
        agent_code=bundle,
        allowed_urls=allowed_urls,
    )

    try:
        result = await run_agent_on_pod(
            agent_env, miner_challenge_json,
            timeout=agent_timeout,
            allowed_urls=allowed_urls,
        )
        if result and isinstance(result, dict) and "code" in result:
            proposal = Proposal(
                code=result.get("code", ""),
                name=result.get("name", ""),
                motivation=result.get("motivation", ""),
            )
            agent_log = result.get("agent_log", "")
            r2.upload_json(
                f"round_{round_id}/proposals/{uid}.json",
                {
                    "code": proposal.code,
                    "name": proposal.name,
                    "motivation": proposal.motivation,
                    "agent_log": agent_log,
                },
            )
            return proposal, agent_log
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
        try:
            await agent_env.cleanup()
        except Exception:
            pass
    return None, ""


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
) -> dict[int, tuple[Optional[Proposal], str]]:
    """Run multiple agents concurrently with a bounded semaphore.

    Returns {uid: (proposal_or_None, agent_log)}. Per-agent exceptions
    are logged and produce (None, "") so one bad agent never aborts the
    batch.
    """
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _bounded(uid: int) -> tuple[int, Optional[Proposal], str]:
        async with sem:
            try:
                proposal, agent_log = await _run_single_agent(
                    uid, commitments[uid], challenge_json, r2, round_id,
                    allowed_urls, bundles[uid],
                )
                return uid, proposal, agent_log
            except Exception as e:
                logger.error("UID %d agent failed: %s", uid, e)
                return uid, None, ""

    logger.info(
        "Running %d agents concurrently (cap=%d)",
        len(uids), max(1, int(concurrency)),
    )
    raw = await asyncio.gather(*[_bounded(uid) for uid in uids])
    return {uid: (proposal, log) for uid, proposal, log in raw}


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
) -> tuple[dict[int, Proposal], dict[int, str]]:
    """Phase A: run assigned miner agents, upload proposals to R2, read all.

    1. Work-split: get_my_assignments() for which agents this validator runs
    2. For each assigned miner:
       a. Fetch agent code bundle from DB server
       b. Verify code_hash against on-chain commitment
       c. launch_agent_pod() with official image + code injection
       d. run_agent_on_pod() with challenge JSON + URL allowlist
       e. Upload proposal + agent_log to R2: round_{id}/proposals/{uid}.json
    3. Wait for other validators to upload
    4. Read all proposals from R2
    5. Dedup by code hash
    6. Return ({uid: Proposal}, {uid: agent_log})
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
    agent_logs: dict[int, str] = {}

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
        for uid, (proposal, agent_log) in results.items():
            if proposal:
                proposals[uid] = proposal
                agent_logs[uid] = agent_log

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
                        )
                        agent_logs[uid] = data.get("agent_log", "")
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
                    )
                    agent_logs[uid] = data.get("agent_log", "")
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
            for uid, (proposal, agent_log) in results.items():
                if proposal:
                    proposals[uid] = proposal
                    agent_logs[uid] = agent_log

    # Dedup by code hash
    seen_hashes: dict[str, int] = {}  # hash -> first uid that claimed it
    deduped: dict[int, Proposal] = {}
    deduped_logs: dict[int, str] = {}
    for uid, p in sorted(proposals.items()):
        h = hashlib.sha256(p.code.strip().encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes[h] = uid
            deduped[uid] = p
            deduped_logs[uid] = agent_logs.get(uid, "")
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
    return deduped, deduped_logs
