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

    # Pre-fetch all agent bundles for my assigned miners
    bundles: dict[int, dict] = {}
    for uid in my_agent_uids:
        if uid not in commitments:
            continue
        commitment = commitments[uid]
        try:
            bundle = await _fetch_agent_bundle(db_client, commitment)
            if bundle:
                bundles[uid] = bundle
        except Exception as e:
            logger.error("UID %d failed to fetch agent code: %s", uid, e)

    # Run my assigned agents
    proposals: dict[int, Proposal] = {}
    agent_logs: dict[int, str] = {}
    for uid in my_agent_uids:
        if uid not in bundles:
            continue
        commitment = commitments[uid]
        try:
            proposal, agent_log = await _run_single_agent(
                uid, commitment, challenge_json, r2, round_id,
                allowed_urls, bundles[uid],
            )
            if proposal:
                proposals[uid] = proposal
                agent_logs[uid] = agent_log
        except Exception as e:
            logger.error("UID %d agent failed: %s", uid, e)

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
        for uid in missing_uids:
            commitment = commitments[uid]
            try:
                bundle = await _fetch_agent_bundle(db_client, commitment)
                if not bundle:
                    continue
                proposal, agent_log = await _run_single_agent(
                    uid, commitment, challenge_json, r2, round_id,
                    allowed_urls, bundle,
                )
                if proposal:
                    proposals[uid] = proposal
                    agent_logs[uid] = agent_log
            except Exception as e:
                logger.error("UID %d agent failed (fallback): %s", uid, e)

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
