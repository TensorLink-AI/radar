"""Phase A: run miner agents and collect architecture proposals.

Validators launch the OFFICIAL agent image with miner code injected,
send challenges, collect proposals.  Work-split across validators —
each runs a subset.  Proposals uploaded to R2 so all validators see
all submissions.

Agent stderr (reasoning trace) is captured and stored alongside proposals.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Optional

from config import Config
from shared.artifacts import generate_scratchpad_urls
from shared.commitment import ImageCommitment, pull_and_verify_image
from shared.protocol import Proposal
from shared.url_gate import parse_allowed_urls
from validator.pod_manager import (
    get_mode, launch_agent_pod, pre_validate_agent_code, run_agent_on_pod,
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

    # Add validator-provided URLs from the challenge
    try:
        data = json.loads(challenge_json)
        for key in ("db_url", "desearch_url"):
            url = data.get(key, "")
            if url:
                # Allow the base URL (e.g. "http://localhost:8080/")
                base = url.rstrip("/") + "/"
                if base not in prefixes:
                    prefixes.append(base)
    except (json.JSONDecodeError, TypeError):
        pass

    return ",".join(prefixes)


def _fetch_agent_code(commitment: ImageCommitment, r2) -> Optional[str]:
    """Retrieve the miner's agent code from R2 or commitment.

    Miners commit their agent code to R2 at:
        agents/{hotkey}/agent.py

    Falls back to pulling the miner's Docker image (legacy path) if
    no agent code is available.

    Returns agent code string, or None if unavailable.
    """
    if not commitment.hotkey:
        return None
    try:
        data = r2.download_json(f"agents/{commitment.hotkey}/agent.json")
        if data and "code" in data:
            return data["code"]
    except Exception:
        pass
    return None


async def _run_single_agent(
    uid: int,
    commitment: ImageCommitment,
    challenge_json: str,
    r2,
    round_id: int,
    allowed_urls: str,
) -> tuple[Optional[Proposal], str]:
    """Run a single miner agent and return (proposal, agent_log) or (None, "")."""
    # Try to get miner's agent code for sandboxed execution
    agent_code = _fetch_agent_code(commitment, r2)

    miner_challenge_json = _attach_scratchpad_urls(
        challenge_json, r2, commitment.hotkey,
    )

    if agent_code is not None:
        # ── Secure path: official image + miner code injection ───
        ok, err = pre_validate_agent_code(agent_code)
        if not ok:
            logger.warning("UID %d: agent code validation failed: %s", uid, err)
            return None, ""

        agent_env = await launch_agent_pod(
            image_url=Config.OFFICIAL_AGENT_IMAGE,
            mem_limit="8192Mi",
            agent_code=agent_code,
            allowed_urls=allowed_urls,
        )
    else:
        # ── Legacy path: miner's Docker image (will be deprecated) ─
        if not commitment.image_url:
            return None, ""
        if not pull_and_verify_image(commitment):
            logger.warning("UID %d: image verification failed", uid)
            return None, ""

        agent_env = await launch_agent_pod(
            image_url=commitment.image_url,
            mem_limit="8192Mi",
        )

    try:
        result = await run_agent_on_pod(
            agent_env, miner_challenge_json,
            timeout=Config.AGENT_TIMEOUT,
            allowed_urls=allowed_urls if agent_code is not None else "",
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
) -> tuple[dict[int, Proposal], dict[int, str]]:
    """Phase A: run assigned miner agents, upload proposals to R2, read all.

    1. Work-split: get_my_assignments() for which agents this validator runs
    2. For each assigned miner:
       a. Fetch agent code from R2 (or fall back to Docker image)
       b. launch_agent_pod() with official image + code injection
       c. run_agent_on_pod() with challenge JSON + URL allowlist
       d. Upload proposal + agent_log to R2: round_{id}/proposals/{uid}.json
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

    # Run my assigned agents
    proposals: dict[int, Proposal] = {}
    agent_logs: dict[int, str] = {}
    for uid in my_agent_uids:
        if uid not in commitments:
            continue
        commitment = commitments[uid]
        try:
            proposal, agent_log = await _run_single_agent(
                uid, commitment, challenge_json, r2, round_id, allowed_urls,
            )
            if proposal:
                proposals[uid] = proposal
                agent_logs[uid] = agent_log
        except Exception as e:
            logger.error("UID %d agent failed: %s", uid, e)

    logger.info(
        "Ran %d agents, got %d proposals", len(my_agent_uids), len(proposals),
    )

    # Poll R2 for other validators' proposals (replaces fixed sleep)
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
        # Single validator or no expected proposals — just try once
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
            "R2 sync missed %d proposals — running agents locally: %s",
            len(missing_uids), missing_uids,
        )
        for uid in missing_uids:
            commitment = commitments[uid]
            try:
                proposal, agent_log = await _run_single_agent(
                    uid, commitment, challenge_json, r2, round_id, allowed_urls,
                )
                if proposal:
                    proposals[uid] = proposal
                    agent_logs[uid] = agent_log
            except Exception as e:
                logger.error("UID %d agent failed (fallback): %s", uid, e)

    # Dedup by code hash
    seen_hashes: set[str] = set()
    deduped: dict[int, Proposal] = {}
    deduped_logs: dict[int, str] = {}
    for uid, p in sorted(proposals.items()):
        h = hashlib.sha256(p.code.strip().encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped[uid] = p
            deduped_logs[uid] = agent_logs.get(uid, "")

    logger.info(
        "Phase A complete: %d total proposals, %d after dedup",
        len(proposals), len(deduped),
    )
    return deduped, deduped_logs
