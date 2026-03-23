"""Phase A: run miner agents and collect architecture proposals.

Validators pull miner Docker images, launch agent pods, send challenges,
collect proposals. Work-split across validators — each runs a subset.
Proposals uploaded to R2 so all validators see all submissions.

Agent stderr (reasoning trace) is captured and stored alongside proposals.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional

from config import Config
from shared.commitment import ImageCommitment, pull_and_verify_image
from shared.protocol import Proposal
from validator.pod_manager import get_mode, launch_agent_pod, run_agent_on_pod

logger = logging.getLogger(__name__)

# Seconds to wait for other validators to upload proposals
_PROPOSAL_SYNC_DELAY = 60


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
       a. pull_and_verify_image()
       b. launch_agent_pod() via affinetes
       c. run_agent_on_pod() with challenge JSON
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

    # Run my assigned agents
    proposals: dict[int, Proposal] = {}
    agent_logs: dict[int, str] = {}
    for uid in my_agent_uids:
        if uid not in commitments:
            continue
        commitment = commitments[uid]
        if not commitment.image_url:
            continue
        try:
            if not pull_and_verify_image(commitment):
                logger.warning("UID %d: image verification failed", uid)
                continue

            agent_env = await launch_agent_pod(
                image_url=commitment.image_url,
                mem_limit="8192Mi",
            )
            try:
                result = await run_agent_on_pod(
                    agent_env, challenge_json,
                    timeout=Config.AGENT_TIMEOUT,
                )
                if result and isinstance(result, dict) and "code" in result:
                    proposal = Proposal(
                        code=result.get("code", ""),
                        name=result.get("name", ""),
                        motivation=result.get("motivation", ""),
                    )
                    proposals[uid] = proposal
                    agent_logs[uid] = result.get("agent_log", "")
                    # Upload proposal + reasoning trace to R2
                    r2.upload_json(
                        f"round_{round_id}/proposals/{uid}.json",
                        {
                            "code": proposal.code,
                            "name": proposal.name,
                            "motivation": proposal.motivation,
                            "agent_log": agent_logs[uid],
                        },
                    )
            finally:
                try:
                    await agent_env.cleanup()
                except Exception:
                    pass
        except Exception as e:
            logger.error("UID %d agent failed: %s", uid, e)

    logger.info(
        "Ran %d agents, got %d proposals", len(my_agent_uids), len(proposals),
    )

    # Wait for other validators to upload their proposals
    if len(validator_uids) > 1:
        await asyncio.sleep(_PROPOSAL_SYNC_DELAY)

    # Read ALL proposals from R2 (including other validators' runs)
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

    # Fallback: if R2 sync missed proposals, run remaining agents locally.
    # This handles the case where the other validator(s) haven't uploaded
    # yet or R2 is unavailable.
    missing_uids = [
        uid for uid in all_uids
        if uid not in proposals and uid not in my_agent_uids
        and uid in commitments and commitments[uid].image_url
    ]
    if missing_uids:
        logger.info(
            "R2 sync missed %d proposals — running agents locally: %s",
            len(missing_uids), missing_uids,
        )
        for uid in missing_uids:
            commitment = commitments[uid]
            try:
                if not pull_and_verify_image(commitment):
                    logger.warning("UID %d: image verification failed (fallback)", uid)
                    continue

                agent_env = await launch_agent_pod(
                    image_url=commitment.image_url,
                    mem_limit="8192Mi",
                )
                try:
                    result = await run_agent_on_pod(
                        agent_env, challenge_json,
                        timeout=Config.AGENT_TIMEOUT,
                    )
                    if result and isinstance(result, dict) and "code" in result:
                        proposal = Proposal(
                            code=result.get("code", ""),
                            name=result.get("name", ""),
                            motivation=result.get("motivation", ""),
                        )
                        proposals[uid] = proposal
                        agent_logs[uid] = result.get("agent_log", "")
                        r2.upload_json(
                            f"round_{round_id}/proposals/{uid}.json",
                            {
                                "code": proposal.code,
                                "name": proposal.name,
                                "motivation": proposal.motivation,
                                "agent_log": agent_logs[uid],
                            },
                        )
                finally:
                    try:
                        await agent_env.cleanup()
                    except Exception:
                        pass
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
