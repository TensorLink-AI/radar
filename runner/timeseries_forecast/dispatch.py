"""Per-task dispatch extras for ts_forecasting.

Produces the forecasting-specific keys (`gift_eval_urls`,
`pretrain_shard_urls`, `pretrain_val_shard_urls`) that get merged into
the trainer dispatch payload. Other tasks supply their own
`dispatch.build_dispatch_extras` or omit the module entirely.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.r2_audit import R2AuditLog
    from shared.task import TaskSpec

logger = logging.getLogger(__name__)


def build_dispatch_extras(
    task: "TaskSpec",
    *,
    gift_r2: "R2AuditLog | None",
    pretrain_r2: "R2AuditLog | None",
    seed: int,
    shards_per_round: int,
    r2_prefixes: dict[str, str],
) -> dict:
    """Generate presigned R2 URLs for the ts_forecasting trainer.

    Returns a dict whose keys are merged into the trainer payload:
      - gift_eval_urls: dict[str, str]   (GIFT-Eval presigned GETs)
      - pretrain_shard_urls: list[str]   (training shards)
      - pretrain_val_shard_urls: list[str] (deterministic val shard)

    Keys are omitted when their data source is unavailable so the
    trainer sees exactly the same payload shape as the pre-refactor
    code path.
    """
    extras: dict = {}

    if gift_r2 is not None:
        try:
            from shared.gift_eval import GiftEvalBenchmark
            gift = GiftEvalBenchmark(
                r2=gift_r2,
                r2_prefix=r2_prefixes.get("gift", ""),
            )
            urls = gift.generate_presigned_get_urls()
            if urls:
                extras["gift_eval_urls"] = urls
        except Exception as e:
            logger.warning("Failed to generate GIFT-Eval presigned URLs: %s", e)

    if pretrain_r2 is not None:
        try:
            from shared.pretrain_data import PretrainBenchmark
            pretrain = PretrainBenchmark(
                r2=pretrain_r2,
                r2_prefix=r2_prefixes.get("pretrain", ""),
            )
            shard_keys = pretrain.select_shards(seed=seed, n=shards_per_round)
            train_urls = pretrain.generate_presigned_shard_urls(shard_keys)
            if train_urls:
                extras["pretrain_shard_urls"] = train_urls
            val_shard_keys = pretrain.get_val_shard_keys()
            if val_shard_keys:
                val_urls = pretrain.generate_presigned_shard_urls(val_shard_keys)
                if val_urls:
                    extras["pretrain_val_shard_urls"] = val_urls
        except Exception as e:
            logger.warning("Failed to generate pretrain shard URLs: %s", e)

    return extras
