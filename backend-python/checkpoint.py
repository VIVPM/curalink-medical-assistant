"""
Per-step pipeline checkpointing in Redis.

Saves each stage's output so a retry can resume from the last checkpoint
instead of re-running the whole pipeline (~15s and 3 LLM calls saved).
No-ops gracefully when Redis is unavailable.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_CHECKPOINT_TTL = 600  # 10 min — checkpoints are transient

STAGES = (
    "query_expansion", "retrieval", "normalization",
    "ranking", "context_build", "llm", "assembly",
)


def save(job_id: str, stage: str, data) -> None:
    try:
        from redis_cache import _get_client
        client = _get_client()
        if client:
            client.set(
                f"ckpt:{job_id}:{stage}",
                json.dumps(data, default=str),
                ex=_CHECKPOINT_TTL,
            )
    except Exception as e:
        logger.debug("[checkpoint] save %s:%s failed: %s", job_id, stage, e)


def load(job_id: str, stage: str):
    try:
        from redis_cache import _get_client
        client = _get_client()
        if client:
            raw = client.get(f"ckpt:{job_id}:{stage}")
            if raw:
                logger.info("[checkpoint] resuming %s from stage %s", job_id, stage)
                return json.loads(raw)
    except Exception as e:
        logger.debug("[checkpoint] load %s:%s failed: %s", job_id, stage, e)
    return None


def clear(job_id: str) -> None:
    try:
        from redis_cache import _get_client
        client = _get_client()
        if client:
            pipe = client.pipeline()
            for stage in STAGES:
                pipe.delete(f"ckpt:{job_id}:{stage}")
            pipe.execute()
    except Exception:
        pass
