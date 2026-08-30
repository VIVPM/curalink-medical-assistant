"""
SSE event buffering in Redis for replay on reconnect.

Worker writes events to a Redis list keyed by job_id regardless of whether
a listener is connected. Client reconnects with Last-Event-ID and replays
from that point forward — no events lost on disconnect.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_BUFFER_TTL = 600  # 10 min


def buffer_event(job_id: str, event_id: int, event_type: str, data: str) -> None:
    try:
        from redis_cache import _get_client
        client = _get_client()
        if client:
            entry = json.dumps({"id": event_id, "event": event_type, "data": data})
            client.rpush(f"sse:{job_id}", entry)
            client.expire(f"sse:{job_id}", _BUFFER_TTL)
    except Exception as e:
        logger.debug("[event_buffer] write failed: %s", e)


def replay_events(job_id: str, last_event_id: int | None = None) -> list[dict]:
    try:
        from redis_cache import _get_client
        client = _get_client()
        if client:
            raw = client.lrange(f"sse:{job_id}", 0, -1)
            events = [json.loads(r) for r in raw]
            if last_event_id is not None:
                for i, e in enumerate(events):
                    if e["id"] == last_event_id:
                        return events[i + 1:]
                return events  # ID not found — replay all
            return events
    except Exception as e:
        logger.debug("[event_buffer] replay failed: %s", e)
    return []


def format_sse(event_id: int, event_type: str, data: str) -> str:
    """Format a single SSE event with an id for replay."""
    return f"id: {event_id}\nevent: {event_type}\ndata: {data}\n\n"
