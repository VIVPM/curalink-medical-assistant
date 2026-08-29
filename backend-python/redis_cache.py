"""
Optional Redis cache for embeddings. No-ops gracefully when REDIS_URL is unset
or Redis is unreachable, so the pipeline runs identically without it.

Keyed per (model, text) so overlapping documents across queries skip re-embedding
on the HF Inference API — a real time + cost saver for the same-disease long tail.
"""

from __future__ import annotations

import hashlib
import json
import os

try:
    import redis as _redis
except ImportError:
    _redis = None

REDIS_URL = os.getenv("REDIS_URL")
EMBED_TTL = 7 * 24 * 3600   # 7 days
PROMPT_TTL = 24 * 3600      # 1 day — LLM responses are less stable than embeddings

_client = None
_tried = False


def _get_client():
    global _client, _tried
    if _tried:
        return _client
    _tried = True
    if not REDIS_URL or _redis is None:
        print("[cache] REDIS_URL not set — embedding cache disabled")
        return None
    try:
        _client = _redis.from_url(
            REDIS_URL,
            socket_timeout=2,
            socket_connect_timeout=2,
            decode_responses=True,
        )
        _client.ping()
        print("[cache] Redis connected (embedding cache on)")
    except Exception as e:
        print(f"[cache] Redis unavailable ({e}); embedding cache disabled")
        _client = None
    return _client


def status() -> str:
    """'connected' / 'disabled' / 'error' — for a /health readout."""
    if not REDIS_URL:
        return "disabled"
    client = _get_client()
    if client is None:
        return "error"
    try:
        client.ping()
        return "connected"
    except Exception:
        return "error"


def _key(model: str, text: str) -> str:
    return "emb:" + hashlib.sha256(f"{model}|{text}".encode("utf-8")).hexdigest()


def get_embeddings(model: str, texts: list[str]) -> dict[int, list[float]]:
    """Return {index: vector} for texts already cached. Empty dict on miss/error."""
    client = _get_client()
    if client is None or not texts:
        return {}
    try:
        vals = client.mget([_key(model, t) for t in texts])
    except Exception as e:
        print(f"[cache] mget failed: {e}")
        return {}
    hits: dict[int, list[float]] = {}
    for i, v in enumerate(vals):
        if v:
            try:
                hits[i] = json.loads(v)
            except Exception:
                pass
    return hits


def set_embeddings(model: str, texts: list[str], vectors: list[list[float]]) -> None:
    """Cache vectors. Callers should skip zero/failed vectors."""
    client = _get_client()
    if client is None or not texts:
        return
    try:
        pipe = client.pipeline()
        for t, vec in zip(texts, vectors):
            pipe.set(_key(model, t), json.dumps(vec), ex=EMBED_TTL)
        pipe.execute()
    except Exception as e:
        print(f"[cache] set failed: {e}")


# ---------------------------------------------------------------------------
# Prompt-level LLM cache: hash(model + system_prompt + user_prompt) -> response
# Saves money on repeated LLM calls with identical prompts (e.g. same query
# expansion hitting the same disease+message combo). 24h TTL.
# ---------------------------------------------------------------------------

def _prompt_key(model: str, system_prompt: str, user_prompt: str) -> str:
    blob = f"{model}|{system_prompt}|{user_prompt}"
    return "llm:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get_prompt_cache(model: str, system_prompt: str, user_prompt: str) -> str | None:
    """Return cached LLM response text, or None on miss."""
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get(_prompt_key(model, system_prompt, user_prompt))
    except Exception:
        return None


def set_prompt_cache(model: str, system_prompt: str, user_prompt: str, response: str) -> None:
    """Cache an LLM response. Short responses (< 10 chars) are not cached."""
    if len(response) < 10:
        return
    client = _get_client()
    if client is None:
        return
    try:
        client.set(_prompt_key(model, system_prompt, user_prompt), response, ex=PROMPT_TTL)
    except Exception as e:
        print(f"[cache] prompt set failed: {e}")
