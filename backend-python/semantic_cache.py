"""
Semantic query cache (SCALE-1). For a FIRST-TURN question, embed it and cosine-
match against recent questions in the same tenant|disease|intent|location bucket; on a close hit
(>= threshold) we return that question's cached response and skip the ENTIRE
pipeline — external APIs, ranking, and the LLM.

Uses the existing HF embedder + Redis (no vector DB needed: buckets are small, so
we brute-force cosine in-process). No-ops when Redis is unavailable.

Safety: the threshold is deliberately high (0.97) and this only applies to
first-turn questions, so a near-identical rephrase hits but a genuinely different
question does not — important for a medical tool that must not answer question A
with question B's cached result.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from redis_cache import _get_client

SIM_THRESHOLD = 0.97
BUCKET_MAX = 50
BUCKET_TTL = 24 * 3600


def _bucket(tenant: str, disease: str, intent: str, location: str) -> str:
    t = (tenant or "").strip().lower()
    context = "|".join(
        value.strip().lower() for value in (disease or "", intent or "", location or "")
    )
    digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
    return f"semq:{t}:{digest}"


def lookup(embedder, tenant: str, disease: str, intent: str, location: str, user_message: str):
    """
    Return (response|None, embedding|None). Embeds the message ONCE (an HF call)
    only when the cache is enabled, so the caller can reuse the embedding for
    store() on a miss. When Redis is off, returns (None, None) — no embed cost.
    """
    client = _get_client()
    if client is None:
        return None, None

    emb = embedder.embed_text(user_message)
    try:
        raw = client.lrange(_bucket(tenant, disease, intent, location), 0, BUCKET_MAX - 1)
    except Exception as e:
        print(f"[semcache] lrange failed: {e}")
        return None, emb

    q = np.asarray(emb, dtype=np.float32)
    best_resp = None
    best_sim = -1.0
    for item in raw:
        try:
            entry = json.loads(item)
            sim = float(np.dot(q, np.asarray(entry["emb"], dtype=np.float32)))
            if sim > best_sim:
                best_sim = sim
                best_resp = entry["resp"]
        except Exception:
            continue

    if best_resp is not None and best_sim >= SIM_THRESHOLD:
        return best_resp, emb
    return None, emb


def store(tenant: str, disease: str, intent: str, location: str, embedding, response: dict) -> None:
    """Cache a response inside one user's de-identified context bucket."""
    client = _get_client()
    if client is None or embedding is None:
        return
    try:
        entry = json.dumps({"emb": embedding, "resp": response})
        key = _bucket(tenant, disease, intent, location)
        pipe = client.pipeline()
        pipe.lpush(key, entry)
        pipe.ltrim(key, 0, BUCKET_MAX - 1)
        pipe.expire(key, BUCKET_TTL)
        pipe.execute()
    except Exception as e:
        print(f"[semcache] store failed: {e}")
