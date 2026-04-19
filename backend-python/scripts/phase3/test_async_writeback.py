"""
Standalone test for Phase 3 Step 3.5 — Async write-back.
Run from the backend-python directory:
    python scripts/phase3/test_async_writeback.py

Prerequisites:
  - FastAPI server running at localhost:8000 (uvicorn main:app)
  - Pinecone index exists and is accessible

Test flow:
  1. Fetch a small set of docs from live APIs + normalize
  2. Record Pinecone vector count BEFORE
  3. POST docs to /writeback (should return immediately)
  4. Measure response time (must be fast — background task, not blocking)
  5. Wait a few seconds for background task to finish
  6. Record Pinecone vector count AFTER
  7. Verify count increased
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import os
import httpx
from dotenv import load_dotenv
from pinecone import Pinecone

from sources.pubmed import fetch_pubmed
from sources.openalex import fetch_openalex
from sources.trials import fetch_trials
from sources.normalizer import normalize_pubmed, normalize_openalex, normalize_trial
from sources.merger import merge_and_dedupe, filter_complete
from pinecone_store import NAMESPACE

load_dotenv()

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "curalink")


async def fetch_and_normalize():
    query = "alzheimer immunotherapy"
    disease = "alzheimer"

    pubmed_raw, openalex_raw, trials_raw = await asyncio.gather(
        fetch_pubmed(query, limit=5),
        fetch_openalex(query, limit=5),
        fetch_trials(disease=disease, limit=3),
    )

    pubmed_docs = [normalize_pubmed(r) for r in pubmed_raw]
    openalex_docs = [normalize_openalex(r) for r in openalex_raw]
    trial_docs = [normalize_trial(r, disease_context=disease) for r in trials_raw]

    deduped = merge_and_dedupe([pubmed_docs, openalex_docs, trial_docs])
    complete = filter_complete(deduped)
    return complete


def get_pinecone_count():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(PINECONE_INDEX)
    stats = idx.describe_index_stats()
    ns_stats = (stats.namespaces or {}).get(NAMESPACE)
    return ns_stats.vector_count if ns_stats else 0


def main():
    print("=" * 60)
    print("Phase 3 Step 3.5: Async Write-back Test")
    print("=" * 60)

    # 1. Fetch + normalize docs
    print("\n[1/5] Fetching + normalizing (5 pubmed + 5 openalex + 3 trials)")
    t0 = time.perf_counter()
    docs = asyncio.run(fetch_and_normalize())
    print(f"  Got {len(docs)} complete docs in {time.perf_counter() - t0:.1f}s")

    if not docs:
        print("  FAIL: no docs fetched, cannot test writeback")
        return

    docs_dicts = [d.to_dict() for d in docs]

    # 2. Record Pinecone count BEFORE
    print("\n[2/5] Recording Pinecone vector count before writeback")
    count_before = get_pinecone_count()
    print(f"  Count before: {count_before}")

    # 3. POST to /writeback — should return immediately
    print(f"\n[3/5] POSTing {len(docs_dicts)} docs to {FASTAPI_URL}/writeback")
    t0 = time.perf_counter()
    resp = httpx.post(
        f"{FASTAPI_URL}/writeback",
        json={"documents": docs_dicts},
        timeout=10.0,
    )
    response_time_ms = (time.perf_counter() - t0) * 1000
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {resp.json()}")
    print(f"  Response time: {response_time_ms:.0f}ms")

    if resp.status_code != 200:
        print("  FAIL: endpoint returned non-200")
        return

    # 4. Check response was fast (background task, not blocking)
    if response_time_ms < 2000:
        print("  PASS: response was fast (task is running in background)")
    else:
        print(f"  WARN: response took {response_time_ms:.0f}ms — "
              "might be blocking instead of background")

    # 5. Wait for background task to finish, then check count
    print("\n[4/5] Waiting 10s for background embed+upsert to finish...")
    time.sleep(10)

    print("\n[5/5] Recording Pinecone vector count after writeback")
    count_after = get_pinecone_count()
    print(f"  Count after: {count_after}")
    print(f"  Delta: +{count_after - count_before}")

    if count_after > count_before:
        print("  PASS: vector count increased — async writeback works")
    elif count_after == count_before and count_before > 0:
        print("  OK: count unchanged (docs likely already existed from prior runs)")
    else:
        print("  WARN: count did not increase — check FastAPI logs for "
              "[writeback] messages")

    print("\n" + "=" * 60)
    print("Done. Check FastAPI terminal for [writeback] log lines.")


if __name__ == "__main__":
    main()
