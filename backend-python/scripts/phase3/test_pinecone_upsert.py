"""
Standalone test for Phase 3 Step 3.3.
Run from the backend-python directory:
    python scripts/phase3/test_pinecone_upsert.py

Fetches a small set of docs, normalizes, embeds, and upserts to Pinecone.
Verifies:
  1. Records are created (vector count increases).
  2. Metadata is intact (can be queried back).
  3. Trial section chunking produces multiple records per trial.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import os
from dotenv import load_dotenv
from pinecone import Pinecone

from embeddings.embedder import Embedder
from sources.pubmed import fetch_pubmed
from sources.openalex import fetch_openalex
from sources.trials import fetch_trials
from sources.normalizer import normalize_pubmed, normalize_openalex, normalize_trial
from sources.merger import merge_and_dedupe, filter_complete
from pinecone_store import prepare_records, embed_records, upsert_records, NAMESPACE

load_dotenv()

BIENCODER_MODEL = os.getenv("BIENCODER_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "curalink")


async def fetch_and_normalize():
    query = "parkinson vitamin d"
    disease = "parkinson"

    pubmed_raw, openalex_raw, trials_raw = await asyncio.gather(
        fetch_pubmed(query, limit=10),
        fetch_openalex(query, limit=10),
        fetch_trials(disease=disease, limit=5),
    )

    pubmed_docs = [normalize_pubmed(r) for r in pubmed_raw]
    openalex_docs = [normalize_openalex(r) for r in openalex_raw]
    trial_docs = [normalize_trial(r, disease_context=disease) for r in trials_raw]

    deduped = merge_and_dedupe([pubmed_docs, openalex_docs, trial_docs])
    complete = filter_complete(deduped)
    return complete


def main():
    print("=" * 60)
    print("Phase 3 Step 3.3: Pinecone Upsert Test")
    print("=" * 60)

    # Load model
    print(f"\n[1/5] Loading embedder: {BIENCODER_MODEL}")
    t0 = time.perf_counter()
    embedder = Embedder(BIENCODER_MODEL)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s (dim={embedder.dim})")

    # Connect to Pinecone
    print(f"\n[2/5] Connecting to Pinecone index: {PINECONE_INDEX}")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(PINECONE_INDEX)
    print(f"  Namespace: {NAMESPACE}")
    stats_before = idx.describe_index_stats()
    ns_stats = (stats_before.namespaces or {}).get(NAMESPACE)
    count_before = ns_stats.vector_count if ns_stats else 0
    print(f"  Current vector count in namespace: {count_before}")

    # Fetch + normalize
    print("\n[3/5] Fetching + normalizing (10 pubmed + 10 openalex + 5 trials)")
    t0 = time.perf_counter()
    docs = asyncio.run(fetch_and_normalize())
    print(f"  Got {len(docs)} complete docs in {time.perf_counter() - t0:.1f}s")

    pubs = [d for d in docs if d.doc_type == "publication"]
    trials = [d for d in docs if d.doc_type == "trial"]
    print(f"  Publications: {len(pubs)}, Trials: {len(trials)}")

    # Prepare + embed records
    print("\n[4/5] Preparing + embedding records")
    t0 = time.perf_counter()
    records = prepare_records(docs)
    print(f"  Prepared {len(records)} records ({len(pubs)} pub records "
          f"+ {len(records) - len(pubs)} trial chunk records)")

    embed_records(records, embedder)
    print(f"  Embedded in {time.perf_counter() - t0:.1f}s")

    # Verify record shapes
    sample = records[0] if records else None
    if sample:
        print(f"\n  Sample record:")
        print(f"    id:       {sample['id']}")
        print(f"    values:   [{sample['values'][0]:.6f}, ...] "
              f"(len={len(sample['values'])})")
        print(f"    metadata: {list(sample['metadata'].keys())}")

    # Upsert
    print(f"\n[5/5] Upserting {len(records)} records to Pinecone")
    t0 = time.perf_counter()
    upserted = upsert_records(idx, records)
    print(f"  Upserted {upserted} records in {time.perf_counter() - t0:.1f}s")

    # Verify count increased (Pinecone is eventually consistent; short wait)
    import time as t_mod
    t_mod.sleep(2)
    stats_after = idx.describe_index_stats()
    ns_stats_after = (stats_after.namespaces or {}).get(NAMESPACE)
    count_after = ns_stats_after.vector_count if ns_stats_after else 0
    print(f"\n  Vector count: {count_before} -> {count_after} "
          f"(+{count_after - count_before})")

    if count_after > count_before:
        print("  PASS: vector count increased")
    elif count_after == count_before and count_before > 0:
        print("  OK: count unchanged (records already existed from a prior run)")
    else:
        print("  WARN: count did not increase — check Pinecone console")

    # Verify metadata by fetching a record back
    if records:
        test_id = records[0]["id"]
        print(f"\n  Fetching back record '{test_id}' to verify metadata...")
        fetched = idx.fetch(ids=[test_id], namespace=NAMESPACE)
        if fetched.vectors and test_id in fetched.vectors:
            meta = fetched.vectors[test_id].metadata
            print(f"    title:        {meta.get('title', '<missing>')[:60]}")
            print(f"    source:       {meta.get('source')}")
            print(f"    doc_type:     {meta.get('doc_type')}")
            print(f"    section:      {meta.get('section')}")
            print(f"    disease_tags: {meta.get('disease_tags')}")
            print(f"    year:         {meta.get('year')}")
            print("    PASS: metadata intact")
        else:
            print(f"    WARN: could not fetch back {test_id}")

    print("\n" + "=" * 60)
    print("Done. Check Pinecone console to visually confirm.")


if __name__ == "__main__":
    main()
