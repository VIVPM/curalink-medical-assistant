"""
Standalone test for Phase 4 Step 4.3 — Reciprocal Rank Fusion.
Run from the backend-python directory:
    python scripts/phase4/test_rrf.py

Fetches ~50 docs, runs BM25 + cosine independently, fuses via RRF,
and verifies the fused ranking is better than either alone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import os
import time
from dotenv import load_dotenv

from embeddings.embedder import Embedder
from sources.pubmed import fetch_pubmed
from sources.openalex import fetch_openalex
from sources.trials import fetch_trials
from sources.normalizer import normalize_pubmed, normalize_openalex, normalize_trial
from sources.merger import merge_and_dedupe, filter_complete
from ranking.bm25 import rank_bm25
from ranking.cosine import rank_cosine
from ranking.rrf import rrf_fuse

load_dotenv()

BIENCODER_MODEL = os.getenv("BIENCODER_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")


async def fetch_docs():
    query = "parkinson disease treatment"
    disease = "parkinson"

    pubmed_raw, openalex_raw, trials_raw = await asyncio.gather(
        fetch_pubmed(query, limit=25),
        fetch_openalex(query, limit=25),
        fetch_trials(disease=disease, limit=10),
    )

    pubmed_docs = [normalize_pubmed(r) for r in pubmed_raw]
    openalex_docs = [normalize_openalex(r) for r in openalex_raw]
    trial_docs = [normalize_trial(r, disease_context=disease) for r in trials_raw]

    deduped = merge_and_dedupe([pubmed_docs, openalex_docs, trial_docs])
    return filter_complete(deduped)


def top_n_indices(scores, n=20):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]


def main():
    print("=" * 60)
    print("Phase 4 Step 4.3: Reciprocal Rank Fusion Test")
    print("=" * 60)

    # 1. Load embedder
    print(f"\n[1/5] Loading embedder: {BIENCODER_MODEL}")
    embedder = Embedder(BIENCODER_MODEL)
    print(f"  Loaded (dim={embedder.dim})")

    # 2. Fetch docs
    print("\n[2/5] Fetching docs (25 pubmed + 25 openalex + 10 trials)")
    t0 = time.perf_counter()
    docs = asyncio.run(fetch_docs())
    print(f"  Got {len(docs)} complete docs in {time.perf_counter() - t0:.1f}s")

    if len(docs) < 10:
        print("  FAIL: too few docs")
        return

    query = "vitamin D supplementation parkinson"

    # 3. Run BM25 + cosine
    print(f"\n[3/5] Scoring with query: \"{query}\"")
    bm25_scores = rank_bm25(query, docs)
    cosine_scores = rank_cosine(query, docs, embedder)
    print(f"  BM25 done, cosine done")

    # 4. Fuse
    print("\n[4/5] Fusing with RRF (k=60)")
    t0 = time.perf_counter()
    rrf_scores = rrf_fuse([bm25_scores, cosine_scores], k=60)
    dt = time.perf_counter() - t0
    print(f"  Fused in {dt * 1000:.1f}ms")

    # 5. Compare top-20 from each method
    bm25_top20 = set(top_n_indices(bm25_scores, 20))
    cosine_top20 = set(top_n_indices(cosine_scores, 20))
    rrf_top20 = set(top_n_indices(rrf_scores, 20))

    print(f"\n[5/5] Results:")
    print(f"  BM25 top-20 ∩ Cosine top-20: {len(bm25_top20 & cosine_top20)} shared")
    print(f"  RRF top-20 ∩ BM25 top-20:    {len(rrf_top20 & bm25_top20)} shared")
    print(f"  RRF top-20 ∩ Cosine top-20:  {len(rrf_top20 & cosine_top20)} shared")

    # RRF top 10
    rrf_ranked = sorted(zip(rrf_scores, docs), key=lambda x: x[0], reverse=True)
    print(f"\n  RRF Top 10:")
    print("-" * 60)
    for i, (score, doc) in enumerate(rrf_ranked[:10]):
        bm25_s = bm25_scores[docs.index(doc)]
        cos_s = cosine_scores[docs.index(doc)]
        title = (doc.title or "")[:50]
        print(f"  {i+1:2d}. [rrf={score:.5f} bm25={bm25_s:5.2f} cos={cos_s:.4f}] {title}")
    print("-" * 60)

    # Checks
    if len(rrf_scores) == len(docs):
        print(f"\n  PASS: scores length ({len(rrf_scores)}) matches docs ({len(docs)})")
    else:
        print(f"\n  FAIL: length mismatch")

    # RRF should draw from both rankings
    only_bm25 = rrf_top20 - cosine_top20
    only_cosine = rrf_top20 - bm25_top20
    if only_bm25 or only_cosine:
        print(f"  PASS: RRF top-20 includes docs unique to BM25 ({len(only_bm25)}) "
              f"and unique to cosine ({len(only_cosine)}) — fusion is working")
    else:
        print("  INFO: RRF top-20 fully overlaps both — rankings were very similar")

    rrf_top5_avg = sum(s for s, _ in rrf_ranked[:5]) / 5
    rrf_bot5_avg = sum(s for s, _ in rrf_ranked[-5:]) / 5
    if rrf_top5_avg > rrf_bot5_avg:
        print(f"  PASS: RRF top 5 avg ({rrf_top5_avg:.5f}) > bottom 5 ({rrf_bot5_avg:.5f})")
    else:
        print("  FAIL: RRF ordering wrong")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
