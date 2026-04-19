"""
Standalone test for Phase 4 Step 4.5 — MedCPT cross-encoder rerank.
Run from the backend-python directory:
    python scripts/phase4/test_cross_encoder.py

Fetches ~50 docs, runs BM25+cosine+RRF+boosts to get top-20, then
reranks with MedCPT cross-encoder and checks that the order changes
(cross-encoder usually rearranges the top-20 noticeably).
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
from ranking.boosts import apply_boosts
from ranking.cross_encoder import MedCPTReranker

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


def main():
    print("=" * 60)
    print("Phase 4 Step 4.5: MedCPT Cross-Encoder Rerank Test")
    print("=" * 60)

    # 1. Load models
    print(f"\n[1/5] Loading bi-encoder: {BIENCODER_MODEL}")
    embedder = Embedder(BIENCODER_MODEL)

    print(f"[1/5] Loading cross-encoder: ncbi/MedCPT-Cross-Encoder")
    t0 = time.perf_counter()
    reranker = MedCPTReranker()
    print(f"  Cross-encoder loaded in {time.perf_counter() - t0:.1f}s")

    # 2. Fetch docs
    print("\n[2/5] Fetching docs")
    docs = asyncio.run(fetch_docs())
    print(f"  Got {len(docs)} docs")

    query = "vitamin D supplementation parkinson"

    # 3. Run BM25+cosine+RRF+boosts to get top-20
    print(f"\n[3/5] Running BM25+cosine+RRF+boosts")
    bm25_scores = rank_bm25(query, docs)
    cosine_scores = rank_cosine(query, docs, embedder)
    rrf_scores = rrf_fuse([bm25_scores, cosine_scores])
    boosted_scores = apply_boosts(rrf_scores, docs)

    # Get top-20 by boosted score
    indexed = sorted(range(len(docs)), key=lambda i: boosted_scores[i], reverse=True)
    top20_indices = indexed[:20]
    top20_docs = [docs[i] for i in top20_indices]

    print(f"  Top-20 selected from {len(docs)} docs")

    # 4. Rerank top-20 with MedCPT
    print(f"\n[4/5] Reranking top-20 with MedCPT cross-encoder")
    t0 = time.perf_counter()
    ce_scores = reranker.rerank(query, top20_docs)
    dt = time.perf_counter() - t0
    print(f"  Reranked in {dt * 1000:.0f}ms")

    # 5. Compare before vs after
    print(f"\n[5/5] Before vs after cross-encoder:")
    print("-" * 70)
    print(f"  {'Rank':>4}  {'Before (boosted)':>16}  {'After (CE)':>10}  Title")
    print("-" * 70)

    # Before order (already sorted by boosted)
    ce_ranked = sorted(range(len(top20_docs)), key=lambda i: ce_scores[i], reverse=True)

    for new_rank, i in enumerate(ce_ranked[:10]):
        old_rank = top20_indices.index(top20_indices[i]) + 1 if i < len(top20_indices) else "?"
        doc = top20_docs[i]
        title = (doc.title or "")[:45]
        print(f"  {new_rank+1:4d}  (was #{i+1:2d}, {boosted_scores[top20_indices[i]]:.5f})"
              f"  [{ce_scores[i]:7.3f}]  {title}")

    print("-" * 70)

    # Check: did CE reorder?
    before_order = list(range(len(top20_docs)))
    after_order = ce_ranked
    reordered = before_order != after_order
    if reordered:
        print("\n  PASS: cross-encoder rearranged the top-20")
    else:
        print("\n  INFO: cross-encoder kept same order (unusual)")

    # Check: scores length
    if len(ce_scores) == len(top20_docs):
        print(f"  PASS: scores length ({len(ce_scores)}) matches top-20 ({len(top20_docs)})")
    else:
        print(f"  FAIL: length mismatch")

    # Check: score variance (CE should produce spread, not flat scores)
    score_range = max(ce_scores) - min(ce_scores)
    print(f"  Score range: {min(ce_scores):.3f} to {max(ce_scores):.3f} (spread={score_range:.3f})")
    if score_range > 0.1:
        print("  PASS: meaningful score spread (CE is discriminating)")
    else:
        print("  WARN: narrow score spread — CE might not be helping")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
