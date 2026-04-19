"""
Standalone test for Phase 4 Step 4.1 — BM25 scoring.
Run from the backend-python directory:
    python scripts/phase4/test_bm25.py

Fetches ~50 docs from live APIs, scores them with BM25 against a query,
and verifies the top-ranked docs contain the query terms in their abstracts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import time
from sources.pubmed import fetch_pubmed
from sources.openalex import fetch_openalex
from sources.trials import fetch_trials
from sources.normalizer import normalize_pubmed, normalize_openalex, normalize_trial
from sources.merger import merge_and_dedupe, filter_complete
from ranking.bm25 import rank_bm25


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
    print("Phase 4 Step 4.1: BM25 Scoring Test")
    print("=" * 60)

    # 1. Fetch docs
    print("\n[1/3] Fetching docs (25 pubmed + 25 openalex + 10 trials)")
    t0 = time.perf_counter()
    docs = asyncio.run(fetch_docs())
    print(f"  Got {len(docs)} complete docs in {time.perf_counter() - t0:.1f}s")

    if len(docs) < 10:
        print("  FAIL: too few docs to test ranking")
        return

    # 2. Run BM25 with a specific query
    query = "vitamin D supplementation"
    print(f"\n[2/3] Running BM25 with query: \"{query}\"")
    t0 = time.perf_counter()
    scores = rank_bm25(query, docs)
    dt = time.perf_counter() - t0
    print(f"  Scored {len(scores)} docs in {dt * 1000:.0f}ms")

    # Pair docs with scores, sort descending
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    # 3. Verify top results
    print(f"\n[3/3] Top 10 results:")
    print("-" * 60)
    query_terms = {"vitamin", "d", "supplementation"}
    top5_hits = 0

    for i, (score, doc) in enumerate(ranked[:10]):
        title = (doc.title or "")[:70]
        text = ((doc.abstract or "") + " " + (doc.title or "")).lower()
        has_term = any(t in text for t in query_terms)
        marker = "<<" if has_term else ""
        print(f"  {i+1:2d}. [{score:6.2f}] {title} {marker}")
        if i < 5 and has_term:
            top5_hits += 1

    print("-" * 60)

    # Check: bottom 5 should have lower scores
    top5_avg = sum(s for s, _ in ranked[:5]) / 5
    bot5_avg = sum(s for s, _ in ranked[-5:]) / 5
    print(f"\n  Top 5 avg score: {top5_avg:.2f}")
    print(f"  Bottom 5 avg score: {bot5_avg:.2f}")

    if top5_avg > bot5_avg:
        print("  PASS: top docs score higher than bottom docs")
    else:
        print("  FAIL: scoring order looks wrong")

    if top5_hits >= 3:
        print(f"  PASS: {top5_hits}/5 top docs contain query terms")
    else:
        print(f"  WARN: only {top5_hits}/5 top docs contain query terms")

    # Sanity: scores list length matches docs
    if len(scores) == len(docs):
        print(f"  PASS: scores length ({len(scores)}) matches docs ({len(docs)})")
    else:
        print(f"  FAIL: length mismatch scores={len(scores)} docs={len(docs)}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
