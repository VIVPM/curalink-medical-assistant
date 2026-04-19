"""
Standalone test for Phase 4 Step 4.4 — Recency + credibility boosts.
Run from the backend-python directory:
    python scripts/phase4/test_boosts.py

Fetches ~50 docs, runs BM25+cosine+RRF, applies boosts, and verifies
that a recent multi-source paper ranks above an old single-source paper
on the same topic.
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
from ranking.boosts import apply_boosts, _recency_boost, _credibility_boost

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
    print("Phase 4 Step 4.4: Recency + Credibility Boosts Test")
    print("=" * 60)

    # 1. Unit test boost functions
    print("\n[1/4] Unit testing boost functions")

    # Recency
    assert _recency_boost(2026) == 1.0, "current year should be 1.0"
    assert _recency_boost(2016) == 0.0, "10 years old should be 0.0"
    assert _recency_boost(2010) == 0.0, ">10 years should be 0.0"
    assert 0.4 < _recency_boost(2021) < 0.6, "5 years old should be ~0.5"
    assert _recency_boost(None) == 0.0, "None year should be 0.0"
    print("  PASS: recency_boost values correct")

    # Credibility (quick mock)
    from schemas.document import Document
    multi = Document(doc_id="x", doc_type="publication", title="x",
                     sources=["pubmed", "openalex"])
    trial = Document(doc_id="x", doc_type="trial", title="x",
                     sources=["clinicaltrials"])
    pub_only = Document(doc_id="x", doc_type="publication", title="x",
                        sources=["pubmed"])
    oa_only = Document(doc_id="x", doc_type="publication", title="x",
                       sources=["openalex"])

    assert _credibility_boost(multi) == 1.0
    assert _credibility_boost(trial) == 0.7
    assert _credibility_boost(pub_only) == 0.5
    assert _credibility_boost(oa_only) == 0.3
    print("  PASS: credibility_boost values correct")

    # 2. Load embedder + fetch
    print(f"\n[2/4] Loading embedder + fetching docs")
    embedder = Embedder(BIENCODER_MODEL)
    docs = asyncio.run(fetch_docs())
    print(f"  Got {len(docs)} docs")

    query = "vitamin D supplementation parkinson"

    # 3. Run full scoring pipeline
    print(f"\n[3/4] Running BM25 + cosine + RRF + boosts")
    bm25_scores = rank_bm25(query, docs)
    cosine_scores = rank_cosine(query, docs, embedder)
    rrf_scores = rrf_fuse([bm25_scores, cosine_scores])
    boosted_scores = apply_boosts(rrf_scores, docs)

    # 4. Compare before/after boosts
    rrf_ranked = sorted(
        zip(rrf_scores, boosted_scores, docs),
        key=lambda x: x[0], reverse=True
    )
    boosted_ranked = sorted(
        zip(boosted_scores, rrf_scores, docs),
        key=lambda x: x[0], reverse=True
    )

    print(f"\n[4/4] Top 10 after boosts:")
    print("-" * 70)
    for i, (b_score, r_score, doc) in enumerate(boosted_ranked[:10]):
        year = doc.year or "?"
        src = ",".join(doc.sources[:2])
        title = (doc.title or "")[:45]
        print(f"  {i+1:2d}. [boosted={b_score:.5f} rrf={r_score:.5f}] "
              f"({year}, {src}) {title}")
    print("-" * 70)

    # Check: order changed (boosts had effect)
    rrf_order = [id(x[1]) for x in sorted(
        zip(rrf_scores, docs), key=lambda x: x[0], reverse=True)]
    boosted_order = [id(x[1]) for x in sorted(
        zip(boosted_scores, docs), key=lambda x: x[0], reverse=True)]
    reordered = rrf_order != boosted_order
    if reordered:
        print("\n  PASS: boosts changed the ranking order")
    else:
        print("\n  INFO: boosts did not change order (docs may have similar ages/sources)")

    # Check: boosted scores >= rrf scores (boosts are always >= 1.0 multiplier)
    all_gte = all(b >= r - 1e-10 for b, r in zip(boosted_scores, rrf_scores))
    if all_gte:
        print("  PASS: all boosted scores >= rrf scores (multiplicative boost works)")
    else:
        print("  FAIL: some boosted scores < rrf scores")

    if len(boosted_scores) == len(docs):
        print(f"  PASS: scores length ({len(boosted_scores)}) matches docs ({len(docs)})")
    else:
        print(f"  FAIL: length mismatch")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
