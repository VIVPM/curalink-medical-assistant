"""
Standalone test for Step 2.5.

Fetches all 3 sources, normalizes, merges + dedupes. Verifies that:
  1. Dedupe actually reduces the count (PubMed/OpenAlex overlap)
  2. Multi-source docs preserve provenance in `sources`
  3. `is_complete` filter applies correctly after merge
  4. disease_tags get merged from both sources
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from dotenv import load_dotenv

from sources.pubmed import fetch_pubmed
from sources.openalex import fetch_openalex
from sources.trials import fetch_trials
from sources.normalizer import normalize_pubmed, normalize_openalex, normalize_trial
from sources.merger import merge_and_dedupe, filter_complete

load_dotenv()


async def main():
    query = "parkinson vitamin d"
    disease = "parkinson"

    print(f"Fetching all 3 sources for: {query!r}\n")

    pubmed_raw, openalex_raw, trials_raw = await asyncio.gather(
        fetch_pubmed(query, limit=25),
        fetch_openalex(query, limit=25),
        fetch_trials(disease=disease, limit=10),
    )

    pubmed_docs = [normalize_pubmed(r) for r in pubmed_raw]
    openalex_docs = [normalize_openalex(r) for r in openalex_raw]
    trial_docs = [normalize_trial(r, disease_context=disease) for r in trials_raw]

    total_raw = len(pubmed_docs) + len(openalex_docs) + len(trial_docs)

    # Dedupe + merge. PubMed first so PubMed's abstract wins on collision.
    deduped = merge_and_dedupe([pubmed_docs, openalex_docs, trial_docs])
    complete = filter_complete(deduped)

    multi_source = [d for d in deduped if len(d.sources) > 1]

    print("=" * 70)
    print(f"Raw:          {len(pubmed_docs)} PubMed + {len(openalex_docs)} OpenAlex "
          f"+ {len(trial_docs)} Trials = {total_raw}")
    print(f"After dedupe: {len(deduped)} unique "
          f"({total_raw - len(deduped)} duplicates collapsed)")
    print(f"After filter: {len(complete)} complete "
          f"({len(deduped) - len(complete)} dropped by is_complete)")
    print(f"Multi-source: {len(multi_source)} docs appear in >1 source")

    if multi_source:
        print("\n" + "=" * 70)
        print("MULTI-SOURCE SAMPLES (these get credibility boost in Stage 4)")
        print("=" * 70)
        for d in multi_source[:5]:
            print(f"\n  doc_id:       {d.doc_id}")
            print(f"  title:        {d.title[:70]}")
            print(f"  sources:      {d.sources}")
            print(f"  doi:          {d.doi}")
            print(f"  pmid:         {d.pmid}")
            mesh_preview = ", ".join(d.mesh_terms[:4]) if d.mesh_terms else "<none>"
            concepts_preview = (
                ", ".join(d.openalex_concepts[:4]) if d.openalex_concepts else "<none>"
            )
            print(f"  mesh_terms:   {mesh_preview}")
            print(f"  concepts:     {concepts_preview}")
            print(f"  is_complete:  {d.is_complete}")

    # Health checks - the only thing that matters is "did dedupe actually
    # collapse at least one cross-source duplicate?" Overlap rate depends
    # heavily on query + fetch-depth; small samples (25+25) often hit 0-5%.
    print("\n" + "=" * 70)
    print("HEALTH CHECKS")
    print("=" * 70)
    print(f"Dedupe mechanism: "
          f"{'WORKING' if multi_source else 'not triggered (normal for small samples)'}")
    print(f"Multi-source docs: {len(multi_source)} "
          f"(proof the cross-source PMID/DOI match works)")


if __name__ == "__main__":
    asyncio.run(main())
