"""
Standalone test for Step 2.4.

Runs all three fetchers on the same query, normalizes each result into a
Document, and prints a sample + stats.

Manual verification:
  1. Every Document has the same shape regardless of source.
  2. DOIs are bare (no https://doi.org/ prefix).
  3. OpenAlex authors are in canonical 'LastName Initials' form.
  4. `is_complete` hard-filter catches no-abstract docs.
  5. disease_tags are lowercase, stripped of suffix noise.
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

load_dotenv()


def print_doc(label: str, doc) -> None:
    print(f"\n--- {label} ---")
    print(f"doc_id:       {doc.doc_id}")
    print(f"doc_type:     {doc.doc_type}")
    print(f"title:        {doc.title[:80]}")
    print(f"authors:      {doc.authors[:3]}{'...' if len(doc.authors) > 3 else ''}")
    print(f"year:         {doc.year}")
    print(f"doi:          {doc.doi}")
    print(f"pmid:         {doc.pmid}")
    print(f"url:          {doc.url}")
    print(f"abstract:     {(doc.abstract or '<none>')[:150]}...")
    if doc.doc_type == "trial":
        print(f"nct_id:       {doc.nct_id}")
        print(f"status:       {doc.status}")
        print(f"eligibility:  {(doc.eligibility_criteria or '<none>')[:100]}...")
        print(f"locations:    {len(doc.locations)} sites")
    print(f"sources:      {doc.sources}")
    print(f"disease_tags: {doc.disease_tags[:6]}")
    print(f"is_complete:  {doc.is_complete}")


def report_stats(name: str, docs: list) -> None:
    total = len(docs)
    complete = sum(1 for d in docs if d.is_complete)
    with_doi = sum(1 for d in docs if d.doi)
    with_year = sum(1 for d in docs if d.year)
    print(f"\n{name}: {total} raw -> {complete} complete "
          f"({total - complete} dropped), "
          f"{with_doi}/{total} have DOI, {with_year}/{total} have year")


async def main():
    query = "parkinson vitamin d"
    disease = "parkinson"

    print(f"Fetching all 3 sources for: {query!r}\n")
    print("=" * 70)

    pubmed_raw, openalex_raw, trials_raw = await asyncio.gather(
        fetch_pubmed(query, limit=5),
        fetch_openalex(query, limit=5),
        fetch_trials(disease=disease, limit=5),
    )

    pubmed_docs = [normalize_pubmed(r) for r in pubmed_raw]
    openalex_docs = [normalize_openalex(r) for r in openalex_raw]
    trial_docs = [normalize_trial(r, disease_context=disease) for r in trials_raw]

    # Show one sample of each type
    if pubmed_docs:
        print_doc("PubMed sample", pubmed_docs[0])
    if openalex_docs:
        print_doc("OpenAlex sample", openalex_docs[0])
    if trial_docs:
        print_doc("Trial sample", trial_docs[0])

    # Stats
    print("\n" + "=" * 70)
    print("NORMALIZATION STATS")
    print("=" * 70)
    report_stats("PubMed", pubmed_docs)
    report_stats("OpenAlex", openalex_docs)
    report_stats("Trials", trial_docs)

    total_raw = len(pubmed_raw) + len(openalex_raw) + len(trials_raw)
    total_docs = len(pubmed_docs) + len(openalex_docs) + len(trial_docs)
    total_complete = sum(
        1
        for d in pubmed_docs + openalex_docs + trial_docs
        if d.is_complete
    )
    print(f"\nGrand total: {total_raw} raw -> {total_docs} normalized -> "
          f"{total_complete} complete")


if __name__ == "__main__":
    asyncio.run(main())
