"""
Standalone test script for Step 2.2.
Run from the backend-python directory:
    python scripts/test_openalex.py

Manual verification targets:
  1. Abstracts are readable plaintext (not JSON gibberish).
  2. Titles match the query intent.
  3. DOI canonicalization worked (no "https://doi.org/" prefix).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from dotenv import load_dotenv

from sources.openalex import fetch_openalex

load_dotenv()


async def main():
    query = "parkinson vitamin d"
    print(f"Fetching OpenAlex for: {query!r}\n")

    docs = await fetch_openalex(query, limit=10)

    print(f"Got {len(docs)} works")
    print("=" * 70)

    if not docs:
        print("No results. Check your OPENALEX_EMAIL in .env.")
        return

    for i, d in enumerate(docs, 1):
        title = (d["title"] or "<no title>")[:80]
        year = d["year"] or "?"
        doi = d["doi"] or "<no doi>"
        authors = d["authors"][:3]
        authors_str = ", ".join(authors)
        if len(d["authors"]) > 3:
            authors_str += f" (+{len(d['authors']) - 3} more)"
        abstract = (d["abstract"] or "<no abstract>")[:180]
        concepts = ", ".join(d["concepts"][:5])

        print(f"\n[{i}] {title}")
        print(f"    OpenAlexID: {d['openalex_id']} | Year: {year} | DOI: {doi}")
        print(f"    Authors: {authors_str}")
        print(f"    Abstract: {abstract}...")
        if concepts:
            print(f"    Concepts: {concepts}")

    with_abstract = sum(1 for d in docs if d["abstract"])
    with_doi = sum(1 for d in docs if d["doi"])
    with_year = sum(1 for d in docs if d["year"])
    with_pmid = sum(1 for d in docs if d["pmid"])

    print("\n" + "=" * 70)
    print(f"Summary: {with_abstract}/{len(docs)} have abstracts, "
          f"{with_doi}/{len(docs)} have DOIs, "
          f"{with_year}/{len(docs)} have years, "
          f"{with_pmid}/{len(docs)} also have PMIDs (for PubMed dedupe)")


if __name__ == "__main__":
    asyncio.run(main())
