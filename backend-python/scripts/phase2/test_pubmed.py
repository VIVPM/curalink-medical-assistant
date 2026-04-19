"""
Standalone test script for Step 2.1.
Run from the backend-python directory:
    python scripts/test_pubmed.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from dotenv import load_dotenv

from sources.pubmed import fetch_pubmed

load_dotenv()


async def main():
    query = "parkinson vitamin d"
    print(f"Fetching PubMed for: {query!r}\n")

    docs = await fetch_pubmed(query, limit=10)

    print(f"Got {len(docs)} papers")
    print("=" * 70)

    if not docs:
        print("No results. Check your NCBI_API_KEY and NCBI_EMAIL in .env.")
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
        mesh = ", ".join(d["mesh_terms"][:5])

        print(f"\n[{i}] {title}")
        print(f"    PMID: {d['pmid']} | Year: {year} | DOI: {doi}")
        print(f"    Authors: {authors_str}")
        print(f"    Abstract: {abstract}...")
        if mesh:
            print(f"    MeSH: {mesh}")

    # Sanity summary
    with_abstract = sum(1 for d in docs if d["abstract"])
    with_doi = sum(1 for d in docs if d["doi"])
    with_year = sum(1 for d in docs if d["year"])

    print("\n" + "=" * 70)
    print(f"Summary: {with_abstract}/{len(docs)} have abstracts, "
          f"{with_doi}/{len(docs)} have DOIs, "
          f"{with_year}/{len(docs)} have years")


if __name__ == "__main__":
    asyncio.run(main())
