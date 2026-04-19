"""
OpenAlex fetcher.

Single JSON endpoint. No API key required — send `User-Agent` with a
mailto to get into the polite pool (~10 req/sec).

CRITICAL gotcha: OpenAlex returns abstracts as an inverted index to save
bandwidth. You MUST reconstruct plaintext before passing to embeddings,
or you'll silently embed JSON gibberish and wreck retrieval quality.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

WORKS_URL = "https://api.openalex.org/works"
DEFAULT_TIMEOUT = 15.0
MAX_RETRIES = 2


def _user_agent() -> str:
    email = os.getenv("OPENALEX_EMAIL", "")
    return f"curalink (mailto:{email})" if email else "curalink"


def _reconstruct_abstract(inverted: dict | None) -> str:
    """
    OpenAlex's `abstract_inverted_index` maps `word -> [positions]`.
    Reconstruct a plaintext string from it.
    """
    if not inverted:
        return ""
    max_pos = max((max(positions) for positions in inverted.values()), default=-1)
    if max_pos < 0:
        return ""
    words: list[str] = [""] * (max_pos + 1)
    for word, positions in inverted.items():
        for pos in positions:
            if 0 <= pos <= max_pos:
                words[pos] = word
    return " ".join(w for w in words if w)


def _canonical_doi(doi_url: str | None) -> str | None:
    if not doi_url:
        return None
    doi = doi_url.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi.org/"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi or None


def _extract_pmid(ids: dict | None) -> str | None:
    if not ids:
        return None
    pmid_url = ids.get("pmid")
    if not pmid_url:
        return None
    # pmid value is typically a URL like "https://pubmed.ncbi.nlm.nih.gov/12345"
    return pmid_url.rstrip("/").split("/")[-1]


def _extract_openalex_id(work_id: str | None) -> str | None:
    if not work_id:
        return None
    # "https://openalex.org/W1234567890" -> "W1234567890"
    return work_id.rstrip("/").split("/")[-1]


def _parse_work(work: dict) -> dict:
    ids = work.get("ids") or {}
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    authorships = work.get("authorships") or []
    concepts = work.get("concepts") or []

    openalex_id = _extract_openalex_id(work.get("id"))
    doi = _canonical_doi(work.get("doi") or ids.get("doi"))

    return {
        "openalex_id": openalex_id,
        "pmid": _extract_pmid(ids),
        "doi": doi,
        "title": (work.get("title") or "").strip(),
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "authors": [
            (a.get("author") or {}).get("display_name", "").strip()
            for a in authorships
            if (a.get("author") or {}).get("display_name")
        ],
        "year": work.get("publication_year"),
        "journal": source.get("display_name"),
        "concepts": [c.get("display_name") for c in concepts if c.get("display_name")],
        "url": work.get("doi") or work.get("id"),
    }


async def fetch_openalex(
    query: str,
    limit: int = 75,
    from_year: int = 2020,
) -> list[dict]:
    """
    Fetch works from OpenAlex matching `query`.

    Args:
        query: free-text query (OpenAlex has no field tags).
        limit: max works to return (OpenAlex caps at 200 per page).
        from_year: publication year lower bound filter.

    Returns:
        List of dicts with: openalex_id, pmid (if known), doi, title,
        abstract (plaintext, reconstructed), authors, year, journal,
        concepts, url.
    """
    params = {
        "search": query,
        "per-page": str(min(limit, 200)),
        "page": "1",
        "filter": f"from_publication_date:{from_year}-01-01,type:article",
        "sort": "relevance_score:desc",
    }
    headers = {"User-Agent": _user_agent()}

    last_error: Exception | None = None
    async with httpx.AsyncClient() as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await client.get(WORKS_URL, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                works = data.get("results", [])
                return [_parse_work(w) for w in works]
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise

    # Unreachable, but keeps the type checker happy
    raise last_error if last_error else RuntimeError("openalex fetch failed")
