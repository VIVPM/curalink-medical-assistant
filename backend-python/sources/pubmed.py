"""
PubMed fetcher (NCBI E-utilities).

Two-step API: esearch returns a list of PMIDs for a query, then efetch
returns the full XML records for those PMIDs in a single batch call.

Rate limits:
  - without NCBI_API_KEY: 3 req/sec
  - with    NCBI_API_KEY: 10 req/sec
NCBI policy requires sending `tool` and `email` identifiers on every call.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from lxml import etree

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

ESEARCH_TIMEOUT = 10.0
EFETCH_TIMEOUT = 15.0


def _common_params() -> dict[str, str]:
    params: dict[str, str] = {
        "tool": os.getenv("NCBI_TOOL", "curalink"),
        "email": os.getenv("NCBI_EMAIL", ""),
    }
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


async def _esearch(
    client: httpx.AsyncClient, query: str, limit: int, sort: str
) -> list[str]:
    params = {
        **_common_params(),
        "db": "pubmed",
        "term": query,
        "retmax": str(limit),
        "retmode": "json",
        "sort": sort,
    }
    resp = await client.get(ESEARCH_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


async def _efetch(client: httpx.AsyncClient, pmids: list[str]) -> bytes:
    params = {
        **_common_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    resp = await client.get(EFETCH_URL, params=params)
    resp.raise_for_status()
    return resp.content


def _text(element: Any, xpath: str, default: str = "") -> str:
    found = element.find(xpath)
    if found is not None and found.text:
        return found.text.strip()
    return default


def _parse_abstract(article: Any) -> str:
    """PubMed abstracts can have multiple labeled sections (Background,
    Methods, Results, Conclusion). Concatenate them with their labels."""
    parts: list[str] = []
    for abstract_text in article.findall(".//Abstract/AbstractText"):
        label = abstract_text.get("Label")
        text = "".join(abstract_text.itertext()).strip()
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


def _parse_authors(article: Any) -> list[str]:
    """Canonical format: `LastName Initials`."""
    authors: list[str] = []
    for author in article.findall(".//AuthorList/Author"):
        last = author.find("LastName")
        initials = author.find("Initials")
        if last is not None and last.text:
            name = last.text.strip()
            if initials is not None and initials.text:
                name += " " + initials.text.strip()
            authors.append(name)
    return authors


def _parse_year(article: Any) -> int | None:
    year_elem = article.find(".//PubDate/Year")
    if year_elem is not None and year_elem.text:
        try:
            return int(year_elem.text.strip())
        except ValueError:
            pass
    # MedlineDate fallback, e.g. "2023 Jan-Feb"
    ml = article.find(".//PubDate/MedlineDate")
    if ml is not None and ml.text:
        for token in ml.text.split():
            if token.isdigit() and len(token) == 4:
                return int(token)
    return None


def _parse_doi(article: Any) -> str | None:
    for article_id in article.findall(".//ArticleIdList/ArticleId"):
        if article_id.get("IdType") == "doi" and article_id.text:
            return article_id.text.strip().lower()
    return None


def _parse_mesh(article: Any) -> list[str]:
    return [
        desc.text.strip()
        for desc in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
        if desc.text
    ]


def _parse_article(article: Any) -> dict:
    pmid = _text(article, ".//MedlineCitation/PMID")
    return {
        "pmid": pmid,
        "title": _text(article, ".//ArticleTitle"),
        "abstract": _parse_abstract(article),
        "authors": _parse_authors(article),
        "year": _parse_year(article),
        "journal": _text(article, ".//Journal/Title"),
        "doi": _parse_doi(article),
        "mesh_terms": _parse_mesh(article),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
    }


async def fetch_pubmed(
    query: str, limit: int = 75, sort: str = "relevance"
) -> list[dict]:
    """
    Fetch publications from PubMed matching `query`.

    Args:
        query: PubMed query string. Can use field tags like
               `"parkinson"[Title/Abstract] AND "vitamin d"[Title/Abstract]`.
        limit: max number of PMIDs to fetch (default 75).
        sort: "relevance" (default) or "pub_date".

    Returns:
        List of dicts with: pmid, title, abstract, authors, year, journal,
        doi, mesh_terms, url. Docs missing an abstract are included here
        and filtered later by the normalizer.
    """
    async with httpx.AsyncClient() as client:
        pmids = await _esearch(client, query, limit, sort)
        if not pmids:
            return []
        xml_bytes = await _efetch(client, pmids)

    root = etree.fromstring(xml_bytes)
    articles = root.findall(".//PubmedArticle")
    return [_parse_article(a) for a in articles]
