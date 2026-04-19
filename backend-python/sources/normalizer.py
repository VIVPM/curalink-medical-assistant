"""
Normalizer - converts raw fetcher output into unified Document objects.

Thin adapter layer, no reasoning. Cleans text, normalizes IDs/authors,
and sets the `is_complete` hard-filter flag. Everything downstream sees
a single Document type and never cares about source specifics.
"""

from __future__ import annotations

import html
import re

from schemas.document import Document

# --- Text cleaning ---

_COPYRIGHT_PATTERNS = [
    re.compile(r"©\s*\d{4}[^.]*\.?", re.IGNORECASE),
    re.compile(r"\(?©\s*\d{4}\)?[^.]*\.?", re.IGNORECASE),
    re.compile(r"Copyright\s*©?[^.]{0,120}\.", re.IGNORECASE),
    re.compile(r"All rights reserved\.?", re.IGNORECASE),
    re.compile(r"Published by\s+[^.]{0,80}\.", re.IGNORECASE),
]

_WHITESPACE = re.compile(r"\s+")


def clean_abstract(text: str | None) -> str:
    """Unescape HTML, strip copyright boilerplate, collapse whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    for pattern in _COPYRIGHT_PATTERNS:
        text = pattern.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


# --- DOI canonicalization ---

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "dx.doi.org/",
)


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    for prefix in _DOI_PREFIXES:
        if d.startswith(prefix):
            d = d[len(prefix):]
            break
    return d or None


# --- Author normalization ---

def normalize_author_display_name(name: str) -> str:
    """
    Convert a display-name style author ('John A. Smith') to canonical
    'Smith JA' form. Handles single names and hyphenated last names.
    """
    if not name:
        return ""
    cleaned = name.strip().replace(".", "")
    if not cleaned:
        return ""
    parts = cleaned.split()
    if len(parts) == 1:
        return parts[0]
    last = parts[-1]
    initials = "".join(p[0].upper() for p in parts[:-1] if p)
    return f"{last} {initials}".strip()


# --- Disease tag normalization ---

_TAG_SUFFIXES = (" diseases", " disease", " syndrome", " disorder", " disorders")


def normalize_disease_tags(tags: list[str] | None) -> list[str]:
    """Lowercase, strip suffix noise, dedupe."""
    if not tags:
        return []
    result: set[str] = set()
    for tag in tags:
        if not tag:
            continue
        t = tag.strip().lower()
        for suffix in _TAG_SUFFIXES:
            if t.endswith(suffix):
                t = t[: -len(suffix)]
                break
        if t:
            result.add(t)
    return sorted(result)


# --- Per-source mappers ---

_MIN_ABSTRACT_LEN = 50
_MIN_TITLE_LEN = 20


def normalize_pubmed(raw: dict) -> Document:
    title = (raw.get("title") or "").strip()
    abstract = clean_abstract(raw.get("abstract"))
    pmid = raw.get("pmid")
    doi = normalize_doi(raw.get("doi"))
    mesh_terms = raw.get("mesh_terms") or []

    is_complete = bool(
        title
        and len(title) >= _MIN_TITLE_LEN
        and abstract
        and len(abstract) >= _MIN_ABSTRACT_LEN
    )

    return Document(
        doc_id=f"pubmed:{pmid}" if pmid else "pubmed:unknown",
        doc_type="publication",
        title=title,
        abstract=abstract or None,
        authors=list(raw.get("authors") or []),  # PubMed fetcher already "LastName Initials"
        year=raw.get("year"),
        journal=raw.get("journal"),
        doi=doi,
        pmid=pmid,
        url=raw.get("url")
        or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""),
        sources=["pubmed"],
        mesh_terms=list(mesh_terms),
        disease_tags=normalize_disease_tags(mesh_terms),
        is_complete=is_complete,
    )


def normalize_openalex(raw: dict) -> Document:
    title = (raw.get("title") or "").strip()
    abstract = clean_abstract(raw.get("abstract"))
    openalex_id = raw.get("openalex_id")
    pmid = raw.get("pmid")
    doi = normalize_doi(raw.get("doi"))
    concepts = raw.get("concepts") or []
    raw_authors = raw.get("authors") or []
    authors = [normalize_author_display_name(a) for a in raw_authors if a]

    is_complete = bool(
        title
        and len(title) >= _MIN_TITLE_LEN
        and abstract
        and len(abstract) >= _MIN_ABSTRACT_LEN
    )

    return Document(
        doc_id=f"openalex:{openalex_id}" if openalex_id else "openalex:unknown",
        doc_type="publication",
        title=title,
        abstract=abstract or None,
        authors=authors,
        year=raw.get("year"),
        journal=raw.get("journal"),
        doi=doi,
        pmid=pmid,
        url=raw.get("url")
        or (f"https://openalex.org/{openalex_id}" if openalex_id else ""),
        sources=["openalex"],
        openalex_concepts=list(concepts),
        disease_tags=normalize_disease_tags(concepts),
        is_complete=is_complete,
    )


def normalize_trial(raw: dict, disease_context: str | None = None) -> Document:
    """
    `disease_context` is the disease the user queried on (from Stage 1's
    `disease_focus`). Trials don't carry MeSH/concepts like papers do, so
    we inherit the tag from the query context.
    """
    title = (raw.get("title") or "").strip()
    brief = clean_abstract(raw.get("brief_summary"))
    detailed = clean_abstract(raw.get("detailed_description"))
    eligibility = clean_abstract(raw.get("eligibility_criteria"))
    nct_id = raw.get("nct_id")

    # Use brief summary as the primary embedding text; fall back to detailed.
    abstract = brief or detailed or None

    # Hard filter: needs title + some description. Trials often have no
    # classic "abstract", so we accept any of brief / detailed.
    is_complete = bool(
        title and len(title) >= _MIN_TITLE_LEN and (brief or detailed)
    )

    disease_tags: list[str] = []
    if disease_context:
        disease_tags = normalize_disease_tags([disease_context])

    return Document(
        doc_id=f"nct:{nct_id}" if nct_id else "nct:unknown",
        doc_type="trial",
        title=title,
        abstract=abstract,
        full_text=detailed or None,
        nct_id=nct_id,
        status=raw.get("status"),
        eligibility_criteria=eligibility or None,
        min_age=raw.get("min_age"),
        max_age=raw.get("max_age"),
        start_date=raw.get("start_date"),
        primary_outcomes=list(raw.get("primary_outcomes") or []),
        secondary_outcomes=list(raw.get("secondary_outcomes") or []),
        locations=list(raw.get("locations") or []),
        contacts=list(raw.get("contacts") or []),
        year=raw.get("year"),
        url=raw.get("url")
        or (f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""),
        sources=["clinicaltrials"],
        disease_tags=disease_tags,
        is_complete=is_complete,
    )
