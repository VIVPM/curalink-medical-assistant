"""
Merge + dedupe across sources.

A paper often appears in BOTH PubMed and OpenAlex (OpenAlex back-fills PMIDs
for ~70% of biomedical papers). This module collapses them into a single
Document with `sources=["pubmed","openalex"]` so downstream ranking can
apply a credibility boost for multi-source provenance.

Composite-key priority: DOI > PMID > NCT-ID > source-prefixed doc_id.
"""

from __future__ import annotations

from schemas.document import Document
from sources.normalizer import normalize_disease_tags


def _composite_id(doc: Document) -> str:
    """Most specific identifier we have, in priority order."""
    if doc.doi:
        return f"doi:{doc.doi}"
    if doc.pmid:
        return f"pmid:{doc.pmid}"
    if doc.nct_id:
        return f"nct:{doc.nct_id}"
    return doc.doc_id


def _merge_into(existing: Document, incoming: Document) -> None:
    """
    Enrich `existing` with any metadata from `incoming` that it's missing.
    Mutates `existing` in place. The first-seen doc remains the "base"
    (its title, abstract, authors win), but provenance and tags merge.
    """
    # Provenance: union of sources, dedupe
    for src in incoming.sources:
        if src not in existing.sources:
            existing.sources.append(src)

    # PubMed has MeSH tags, OpenAlex has concepts - both are useful metadata
    if incoming.mesh_terms and not existing.mesh_terms:
        existing.mesh_terms = list(incoming.mesh_terms)
    if incoming.openalex_concepts and not existing.openalex_concepts:
        existing.openalex_concepts = list(incoming.openalex_concepts)

    # Recompute disease_tags from the merged metadata
    existing.disease_tags = normalize_disease_tags(
        existing.mesh_terms + existing.openalex_concepts
    )

    # Fill in missing cross-references
    if not existing.pmid and incoming.pmid:
        existing.pmid = incoming.pmid
    if not existing.doi and incoming.doi:
        existing.doi = incoming.doi
    if not existing.journal and incoming.journal:
        existing.journal = incoming.journal

    # If the base doc had no abstract but the incoming one does, adopt it.
    # This handles the case where OpenAlex (seen first) lacked an abstract
    # but PubMed (seen second) has one.
    if (not existing.abstract) and incoming.abstract:
        existing.abstract = incoming.abstract
        # A doc that just gained an abstract might now pass the completeness
        # check that previously failed it.
        if incoming.is_complete:
            existing.is_complete = True


def merge_and_dedupe(doc_lists: list[list[Document]]) -> list[Document]:
    """
    Flatten multiple source lists, dedupe by composite ID, merge metadata
    on collision. Order of input lists matters: earlier lists "win" for
    ambiguous fields (title, abstract, authors) when duplicates exist.

    Recommended call:
        merge_and_dedupe([pubmed_docs, openalex_docs, trial_docs])

    PubMed is passed first so PubMed's cleaner abstract wins over
    OpenAlex's version when the same paper appears in both.
    """
    seen: dict[str, Document] = {}
    for docs in doc_lists:
        for doc in docs:
            uid = _composite_id(doc)
            if uid not in seen:
                seen[uid] = doc
            else:
                _merge_into(seen[uid], doc)
    return list(seen.values())


def filter_complete(docs: list[Document]) -> list[Document]:
    """Apply the is_complete hard filter. Apply AFTER merging - a doc can
    gain completeness via metadata merge from a duplicate."""
    return [d for d in docs if d.is_complete]
