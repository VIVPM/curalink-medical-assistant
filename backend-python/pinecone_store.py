"""
Pinecone upsert module.

Converts Documents into Pinecone records with embeddings and metadata.
Publications = one record each (title + abstract as embed text).
Trials = section-chunked (summary, eligibility, outcomes, description).

Records are idempotent by deterministic ID, so re-upserting the same doc
is a safe no-op on Pinecone's side.
"""

from __future__ import annotations

from schemas.document import Document
from embeddings.embedder import Embedder

NAMESPACE = "medical_research"


def _pub_embed_text(doc: Document) -> str:
    """Build embedding text for a publication: Title | Year | Abstract."""
    parts: list[str] = []
    if doc.title:
        parts.append(f"Title: {doc.title}")
    if doc.year:
        parts.append(f"Year: {doc.year}")
    if doc.abstract:
        parts.append(f"Abstract: {doc.abstract}")
    return " | ".join(parts)


def _base_metadata(doc: Document) -> dict:
    """Fields shared across all chunks of a doc."""
    meta = {
        "parent_id": doc.doc_id,
        "source": doc.sources[0] if doc.sources else "unknown",
        "doc_type": doc.doc_type,
        "title": (doc.title or "")[:200],
        "url": (doc.url or "")[:500],
        "disease_tags": doc.disease_tags[:10],
    }
    if doc.year is not None:
        meta["year"] = doc.year
    return meta


def _chunk_trial(doc: Document) -> list[dict]:
    """
    Break a trial into semantic section chunks. Each chunk becomes its own
    Pinecone record with a `section` metadata field and a `parent_id` link
    so retrieval can merge multiple high-scoring chunks from the same trial.
    """
    base = _base_metadata(doc)
    base["status"] = doc.status
    chunks: list[dict] = []

    if doc.abstract and len(doc.abstract) >= 50:
        chunks.append({
            "id": f"{doc.doc_id}#summary",
            "text": f"Title: {doc.title} | Summary: {doc.abstract}",
            "metadata": {**base, "section": "summary",
                         "content_text": doc.abstract[:2000]},
        })

    if doc.eligibility_criteria and len(doc.eligibility_criteria) >= 50:
        chunks.append({
            "id": f"{doc.doc_id}#eligibility",
            "text": f"Eligibility Criteria: {doc.eligibility_criteria}",
            "metadata": {**base, "section": "eligibility",
                         "content_text": doc.eligibility_criteria[:2000]},
        })

    outcomes_parts: list[str] = []
    if doc.primary_outcomes:
        outcomes_parts.append("Primary: " + "; ".join(doc.primary_outcomes))
    if doc.secondary_outcomes:
        outcomes_parts.append("Secondary: " + "; ".join(doc.secondary_outcomes))
    outcomes_text = " | ".join(outcomes_parts)
    if outcomes_text and len(outcomes_text) >= 50:
        chunks.append({
            "id": f"{doc.doc_id}#outcomes",
            "text": f"Outcomes: {outcomes_text}",
            "metadata": {**base, "section": "outcomes",
                         "content_text": outcomes_text[:2000]},
        })

    if (
        doc.full_text
        and doc.full_text != doc.abstract
        and len(doc.full_text) >= 50
    ):
        chunks.append({
            "id": f"{doc.doc_id}#description",
            "text": f"Description: {doc.full_text}",
            "metadata": {**base, "section": "description",
                         "content_text": doc.full_text[:2000]},
        })

    return chunks


def prepare_records(docs: list[Document]) -> list[dict]:
    """
    Convert Documents into upsert-ready records (with `text` for embedding,
    not yet embedded). Publications → 1 record. Trials → section chunks.

    Returns list of dicts: {id, text, metadata}.
    """
    records: list[dict] = []
    for doc in docs:
        if doc.doc_type == "publication":
            text = _pub_embed_text(doc)
            if len(text) < 50:
                continue
            meta = {
                **_base_metadata(doc),
                "section": "abstract",
                "content_text": (doc.abstract or "")[:2000],
                "authors": ", ".join(doc.authors[:5]),
                "journal": (doc.journal or "")[:200],
            }
            records.append({"id": doc.doc_id, "text": text, "metadata": meta})
        elif doc.doc_type == "trial":
            records.extend(_chunk_trial(doc))
    return records


def embed_records(records: list[dict], embedder: Embedder) -> list[dict]:
    """
    Batch-embed all record texts. Mutates records in place: adds `values`
    (the embedding vector) and moves `text` out (was only needed for embedding;
    the retrievable text is now in metadata.content_text).
    """
    if not records:
        return records
    texts = [r["text"] for r in records]
    vectors = embedder.embed_batch(texts)
    for rec, vec in zip(records, vectors):
        rec["values"] = vec
        rec.pop("text", None)
    return records


def upsert_records(
    index, records: list[dict], batch_size: int = 100, namespace: str = NAMESPACE
) -> int:
    """
    Upsert embedded records to Pinecone in batches.
    All records go into the `medical_research` namespace by default.
    Returns the number of records upserted.
    """
    count = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        index.upsert(
            vectors=[
                {"id": r["id"], "values": r["values"], "metadata": r["metadata"]}
                for r in batch
            ],
            namespace=namespace,
        )
        count += len(batch)
    return count


def query_pinecone(
    index,
    query_vector: list[float],
    top_k: int = 40,
    filter_dict: dict | None = None,
    namespace: str = NAMESPACE,
) -> list[dict]:
    """
    Query Pinecone and return results with metadata. Used by Stage 2
    retrieval and Step 3.4 test. Pre-wired with the default namespace.
    """
    kwargs = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
        "namespace": namespace,
    }
    if filter_dict:
        kwargs["filter"] = filter_dict
    results = index.query(**kwargs)
    return [
        {
            "id": m.id,
            "score": m.score,
            "metadata": m.metadata,
        }
        for m in results.matches
    ]
