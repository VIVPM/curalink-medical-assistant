"""
Unified Document type.

One dataclass covers both publications and trials. A `doc_type` discriminator
and nullable source-specific fields handle the divergence cleanly. Every
downstream pipeline stage (rank, context build, LLM, assemble) works against
this single type and never has to care which source a doc came from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

DocType = Literal["publication", "trial"]


@dataclass
class Document:

    doc_id: str
    doc_type: DocType


    title: str
    abstract: str | None = None
    full_text: str | None = None


    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    pmid: str | None = None
    url: str = ""


    nct_id: str | None = None
    status: str | None = None
    eligibility_criteria: str | None = None
    min_age: str | None = None
    max_age: str | None = None
    start_date: str | None = None
    primary_outcomes: list[str] = field(default_factory=list)
    secondary_outcomes: list[str] = field(default_factory=list)
    locations: list[dict] = field(default_factory=list)
    contacts: list[dict] = field(default_factory=list)


    sources: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    openalex_concepts: list[str] = field(default_factory=list)
    disease_tags: list[str] = field(default_factory=list)


    is_complete: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
