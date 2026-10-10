"""``:Evidence`` -- one verifiable piece of proof produced by the retrieval step.

Every element of an :class:`~models.answer.EvidencePath` is an ``Evidence`` node,
so an answer can always be replayed and audited: each evidence points at concrete
graph nodes (``:REFERENCES``) and, when it came from the vector store, at a
pgvector row (``VectorRef``) that holds the embedding.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import Edge, NexusBaseModel, Provenance
from .enums import (
    EvidenceKind,
    EvidenceRole,
    NodeKind,
    RelationType,
    RetrievalStrategy,
)

__all__ = ["Evidence", "EvidenceRef", "VectorRef"]


class VectorRef(BaseModel):
    """Pointer to the pgvector row holding this evidence's embedding."""

    model_config = ConfigDict(extra="forbid")

    table: str = Field(default="evidence_embeddings")
    row_id: str = Field(..., description="Primary key (uuid) of the pgvector row.")
    model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model name.")
    dim: int = Field(default=384, ge=1)
    distance: float | None = Field(
        default=None, ge=0.0, description="Distance returned by the similarity search."
    )


class EvidenceRef(BaseModel):
    """``(:Evidence)-[:REFERENCES]->(any node)`` payload."""

    model_config = ConfigDict(extra="forbid")

    node_kind: NodeKind
    node_id: str
    role: EvidenceRole = EvidenceRole.CONTEXT
    hop_index: int | None = Field(
        default=None, ge=0, description="Position in the EvidencePath."
    )
    relation: str | None = Field(
        default=None, description="Graph relation used to reach this node (e.g. ``IMPORTS``)."
    )
    rationale: str | None = None
    quote: str | None = Field(default=None, description="Verbatim excerpt supporting the hop.")
    url: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)


class Evidence(NexusBaseModel):
    """A retrievable, citable unit of proof."""

    NEO4J_LABEL = "Evidence"
    EDGE_FIELDS = ("node_references", "derived_from")

    id: str = Field(..., min_length=1, description="Stable id, e.g. ``ev-httpie-1583-1``.")
    repository_id: str | None = None
    #: Answer this evidence was retrieved for; ``None`` while still in the pool.
    answer_id: str | None = None
    kind: EvidenceKind = EvidenceKind.NODE
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.GRAPH_TRAVERSAL
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance/weight used by the confidence aggregator.",
    )
    rank: int | None = Field(default=None, ge=1, description="1-based rank in the result list.")
    text: str | None = Field(
        default=None, description="Exact text that was embedded and is shown to the LLM."
    )
    rationale: str | None = Field(
        default=None, description="Why the retriever picked this (explainability)."
    )
    hop_index: int | None = Field(default=None, ge=0)
    path_id: str | None = Field(
        default=None, description="Groups the evidence belonging to one EvidencePath."
    )
    node_references: list[EvidenceRef] = Field(default_factory=list)
    derived_from: list[str] = Field(
        default_factory=list, description="Ids of Evidence nodes this one was built from."
    )
    vector: VectorRef | None = None
    retrieved_at: datetime | None = None

    @model_validator(mode="after")
    def _sync_path(self) -> Evidence:
        if self.path_id is None:
            object.__setattr__(self, "path_id", self.answer_id or self.id)
        return self

    def edges(self) -> list[Edge]:
        edges: list[Edge] = []
        for ref in self.node_references:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.EVIDENCE,
                    type=RelationType.REFERENCES,
                    target_id=ref.node_id,
                    target_label=ref.node_kind,
                    role=ref.role.value,
                    hop_index=ref.hop_index,
                    relation=ref.relation,
                    rationale=ref.rationale,
                    quote=ref.quote,
                    url=ref.url,
                    line_start=ref.line_start,
                    line_end=ref.line_end,
                )
            )
        for parent_id in self.derived_from:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.EVIDENCE,
                    type=RelationType.DERIVED_FROM,
                    target_id=parent_id,
                    target_label=NodeKind.EVIDENCE,
                )
            )
        return edges
