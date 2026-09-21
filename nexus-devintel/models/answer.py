"""``:Answer`` -- what the LangGraph agent returns, with its evidence and confidence.

The contract is deliberately "auditable first": an answer is only credible if it
exposes *how* it was produced. Hence :class:`EvidencePath` (the ordered hops the
agent traversed), the per-signal :attr:`Answer.confidence_breakdown` and the
citations that map back to graph ids.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import Edge, NexusBaseModel
from .enums import (
    AnswerIntent,
    AnswerStatus,
    ConfidenceBand,
    EvidenceRole,
    NodeKind,
    RelationType,
)

__all__ = ["Answer", "Citation", "EvidenceHop", "EvidencePath"]


class EvidenceHop(BaseModel):
    """One step of the reasoning path, mirroring an ``Evidence`` node."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(..., ge=0, description="0-based position in the path.")
    node_kind: NodeKind
    node_id: str
    relation_in: str | None = Field(
        default=None, description="Relation traversed to reach this node."
    )
    role: EvidenceRole = EvidenceRole.CONTEXT
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str | None = None


class EvidencePath(BaseModel):
    """The end-to-end proof chain returned with an answer."""

    model_config = ConfigDict(extra="forbid")

    path_id: str = Field(..., min_length=1)
    intent: AnswerIntent = AnswerIntent.ROOT_CAUSE
    hops: list[EvidenceHop] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_hops: int = Field(default=0, ge=0)
    is_valid: bool = Field(
        default=False, description="True when the graph validated every hop."
    )
    cycle_detected: bool = False
    validation_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_total(self) -> EvidencePath:
        object.__setattr__(self, "total_hops", len(self.hops))
        return self


class Citation(BaseModel):
    """A human-readable pointer shown next to the answer text."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., min_length=1, description="``httpie/utils.py:230``.")
    node_kind: NodeKind
    node_id: str
    url: str | None = None


class Answer(NexusBaseModel):
    """Final agent output for one question."""

    NEO4J_LABEL = "Answer"
    EDGE_FIELDS = ("evidence_ids",)

    id: str = Field(..., min_length=1, description="Answer id, e.g. ``ans-2026-09-18-0001``.")
    question: str = Field(..., min_length=1)
    intent: AnswerIntent = AnswerIntent.GENERAL
    answer_text: str = Field(default="", description="Markdown answer shown to the user.")
    status: AnswerStatus = AnswerStatus.OK
    repository_id: str | None = None
    #: 0..1 aggregate confidence; the API surface also exposes the band.
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_band: ConfidenceBand = ConfidenceBand.UNKNOWN
    confidence_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-signal contributions, e.g. ``{'graph_coverage': 0.8, "
            "'path_length_penalty': -0.15}``. Stored as a JSON string in Neo4j."
        ),
    )
    evidence_path: EvidencePath | None = None
    evidence_ids: list[str] = Field(
        default_factory=list, description="Ordered ids -> ``(:Answer)-[:SUPPORTED_BY]->(:Evidence)``."
    )
    citations: list[Citation] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    model: str | None = Field(default=None, description="LLM used, e.g. ``gpt-4.1-mini``.")
    prompt_version: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    token_usage: dict[str, int] = Field(default_factory=dict)
    tool_calls: int | None = Field(default=None, ge=0)
    created_at: datetime | None = None
    graph_snapshot_at: datetime | None = Field(
        default=None, description="Ingestion watermark the answer was computed against."
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> Answer:
        if not self.confidence_band or self.confidence_band == ConfidenceBand.UNKNOWN:
            object.__setattr__(
                self, "confidence_band", ConfidenceBand.from_score(self.confidence_score)
            )
        if self.answer_text and not self.evidence_ids and self.status == AnswerStatus.OK:
            # An "ok" answer without evidence is a bug in the agent, not in the data.
            object.__setattr__(self, "status", AnswerStatus.INSUFFICIENT_EVIDENCE)
        return self

    def edges(self) -> list[Edge]:
        edges: list[Edge] = []
        for rank, evidence_id in enumerate(self.evidence_ids, start=1):
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.ANSWER,
                    type=RelationType.SUPPORTED_BY,
                    target_id=evidence_id,
                    target_label=NodeKind.EVIDENCE,
                    rank=rank,
                )
            )
        return edges
