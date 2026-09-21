"""``:Incident`` -- the entry point of Root Cause Analysis.

An incident is a *production-visible defect*, modelled on a GitHub issue. The
:class:`IssueLink` payload carried by a PR is the placeholder for the
``(:PR)-[:CLOSES]->(:Incident)`` edge that the Week-2 ingestion will populate from
the GitHub GraphQL ``closingIssuesReferences`` field (see ``schema/README.md``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import NexusBaseModel, PersonRef, Provenance
from .enums import (
    DetectionSource,
    EvidenceRole,
    IncidentSeverity,
    IncidentStatus,
    LinkMethod,
    NodeKind,
    RelationType,
)
from .file import FileChange  # noqa: F401  (re-exported convenience)

__all__ = [
    "Incident",
    "IncidentCause",
    "IncidentFileLink",
    "IssueLink",
    "IssueRef",
    "make_incident_id",
]

#: Where a reference to an incident was found.
ReferenceLocation = Literal["subject", "body", "merge_message", "pr_body", "manual"]


def make_incident_id(repository_id: str, number: int) -> str:
    """``httpie/cli`` + ``1583`` -> ``httpie/cli#issue-1583``.

    The ``issue-`` prefix matters: GitHub shares one numbering space between
    issues and pull requests, so ``#1583`` may be either one. Prefixing removes
    that ambiguity in the graph.
    """
    return f"{repository_id}#issue-{number}"


class IssueLink(BaseModel):
    """``(:PR)-[:CLOSES]->(:Incident)`` payload, carried by a PullRequest."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str
    issue_number: int = Field(..., ge=1)
    method: LinkMethod = LinkMethod.GRAPHQL_CLOSING_REFERENCE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    referenced_in: ReferenceLocation = "pr_body"
    evidence_text: str | None = Field(
        default=None, description="Raw snippet that justified the link, for auditability."
    )
    provenance: Provenance | None = None


class IssueRef(BaseModel):
    """``(:Commit)-[:CLOSES|:REFERENCES]->(:Incident)`` payload, carried by a Commit.

    httpie/cli gave us 113 such references straight from commit messages
    (e.g. ``7f03c52`` body: ``Close #1583``), i.e. an ``Incident -> Commit`` link
    that exists *without* any pull request.
    """

    model_config = ConfigDict(extra="forbid")

    incident_id: str
    issue_number: int = Field(..., ge=1)
    relation: Literal["CLOSES", "REFERENCES"] = "CLOSES"
    method: LinkMethod = LinkMethod.COMMIT_MESSAGE
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    referenced_in: ReferenceLocation = "body"
    evidence_text: str | None = None
    provenance: Provenance | None = None


class IncidentFileLink(BaseModel):
    """``(:Incident)-[:AFFECTS]->(:File)`` payload (candidate root-cause file)."""

    model_config = ConfigDict(extra="forbid")

    file_id: str
    path: str
    role: EvidenceRole = EvidenceRole.ROOT_CAUSE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    method: str | None = Field(
        default=None,
        description=(
            "How the file was implicated: ``commit_overlap``, ``import_blast_radius``, "
            "``stack_trace``, ``manual_triage``."
        ),
    )
    rationale: str | None = None


class IncidentCause(BaseModel):
    """``(:Incident)-[:CAUSED_BY]->(:Incident)`` payload.

    Models cross-repository root causes. Real example harvested from httpie/cli:
    httpie's ``#1583`` (SSL failures) is caused by ``psf/requests#6730`` (requests
    2.32.3 stopped loading system certificates) -- which is why the fixtures
    contain a second repository.
    """

    model_config = ConfigDict(extra="forbid")

    incident_id: str
    repository_id: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    method: str | None = Field(
        default=None,
        description="``issue_body_reference``, ``dependency_upgrade``, ``manual_triage``.",
    )
    rationale: str | None = None
    evidence_url: str | None = None


class Incident(NexusBaseModel):
    """A defect/outage whose fix must be traced back to code."""

    NEO4J_LABEL = "Incident"
    EDGE_FIELDS = ("affected_files", "observed_in", "caused_by", "reporter")

    id: str = Field(..., description="``owner/name#issue-N``.")
    repository_id: str = Field(..., min_length=1)
    number: int = Field(..., ge=1, description="Issue number on the forge.")
    title: str = Field(..., min_length=1)
    body: str | None = None
    url: str | None = None
    status: IncidentStatus = IncidentStatus.OPEN
    severity: IncidentSeverity = IncidentSeverity.UNKNOWN
    severity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    labels: list[str] = Field(default_factory=list)
    reporter: PersonRef | None = None
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    updated_at: datetime | None = None
    detection_source: DetectionSource = DetectionSource.UNKNOWN
    is_regression: bool = False
    #: Commit SHAs that closed the issue (also emitted as ``:CLOSES`` by the Commit).
    resolution_commit_ids: list[str] = Field(default_factory=list)
    #: PR numbers that closed the issue (also emitted as ``:CLOSES`` by the PR).
    resolution_pr_numbers: list[int] = Field(default_factory=list)
    first_affected_version: str | None = None
    fixed_in_version: str | None = Field(
        default=None, description="Tag/version of the Deployment that fixed it."
    )
    affected_files: list[IncidentFileLink] = Field(default_factory=list)
    observed_in: list[str] = Field(
        default_factory=list, description="Deployment ids where the symptom was seen."
    )
    caused_by: list[IncidentCause] = Field(default_factory=list)
    related_incident_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_id(self) -> Incident:
        expected = make_incident_id(self.repository_id, self.number)
        if self.id != expected:
            raise ValueError(f"Incident.id must be '{expected}', got '{self.id}'")
        return self

    def edges(self) -> list:
        from .base import Edge

        edges: list[Edge] = []
        if self.reporter is not None:
            edges.append(
                Edge.link(
                    source_id=self.reporter.id,
                    source_label=NodeKind.PERSON,
                    type=RelationType.REPORTED,
                    target_id=self.id,
                    target_label=NodeKind.INCIDENT,
                )
            )
        for link in self.affected_files:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.INCIDENT,
                    type=RelationType.AFFECTS,
                    target_id=link.file_id,
                    target_label=NodeKind.FILE,
                    confidence=link.confidence,
                    role=link.role.value,
                    method=link.method,
                    rationale=link.rationale,
                )
            )
        for deployment_id in self.observed_in:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.INCIDENT,
                    type=RelationType.OBSERVED_IN,
                    target_id=deployment_id,
                    target_label=NodeKind.DEPLOYMENT,
                )
            )
        for cause in self.caused_by:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.INCIDENT,
                    type=RelationType.CAUSED_BY,
                    target_id=cause.incident_id,
                    target_label=NodeKind.INCIDENT,
                    confidence=cause.confidence,
                    method=cause.method,
                    rationale=cause.rationale,
                    evidence_url=cause.evidence_url,
                )
            )
        return edges
