"""``:PR`` -- the middle hop of the ``Incident -> PR -> Commit -> Deployment`` chain."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from .base import Edge, NexusBaseModel, PersonRef
from .enums import MergeStrategy, NodeKind, PRState, RelationType
from .file import FileChange
from .incident import IssueLink

__all__ = ["PullRequest", "make_pr_id"]


def make_pr_id(repository_id: str, number: int) -> str:
    """``httpie/cli`` + ``1611`` -> ``httpie/cli#1611``."""
    return f"{repository_id}#{number}"


class PullRequest(NexusBaseModel):
    """A pull request, including how it landed on the default branch."""

    NEO4J_LABEL = "PR"
    EDGE_FIELDS = ("author", "closes_issues", "files_changed")

    id: str = Field(..., description="``owner/name#N``.")
    repository_id: str = Field(..., min_length=1)
    number: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    body: str | None = None
    url: str | None = None
    state: PRState = PRState.MERGED
    is_draft: bool = False
    author: PersonRef | None = None
    merged_by: PersonRef | None = None
    base_branch: str | None = Field(default=None, description="Target branch.")
    head_branch: str | None = Field(default=None, description="Source branch.")
    created_at: datetime | None = None
    updated_at: datetime | None = None
    merged_at: datetime | None = None
    closed_at: datetime | None = None
    merge_commit_id: str | None = Field(
        default=None, description="SHA of the commit that landed the PR (``:MERGED_INTO``)."
    )
    merge_strategy: MergeStrategy = MergeStrategy.UNKNOWN
    commits: list[str] = Field(
        default_factory=list,
        description="Head-branch SHAs; for squash merges this holds the single squashed SHA.",
    )
    files_changed: list[FileChange] = Field(default_factory=list)
    changed_files_count: int | None = Field(default=None, ge=0)
    additions: int | None = Field(default=None, ge=0)
    deletions: int | None = Field(default=None, ge=0)
    review_count: int | None = Field(default=None, ge=0)
    comment_count: int | None = Field(default=None, ge=0)
    labels: list[str] = Field(default_factory=list)
    closes_issues: list[IssueLink] = Field(
        default_factory=list,
        description=(
            "Populated in Week 2 from the GitHub GraphQL ``closingIssuesReferences`` "
            "field; the fixtures carry regex-derived links in the meantime."
        ),
    )

    @model_validator(mode="after")
    def _check_id(self) -> PullRequest:
        expected = make_pr_id(self.repository_id, self.number)
        if self.id != expected:
            raise ValueError(f"PullRequest.id must be '{expected}', got '{self.id}'")
        return self

    def edges(self) -> list[Edge]:
        edges: list[Edge] = []
        if self.author is not None:
            edges.append(
                Edge.link(
                    source_id=self.author.id,
                    source_label=NodeKind.PERSON,
                    type=RelationType.OPENED,
                    target_id=self.id,
                    target_label=NodeKind.PR,
                    at=self.created_at,
                )
            )
        if self.merge_commit_id:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.PR,
                    type=RelationType.MERGED_INTO,
                    target_id=self.merge_commit_id,
                    target_label=NodeKind.COMMIT,
                    merge_strategy=self.merge_strategy.value,
                    merged_at=self.merged_at,
                )
            )
        for change in self.files_changed:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.PR,
                    type=RelationType.TOUCHES,
                    target_id=change.file_id,
                    target_label=NodeKind.FILE,
                    change_type=change.change_type.value,
                    additions=change.additions,
                    deletions=change.deletions,
                )
            )
        for link in self.closes_issues:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.PR,
                    type=RelationType.CLOSES,
                    target_id=link.incident_id,
                    target_label=NodeKind.INCIDENT,
                    confidence=link.confidence,
                    method=link.method,
                    referenced_in=link.referenced_in,
                    evidence_text=link.evidence_text,
                    provenance=link.provenance,
                )
            )
        return edges
