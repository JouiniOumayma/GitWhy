"""``:Commit`` -- the join point between code, people, incidents and deployments."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from .base import Edge, NexusBaseModel, PersonRef
from .enums import NodeKind, RelationType
from .file import FileChange
from .incident import IssueRef

__all__ = ["Commit"]

SHA_PATTERN = r"^[0-9a-f]{40}$"


class Commit(NexusBaseModel):
    """A single git commit.

    ``id`` *is* the full SHA (globally unique, therefore already repo-scoped), by
    design: it makes ``(:Commit)`` nodes mergeable across repositories and keeps
    ``:PARENT_OF`` traversals trivial.
    """

    NEO4J_LABEL = "Commit"
    EDGE_FIELDS = ("author", "committer", "parents", "files_changed", "issue_refs")

    id: str = Field(..., pattern=SHA_PATTERN, description="Full 40-char git SHA.")
    repository_id: str = Field(..., min_length=1)
    sha: str | None = Field(default=None, description="Mirror of ``id`` (convenience).")
    short_sha: str | None = Field(default=None, min_length=4, max_length=12)
    subject: str = Field(..., min_length=1, description="First line of the message.")
    body: str | None = Field(default=None, description="Message body (may be empty).")
    is_merge: bool = False
    is_release_commit: bool = Field(
        default=False,
        description="Matches the `vX.Y.Z` / release-prep pattern; used to spot deployments.",
    )
    branch: str | None = None
    authored_at: datetime | None = None
    committed_at: datetime | None = None
    author: PersonRef | None = None
    committer: PersonRef | None = None
    parents: list[str] = Field(default_factory=list, description="Parent SHAs.")
    files_changed: list[FileChange] = Field(default_factory=list)
    changed_files_count: int | None = Field(default=None, ge=0)
    additions: int | None = Field(default=None, ge=0)
    deletions: int | None = Field(default=None, ge=0)
    tags: list[str] = Field(
        default_factory=list,
        description="Tags pointing at this commit; the Deployment emits :DEPLOYED_AS.",
    )
    issue_refs: list[IssueRef] = Field(
        default_factory=list,
        description="``Closes/Fixes/References #N`` found in subject or body.",
    )
    involves_pull_request: bool | None = Field(
        default=None,
        description=(
            "True when the subject ends with ``(#N)``: httpie/cli squash-merges "
            "281 of its 1797 commits this way, which is how the PR hop of an "
            "Incident->PR->Commit chain is recovered from git alone."
        ),
    )
    pull_request_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _sync_sha(self) -> Commit:
        if self.sha is not None and self.sha != self.id:
            raise ValueError(f"Commit.sha must equal Commit.id ({self.id}), got {self.sha}")
        object.__setattr__(self, "sha", self.id)
        if self.short_sha is None:
            object.__setattr__(self, "short_sha", self.id[:7])
        return self

    # ---- derived helpers --------------------------------------------------- #
    @property
    def closes_issues(self) -> list[IssueRef]:
        return [ref for ref in self.issue_refs if ref.relation == "CLOSES"]

    @property
    def is_root_commit(self) -> bool:
        return not self.parents

    def edges(self) -> list[Edge]:
        edges: list[Edge] = [
            Edge.link(
                source_id=self.repository_id,
                source_label=NodeKind.REPOSITORY,
                type=RelationType.HAS_COMMIT,
                target_id=self.id,
                target_label=NodeKind.COMMIT,
            )
        ]
        if self.author is not None:
            edges.append(
                Edge.link(
                    source_id=self.author.id,
                    source_label=NodeKind.PERSON,
                    type=RelationType.AUTHORED,
                    target_id=self.id,
                    target_label=NodeKind.COMMIT,
                    at=self.authored_at,
                    email=self.author.email,
                )
            )
        if self.committer is not None and self.committer.id != (
            self.author.id if self.author else None
        ):
            edges.append(
                Edge.link(
                    source_id=self.committer.id,
                    source_label=NodeKind.PERSON,
                    type=RelationType.COMMITTED,
                    target_id=self.id,
                    target_label=NodeKind.COMMIT,
                    at=self.committed_at,
                )
            )
        for parent_sha in self.parents:
            edges.append(
                Edge.link(
                    source_id=parent_sha,
                    source_label=NodeKind.COMMIT,
                    type=RelationType.PARENT_OF,
                    target_id=self.id,
                    target_label=NodeKind.COMMIT,
                )
            )
        for change in self.files_changed:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.COMMIT,
                    type=RelationType.MODIFIES,
                    target_id=change.file_id,
                    target_label=NodeKind.FILE,
                    change_type=change.change_type.value,
                    additions=change.additions,
                    deletions=change.deletions,
                    churn=change.churn,
                    previous_path=change.previous_path,
                )
            )
        for ref in self.issue_refs:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.COMMIT,
                    type=RelationType.CLOSES if ref.relation == "CLOSES" else RelationType.REFERENCES,
                    target_id=ref.incident_id,
                    target_label=NodeKind.INCIDENT,
                    confidence=ref.confidence,
                    method=ref.method,
                    referenced_in=ref.referenced_in,
                    evidence_text=ref.evidence_text,
                    provenance=ref.provenance,
                )
            )
        return edges
