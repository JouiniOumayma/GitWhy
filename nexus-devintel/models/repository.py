"""``:Repository`` -- the root of every graph."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import NexusBaseModel

__all__ = ["Repository", "RepositoryAuditMetrics", "make_repository_id"]


def make_repository_id(owner: str, name: str) -> str:
    """``httpie`` + ``cli`` -> ``httpie/cli``."""
    return f"{owner}/{name}"


class RepositoryAuditMetrics(BaseModel):
    """Static-analysis snapshot of the repository.

    These are the numbers NEXUS-DevIntel uses to decide whether a repository is
    rich enough for Change Impact Analysis (see ``repo_audit.py`` produced during
    Week 0 of the project).
    """

    model_config = ConfigDict(extra="forbid")

    total_commits: int | None = Field(default=None, ge=0)
    merge_commits: int | None = Field(default=None, ge=0)
    python_files: int | None = Field(default=None, ge=0)
    total_files: int | None = Field(default=None, ge=0)
    tag_count: int | None = Field(default=None, ge=0)
    size_worktree_mb: float | None = Field(default=None, ge=0)
    size_incl_git_mb: float | None = Field(default=None, ge=0)
    contributors: int | None = Field(default=None, ge=0)
    # graph shape
    import_edges: int | None = Field(default=None, ge=0)
    files_importing: int | None = Field(default=None, ge=0)
    files_depended_upon: int | None = Field(default=None, ge=0)
    impact_max_depth: int | None = Field(
        default=None, ge=0, description="Longest transitive dependency chain (hops)."
    )
    impact_avg_blast_radius: float | None = Field(
        default=None, ge=0, description="Avg. number of files impacted by a one-file change."
    )
    # issue <-> PR <-> commit linkage
    linked_commit_pct: float | None = Field(default=None, ge=0, le=100)
    closing_keyword_commit_pct: float | None = Field(default=None, ge=0, le=100)
    unique_issue_or_pr_refs: int | None = Field(default=None, ge=0)
    audit_script: str | None = Field(default="repo_audit.py")
    audit_version: str | None = "0.1.0"
    computed_at: datetime | None = None


class Repository(NexusBaseModel):
    """A source repository, identified by ``owner/name``."""

    NEO4J_LABEL = "Repository"

    id: str = Field(..., description="``owner/name``, e.g. ``httpie/cli``.")
    host: str = Field(default="github.com", description="Forge host.")
    owner: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    url: str | None = None
    description: str | None = None
    default_branch: str = Field(default="main")
    primary_language: str | None = Field(default="Python")
    license_spdx: str | None = None
    stars: int | None = Field(default=None, ge=0)
    forks: int | None = Field(default=None, ge=0)
    is_archived: bool = False
    is_fork: bool = False
    created_at: datetime | None = None
    pushed_at: datetime | None = None
    repo_created_at: datetime | None = None
    #: Local clone path used by the ingestion pipeline (never queried in Cypher).
    local_path: str | None = None
    metrics: RepositoryAuditMetrics | None = None

    @model_validator(mode="after")
    def _check_id(self) -> Repository:
        expected = make_repository_id(self.owner, self.name)
        if self.id != expected:
            raise ValueError(f"Repository.id must be '{expected}', got '{self.id}'")
        return self

    def edges(self) -> list:
        """``HAS_COMMIT`` / ``CONTAINS`` / ``HAS_DEPLOYMENT`` are emitted by the
        Commits, Files and Deployments themselves (they own the local ids), so a
        Repository carries no outgoing relationship of its own.
        """
        return []
