"""``:Deployment`` -- how the project models a release (git tag / GitHub release).

httpie/cli has 50 semver tags spanning 2012-2024, which gives the agent a clean
notion of "deployment window" for temporal root-cause analysis.
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import Field, model_validator

from .base import Edge, NexusBaseModel
from .enums import DeploymentKind, Environment, NodeKind, RelationType

__all__ = ["Deployment", "make_deployment_id", "parse_semver"]

SEMVER_RE = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?(?:[-+].*)?$")


def make_deployment_id(repository_id: str, tag: str) -> str:
    """``httpie/cli`` + ``3.2.4`` -> ``httpie/cli@3.2.4``."""
    return f"{repository_id}@{tag}"


def parse_semver(tag: str) -> tuple[int, int, int] | None:
    """``v3.2.2`` -> ``(3, 2, 2)``; returns ``None`` for non-semver tags."""
    match = SEMVER_RE.match(tag.strip())
    if match is None:
        return None
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch") or 0),
    )


class Deployment(NexusBaseModel):
    """A released artefact, anchored on the commit it was built from."""

    NEO4J_LABEL = "Deployment"
    EDGE_FIELDS = ("commit_id",)

    id: str = Field(..., description="``owner/name@tag``.")
    repository_id: str = Field(..., min_length=1)
    tag: str = Field(..., min_length=1, description="Raw tag name, e.g. ``v3.2.2``.")
    version: str | None = Field(default=None, description="Normalized, e.g. ``3.2.4``.")
    name: str | None = Field(default=None, description="Release title.")
    kind: DeploymentKind = DeploymentKind.TAG
    environment: Environment = Environment.PRODUCTION
    url: str | None = None
    #: ``id`` of the commit the tag points at -> ``:DEPLOYED_AT`` / ``:DEPLOYED_AS``.
    commit_id: str | None = Field(default=None, description="Full SHA (``created_at`` tag).")
    commit_sha: str | None = None
    created_at: datetime | None = None
    published_at: datetime | None = None
    is_prerelease: bool = False
    is_draft: bool = False
    is_head: bool = Field(
        default=False, description="True if the tag points at the current default-branch HEAD."
    )
    version_major: int | None = Field(default=None, ge=0)
    version_minor: int | None = Field(default=None, ge=0)
    version_patch: int | None = Field(default=None, ge=0)
    release_notes: str | None = None
    changelog_url: str | None = None
    previous_deployment_id: str | None = Field(
        default=None, description="Previous tag -> ``:PRECEDES``."
    )
    previous_tag: str | None = None
    commit_delta: int | None = Field(
        default=None, ge=0, description="Commits since the previous deployment."
    )
    days_since_previous: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_id(self) -> Deployment:
        expected = make_deployment_id(self.repository_id, self.tag)
        if self.id != expected:
            raise ValueError(f"Deployment.id must be '{expected}', got '{self.id}'")
        return self

    @model_validator(mode="after")
    def _derive_semver(self) -> Deployment:
        parsed = parse_semver(self.tag)
        if parsed is not None:
            major, minor, patch = parsed
            object.__setattr__(self, "version_major", self.version_major or major)
            object.__setattr__(self, "version_minor", self.version_minor or minor)
            object.__setattr__(self, "version_patch", self.version_patch or patch)
            if self.version is None:
                object.__setattr__(self, "version", f"{major}.{minor}.{patch}")
        if self.commit_sha is None and self.commit_id is not None:
            object.__setattr__(self, "commit_sha", self.commit_id)
        return self

    def edges(self) -> list[Edge]:
        edges = [
            Edge.link(
                source_id=self.repository_id,
                source_label=NodeKind.REPOSITORY,
                type=RelationType.HAS_DEPLOYMENT,
                target_id=self.id,
                target_label=NodeKind.DEPLOYMENT,
            )
        ]
        if self.commit_id:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.DEPLOYMENT,
                    type=RelationType.DEPLOYED_AT,
                    target_id=self.commit_id,
                    target_label=NodeKind.COMMIT,
                    at=self.created_at,
                    tag=self.tag,
                )
            )
            # Reverse convenience edge: "this commit shipped as that release".
            edges.append(
                Edge.link(
                    source_id=self.commit_id,
                    source_label=NodeKind.COMMIT,
                    type=RelationType.DEPLOYED_AS,
                    target_id=self.id,
                    target_label=NodeKind.DEPLOYMENT,
                    tag=self.tag,
                    environment=self.environment.value,
                )
            )
        if self.previous_deployment_id:
            edges.append(
                Edge.link(
                    source_id=self.previous_deployment_id,
                    source_label=NodeKind.DEPLOYMENT,
                    type=RelationType.PRECEDES,
                    target_id=self.id,
                    target_label=NodeKind.DEPLOYMENT,
                    days_since_previous=self.days_since_previous,
                    commit_delta=self.commit_delta,
                )
            )
        return edges
