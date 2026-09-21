"""Map GitHub REST payloads onto the NEXUS-DevIntel Pydantic contracts.

This module is the *only* place that knows both dialects. The graph never sees a
raw API dict: everything goes through these factories, which

* build the ids exactly like the fixtures do (``owner/name::path``,
  ``owner/name#N``, ``owner/name#issue-N``), so API-ingested nodes ``MERGE``
  onto fixture nodes instead of duplicating them;
* tag every node with ``source=github_api`` provenance and the API URL, unlike
  ``build_fixtures.py`` which tags with ``git`` / ``synthetic_fixture``;
* reject payloads whose shape drifted (``KeyError`` surfaces loudly instead of
  writing half a node).

PR-numbering ambiguity: ``/issues/N`` also serves PRs, so
:func:`incident_from_api` raises :class:`ValueError` on a PR payload rather than
creating an ``(:Incident)`` node for a pull request.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from models import (
    Commit,
    Deployment,
    DeploymentKind,
    Environment,
    File,
    FileChange,
    Incident,
    IncidentStatus,
    PersonRef,
    Provenance,
    PullRequest,
    PRState,
    Repository,
    ChangeType,
    make_deployment_id,
    make_file_id,
    make_incident_id,
    make_pr_id,
)

EXTRACTOR = "nexus-devintel.ingestion.models_adapter"


def _provenance(source_uri: str, confidence: float = 1.0) -> Provenance:
    return Provenance(
        source="github_api",
        source_uri=source_uri,
        extractor=EXTRACTOR,
        confidence=confidence,
    )


def _person(payload: dict[str, Any] | None) -> PersonRef | None:
    if not payload:
        return None
    login = payload.get("login") or "unknown"
    return PersonRef(
        id=str(login).lower(),
        login=login,
        name=payload.get("name") or login,
        email=payload.get("email"),
    )


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def repository_from_api(payload: dict[str, Any], url: str) -> Repository:
    owner = payload["owner"]["login"]
    name = payload["name"]
    return Repository(
        id=f"{owner}/{name}",
        owner=owner,
        name=name,
        url=payload.get("html_url"),
        description=payload.get("description"),
        default_branch=payload.get("default_branch") or "main",
        primary_language=payload.get("language"),
        stars=payload.get("stargazers_count"),
        forks=payload.get("forks_count"),
        is_archived=bool(payload.get("archived")),
        is_fork=bool(payload.get("fork")),
        created_at=_dt(payload.get("created_at")),
        pushed_at=_dt(payload.get("pushed_at")),
        provenance=_provenance(url),
    )


def file_from_api(
    tree_entry: dict[str, Any], repository_id: str, ref: str = "HEAD"
) -> File:
    """One ``git/trees`` entry (``blob``) -> ``(:File)``.

    Import edges are NOT built here: the REST tree carries no import
    information. They come from the AST pass (``build_fixtures.parse_imports``)
    or from the Week-2 pipeline; a ``:File`` without ``:IMPORTS`` still merges
    cleanly into an existing graph.
    """
    path = tree_entry["path"]
    return File(
        id=make_file_id(repository_id, path),
        repository_id=repository_id,
        path=path,
        extension=("." + path.rsplit(".", 1)[-1]) if "." in path else None,
        language="Python" if path.endswith(".py") else None,
        is_python=path.endswith(".py"),
        is_package_init=path.endswith("__init__.py"),
        size_bytes=tree_entry.get("size"),
        blob_sha=tree_entry.get("sha"),
        provenance=_provenance(f"git:ref/{ref}:{path}"),
    )


def _changes_from_commit(payload: dict[str, Any], repository_id: str) -> list[FileChange]:
    changes: list[FileChange] = []
    for entry in payload.get("files") or []:
        status = entry.get("status", "modified")
        if status not in {"added", "removed", "modified", "renamed", "copied"}:
            status = "modified"
        changes.append(
            FileChange(
                file_id=make_file_id(repository_id, entry["filename"]),
                path=entry["filename"],
                change_type=ChangeType({"removed": "deleted"}.get(status, status)),
                additions=int(entry.get("additions") or 0),
                deletions=int(entry.get("deletions") or 0),
                previous_path=entry.get("previous_filename"),
            )
        )
    return changes


def commit_from_api(payload: dict[str, Any], repository_id: str, url: str) -> Commit:
    core = payload["commit"]
    author_login = (payload.get("author") or {}).get("login")
    committer_login = (payload.get("committer") or {}).get("login")
    return Commit(
        id=payload["sha"],
        repository_id=repository_id,
        subject=core["message"].split("\n", 1)[0],
        body=core["message"].split("\n", 1)[1].strip() or None
        if "\n" in core["message"] else None,
        is_merge=len(payload.get("parents") or []) > 1,
        authored_at=_dt(core["author"]["date"]),
        committed_at=_dt(core["committer"]["date"]),
        author=PersonRef(
            id=(author_login or core["author"]["email"] or "unknown").lower(),
            login=author_login,
            name=core["author"]["name"],
            email=core["author"].get("email"),
        ),
        committer=PersonRef(
            id=(committer_login or core["committer"]["email"] or "unknown").lower(),
            login=committer_login,
            name=core["committer"]["name"],
            email=core["committer"].get("email"),
        ),
        parents=[parent["sha"] for parent in payload.get("parents") or []],
        files_changed=_changes_from_commit(payload, repository_id),
        tags=[tag["name"] for tag in payload.get("tags") or []]
        if isinstance(payload.get("tags"), list) else [],
        provenance=_provenance(url),
    )


def pull_request_from_api(payload: dict[str, Any], repository_id: str, url: str) -> PullRequest:
    number = payload["number"]
    state = "merged" if payload.get("merged") else payload.get("state", "closed")
    return PullRequest(
        id=make_pr_id(repository_id, number),
        repository_id=repository_id,
        number=number,
        title=payload["title"],
        body=payload.get("body") or None,
        url=payload.get("html_url"),
        state=PRState(state),
        is_draft=bool(payload.get("draft")),
        author=_person(payload.get("user")),
        merged_by=_person(payload.get("merged_by")),
        base_branch=payload.get("base", {}).get("ref"),
        head_branch=payload.get("head", {}).get("ref"),
        created_at=_dt(payload.get("created_at")),
        updated_at=_dt(payload.get("updated_at")),
        merged_at=_dt(payload.get("merged_at")),
        closed_at=_dt(payload.get("closed_at")),
        merge_commit_id=payload.get("merge_commit_sha") or None,
        merge_strategy="squash" if payload.get("merge_commit_sha") else "unknown",
        additions=payload.get("additions"),
        deletions=payload.get("deletions"),
        changed_files_count=payload.get("changed_files"),
        labels=[label["name"] for label in payload.get("labels") or []],
        provenance=_provenance(url),
    )


def incident_from_api(payload: dict[str, Any], repository_id: str, url: str) -> Incident:
    """``/issues/N`` -> ``(:Incident)``. Raises on a PR payload (same numbering)."""
    if payload.get("pull_request") is not None:
        raise ValueError(
            f"issue #{payload['number']} is a pull request; use pull_request_from_api"
        )
    return Incident(
        id=make_incident_id(repository_id, payload["number"]),
        repository_id=repository_id,
        number=payload["number"],
        title=payload["title"],
        body=payload.get("body") or None,
        url=payload.get("html_url"),
        status=IncidentStatus.CLOSED if payload.get("closed_at") else IncidentStatus.OPEN,
        labels=[label["name"] for label in payload.get("labels") or []],
        reporter=_person(payload.get("user")),
        opened_at=_dt(payload.get("created_at")),
        closed_at=_dt(payload.get("closed_at")),
        updated_at=_dt(payload.get("updated_at")),
        provenance=_provenance(url),
    )


def deployment_from_tag(repository_id: str, tag: str, commit_sha: str | None,
                        url: str, created_at: datetime | None = None) -> Deployment:
    """A git tag as a minimal deployment anchor (release payload optional)."""
    return Deployment(
        id=make_deployment_id(repository_id, tag),
        repository_id=repository_id,
        tag=tag,
        kind=DeploymentKind.TAG,
        environment=Environment.PRODUCTION,
        commit_id=commit_sha,
        created_at=created_at,
        published_at=created_at,
        url=url,
        provenance=_provenance(url or f"git:refs/tags/{tag}"),
    )
