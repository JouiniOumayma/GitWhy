"""Phase 1 connector tests -- all offline (no network, no Neo4j).

Pinned invariants:

* the API -> contract mapping reproduces the fixture id scheme exactly, so
  API-ingested nodes ``MERGE`` onto fixture nodes instead of duplicating them;
* every mapped node carries ``github_api`` provenance;
* webhook signatures fail closed, and a push event flows ``fetch -> models ->
  write`` without any real HTTP call;
* the graph writer whitelists labels/relations and skips unresolved edges
  instead of creating dangling nodes.
"""

from __future__ import annotations

import hmac
import hashlib
import json
from typing import Any

import pytest

from ingestion import (
    GitHubClient,
    GraphWriter,
    WebhookServer,
    commit_from_api,
    file_from_api,
    incident_from_api,
    pull_request_from_api,
    repository_from_api,
    verify_signature,
)
from ingestion.webhook import PushEvent
from models import (
    Commit,
    Edge,
    NodeKind,
    PullRequest,
    RelationType,
    SourceKind,
    make_file_id,
)

REPO = "httpie/cli"
SHA = "fd30c4ef6230a927f9dcfad6301c40e8bf846156"


# --------------------------------------------------------------------------- #
# GitHub client: read-only surface
# --------------------------------------------------------------------------- #
def test_client_only_uses_get() -> None:
    mutating = [name for name in dir(GitHubClient) if name.startswith(("post", "put", "patch", "delete"))]
    assert mutating == [], "the Phase 1 connector must be read-only"


def test_client_builds_paginated_urls() -> None:
    client = GitHubClient(token=None)
    assert client.api_root == "https://api.github.com"
    assert client.per_page == 100


# --------------------------------------------------------------------------- #
# API payload -> contracts
# --------------------------------------------------------------------------- #
@pytest.fixture()
def api_repository() -> dict[str, Any]:
    return {
        "owner": {"login": "httpie"},
        "name": "cli",
        "html_url": "https://github.com/httpie/cli",
        "description": "HTTP client",
        "default_branch": "master",
        "language": "Python",
        "stargazers_count": 35000,
        "forks_count": 2200,
        "archived": False,
        "fork": False,
        "created_at": "2012-02-25T18:39:38Z",
        "pushed_at": "2024-12-17T17:30:35Z",
    }


@pytest.fixture()
def api_commit() -> dict[str, Any]:
    return {
        "sha": SHA,
        "commit": {
            "message": "Fix SSL context creation\n\nExplicitly load default certificates.\n",
            "author": {"name": "A U Thor", "email": "author@example.com", "date": "2024-06-05T10:00:00Z"},
            "committer": {"name": "A U Thor", "email": "author@example.com", "date": "2024-06-05T10:00:00Z"},
        },
        "author": {"login": "author"},
        "committer": {"login": "author"},
        "parents": [{"sha": "0" * 40}],
        "files": [
            {"filename": "httpie/ssl_.py", "status": "modified", "additions": 12, "deletions": 3},
            {"filename": "old_path.py", "status": "removed", "additions": 0, "deletions": 40},
        ],
    }


def test_repository_mapping_keeps_the_id_scheme(api_repository) -> None:
    repo = repository_from_api(api_repository, "https://api.github.com/repos/httpie/cli")
    assert repo.id == "httpie/cli"
    assert repo.default_branch == "master"
    assert repo.provenance.source is SourceKind.GITHUB_API
    assert repo.provenance.source_uri == "https://api.github.com/repos/httpie/cli"


def test_commit_mapping_builds_fixture_compatible_ids(api_commit) -> None:
    commit = commit_from_api(api_commit, REPO, "https://api.github.com/commits/x")
    assert commit.id == SHA
    assert commit.subject == "Fix SSL context creation"
    assert commit.body == "Explicitly load default certificates."
    assert commit.is_merge is False
    assert commit.provenance.source is SourceKind.GITHUB_API
    # :MODIFIES edges target the same file ids the fixtures created
    assert commit.files_changed[0].file_id == make_file_id(REPO, "httpie/ssl_.py")
    assert commit.files_changed[1].change_type.value == "deleted"


def test_file_mapping_targets_fixture_file_ids() -> None:
    entry = {"path": "httpie/ssl_.py", "sha": "abc123", "size": 4321}
    file = file_from_api(entry, REPO, "HEAD")
    assert file.id == "httpie/cli::httpie/ssl_.py"
    assert file.is_python is True
    assert file.blob_sha == "abc123"


def test_pull_request_mapping() -> None:
    payload = {
        "number": 1596,
        "title": "Fix SSL context",
        "state": "closed",
        "merged": True,
        "html_url": "https://github.com/httpie/cli/pull/1596",
        "user": {"login": "Author"},
        "merged_by": {"login": "Maintainer"},
        "base": {"ref": "master"},
        "head": {"ref": "fix-ssl"},
        "created_at": "2024-06-01T00:00:00Z",
        "merged_at": "2024-06-05T00:00:00Z",
        "merge_commit_sha": SHA,
        "additions": 15,
        "deletions": 5,
        "changed_files": 2,
        "labels": [{"name": "bug"}],
    }
    pr = pull_request_from_api(payload, REPO, "url")
    assert pr.id == "httpie/cli#1596"
    assert pr.state.value == "merged"
    assert pr.merge_commit_id == SHA
    assert pr.author is not None and pr.author.login == "Author"


def test_issue_mapped_to_pr_is_rejected() -> None:
    payload = {"number": 1596, "title": "t", "pull_request": {"url": "https://..."}}
    with pytest.raises(ValueError, match="pull request"):
        incident_from_api(payload, REPO, "url")


def test_incident_mapping_uses_issue_prefix() -> None:
    payload = {"number": 1583, "title": "SSL failures", "closed_at": "2024-06-10T00:00:00Z",
               "labels": [], "user": {"login": "reporter"}}
    incident = incident_from_api(payload, REPO, "url")
    assert incident.id == "httpie/cli#issue-1583"
    assert incident.status.value == "closed"


# --------------------------------------------------------------------------- #
# Webhook
# --------------------------------------------------------------------------- #
def test_signature_verification_is_exact_and_constant_time() -> None:
    secret = "hunt"
    body = b'{"zen": "Keep it logically awesome."}'
    good = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_signature(secret, good, body) is True
    assert verify_signature(secret, "sha256=" + "0" * 64, body) is False
    assert verify_signature(secret, None, body) is False
    assert verify_signature(secret, "sha1=deadbeef", body) is False


def test_push_event_requires_the_repository() -> None:
    with pytest.raises(ValueError):
        PushEvent.from_payload({"ref": "refs/heads/master"})


def test_push_event_flows_through_the_pipeline(api_commit) -> None:
    """fetch (fake) -> adapter -> write (recording fake), zero network."""
    pushes: list[list[Any]] = []

    def fake_fetch(repository_id: str, sha: str) -> dict[str, Any]:
        assert repository_id == REPO and sha == SHA
        return api_commit

    def fake_write(models: list[Any]) -> dict[str, int]:
        pushes.append(models)
        return {"node:Commit": len(models)}

    payload = {
        "repository": {"full_name": REPO},
        "ref": "refs/heads/master",
        "after": SHA,
        "commits": [{"id": SHA, "message": api_commit["commit"]["message"]}],
    }
    server = WebhookServer(fetch_commit=fake_fetch, write_models=fake_write, secret="")
    result = server.handle_push(payload)

    assert result == {"repository": REPO, "ref": "refs/heads/master", "commits_written": 1}
    written = pushes[0][0]
    assert isinstance(written, Commit)
    assert written.id == SHA
    assert written.files_changed[0].file_id == "httpie/cli::httpie/ssl_.py"


# --------------------------------------------------------------------------- #
# Graph writer (offline: fake driver recording Cypher calls)
# --------------------------------------------------------------------------- #
class _RecordingSession:
    def __init__(self, existing_ids: set[str]) -> None:
        self.existing_ids = existing_ids
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, query: str, **parameters: Any):
        self.calls.append((query, parameters))
        if "RETURN count(r) AS linked" in query:
            a = parameters["source_id"] in self.existing_ids
            b = parameters["target_id"] in self.existing_ids
            linked = 1 if (a and b) else 0
            return _Result({"linked": linked})
        return _Result({"id": parameters.get("id")})

    def __enter__(self) -> _RecordingSession:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _Result:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def single(self) -> _Result:
        return self

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


class _FakeDriver:
    def __init__(self, existing_ids: set[str]) -> None:
        self._session = _RecordingSession(existing_ids)

    def session(self, database: str) -> _RecordingSession:
        return self._session


def _commit_model() -> Commit:
    return Commit(
        id=SHA,
        repository_id=REPO,
        subject="Fix SSL context creation",
        author=None,
        parents=[],
    )


def test_writer_merges_nodes_on_id() -> None:
    driver = _FakeDriver({SHA, REPO})
    writer = GraphWriter(driver)
    stats = writer.write_model(_commit_model())
    assert stats["node:Commit"] == 1
    # the node query MERGEs on id
    node_calls = [q for q, _ in driver._session.calls if "MERGE (n:`Commit`" in q]  # noqa: SLF001
    assert node_calls, "expected a MERGE on :Commit"
    assert "{id: $id}" in node_calls[0]

def test_writer_skips_edges_with_missing_endpoints() -> None:
    driver = _FakeDriver({SHA})  # REPO node absent
    writer = GraphWriter(driver)
    stats = writer.write_model(_commit_model())
    assert stats["rel:HAS_COMMIT"] == 0
    assert stats["rel_skipped_unresolved"] == 1


def test_writer_resolves_edges_when_endpoints_exist() -> None:
    driver = _FakeDriver({SHA, REPO})
    writer = GraphWriter(driver)
    stats = writer.write_model(_commit_model())
    assert stats["rel:HAS_COMMIT"] == 1
    assert "rel_skipped_unresolved" not in stats


def test_writer_refuses_unknown_labels_and_relations() -> None:
    writer = GraphWriter(_FakeDriver(set()))

    class Impostor:
        NEO4J_LABEL = "DropTable"
        id = "x"

        def to_neo4j_properties(self) -> dict[str, Any]:
            return {}

    with pytest.raises(ValueError, match="unknown label"):
        writer.write_node(Impostor())

    class BadEdge:
        source_id = REPO
        source_label = NodeKind.REPOSITORY
        target_id = SHA
        target_label = NodeKind.COMMIT
        type = "DROP"  # not a RelationType
        provenance = None

        def to_parameters(self) -> dict[str, Any]:
            return {}

    with pytest.raises(ValueError, match="unknown relation type"):
        writer.write_edge(BadEdge())


def test_writer_writes_pr_edges_from_the_contract() -> None:
    """The composite path: a PR model declares its edges, the writer applies them."""
    pr = PullRequest(
        id="httpie/cli#1596", repository_id=REPO, number=1596, title="Fix SSL",
        merge_commit_id=SHA, state="merged",
    )
    declared = pr.edges()
    assert any(edge.type == RelationType.MERGED_INTO for edge in declared)

    driver = _FakeDriver({pr.id, SHA})
    writer = GraphWriter(driver)
    stats = writer.write_model(pr)
    assert stats["rel:MERGED_INTO"] == 1
    assert stats["node:PR"] == 1
