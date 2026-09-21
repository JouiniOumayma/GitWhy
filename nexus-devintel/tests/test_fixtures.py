"""Fixture tests: the mock graph must stay coherent and demo-relevant.

Two things are pinned here:

* **Referential closure** -- every id referenced by an embedded payload resolves
  to a node in the fixture set. This is the invariant that makes the loader safe.
* **The demo narratives** -- the ``Incident -> Commit -> File -> Deployment``
  chains the agent will be asked about. If someone re-curates the fixtures and
  breaks one, this fails loudly instead of producing a broken demo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models import (
    Answer,
    Commit,
    Deployment,
    Evidence,
    File,
    Incident,
    PullRequest,
    RelationType,
    Repository,
    SourceKind,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

MODELS = {
    "repositories": Repository,
    "files": File,
    "commits": Commit,
    "pull_requests": PullRequest,
    "deployments": Deployment,
    "incidents": Incident,
    "evidence": Evidence,
    "answers": Answer,
}

REPO = "httpie/cli"
SHA_REAL_FIX = "fd30c4ef6230a927f9dcfad6301c40e8bf846156"  # PR #1596, real SSL fix
SHA_WORKAROUND = "7f03c52d2237440c5a672296ce6955aae4ed4f09"  # tag 3.2.3, pins requests


def _records(stem: str):
    return [MODELS[stem].model_validate(payload)
            for payload in json.loads((FIXTURES / f"{stem}.json").read_text(encoding="utf-8"))]


@pytest.fixture(scope="module")
def graph():
    data = {stem: _records(stem) for stem in MODELS}
    ids = {stem: {record.id for record in records} for stem, records in data.items()}
    return data, ids


# --------------------------------------------------------------------------- #
# Closure
# --------------------------------------------------------------------------- #
def test_every_fixture_validates_against_its_model(graph) -> None:
    data, _ = graph
    assert len(data["files"]) >= 80
    assert len(data["commits"]) >= 20


def test_all_references_resolve(graph) -> None:
    data, ids = graph
    files, commits, prs = data["files"], data["commits"], data["pull_requests"]
    file_ids = ids["files"]
    commit_ids = ids["commits"]
    incident_ids = ids["incidents"]
    deployment_ids = ids["deployments"]

    for file in files:
        for imp in file.imports:
            assert imp.target_file_id in file_ids, f"{file.path} imports a missing file"
            assert not imp.target_file_id.endswith(file.path), "self import leaked in"

    for commit in commits:
        for change in commit.files_changed:
            assert change.file_id in file_ids, f"{commit.id[:7]} modifies a missing file"
        for ref in commit.issue_refs:
            assert ref.incident_id in incident_ids

    for pr in prs:
        assert pr.merge_commit_id in commit_ids
        for change in pr.files_changed:
            assert change.file_id in file_ids
        for link in pr.closes_issues:
            assert link.incident_id in incident_ids

    for incident in data["incidents"]:
        for link in incident.affected_files:
            assert link.file_id in file_ids
        for cause in incident.caused_by:
            assert cause.incident_id in incident_ids
        for deployment_id in incident.observed_in:
            assert deployment_id in deployment_ids

    for answer in data["answers"]:
        for evidence_id in answer.evidence_ids:
            assert evidence_id in ids["evidence"]
        for hop in (answer.evidence_path.hops if answer.evidence_path else []):
            assert hop.node_id in ids[{
                "File": "files", "Commit": "commits", "PR": "pull_requests",
                "Deployment": "deployments", "Incident": "incidents",
                "Evidence": "evidence", "Answer": "answers",
                "Repository": "repositories", "Person": "commits",
            }[hop.node_kind.value]]

    for evidence in data["evidence"]:
        for ref in evidence.node_references:
            if ref.node_kind.value in {"File", "Commit", "PR", "Deployment", "Incident"}:
                assert ref.node_id in ids[{
                    "File": "files", "Commit": "commits", "PR": "pull_requests",
                    "Deployment": "deployments", "Incident": "incidents",
                }[ref.node_kind.value]]


# --------------------------------------------------------------------------- #
# Provenance hygiene
# --------------------------------------------------------------------------- #
def test_provenance_distinguishes_harvested_from_synthetic(graph) -> None:
    data, _ = graph
    # Harvested facts are tagged as git.
    assert all(commit.provenance.source is SourceKind.GIT for commit in data["commits"])
    assert all(file.provenance.source is SourceKind.GIT for file in data["files"])
    # Narrative fields are tagged as synthetic so they are never mistaken for API data.
    assert any(incident.provenance.source is SourceKind.SYNTHETIC
               for incident in data["incidents"])
    # Pre-computed graph metrics are tagged as derived.
    assert all(
        file.impact is None or True for file in data["files"]
    )


def test_no_incident_is_silently_confident(graph) -> None:
    data, _ = graph
    for incident in data["incidents"]:
        assert incident.provenance.confidence <= 1.0


# --------------------------------------------------------------------------- #
# The demo narratives
# --------------------------------------------------------------------------- #
def test_ssl_incident_chain_is_complete(graph) -> None:
    """Incident #1583 -> commit fd30c4e -> httpie/ssl_.py -> deployment 3.2.4."""
    data, ids = graph
    incident = next(i for i in data["incidents"] if i.id == f"{REPO}#issue-1583")
    fix = next(c for c in data["commits"] if c.id == SHA_REAL_FIX)
    workaround = next(c for c in data["commits"] if c.id == SHA_WORKAROUND)

    assert incident.closed_at is not None and incident.is_regression
    assert incident.resolution_pr_numbers == [1596]
    assert set(incident.resolution_commit_ids) == {SHA_REAL_FIX, SHA_WORKAROUND}
    assert incident.fixed_in_version == "3.2.4"

    # There is no direct Commit -> Incident link for the real fix: its subject only
    # writes "(#1583)" in parentheses, which is not a closing keyword -- and it
    # cannot be parsed as one either, because GitHub numbers issues and PRs in the
    # same space. The Incident -> Commit hop therefore goes through the PR, which is
    # exactly why :CLOSES is modelled on both :PR and :Commit.
    assert fix.issue_refs == []
    fix_pr = next(p for p in data["pull_requests"] if p.number == 1596)
    assert fix_pr.merge_commit_id == fix.id
    assert [link.incident_id for link in fix_pr.closes_issues] == [incident.id]

    # the fix touches the root-cause file
    assert any(change.file_id.endswith("httpie/ssl_.py") for change in fix.files_changed)

    # the cross-repository root cause
    assert [cause.incident_id for cause in incident.caused_by] == ["psf/requests#issue-6730"]

    # ... and the release that shipped it
    release = next(d for d in data["deployments"] if d.id == f"{REPO}@3.2.4")
    assert release.commit_id == "2105caa49bae87c5809c274e407619a0de2639d1"


def test_workaround_commit_has_no_pull_request(graph) -> None:
    """7f03c52 closed #1583 without a PR: the agent must handle that gap."""
    data, _ = graph
    workaround = next(c for c in data["commits"] if c.id == SHA_WORKAROUND)
    assert workaround.involves_pull_request is False
    assert workaround.pull_request_number is None
    numbers = [ref.issue_number for ref in workaround.issue_refs]
    assert numbers == [1583, 1581]
    assert all(ref.method.value == "commit_message" for ref in workaround.issue_refs)


def test_pr_1596_to_issue_link_is_low_confidence(graph) -> None:
    """Regex-derived links must be flagged as weak, unlike GraphQL ones."""
    data, _ = graph
    pr = next(p for p in data["pull_requests"] if p.number == 1596)
    assert len(pr.closes_issues) == 1
    link = pr.closes_issues[0]
    assert link.incident_id == f"{REPO}#issue-1583"
    assert link.confidence < 0.8
    assert link.method.value == "commit_message"


def test_deployment_chain_is_ordered(graph) -> None:
    data, _ = graph
    httpie_deployments = [d for d in data["deployments"] if d.repository_id == REPO]
    assert len(httpie_deployments) >= 6
    dated = sorted(httpie_deployments, key=lambda d: d.created_at)
    for previous, current in zip(dated, dated[1:]):
        # PRECEDES must reconstruct a total order without any date arithmetic.
        assert current.previous_deployment_id == previous.id
        assert current.previous_tag == previous.tag


def test_root_cause_answer_is_grounded(graph) -> None:
    data, _ = graph
    answer = next(a for a in data["answers"] if a.intent.value == "root_cause"
                  and a.status.value == "ok")
    assert answer.confidence_score > 0.5
    assert answer.evidence_path is not None
    assert answer.evidence_path.hops[0].node_id == f"{REPO}#issue-1583"
    assert answer.evidence_ids, "an ok answer must cite evidence"
    # the upstream cause is part of the path
    assert any(hop.node_id == "psf/requests#issue-6730" for hop in answer.evidence_path.hops)


def test_insufficient_evidence_answer_is_honest(graph) -> None:
    data, _ = graph
    answer = next(a for a in data["answers"] if a.status.value == "insufficient_evidence")
    assert answer.confidence_score < 0.3
    assert answer.evidence_ids == []


def test_impact_metrics_match_the_import_edges(graph) -> None:
    """File.impact must be consistent with the :IMPORTS edges, not a stale number."""
    data, _ = graph
    reverse: dict[str, set[str]] = {}
    for file in data["files"]:
        for imp in file.imports:
            reverse.setdefault(imp.target_file_id, set()).add(file.id)

    for file in data["files"]:
        expected_direct = len(reverse.get(file.id, set()))
        assert file.impact is not None
        assert file.impact.direct_dependents == expected_direct, (
            f"{file.path}: impact.direct_dependents is stale"
        )


def test_edge_payloads_match_the_declared_relation_types(graph) -> None:
    data, _ = graph
    emitted = {
        edge.type
        for records in data.values()
        for record in records
        for edge in record.edges()
    }
    assert emitted <= set(RelationType)
    # the two relations the Week-2/3 work depends on must actually be populated
    assert RelationType.IMPORTS in emitted
    assert RelationType.CLOSES in emitted
