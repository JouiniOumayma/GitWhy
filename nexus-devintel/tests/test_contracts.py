"""Contract tests: the guarantees other NEXUS-DevIntel components rely on.

These are not "does it run" tests, they pin the invariants of the Week-1
foundations:

* the id scheme (so ``MERGE`` is idempotent and ids are joinable across stores)
* Neo4j-ready serialization (no ``None``, no nested maps, real temporals)
* relationship derivation (the loader must not have to know the schema)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models import (
    Answer,
    Commit,
    Deployment,
    Evidence,
    EvidencePath,
    File,
    FileImport,
    Incident,
    IssueRef,
    LinkMethod,
    NodeKind,
    PRState,
    PersonRef,
    Provenance,
    PullRequest,
    RelationType,
    Repository,
    SourceKind,
    make_deployment_id,
    make_file_id,
    make_incident_id,
    make_pr_id,
    parse_semver,
)

SHA = "fd30c4ef6230a927f9dcfad6301c40e8bf846156"


# --------------------------------------------------------------------------- #
# Id scheme
# --------------------------------------------------------------------------- #
def test_file_id_is_repository_scoped() -> None:
    identity = make_file_id("httpie/cli", "httpie/utils.py")
    assert identity == "httpie/cli::httpie/utils.py"
    file = File(id=identity, repository_id="httpie/cli", path="httpie/utils.py")
    assert file.id == identity


def test_file_id_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="File.id must be"):
        File(id="httpie/utils.py", repository_id="httpie/cli", path="httpie/utils.py")


def test_repository_id_is_owner_slash_name() -> None:
    assert Repository(id="httpie/cli", owner="httpie", name="cli").id == "httpie/cli"
    with pytest.raises(ValidationError):
        Repository(id="cli", owner="httpie", name="cli")


def test_commit_id_is_the_sha() -> None:
    commit = Commit(id=SHA, repository_id="httpie/cli", subject="Fix (#1596)")
    assert commit.sha == SHA and commit.short_sha == SHA[:7]
    with pytest.raises(ValidationError):
        Commit(id=SHA, sha="0" * 40, repository_id="httpie/cli", subject="x")
    with pytest.raises(ValidationError):
        Commit(id="not-a-sha", repository_id="httpie/cli", subject="x")


def test_pr_deployment_incident_ids_are_namespaced() -> None:
    pr = PullRequest(id=make_pr_id("httpie/cli", 1596), repository_id="httpie/cli",
                     number=1596, title="t", state=PRState.MERGED)
    assert pr.id == "httpie/cli#1596"
    with pytest.raises(ValidationError):
        PullRequest(id="httpie/cli#1595", repository_id="httpie/cli", number=1596, title="t")

    deployment = Deployment(id=make_deployment_id("httpie/cli", "v3.2.2"),
                            repository_id="httpie/cli", tag="v3.2.2")
    assert (deployment.version, deployment.version_major, deployment.version_minor) == ("3.2.2", 3, 2)

    incident = Incident(id=make_incident_id("httpie/cli", 1583), repository_id="httpie/cli",
                        number=1583, title="SSL")
    # "issue-" disambiguates issues from PRs, which share one numbering space.
    assert incident.id == "httpie/cli#issue-1583"
    with pytest.raises(ValidationError):
        Incident(id="httpie/cli#1583", repository_id="httpie/cli", number=1583, title="SSL")


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("3.2.4", (3, 2, 4)), ("v3.2.2", (3, 2, 2)), ("2.6.0", (2, 6, 0)),
     ("3.2", (3, 2, 0)), ("not-a-version", None)],
)
def test_semver_parsing(tag: str, expected) -> None:
    assert parse_semver(tag) == expected


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def test_every_node_defaults_to_an_auditable_provenance() -> None:
    repository = Repository(id="httpie/cli", owner="httpie", name="cli")
    assert isinstance(repository.provenance, Provenance)
    assert repository.provenance.source is SourceKind.GIT
    assert repository.provenance.confidence == 1.0
    assert repository.provenance.ingested_at.tzinfo is not None


def test_provenance_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Provenance(confidence=1.5)
    with pytest.raises(ValidationError):
        Provenance(confidence=-0.1)


def test_person_ref_normalizes_email() -> None:
    person = PersonRef.from_email("Adam.Williamson@RedHat.com", name="Adam")
    assert person.id == "adam.williamson@redhat.com"
    assert person.login == "adam.williamson"


# --------------------------------------------------------------------------- #
# Neo4j serialization
# --------------------------------------------------------------------------- #
def test_neo4j_properties_are_flat_and_null_free() -> None:
    file = File(
        id=make_file_id("httpie/cli", "httpie/utils.py"),
        repository_id="httpie/cli",
        path="httpie/utils.py",
        loc=120,
        imports=[FileImport(target_file_id=make_file_id("httpie/cli", "httpie/compat.py"),
                            target_path="httpie/compat.py", line=230)],
    )
    properties = file.to_neo4j_properties(native_temporal=True)
    assert None not in properties.values(), "Neo4j cannot store null properties"
    assert properties["prov_source"] == "git"
    assert isinstance(properties["prov_ingested_at"], datetime)
    # the edge payload must NOT leak into the node properties
    assert "imports" not in properties
    # every value is a Neo4j-storable primitive, list of primitives, or datetime
    for key, value in properties.items():
        assert isinstance(value, (str, int, float, bool, datetime, list)), key


def test_neo4j_temporal_can_be_iso_strings() -> None:
    file = File(id=make_file_id("r/x", "a.py"), repository_id="r/x", path="a.py")
    properties = file.to_neo4j_properties(native_temporal=False)
    assert isinstance(properties["prov_ingested_at"], str)


def test_neo4j_node_carries_its_label() -> None:
    node = File(id=make_file_id("httpie/cli", "httpie/utils.py"),
                repository_id="httpie/cli", path="httpie/utils.py").to_neo4j_node()
    assert node["labels"] == ["File"] and node["id"] == "httpie/cli::httpie/utils.py"


# --------------------------------------------------------------------------- #
# Relationship derivation
# --------------------------------------------------------------------------- #
def test_file_derives_contains_and_imports() -> None:
    file = File(
        id=make_file_id("httpie/cli", "httpie/cli/requestitems.py"),
        repository_id="httpie/cli",
        path="httpie/cli/requestitems.py",
        imports=[FileImport(target_file_id=make_file_id("httpie/cli", "httpie/utils.py"),
                            target_path="httpie/utils.py", imported_names=["split_iterable"],
                            line=21)],
    )
    edges = {edge.type: edge for edge in file.edges()}
    assert edges[RelationType.CONTAINS].source_id == "httpie/cli"
    imports = edges[RelationType.IMPORTS]
    assert (imports.source_label, imports.target_label) == (NodeKind.FILE, NodeKind.FILE)
    # the edge carries the line number: this is what lets the agent point at code
    assert imports.properties["line"] == 21
    assert imports.properties["imported_names"] == ["split_iterable"]


def test_commit_derives_author_modifies_and_closes() -> None:
    commit = Commit(
        id=SHA,
        repository_id="httpie/cli",
        subject="Explicitly load default certificates when creating SSL context (#1583) (#1596)",
        author=PersonRef(id="adam@blueradius.ca", email="adam@blueradius.ca"),
        committer=PersonRef(id="noreply@github.com", email="noreply@github.com"),
        involves_pull_request=True,
        pull_request_number=1596,
        issue_refs=[IssueRef(incident_id=make_incident_id("httpie/cli", 1583), issue_number=1583,
                             method=LinkMethod.COMMIT_MESSAGE, confidence=0.6)],
    )
    edges = commit.edges()
    types = [edge.type for edge in edges]
    assert RelationType.AUTHORED in types
    assert RelationType.COMMITTED in types
    assert RelationType.CLOSES in types
    # the link confidence survives as a relationship property, for the scorer
    closes = next(edge for edge in edges if edge.type is RelationType.CLOSES)
    assert closes.properties["confidence"] == 0.6
    assert closes.properties["link_method"] == "commit_message"


def test_deployment_derives_both_directions() -> None:
    deployment = Deployment(id=make_deployment_id("httpie/cli", "3.2.4"),
                            repository_id="httpie/cli", tag="3.2.4", commit_id=SHA)
    types = {edge.type for edge in deployment.edges()}
    assert {RelationType.HAS_DEPLOYMENT, RelationType.DEPLOYED_AT,
            RelationType.DEPLOYED_AS} <= types


def test_incident_derives_affects_observed_in_and_cause() -> None:
    from models import IncidentCause, IncidentFileLink

    bare = Incident(id=make_incident_id("httpie/cli", 1583), repository_id="httpie/cli",
                    number=1583, title="SSL")
    assert bare.edges() == [], "an incident with no payload must not invent a relationship"

    incident = Incident(
        id=make_incident_id("httpie/cli", 1583),
        repository_id="httpie/cli",
        number=1583,
        title="SSL",
        reporter=PersonRef(id="awilliam@redhat.com", email="awilliam@redhat.com"),
        affected_files=[IncidentFileLink(
            file_id=make_file_id("httpie/cli", "httpie/ssl_.py"),
            path="httpie/ssl_.py", confidence=0.9,
        )],
        observed_in=[make_deployment_id("httpie/cli", "3.2.3")],
        caused_by=[IncidentCause(
            incident_id=make_incident_id("psf/requests", 6730), confidence=0.9,
        )],
    )
    by_type = {edge.type: edge for edge in incident.edges()}
    assert set(by_type) == {
        RelationType.REPORTED, RelationType.AFFECTS,
        RelationType.OBSERVED_IN, RelationType.CAUSED_BY,
    }
    # the cross-repository root cause points at another repository's incident
    assert by_type[RelationType.CAUSED_BY].target_id == "psf/requests#issue-6730"
    assert by_type[RelationType.AFFECTS].properties["confidence"] == 0.9


def test_relation_types_are_closed() -> None:
    """Every emitted relationship must belong to the published whitelist."""
    commit = Commit(id=SHA, repository_id="httpie/cli", subject="x",
                    author=PersonRef(id="a@b.c", email="a@b.c"))
    for edge in commit.edges():
        assert edge.type in RelationType


# --------------------------------------------------------------------------- #
# Answer / evidence invariants
# --------------------------------------------------------------------------- #
def test_answer_without_evidence_cannot_be_ok() -> None:
    answer = Answer(id="ans-1", question="why?", answer_text="because", confidence_score=0.9)
    assert answer.status.value == "insufficient_evidence"
    assert answer.confidence_band.value == "high"


def test_evidence_path_total_hops_is_derived() -> None:
    path = EvidencePath(path_id="p1", hops=[])
    assert path.total_hops == 0
    evidence = Evidence(id="ev-1", node_references=[])
    assert evidence.path_id == "ev-1", "path_id falls back to the evidence id"


def test_answer_supported_by_ranks_evidence_in_order() -> None:
    answer = Answer(id="ans-1", question="q", answer_text="a", confidence_score=0.5,
                    evidence_ids=["ev-1", "ev-2"])
    ranks = [edge.properties["rank"] for edge in answer.edges()]
    assert ranks == [1, 2]
