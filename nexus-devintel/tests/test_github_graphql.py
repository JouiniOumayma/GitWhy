"""GitHub GraphQL enrichment tests -- offline, no network and no Neo4j.

Pinned invariants:

* **read-only** -- a query containing ``mutation`` never reaches the transport
  (``_guard_read_only``), and the real query is a module-level constant whose
  parameters are bound values, so callers cannot reshape it;
* **mapping** -- ``closingIssuesReferences`` nodes become ``(:PR)-[:CLOSES]->
  (:Incident)`` edges with ``source=github_graphql``,
  ``link_method=graphql_closing_issues_reference`` and ``confidence=1.0``;
* **ids** -- incident ids follow the shared ``owner/name#issue-N`` scheme, so
  enrichment ``MERGE``s onto fixture nodes instead of duplicating them;
* **pipeline** -- one paged API pass flows ``client -> enricher -> writer``
  with a scripted fake writer, including stub incident creation.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ingestion import github_graphql as gql
from ingestion.github_graphql import (
    ClosingIssue,
    ClosingIssuesEnricher,
    EnrichmentStats,
    GitHubGraphQLClient,
    GraphQLMutationBlocked,
    PullRequestLinks,
    build_close_edges,
    build_issue_links,
)
from models import (
    Edge,
    Incident,
    LinkMethod,
    NodeKind,
    PullRequest,
    RelationType,
    SourceKind,
    make_incident_id,
)

REPO = "httpie/cli"


# --------------------------------------------------------------------------- #
# Read-only guard
# --------------------------------------------------------------------------- #
def test_mutation_is_blocked_before_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []

    def boom(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise AssertionError("transport must not be reached")

    monkeypatch.setattr(gql.urllib.request, "urlopen", boom)
    client = GitHubGraphQLClient(token="t")
    with pytest.raises(GraphQLMutationBlocked, match="read-only"):
        client.execute('mutation { createIssue(input: {}) { issue { id } } }')
    assert sent == []


def test_guard_accepts_the_real_query() -> None:
    assert gql._guard_read_only(gql.PR_CLOSING_ISSUES_QUERY) == gql.PR_CLOSING_ISSUES_QUERY


def test_guard_rejects_operationName_mutation() -> None:
    with pytest.raises(GraphQLMutationBlocked):
        gql._guard_read_only('query { a }  # operationName: "mutation"')


def test_client_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(gql.GraphQLClientError, match="GITHUB_TOKEN is required"):
        GitHubGraphQLClient(token=None)


def test_query_is_a_constant_not_built_from_input() -> None:
    """The owner/name never leak into the query text: they are bound variables."""
    assert "$owner: String!" in gql.PR_CLOSING_ISSUES_QUERY
    assert "closingIssuesReferences" in gql.PR_CLOSING_ISSUES_QUERY
    assert "mutation" not in gql.PR_CLOSING_ISSUES_QUERY.lower()


# --------------------------------------------------------------------------- #
# Transport (offline: urlopen monkeypatched)
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def test_execute_posts_json_and_parses_data(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["data"] = json.loads(request.data.decode("utf-8"))
        captured["method"] = request.get_method()
        captured["auth"] = request.headers.get("Authorization")
        return _FakeResponse({"data": {"repository": {"id": "R_1"}}})

    monkeypatch.setattr(gql.urllib.request, "urlopen", fake_urlopen)
    client = GitHubGraphQLClient(token="secret")
    data = client.execute(gql.PR_CLOSING_ISSUES_QUERY,
                          {"owner": "httpie", "name": "cli", "cursor": None})
    assert data == {"repository": {"id": "R_1"}}
    assert captured["method"] == "POST"  # GraphQL protocol requirement
    assert captured["auth"] == "Bearer secret"
    assert captured["data"]["variables"] == {"owner": "httpie", "name": "cli", "cursor": None}


def test_execute_raises_on_graphql_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        return _FakeResponse({"errors": [{"message": "not found"}]})

    monkeypatch.setattr(gql.urllib.request, "urlopen", fake_urlopen)
    client = GitHubGraphQLClient(token="t")
    with pytest.raises(gql.GraphQLClientError, match="GraphQL errors"):
        client.execute(gql.PR_CLOSING_ISSUES_QUERY, {})


# --------------------------------------------------------------------------- #
# Mapping: GraphQL -> edges
# --------------------------------------------------------------------------- #
def _pr_links(numbers: list[int], pr_number: int = 1596,
              state: str | None = "MERGED") -> PullRequestLinks:
    return PullRequestLinks(
        number=pr_number,
        title="Fix SSL context creation",
        url=f"https://github.com/{REPO}/pull/{pr_number}",
        closed_at="2024-07-01T00:00:00Z",
        state=state,
        closing_issues=[
            ClosingIssue(number=n, title=f"issue {n}", url=None, closed_at=None)
            for n in numbers
        ],
    )


def test_build_close_edges_provenance_and_payload() -> None:
    edges = build_close_edges(_pr_links([1583]), repository_id=REPO)
    assert len(edges) == 1
    edge = edges[0]
    assert isinstance(edge, Edge)
    assert edge.source_id == f"{REPO}#1596"
    assert edge.target_id == make_incident_id(REPO, 1583) == f"{REPO}#issue-1583"
    assert edge.type == RelationType.CLOSES
    assert edge.source_label == NodeKind.PR
    assert edge.target_label == NodeKind.INCIDENT
    provenance = edge.provenance
    assert provenance is not None
    assert provenance.source == SourceKind.GITHUB_GRAPHQL
    assert provenance.confidence == 1.0
    assert provenance.extractor == "nexus-devintel.ingestion.github_graphql"
    assert edge.properties["link_method"] == LinkMethod.GRAPHQL_CLOSING_REFERENCE.value
    assert edge.properties["confidence"] == 1.0


def test_build_issue_links_matches_issue_link_contract() -> None:
    links = build_issue_links(_pr_links([1583, 1600]), REPO)
    assert [link.incident_id for link in links] == [
        f"{REPO}#issue-1583", f"{REPO}#issue-1600"]
    assert all(link.method == LinkMethod.GRAPHQL_CLOSING_REFERENCE for link in links)
    assert all(link.confidence == 1.0 for link in links)


def test_edges_emit_graphql_source_distinct_from_rest() -> None:
    rest_style = SourceKind.GITHUB_API
    edges = build_close_edges(_pr_links([1583]), repository_id=REPO)
    assert edges[0].provenance is not None
    assert edges[0].provenance.source != rest_style
    assert edges[0].provenance.source == SourceKind.GITHUB_GRAPHQL


# --------------------------------------------------------------------------- #
# Pagination (offline: scripted pages)
# --------------------------------------------------------------------------- #
def _scripted_client(pages: list[dict[str, Any]]) -> GitHubGraphQLClient:
    client = GitHubGraphQLClient(token="t")
    responses = iter(pages)
    monkeypatched = object.__new__(GitHubGraphQLClient)
    monkeypatched.token = "t"
    monkeypatched.endpoint = gql.GRAPHQL_ENDPOINT

    def fake_execute(self: GitHubGraphQLClient, query: str,
                     variables: dict[str, Any] | None = None) -> dict[str, Any]:
        return next(responses)

    client.execute = fake_execute.__get__(client)  # type: ignore[method-assign]
    return client


def test_pagination_follows_cursor_until_exhausted() -> None:
    page1 = {"repository": {"pullRequests": {
        "pageInfo": {"hasNextPage": True, "endCursor": "CUR1"},
        "nodes": [{"number": 1, "closingIssuesReferences": {"totalCount": 0, "nodes": []}}],
    }}}
    page2 = {"repository": {"pullRequests": {
        "pageInfo": {"hasNextPage": False},
        "nodes": [{"number": 1596, "closingIssuesReferences": {
            "totalCount": 1,
            "nodes": [{"number": 1583, "title": "SSL verify failed", "url": None,
                       "closedAt": "2024-11-01T00:00:00Z"}],
        }}],
    }}}
    client = _scripted_client([page1, page2])
    links = list(client.iter_pull_requests_with_closing_issues(REPO))
    assert [pr.number for pr in links] == [1, 1596]
    assert links[1].closing_issues[0].number == 1583


# --------------------------------------------------------------------------- #
# Enricher -> writer pipeline (fake writer, like the fake cursor)
# --------------------------------------------------------------------------- #
class _FakeWriter:
    """Mimics the real GraphWriter: an edge resolves only when both endpoint
    nodes have been written (otherwise it would be ``rel_skipped_unresolved``)."""

    def __init__(self) -> None:
        self.nodes: list[Any] = []
        self.edges: list[Any] = []
        self.node_ids: set[str] = set()

    def write_node(self, model: Any) -> str:
        self.nodes.append(model)
        self.node_ids.add(model.id)
        return model.id

    def write_edge(self, edge: Any) -> bool:
        self.edges.append(edge)
        return edge.source_id in self.node_ids and edge.target_id in self.node_ids


def test_enricher_creates_stub_pr_incident_and_edge() -> None:
    client = _scripted_client([{"repository": {"pullRequests": {
        "pageInfo": {"hasNextPage": False},
        "nodes": [{
            "number": 1596,
            "state": "MERGED",
            "title": "Fix SSL context creation",
            "url": f"https://github.com/{REPO}/pull/1596",
            "closedAt": "2024-07-01T00:00:00Z",
            "closingIssuesReferences": {"totalCount": 1, "nodes": [
                {"number": 1583, "title": "SSL verify failed", "url":
                 "https://github.com/httpie/cli/issues/1583", "closedAt": "2024-11-01"},
            ]},
        }],
    }}}])
    writer = _FakeWriter()
    enricher = ClosingIssuesEnricher(client, writer)
    stats = enricher.enrich(REPO)
    assert stats.prs_seen == 1
    assert stats.prs_with_closing_issues == 1
    assert stats.prs_created == 1
    assert stats.edges_built == 1
    assert stats.incidents_discovered == {1583}
    # the stub PR is MERGE'd first so the :CLOSES edge resolves
    assert len(writer.nodes) == 2
    pr_stub, incident_stub = writer.nodes
    from models import PullRequest

    assert isinstance(pr_stub, PullRequest)
    assert pr_stub.id == f"{REPO}#1596"
    assert pr_stub.state.value == "merged"
    assert pr_stub.provenance.source == SourceKind.GITHUB_GRAPHQL
    assert isinstance(incident_stub, Incident)
    assert incident_stub.id == f"{REPO}#issue-1583"
    assert incident_stub.provenance.source == SourceKind.GITHUB_GRAPHQL
    edge = writer.edges[0]
    assert edge.source_id == f"{REPO}#1596"


def test_pr_stub_state_mapping() -> None:
    from models import PRState

    for state, expected in (("MERGED", PRState.MERGED), ("OPEN", PRState.OPEN),
                            ("CLOSED", PRState.CLOSED), (None, PRState.CLOSED)):
        stub = ClosingIssuesEnricher._pr_stub(_pr_links([1583], state=state), REPO)
        assert stub.state == expected


def test_enricher_without_stub_creation_reports_the_skip() -> None:
    """No stubs -> the writer cannot resolve the edge -> skipped, counted."""
    client = _scripted_client([{"repository": {"pullRequests": {
        "pageInfo": {"hasNextPage": False},
        "nodes": [{"number": 1596, "state": "MERGED", "closingIssuesReferences": {
            "totalCount": 1, "nodes": [{"number": 1583, "title": None, "url": None,
                                        "closedAt": None}]},
        }],
    }}}])
    writer = _FakeWriter()
    stats = ClosingIssuesEnricher(client, writer).enrich(
        REPO, create_incidents=False, create_prs=False)
    assert writer.nodes == []
    assert stats.edges_built == 0  # skipped: both endpoints were unknown


def test_enricher_counts_prs_without_closing_issues() -> None:
    client = _scripted_client([{"repository": {"pullRequests": {
        "pageInfo": {"hasNextPage": False},
        "nodes": [
            {"number": 1, "closingIssuesReferences": {"totalCount": 0, "nodes": []}},
            {"number": 2, "closingIssuesReferences": {"totalCount": 0, "nodes": []}},
            {"number": 3, "state": "CLOSED", "closingIssuesReferences": {
                "totalCount": 1, "nodes": [
                    {"number": 42, "title": None, "url": None, "closedAt": None}]}},
        ],
    }}}])
    writer = _FakeWriter()
    stats = ClosingIssuesEnricher(client, writer).enrich(REPO)
    assert stats.prs_seen == 3
    assert stats.prs_with_closing_issues == 1
    assert stats.edges_built == 1
    assert stats.prs_created == 1


def test_enrichment_stats_to_dict() -> None:
    stats = EnrichmentStats(prs_seen=2, prs_with_closing_issues=1, edges_built=1,
                            incidents_discovered={1583})
    payload = stats.to_dict()
    assert payload["incidents_discovered"] == [1583]
    assert payload["edges_built"] == 1
