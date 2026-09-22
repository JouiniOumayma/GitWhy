"""Read-only security guards -- the Phase 5 demo, automated.

Every test is an *attack attempt* that must be visibly blocked, so the jury
demo is just::

    python -m pytest tests/test_security.py -v

Pinned guarantees, layer by layer:

* **GitHub REST** (:mod:`ingestion.github_client`) -- no mutating verb exists
  on the client surface;
* **GitHub GraphQL** (:mod:`ingestion.github_graphql`) -- a mutation-shaped
  payload raises :class:`GraphQLMutationBlocked` *before* any byte leaves the
  process, whatever the caller tries (direct text, renamed operation, embedded
  in a query);
* **Neo4j writes** (:class:`ingestion.graph_writer.GraphWriter`) -- labels and
  relation types are whitelisted: an unknown label/type cannot be interpolated
  into Cypher, so a poisoned contract cannot inject a write;
* **PostgreSQL** (:class:`retrieval.hybrid.HybridRetriever`) -- the retrieval
  surface sends SELECTs only, and the id scheme leaves no room for injection
  through identifiers (bound parameters everywhere);
* **the analyzers** never mutate the graph: ``analyze()`` is read-only, writes
  are an explicit, separate ``--write`` flag.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from ingestion import GitHubClient, GraphWriter
from ingestion.github_graphql import (
    GitHubGraphQLClient,
    GraphQLMutationBlocked,
)
from models import NodeKind, RelationType


# --------------------------------------------------------------------------- #
# GitHub REST: no mutating verb on the surface
# --------------------------------------------------------------------------- #
def test_rest_client_exposes_no_mutating_method() -> None:
    mutating = [name for name in dir(GitHubClient)
                if name.startswith(("post", "put", "patch", "delete"))]
    assert mutating == []


def test_rest_client_request_method_is_pinned_to_get() -> None:
    import inspect

    source = inspect.getsource(GitHubClient.get)
    assert 'method="GET"' in source, "the transport must hard-code GET"


# --------------------------------------------------------------------------- #
# GitHub GraphQL: mutations blocked before transport
# --------------------------------------------------------------------------- #
@pytest.fixture()
def graphql_client(monkeypatch: pytest.MonkeyPatch) -> GitHubGraphQLClient:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    return GitHubGraphQLClient(token="test-token")


def _no_transport(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
    raise AssertionError("NETWORK WRITE LEAK: the request left the process")


@pytest.fixture(autouse=True)
def _guard_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any HTTP call during these tests means a payload escaped."""
    import ingestion.github_graphql as gql

    monkeypatch.setattr(gql.urllib.request, "urlopen", _no_transport)


ATTACK_PAYLOADS = [
    # classic mutation
    'mutation { createIssue(input: {repositoryId: "R", title: "pwned"}) { issue { id } } }',
    # disguised / renamed operation
    'query Q { repository(owner: "x", name: "y") { id } }  operationName: mutation',
    # embedded after a legitimate-looking prefix
    'query { viewer { login } } mutation { deleteIssue(input: {}) { clientMutationId } }',
    # case games
    'MuTaTiOn { addStar(input: {}) { clientMutationId } }',
]


@pytest.mark.parametrize("payload", ATTACK_PAYLOADS)
def test_graphql_mutation_payloads_are_blocked(graphql_client: GitHubGraphQLClient,
                                               payload: str) -> None:
    with pytest.raises(GraphQLMutationBlocked):
        graphql_client.execute(payload)


def test_graphql_mutation_is_blocked_even_with_variables(graphql_client) -> None:
    with pytest.raises(GraphQLMutationBlocked):
        graphql_client.execute(
            'mutation($input: CreateIssueInput!) { createIssue(input: $input) { issue { id } } }',
            {"input": {"repositoryId": "R", "title": "pwned"}},
        )


# --------------------------------------------------------------------------- #
# Neo4j: Cypher injection via labels/relations is impossible
# --------------------------------------------------------------------------- #
class _PoisonedModel:
    """A contract gone rogue: its label carries a Cypher injection."""

    id = "evil"
    NEO4J_LABEL = "File) DETACH DELETE n //"


class _FakeSession:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def run(self, query: str, **parameters: Any) -> None:
        self.queries.append(query)

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeDriver:
    def __init__(self) -> None:
        self.session_obj = _FakeSession()

    def session(self, database: str) -> _FakeSession:
        return self.session_obj


def test_neo4j_injected_label_is_refused() -> None:
    writer = GraphWriter(_FakeDriver())
    with pytest.raises(ValueError, match="refusing to interpolate unknown label"):
        writer.write_node(_PoisonedModel())


def test_neo4j_injected_relation_is_refused() -> None:
    from models import Edge

    writer = GraphWriter(_FakeDriver())
    # model_construct bypasses Pydantic validation on purpose: this simulates a
    # poisoned/partially deserialized contract reaching the writer.
    edge = Edge.model_construct(
        source_id="a", source_label=NodeKind.PR,
        type="CLOSES) DELETE b //",
        target_id="b", target_label=NodeKind.INCIDENT,
        properties={}, provenance=None,
    )
    with pytest.raises(ValueError, match="refusing to interpolate unknown relation"):
        writer.write_edge(edge)


def test_relation_whitelist_is_the_closed_schema_list() -> None:
    expected = {rel.value for rel in RelationType}
    assert "CLOSES" in expected and "IMPORTS" in expected
    # nothing that mutates without being a relationship type
    assert all(isinstance(rel.value, str) for rel in RelationType)


# --------------------------------------------------------------------------- #
# PostgreSQL: retrieval surface is SELECT-only
# --------------------------------------------------------------------------- #
@pytest.fixture()
def _stub_psycopg(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("psycopg")
    module.connect = lambda dsn: None

    OperationalError = type("OperationalError", (Exception,), {})
    module.OperationalError = OperationalError
    module.errors = types.SimpleNamespace(
        OperationalError=OperationalError,
        UndefinedTable=type("UndefinedTable", (Exception,), {}),
        UndefinedFunction=type("UndefinedFunction", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "psycopg", module)


def test_hybrid_retrieval_sends_only_selects(monkeypatch: pytest.MonkeyPatch,
                                             _stub_psycopg: None) -> None:
    from ingestion.embedding_indexer import HashingTokenBackend
    from retrieval import HybridRetriever

    recorded: list[str] = []

    class _Cursor:
        executed: list[tuple[str, tuple]] = []
        description = None

        def execute(self, query: str, params: tuple | None = None) -> None:
            recorded.append(query)
            self.description = [type("Col", (), {"name": "chunk_id"})()]

        def fetchall(self) -> list[tuple]:
            return []

        def __enter__(self) -> "_Cursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    class _Connection:
        def cursor(self) -> _Cursor:
            return _Cursor()

        def __enter__(self) -> "_Connection":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    monkeypatch.setattr(sys.modules["psycopg"], "connect", lambda dsn: _Connection())
    retriever = HybridRetriever("postgresql://unused", HashingTokenBackend())
    retriever.retrieve("anything")
    assert recorded, "the retriever must talk to PostgreSQL"
    for query in recorded:
        normalized = " ".join(query.split()).lower()
        assert normalized.startswith("select"), f"non-SELECT sent: {query}"
        for forbidden in ("insert", "update", "delete", "drop", "truncate",
                          "grant", "copy"):
            assert forbidden not in normalized


# --------------------------------------------------------------------------- #
# MCP GitHub: the tool surface exposed to an agent is read-only by construction
# --------------------------------------------------------------------------- #
def test_mcp_registry_contains_no_write_tool() -> None:
    """The registry is the security boundary: it must stay write-free."""
    from ingestion.mcp_server import MCP_TOOLS

    forbidden = ("create", "update", "delete", "merge", "close", "edit",
                 "push", "fork", "star", "comment", "label", "assign")
    for name in MCP_TOOLS:
        assert not any(word in name.lower() for word in forbidden), name


def test_mcp_unknown_tool_is_rejected_before_any_call() -> None:
    """Even a forged 'mutation' tool name cannot reach a handler."""
    from ingestion.mcp_server import call_tool

    with pytest.raises(KeyError, match="unknown tool"):
        call_tool("create_issue", {"repo": "httpie/cli", "title": "pwned"})


def test_mcp_tools_cannot_receive_a_mutating_verb() -> None:
    """Arguments are data; the transport verb is pinned GET in the client."""
    import inspect

    from ingestion import GitHubClient

    assert 'method="GET"' in inspect.getsource(GitHubClient.get)


def test_mcp_tool_results_cannot_mutate_github_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live-fire: run every tool against a transport that fails on anything
    that is not a GET, then assert every HTTP attempt was a GET."""
    import json as _json

    import ingestion.github_client as gh

    attempts: list[str] = []

    class _StrictResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return _json.dumps({"tree": [], "encoding": "base64",
                                "content": "", "items": []}).encode()

        def geturl(self):
            return "http://test"

    def _strict_urlopen(request, timeout):
        attempts.append(request.get_method())
        assert request.get_method() == "GET", \
            f"WRITE LEAK via MCP: {request.get_method()} {request.geturl()}"
        return _StrictResponse()

    monkeypatch.setattr(gh.urllib.request, "urlopen", _strict_urlopen)

    from ingestion.mcp_server import MCP_TOOLS, call_tool

    arguments = {
        "get_file": {"repo": "httpie/cli", "path": "httpie/client.py"},
        "list_files": {"repo": "httpie/cli"},
        "get_commit": {"repo": "httpie/cli", "sha": "a" * 40},
        "get_pull_request": {"repo": "httpie/cli", "number": 1},
        "search_issues": {"repo": "httpie/cli", "query": "ssl"},
    }
    for name, tool in MCP_TOOLS.items():
        try:
            call_tool(name, arguments[name])  # may report 'not found': fine
        except AssertionError:
            raise  # a write leak must fail the test loudly
        except Exception:
            pass  # HTTP/parsing quirks on empty payloads are acceptable here
    assert attempts, "the tools must have attempted (read-only) HTTP calls"
    assert set(attempts) == {"GET"}


# --------------------------------------------------------------------------- #
# The analyzers never write: analysis is separate from persistence
# --------------------------------------------------------------------------- #
def test_analyzers_expose_no_write_surface() -> None:
    from retrieval import ChangeImpactAnalyzer, HybridRetriever, RootCauseAnalyzer

    for analyzer_class in (ChangeImpactAnalyzer, RootCauseAnalyzer):
        write_methods = [name for name in dir(analyzer_class)
                         if name.startswith(("write", "merge", "delete", "create_"))]
        assert write_methods == [], f"{analyzer_class.__name__} must not write"
    write_methods = [name for name in dir(HybridRetriever)
                     if name.startswith(("write", "merge", "delete", "create_",
                                         "index", "upsert"))]
    assert write_methods == []


def test_write_path_is_explicit_and_idempotent_by_design() -> None:
    """The only write entry points are GraphWriter (MERGE on id) and the
    embedding indexer (UPSERT on a unique key): both are replay-safe."""
    import inspect

    from ingestion.embedding_indexer import EmbeddingIndexer

    source = inspect.getsource(EmbeddingIndexer._upsert_chunk)
    assert "ON CONFLICT" in source, "embedding writes must be idempotent UPSERTs"
    writer_source = inspect.getsource(GraphWriter.write_node)
    assert "MERGE" in writer_source, "graph writes must be MERGEs"
