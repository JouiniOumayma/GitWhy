"""Hybrid retriever tests -- offline, against a scripted fake connection.

Pinned invariants:

* the SQL calls the existing ``hybrid_search_code_chunks`` function with the
  repository filter (the contract Person A wrote in ``001_init.sql``);
* rows are packaged into ``code_chunk`` Evidence nodes with hybrid strategy,
  RRF score, both ranks and a graph anchor (File / Commit / Incident);
* dimension drift between the backend and ``vector(384)`` is rejected with an
  actionable message;
* evidence ids are stable across identical runs.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from ingestion.embedding_indexer import HashingTokenBackend
from retrieval import HybridRetriever
from retrieval.hybrid import HybridRetrievalError

QUERY = "SSL certificate verify failed"


@pytest.fixture(autouse=True)
def _stub_psycopg(monkeypatch: pytest.MonkeyPatch) -> None:
    """psycopg may not be installed in the test env: stub the module and let
    ``monkeypatch.setattr("psycopg.connect", ...)`` bind our fake connection."""
    module = types.ModuleType("psycopg")
    module.connect = lambda dsn: None  # replaced per-test

    class _Errors:
        class OperationalError(Exception):
            pass

        class UndefinedTable(Exception):
            pass

        class UndefinedFunction(Exception):
            pass

    module.errors = _Errors
    monkeypatch.setitem(sys.modules, "psycopg", module)


class _FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, tuple]] = []
        self.description = None

    def execute(self, query: str, params: tuple | None = None) -> None:
        self.executed.append((query, params))
        if self.rows:
            self.description = [
                type("Col", (), {"name": name})()
                for name in ("chunk_id", "file_id", "path", "symbol", "start_line",
                             "end_line", "content", "rrf_score", "vector_rank",
                             "lexical_rank")
            ]

    def fetchall(self) -> list[tuple]:
        return self.rows

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeConnection:
    def __init__(self, rows: list[tuple]) -> None:
        self.cursor_obj = _FakeCursor(rows)

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _row(chunk_suffix: str, rrf: float = 0.0328, vector_rank: int | None = 1,
         lexical_rank: int | None = 1) -> tuple:
    path = f"httpie/ssl_{chunk_suffix}.py"
    return (
        f"00000000-0000-0000-0000-{chunk_suffix.zfill(12)}",  # chunk_id uuid
        f"httpie/cli::{path}",                                # file_id
        path,                                                 # path
        "create_ssl_context",                                 # symbol
        221, 245,                                             # lines
        "def create_ssl_context(): ...",                      # content
        rrf, vector_rank, lexical_rank,
    )


@pytest.fixture()
def retriever() -> HybridRetriever:
    return HybridRetriever("postgresql://unused", HashingTokenBackend(),
                           repository_id="httpie/cli")


def test_calls_the_existing_sql_function(monkeypatch: pytest.MonkeyPatch,
                                         retriever: HybridRetriever) -> None:
    connection = _FakeConnection([_row("a")])
    monkeypatch.setattr("psycopg.connect", lambda dsn: connection)
    report = retriever.retrieve(QUERY)
    query, params = connection.cursor_obj.executed[0]
    assert "FROM hybrid_search_code_chunks" in query
    # (text, vector, match_count, rrf_k, repository)
    assert params[2] == 10
    assert params[3] == 60
    assert params[4] == "httpie/cli"
    assert params[0] == QUERY
    assert params[1].startswith("[")  # pgvector literal


def test_rows_are_packaged_as_hybrid_evidence(monkeypatch: pytest.MonkeyPatch,
                                              retriever: HybridRetriever) -> None:
    monkeypatch.setattr("psycopg.connect", lambda dsn: _FakeConnection([_row("a")]))
    report = retriever.retrieve(QUERY)
    assert report.backend_name == "hashing-token-384"
    match = report.matches[0]
    assert match.citation == "httpie/ssl_a.py:221-245 (create_ssl_context)"
    evidence = report.evidences[0]
    assert evidence.kind.value == "code_chunk"
    assert evidence.retrieval_strategy.value == "hybrid"
    assert evidence.rank == 1
    assert evidence.node_references[0].node_kind.value == "File"
    assert evidence.node_references[0].node_id == match.file_id
    assert evidence.node_references[0].line_start == 221
    assert 0.0 < evidence.score <= 1.0
    assert "vector_rank=1" in (evidence.rationale or "")


def test_commit_and_incident_chunks_anchor_to_the_right_node(
        monkeypatch: pytest.MonkeyPatch, retriever: HybridRetriever) -> None:
    commit_row = _row("a")
    commit_row = (
        commit_row[0], "httpie/cli::commits/7f03c52d", "commits/7f03c52d",
        None, 1, 3, "Close #1583", 0.03, 2, 1,
    )
    incident_row = (
        "00000000-0000-0000-0000-0000000000ff", "httpie/cli::incidents/1583",
        "incidents/1583", None, 1, 5, "SSL verify failed", 0.02, None, 2,
    )
    monkeypatch.setattr("psycopg.connect",
                        lambda dsn: _FakeConnection([commit_row, incident_row]))
    report = retriever.retrieve(QUERY)
    anchors = {(e.node_references[0].node_kind.value, e.node_references[0].node_id)
               for e in report.evidences}
    assert ("Commit", "7f03c52d") in anchors
    assert ("Incident", "httpie/cli#issue-1583") in anchors


def test_dimension_drift_is_rejected(retriever: HybridRetriever) -> None:
    class WrongDimBackend:
        name = "wrong"

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 768]

    retriever._backend = WrongDimBackend()
    with pytest.raises(HybridRetrievalError, match="768"):
        retriever.retrieve(QUERY)


def test_missing_backend_loads_the_real_minilm(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """No backend configured -> real MiniLM default (dim 384).

    Needs torch + the model: skipped offline, pinned live during execution.
    """
    pytest.importorskip("sentence_transformers")

    monkeypatch.setattr("psycopg.connect", lambda dsn: _FakeConnection([_row("a")]))
    retriever = HybridRetriever("postgresql://unused", None)
    report = retriever.retrieve(QUERY)
    assert report.backend_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_evidence_ids_are_stable(monkeypatch: pytest.MonkeyPatch,
                                 retriever: HybridRetriever) -> None:
    monkeypatch.setattr("psycopg.connect", lambda dsn: _FakeConnection([_row("a")]))
    first = retriever.retrieve(QUERY)
    second = retriever.retrieve(QUERY)
    assert [e.id for e in first.evidences] == [e.id for e in second.evidences]


def test_no_writes_are_ever_sent(monkeypatch: pytest.MonkeyPatch,
                                 retriever: HybridRetriever) -> None:
    """The retriever is SELECT-only: nothing else may hit the database."""
    connection = _FakeConnection([_row("a")])
    monkeypatch.setattr("psycopg.connect", lambda dsn: connection)
    retriever.retrieve(QUERY)
    for query, _ in connection.cursor_obj.executed:
        normalized = " ".join(query.split()).lower()
        assert normalized.startswith("select")
        assert "insert" not in normalized and "update" not in normalized
        assert "delete" not in normalized and "merge" not in normalized
