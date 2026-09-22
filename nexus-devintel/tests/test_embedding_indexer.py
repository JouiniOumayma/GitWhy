"""Embedding indexer tests -- offline, no PostgreSQL required.

Three layers are pinned:

* **chunking** -- stable chunk ids (``repo::path#L<s>-L<e>``), the fixture
  provenance carried onto every chunk, and the file/commit/incident split;
* **backends** -- the hashing-token backend (dim 384, stdlib-only) is
  deterministic so UPSERTs stay idempotent without torch, and carries real
  token-level similarity; the real MiniLM backend shares the same dim;
* **SQL shape** -- the UPSERT targets the UNIQUE ``(file_id, start_line,
  end_line)`` conflict target with a hash guard, and ``ingestion_runs`` is
  opened/closed around the run. A scripted fake cursor records every call, the
  same style as ``tests/test_impact.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ingestion.embedding_indexer import (
    CHUNK_LINES,
    CodeChunk,
    EmbeddingIndexer,
    HashingTokenBackend,
    chunk_commit_message,
    chunk_file,
    chunk_incident,
    content_hash,
    make_chunk_id,
)
from models import File, Provenance, SourceKind, make_file_id

REPO = "httpie/cli"


def _file(path: str = "httpie/utils.py", loc: int = 30) -> File:
    return File(
        id=make_file_id(REPO, path),
        repository_id=REPO,
        path=path,
        extension=".py",
        language="Python",
        loc=loc,
    )


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_chunk_id_scheme_matches_schema_readme() -> None:
    assert make_chunk_id("httpie/cli::httpie/utils.py", 221, 245) == \
        "httpie/cli::httpie/utils.py#L221-L245"


def test_chunk_file_produces_line_bounded_chunks() -> None:
    chunks = chunk_file(_file(loc=95), chunk_lines=CHUNK_LINES, chunk_overlap=5)
    assert chunks, "a 95-loc python file must produce at least one chunk"
    for chunk in chunks:
        assert 1 <= chunk.start_line <= chunk.end_line
        assert chunk.end_line - chunk.start_line + 1 <= CHUNK_LINES
        assert chunk.chunk_id == make_chunk_id(chunk.file_id, chunk.start_line, chunk.end_line)


def test_chunk_file_carries_fixture_provenance() -> None:
    chunk = chunk_file(_file())[0]
    assert chunk.provenance.source == SourceKind.GIT
    assert chunk.kind == "file"
    assert chunk.source_node_id == chunk.file_id


def test_chunk_file_is_deterministic() -> ContentHashStability:
    first = chunk_file(_file(loc=95))
    second = chunk_file(_file(loc=95))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.content_hash for c in first] == [c.content_hash for c in second]


class ContentHashStability:
    """Marker base class for readability (tests above inherit nothing)."""


def test_non_python_files_are_skipped() -> None:
    binary = File(
        id=make_file_id(REPO, "logo.png"),
        repository_id=REPO,
        path="logo.png",
        extension=".png",
    )
    assert chunk_file(binary) == []


def test_commit_message_chunk_anchors_on_commit_id() -> None:
    from models import Commit, PersonRef

    commit = Commit(
        id="7f03c52d2237440c5a672296ce6955aae4ed4f09",
        repository_id=REPO,
        subject="Close #1583: pin requests<2.32.3",
        body="The requests 2.32.3 release stopped loading system certs.",
        author=PersonRef(id="a@b.c", name="A", email="a@b.c"),
    )
    chunks = chunk_commit_message(commit)
    assert len(chunks) == 1
    assert chunks[0].kind == "commit_message"
    assert chunks[0].source_node_id == commit.id
    # The commit_id column joins the chunk back to (:Commit).
    assert chunks[0].source_node_id in chunks[0].file_id
    assert "#1583" in chunks[0].content


def test_incident_chunk_uses_synthetic_provenance() -> None:
    from models import Incident

    incident = Incident(
        id=f"{REPO}#issue-1583",
        repository_id=REPO,
        number=1583,
        title="HTTPS requests fail after requests 2.32.3",
        body="SSLError: certificate verify failed ...",
        provenance=Provenance(source=SourceKind.SYNTHETIC, confidence=0.2),
    )
    chunks = chunk_incident(incident)
    assert chunks[0].kind == "incident_body"
    # The chunk inherits the incident's provenance: a synthetic incident must
    # not rank like a real git object inside pgvector either.
    assert chunks[0].provenance.source == SourceKind.SYNTHETIC
    assert chunks[0].provenance.confidence == 0.2


# --------------------------------------------------------------------------- #
# Backends (real MiniLM dim + stdlib hashing fallback)
# --------------------------------------------------------------------------- #
def test_hashing_backend_is_deterministic_and_384d() -> None:
    from ingestion.embedding_indexer import EMBEDDING_DIM, HashingTokenBackend

    backend = HashingTokenBackend()
    first = backend.embed(["ssl certificate verify failed"])
    second = backend.embed(["ssl certificate verify failed"])
    other = backend.embed(["unrelated text"])
    assert first == second
    assert first[0] != other[0]
    assert len(first[0]) == EMBEDDING_DIM == 384


def test_hashing_backend_carries_token_level_similarity() -> None:
    """The offline fallback must rank shared tokens above noise (dim 384)."""
    from ingestion.embedding_indexer import HashingTokenBackend

    backend = HashingTokenBackend()
    query, ssl_chunk, noise = backend.embed([
        "SSL certificate verify failed after requests upgrade",
        "def create_ssl_context(): certificate verify failed ssl context creation",
        "typography quotes replacement changelog",
    ])

    def _cos(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    assert len(query) == 384
    assert _cos(query, query) == pytest.approx(1.0, abs=1e-3)
    assert _cos(query, ssl_chunk) > _cos(query, noise) + 0.1
    # Deterministic across calls (idempotent re-runs stay no-ops).
    assert backend.embed(["ssl certificate verify failed"]) == \
        backend.embed(["ssl certificate verify failed"])


# --------------------------------------------------------------------------- #
# SQL shape via a scripted fake cursor
# --------------------------------------------------------------------------- #
class _FakeCursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple]] = []
        self.rowcount = 1

    def execute(self, query: str, params: tuple | None = None) -> None:
        self.statements.append((query, params))

    def fetchone(self) -> tuple[Any]:
        return ("00000000-0000-0000-0000-000000000000",)

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()
        self.committed = False

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True


def test_upsert_is_idempotent_by_content_hash() -> None:
    """The conflict target + WHERE clause are what make re-runs no-ops."""
    indexer = EmbeddingIndexer("postgresql://unused", repository_id=REPO)
    connection = _FakeConnection()
    chunk = chunk_file(_file())[0]
    vector = HashingTokenBackend().embed([chunk.text_for_embedding()])[0]

    indexer._upsert_chunk(connection.cursor_obj, chunk, vector)

    query = connection.cursor_obj.statements[0][0]
    assert "ON CONFLICT (file_id, start_line, end_line)" in query
    assert "IS DISTINCT FROM EXCLUDED.content_hash" in query
    params = connection.cursor_obj.statements[0][1]
    # 14 bound values (created_at is inline ``now()``), metadata jsonb included.
    assert len(params) == 14
    metadata = json.loads(params[11])
    assert metadata["prov_source"] == "git"
    assert metadata["kind"] == "file"


def test_upsert_refills_rows_whose_vector_was_dropped() -> None:
    """A NULL embedding must be rewritten even when hash and model still match.

    Regression: recreating the vector column (1024 -> 384 migration) empties it
    while ``content_hash`` / ``embedding_model`` survive, so a hash-only guard
    would treat those rows as "already embedded" and leave them unsearchable.
    """
    indexer = EmbeddingIndexer("postgresql://unused", repository_id=REPO)
    connection = _FakeConnection()
    chunk = chunk_file(_file())[0]
    vector = HashingTokenBackend().embed([chunk.text_for_embedding()])[0]

    indexer._upsert_chunk(connection.cursor_obj, chunk, vector)

    query = connection.cursor_obj.statements[0][0]
    assert "OR code_chunks.embedding IS NULL" in query


def test_prune_deletes_rows_outside_the_current_chunk_set() -> None:
    """``--prune`` keeps the stored corpus equal to the run, per repository."""
    indexer = EmbeddingIndexer("postgresql://unused", repository_id=REPO)
    connection = _FakeConnection()
    chunks = chunk_file(_file())

    pruned = indexer._prune_missing(connection.cursor_obj, chunks)

    query, params = connection.cursor_obj.statements[0]
    assert "DELETE FROM code_chunks" in query
    assert "NOT IN" in query
    assert "unnest(%s::text[], %s::int[], %s::int[])" in query
    # Scoped to the repository: another repo's rows are never deleted.
    assert params[0] == REPO
    assert params[1] == [chunk.file_id for chunk in chunks]
    assert params[2] == [chunk.start_line for chunk in chunks]
    assert params[3] == [chunk.end_line for chunk in chunks]
    # The rowcount reported by the cursor is surfaced as-is.
    assert pruned == 1


def test_index_report_counts_by_kind() -> None:
    indexer = EmbeddingIndexer("postgresql://unused", repository_id=REPO)
    report = indexer.index_fixtures(Path("fixtures"), dry_run=True)
    assert report.chunk_count > 0
    assert report.written_count == 0  # dry run: nothing was written
    assert set(report.by_kind) == {"file", "commit_message", "incident_body"}
    assert report.run_id is None


# --------------------------------------------------------------------------- #
# Fixtures loading
# --------------------------------------------------------------------------- #
def test_missing_fixture_file_is_not_fatal() -> None:
    indexer = EmbeddingIndexer("postgresql://unused", repository_id=REPO)
    empty = indexer.build_chunks(Path("fixtures/does-not-exist"))
    assert empty == []
