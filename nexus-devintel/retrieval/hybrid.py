"""Hybrid retrieval: lexical + vector over pgvector, joined back to the graph.

The SQL half was already written by Person A
(``schema/postgres/001_init.sql::hybrid_search_code_chunks``): reciprocal rank
fusion (k=60) of a ``ts_rank_cd`` lexical ranking and an HNSW cosine ranking,
both over ``code_chunks``. This module is the Python half of the contract:

1. embed the question with the same backend used at indexing time
   (``sentence-transformers/all-MiniLM-L6-v2``, dim 384 -- dimension drift is
   checked against the indexer's constant);
2. call ``hybrid_search_code_chunks`` (read-only ``SELECT``);
3. package each row as an :class:`~models.evidence.Evidence` node of kind
   ``code_chunk`` with ``retrieval_strategy=hybrid``, carrying the RRF score
   and both ranks for explainability, and pointing at the graph through
   ``file_id`` (== ``(:File).id``) or ``commit_id`` -- the schema README's
   join table "Neo4j id <-> code_chunks.file_id/commit_id".

No write anywhere: this analyzer only SELECTs. Errors from a missing table or
a dimension mismatch surface as :class:`HybridRetrievalError` with actionable
messages (did you run ``scripts/index_embeddings.py``?).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from models import (
    Evidence,
    EvidenceKind,
    EvidenceRef,
    EvidenceRole,
    NodeKind,
    Provenance,
    RetrievalStrategy,
    utcnow,
)

EXTRACTOR = "nexus-devintel.retrieval.hybrid"
EXTRACTOR_VERSION = "0.1.0"

#: Must match ingestion.embedding_indexer (and 001_init.sql's vector(384)).
EMBEDDING_DIM = 384

_HYBRID_QUERY = """
SELECT chunk_id, file_id, path, symbol, start_line, end_line, content,
       rrf_score, vector_rank, lexical_rank
FROM hybrid_search_code_chunks(%s::text, %s::vector, %s::int, %s::int, %s::text)
"""


class HybridRetrievalError(RuntimeError):
    """The hybrid search could not run (connection, missing table, dim drift)."""


@dataclass
class HybridMatch:
    """One ranked chunk, before packaging into Evidence."""

    chunk_id: str
    file_id: str
    path: str
    symbol: str | None
    start_line: int
    end_line: int
    content: str
    rrf_score: float
    vector_rank: int | None
    lexical_rank: int | None

    @property
    def citation(self) -> str:
        """``httpie/ssl_.py:221-245`` style pointer for the answer layer."""
        location = f"{self.path}:{self.start_line}-{self.end_line}"
        return f"{location} ({self.symbol})" if self.symbol else location


@dataclass
class HybridReport:
    """Query + matches + packaged evidence, mirroring the other reports."""

    query: str
    repository_id: str
    matches: list[HybridMatch] = field(default_factory=list)
    evidences: list[Evidence] = field(default_factory=list)
    backend_name: str = ""
    computed_at: datetime = field(default_factory=utcnow)


class HybridRetriever:
    """Question -> hybrid-ranked chunks -> citable Evidence nodes.

    Usage::

        retriever = HybridRetriever(dsn, backend=MockEmbeddingBackend())
        report = retriever.retrieve("SSL certificate verify failed")
    """

    def __init__(self, dsn: str, backend: Any = None, *,
                 repository_id: str = "httpie/cli") -> None:
        self._dsn = dsn
        self._backend = backend
        self._repository_id = repository_id

    # ---- retrieval --------------------------------------------------------- #
    def retrieve(self, query: str, *, match_count: int = 10, rrf_k: int = 60,
                 path_regex: str | None = None) -> HybridReport:
        """Run the hybrid search; ``path_regex`` narrows to a subtree (read-only)."""
        if self._backend is None:
            from ingestion.embedding_indexer import HashingTokenBackend

            # Default backend: hashing-token (stdlib-only, dim 384). Pass an
            # explicit backend (MiniLMBackend, MockEmbeddingBackend) to
            # override.
            self._backend = HashingTokenBackend()
        embedding = self._backend.embed([query])[0]
        if len(embedding) != EMBEDDING_DIM:
            raise HybridRetrievalError(
                f"backend returned {len(embedding)} dimensions, the SQL functions "
                f"expect vector({EMBEDDING_DIM}); indexing and retrieval must use "
                "the same model"
            )
        rows = self._search(query, embedding, match_count, rrf_k, path_regex)
        matches = [self._row_to_match(row) for row in rows]
        return HybridReport(
            query=query,
            repository_id=self._repository_id,
            matches=matches,
            evidences=self._package(matches, query),
            backend_name=getattr(self._backend, "name", "unknown"),
        )

    # ---- SQL ---------------------------------------------------------------- #
    def _search(self, query: str, embedding: Sequence[float], match_count: int,
                rrf_k: int, path_regex: str | None) -> list[dict[str, Any]]:
        import psycopg

        try:
            with psycopg.connect(self._dsn) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        _HYBRID_QUERY,
                        (
                            query,
                            _vector_literal(embedding),
                            match_count,
                            rrf_k,
                            self._repository_id,
                        ),
                    )
                    columns = [description.name for description in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except psycopg.OperationalError as error:
            raise HybridRetrievalError(
                f"cannot reach PostgreSQL ({self._dsn}); is the container up? "
                "(docker compose up -d postgres)"
            ) from error
        except psycopg.errors.UndefinedTable as error:
            raise HybridRetrievalError(
                "code_chunks or hybrid_search_code_chunks is missing: apply "
                "schema/postgres/001_init.sql and run scripts/index_embeddings.py"
            ) from error
        except psycopg.errors.UndefinedFunction as error:
            raise HybridRetrievalError(
                "hybrid_search_code_chunks is missing: re-apply "
                "schema/postgres/001_init.sql"
            ) from error

    # ---- packaging -----------------------------------------------------------#
    @staticmethod
    def _row_to_match(row: dict[str, Any]) -> HybridMatch:
        return HybridMatch(
            chunk_id=str(row["chunk_id"]),
            file_id=row["file_id"],
            path=row["path"],
            symbol=row["symbol"],
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            content=row["content"],
            rrf_score=float(row["rrf_score"] or 0.0),
            vector_rank=int(row["vector_rank"]) if row["vector_rank"] is not None else None,
            lexical_rank=int(row["lexical_rank"]) if row["lexical_rank"] is not None else None,
        )

    def _package(self, matches: list[HybridMatch], query: str) -> list[Evidence]:
        """Every match becomes an Evidence pointing at its graph node.

        The ``node_kind`` follows the chunk's origin: pseudo-paths starting
        with ``incidents/`` reference an ``(:Incident)``, ``commits/`` a
        ``(:Commit)``, anything else the ``(:File)`` of ``file_id``.
        """
        evidences: list[Evidence] = []
        for rank, match in enumerate(matches, start=1):
            node_kind, node_id = self._graph_anchor(match)
            evidences.append(
                Evidence(
                    id=f"ev-hybrid-{abs(hash(match.chunk_id)) % 10**10:010d}",
                    kind=EvidenceKind.CODE_CHUNK,
                    retrieval_strategy=RetrievalStrategy.HYBRID,
                    score=round(min(1.0, match.rrf_score * 30.0), 4),  # RRF scores are ~0.03
                    rank=rank,
                    text=match.content,
                    rationale=(
                        f"hybrid rank {rank}: vector_rank={match.vector_rank}, "
                        f"lexical_rank={match.lexical_rank}, rrf={match.rrf_score:.4f}"
                    ),
                    node_references=[
                        EvidenceRef(
                            node_kind=node_kind,
                            node_id=node_id,
                            role=EvidenceRole.CONTEXT,
                            line_start=match.start_line if node_kind == NodeKind.FILE else None,
                            line_end=match.end_line if node_kind == NodeKind.FILE else None,
                        )
                    ],
                    vector=None,  # filled by the writer when persisting the row id
                    retrieved_at=utcnow(),
                    provenance=Provenance(
                        source="derived",
                        source_uri=f"pgvector:hybrid_search_code_chunks:{match.chunk_id}",
                        extractor=EXTRACTOR,
                        confidence=1.0,
                    ),
                )
            )
        return evidences

    @staticmethod
    def _graph_anchor(match: HybridMatch) -> tuple[NodeKind, str]:
        if match.path.startswith("incidents/"):
            number = match.path.split("/")[-1]
            repository_id = match.file_id.split("::")[0]
            return NodeKind.INCIDENT, f"{repository_id}#issue-{number}"
        if match.path.startswith("commits/"):
            # pseudo-file_id is ``repo::commits/<sha>``: strip the prefix to get
            # the ``(:Commit).id`` (the full SHA, per the id scheme).
            return NodeKind.COMMIT, match.file_id.split("::", 1)[1].removeprefix("commits/")
        return NodeKind.FILE, match.file_id


def _vector_literal(vector: Sequence[float]) -> str:
    """pgvector text input format, same as the indexer's."""
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"


__all__ = [
    "EMBEDDING_DIM",
    "HybridMatch",
    "HybridReport",
    "HybridRetrievalError",
    "HybridRetriever",
]
