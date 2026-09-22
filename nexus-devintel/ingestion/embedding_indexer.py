"""Index the fixtures into pgvector: chunking, embeddings, idempotent upserts.

This is the Week-1 Phase 2 deliverable: prove the whole indexing pipeline end
to end on *fixture* data, without touching the real graph Person A is loading.

Pipeline (``EmbeddingIndexer``):

1. **chunk** -- every ``File`` fixture becomes line-bounded ``CodeChunk`` rows
   whose ids follow the scheme reserved in ``schema/README.md``
   (``owner/name::path#L<start>-L<end>``), so a chunk always re-attaches to its
   ``(:File)`` node through ``code_chunks.file_id``. Commit messages and
   incident bodies get deterministic pseudo-chunks (``#L1``) because the
   ``code_chunks`` table requires line bounds;
2. **embed** -- ``BAAI/bge-m3`` through ``sentence-transformers``, dimension
   1024, matching ``vector(1024)`` in ``schema/postgres/001_init.sql``. A
   deterministic hash-based backend (``--mock-embeddings``) keeps the tests
   offline and instant: same chunk text -> same vector;
3. **write** -- one idempotent ``UPSERT`` into ``code_chunks`` keyed on
   ``(file_id, start_line, end_line)`` (the table's UNIQUE constraint), with
   ``content_hash`` guarding against embedding a stale body, plus one ledger
   row in ``ingestion_runs`` -- the pgvector counterpart of ``prov_*``.

Text provenance is preserved per row: the ``metadata`` jsonb column carries
``prov_source`` / ``prov_confidence`` copied from the fixture node, so a
synthetic incident (``prov_confidence = 0.2``) never ranks as high as a real
source file -- the vector store mirrors the graph's trust model.

Read-only guarantee: this module never touches Neo4j and never touches GitHub;
its only write target is the local PostgreSQL container.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from ingestion.content import ORIGIN_SYNTHETIC, content_with_origin
from models import (
    Commit,
    File,
    Incident,
    Provenance,
    SourceKind,
    utcnow,
)

EXTRACTOR = "nexus-devintel.ingestion.embedding_indexer"
EXTRACTOR_VERSION = "0.1.0"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

#: Lines per chunk: ~40 lines matches the granularity the hybrid retriever
#: needs (a function plus its docstring) without blowing up the vector count.
CHUNK_LINES = 40
#: Overlap between consecutive chunks, so a symbol split across a boundary
#: is still retrievable whole from at least one chunk.
CHUNK_OVERLAP = 5
#: Only source files are chunked; a 613-byte YAML workflow is noise for the
#: vector half of hybrid retrieval (its path is still matched by trigram/lexical).
EMBEDDABLE_EXTENSIONS = {".py", ".md", ".rst", ".txt"}


# --------------------------------------------------------------------------- #
# Chunk model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CodeChunk:
    """One insertable ``code_chunks`` row (embedding not yet attached)."""

    chunk_id: str          # owner/name::path#L<start>-L<end>
    repository_id: str
    file_id: str           # (:File).id -- the join key back to Neo4j
    path: str
    module_path: str | None
    language: str | None
    symbol: str | None
    start_line: int
    end_line: int
    content: str
    content_hash: str      # sha256(content), for idempotent re-ingestion
    kind: str              # file | commit_message | incident_body
    source_node_id: str    # id of the fixture node the text was taken from
    provenance: Provenance
    #: What the text actually is (``synthetic`` / ``git_blob_verified`` /
    #: ``api_blob``), surfaced in the row metadata for honesty.
    content_origin: str = ORIGIN_SYNTHETIC

    def text_for_embedding(self) -> str:
        """``bge-m3`` is trained on raw text: no prefix, no markdown scaffolding."""
        return self.content


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def make_chunk_id(file_id: str, start_line: int, end_line: int) -> str:
    """``httpie/cli::httpie/utils.py`` + 221 + 245 -> ``...#L221-L245``."""
    return f"{file_id}#L{start_line}-L{end_line}"


def _module_path(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    dotted = path[:-len(".py")].replace("/", ".")
    return dotted.removesuffix(".__init__")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_file(file: File, *, content_provider: "ContentProvider | None" = None,
               chunk_lines: int = CHUNK_LINES,
               chunk_overlap: int = CHUNK_OVERLAP) -> list[CodeChunk]:
    """Split one ``(:File)`` fixture into line-bounded chunks.

    Fixture ``File`` nodes carry metadata but not the file *body*, so the
    chunker works either from a real blob (``content_provider``) or from a
    deterministic synthetic body derived from the file's import graph -- the
    point of Week 1 is to prove the pipeline, and the real bodies arrive with
    the Week-3 full ingestion. Deterministic synthetic text keeps the
    ``content_hash`` (and therefore the UPSERT) stable across runs.
    """
    if content_provider is not None:
        content, origin = content_with_origin(content_provider, file)
    else:
        content, origin = _synthetic_file_body(file), ORIGIN_SYNTHETIC
    if not content.strip():
        return []

    if not _is_embeddable(file):
        return []

    # Strip the fixture's null-style padding so empty tails don't become chunks.
    lines = content.splitlines()
    chunks: list[CodeChunk] = []
    start = 1
    total = len(lines)
    step = max(1, chunk_lines - chunk_overlap)
    while start <= total:
        end = min(start + chunk_lines - 1, total)
        body = "\n".join(lines[start - 1:end])
        if body.strip():
            chunks.append(
                CodeChunk(
                    chunk_id=make_chunk_id(file.id, start, end),
                    repository_id=file.repository_id,
                    file_id=file.id,
                    path=file.path,
                    module_path=file.module_path or _module_path(file.path),
                    language=file.language,
                    symbol=_first_symbol(body),
                    start_line=start,
                    end_line=end,
                    content=body,
                    content_hash=content_hash(body),
                    kind="file",
                    source_node_id=file.id,
                    provenance=file.provenance,
                    content_origin=origin,
                )
            )
        if end >= total:
            break
        start += step
    return chunks


def chunk_commit_message(commit: Commit) -> list[CodeChunk]:
    """A commit's ``subject + body`` as one pseudo-chunk anchored at line 1.

    Commit messages are the strongest textual signal for root cause ("Close
    #1583: pin requests<2.32.3"), and the ``code_chunks`` table joins them to
    the graph through ``commit_id`` -- no ``(:File)`` involved.
    """
    text = "\n".join(part for part in (commit.subject, commit.body) if part)
    if not text.strip():
        return []
    pseudo_file_id = f"{commit.repository_id}::commits/{commit.id}"
    return [
        CodeChunk(
            chunk_id=make_chunk_id(pseudo_file_id, 1, text.count("\n") + 1),
            repository_id=commit.repository_id,
            file_id=pseudo_file_id,
            path=f"commits/{commit.id}",
            module_path=None,
            language="text",
            symbol=None,
            start_line=1,
            end_line=text.count("\n") + 1,
            content=text,
            content_hash=content_hash(text),
            kind="commit_message",
            source_node_id=commit.id,
            provenance=commit.provenance,
            content_origin="commit_message",
        )
    ]


def chunk_incident(incident: Incident) -> list[CodeChunk]:
    """An incident's ``title + body`` -- the query side of root cause analysis."""
    text = "\n".join(part for part in (incident.title, incident.body) if part)
    if not text.strip():
        return []
    pseudo_file_id = f"{incident.repository_id}::incidents/{incident.number}"
    return [
        CodeChunk(
            chunk_id=make_chunk_id(pseudo_file_id, 1, text.count("\n") + 1),
            repository_id=incident.repository_id,
            file_id=pseudo_file_id,
            path=f"incidents/{incident.number}",
            module_path=None,
            language="text",
            symbol=None,
            start_line=1,
            end_line=text.count("\n") + 1,
            content=text,
            content_hash=content_hash(text),
            kind="incident_body",
            source_node_id=incident.id,
            provenance=incident.provenance,
            content_origin="incident_body",
        )
    ]


def _is_embeddable(file: File) -> bool:
    if file.extension is None:
        return False
    return file.extension in EMBEDDABLE_EXTENSIONS


def _first_symbol(body: str) -> str | None:
    """The first ``def``/``class`` in the chunk, so results display a symbol."""
    for line in body.splitlines():
        stripped = line.lstrip()
        for keyword in ("def ", "class "):
            if stripped.startswith(keyword):
                return stripped[len(keyword):].split("(")[0].split(":")[0].strip() or None
    return None


def _synthetic_file_body(file: File) -> str:
    """A stable, textual stand-in for the file body.

    Deterministic from the fixture's own fields (imports, path, loc): the same
    fixture always produces the same text, hence the same hash and the same
    vector. It *is* labelled as synthetic through the node's provenance, which
    the metadata column carries into pgvector.
    """
    header = f"# {file.path} ({file.language or 'text'}, {file.loc or 0} loc)"
    import_lines = [
        f"# imports {imp.target_path}: {', '.join(imp.imported_names) or 'module'}"
        for imp in file.imports
    ]
    body_lines = max(int(file.loc or 1), len(import_lines) + 2, 3)
    filler = [f"# line {i}: {file.path}#L{i}" for i in range(1, body_lines + 1)]
    return "\n".join([header, *import_lines, *filler])


class ContentProvider(Protocol):
    """Callable protocol: ``content_provider(file) -> file body text``."""

    def __call__(self, file: File) -> str: ...


# --------------------------------------------------------------------------- #
# Embedding backends
# --------------------------------------------------------------------------- #
class EmbeddingBackend:
    """Strategy interface: ``embed(texts) -> list[list[float]]``."""

    name: str = "abstract"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError


class BgeM3Backend(EmbeddingBackend):
    """The real backend: ``sentence-transformers`` + ``BAAI/bge-m3`` (dim 1024).

    Loaded lazily: importing sentence-transformers pulls torch (~2 GB), so the
    module must stay importable for tests and ``--dry-run`` without the model
    on disk. Normalization is on (bge models are trained for cosine similarity
    on normalized vectors, which is exactly what pgvector's ``<=>`` expects).

    ``NEXUS_EMBEDDING_MODEL_PATH`` points at a local snapshot directory (e.g.
    ``models/bge-m3`` filled by ``scripts/fetch_bge_m3.sh``) and skips the
    hub entirely -- useful when HF connections stall.
    """

    name = EMBEDDING_MODEL

    def __init__(self, model_name: str | None = None, batch_size: int = 32) -> None:
        import os

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "sentence-transformers is not installed. Install the embedding "
                "extra: pip install -r requirements-embeddings.txt"
            ) from error
        local_path = model_name or os.environ.get("NEXUS_EMBEDDING_MODEL_PATH")
        source = local_path or EMBEDDING_MODEL
        self._model = SentenceTransformer(source)
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result: list[list[float]] = [list(map(float, vector)) for vector in vectors]
        if result and len(result[0]) != EMBEDDING_DIM:
            raise ValueError(
                f"model returned {len(result[0])} dimensions, schema expects "
                f"vector({EMBEDDING_DIM}); change 001_init.sql and re-create the "
                "volume before switching models"
            )
        return result


class MockEmbeddingBackend(EmbeddingBackend):
    """Deterministic hash vectors for offline tests and smoke runs.

    Not semantically meaningful -- only stable: the same text always yields the
    same vector, so UPSERTs stay idempotent and tests can assert exact rows
    without torch or a model download.
    """

    name = "mock-hash-1024"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [byte for index in range(EMBEDDING_DIM) for byte in
                   digest[(index * 3) % len(digest):][:1]]
            # Normalize to unit length, like the real backend does.
            norm = sum(value * value for value in raw) ** 0.5 or 1.0
            vectors.append([round(value / norm, 6) for value in raw])
        return vectors


class HashingTokenBackend(EmbeddingBackend):
    """Sparse bag-of-tokens vectors via feature hashing (stdlib-only, dim 1024).

    The efficient alternative to the 2.3 GB ``bge-m3`` download when the goal
    is a *working* hybrid retrieval rather than state-of-the-art semantics:

    * **zero download, zero dependency** -- ``hashlib`` + ``re`` only, so it
      runs wherever the test suite runs (774 chunks embed in ~1 second);
    * **real token-level signal** -- each token (plus adjacent-token bigrams)
      owns hashed dimension(s), so two texts sharing ``ssl`` / ``verify`` /
      ``certificate`` get a high cosine while unrelated texts stay near zero.
      The mock backend hashes the *whole text* instead, which carries no
      token-level similarity at all;
    * **pgvector-compatible** -- dim 1024, L2-normalized, cosine-ready, the
      same contract as ``BgeM3Backend`` (signed hashing à la Vowpal Wabbit
      to dampen collision bias);
    * **deterministic** -- ``sha256``-based (never ``hash()``), so re-runs
      are idempotent and the ``content_hash`` guards keep working.

    Honest labelling: ``name = "hashing-token-1024"``, stored in
    ``code_chunks.embedding_model`` -- never masquerading as ``BAAI/bge-m3``.
    A later bge-m3 pass simply re-embeds (``_filter_pending`` keys on the
    model name, ``_upsert_chunk`` rewrites on model drift).
    """

    name = "hashing-token-1024"

    def __init__(self, use_bigrams: bool = True) -> None:
        self._use_bigrams = use_bigrams

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            tokens = self._tokens(text)
            if self._use_bigrams and len(tokens) >= 2:
                tokens = tokens + [f"{first} {second}"
                                   for first, second in zip(tokens, tokens[1:])]
            dense = [0.0] * EMBEDDING_DIM
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
                sign = 1.0 if digest[4] & 1 else -1.0
                weight = 0.5 if " " in token else 1.0  # bigrams count half
                dense[index] += sign * weight
            norm = sum(value * value for value in dense) ** 0.5 or 1.0
            vectors.append([round(value / norm, 6) for value in dense])
        return vectors


# --------------------------------------------------------------------------- #
# Indexer
# --------------------------------------------------------------------------- #
@dataclass
class IndexReport:
    """What one run did -- printed by the CLI, asserted by the tests."""

    repository_id: str
    chunk_count: int = 0
    embedded_count: int = 0
    written_count: int = 0
    skipped_hashes: int = 0
    already_embedded: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    run_id: str | None = None


class EmbeddingIndexer:
    """Fixtures -> chunks -> embeddings -> ``code_chunks`` (idempotent).

    Usage::

        indexer = EmbeddingIndexer(dsn, backend=MockEmbeddingBackend())
        report = indexer.index_fixtures(Path("fixtures"))
    """

    def __init__(self, dsn: str, backend: EmbeddingBackend | None = None,
                 *, repository_id: str = "httpie/cli") -> None:
        self._dsn = dsn
        self._backend = backend
        self._repository_id = repository_id

    # ---- chunking --------------------------------------------------------- #
    def build_chunks(self, fixtures_dir: Path, *, content_provider: Any = None,
                     extra_files: list[File] | None = None) -> list[CodeChunk]:
        """Chunk every fixture worth indexing, in a stable order.

        ``content_provider`` (see :mod:`ingestion.content`) upgrades file
        chunks from synthetic bodies to real, blob-verified content.
        ``extra_files`` indexes additional ``(:File)`` models (e.g. the real
        HEAD tree ingested separately) that the fixtures do not carry.
        """
        chunks: list[CodeChunk] = []
        files = self._load_fixtures(fixtures_dir, "files", File)
        commits = self._load_fixtures(fixtures_dir, "commits", Commit)
        incidents = self._load_fixtures(fixtures_dir, "incidents", Incident)

        for file in sorted(files, key=lambda item: item.id):
            chunks.extend(chunk_file(file, content_provider=content_provider))
        for commit in sorted(commits, key=lambda item: item.id):
            chunks.extend(chunk_commit_message(commit))
        for incident in sorted(incidents, key=lambda item: item.id):
            chunks.extend(chunk_incident(incident))

        fixture_ids = {file.id for file in files}
        for file in sorted(extra_files or [], key=lambda item: item.id):
            if file.id in fixture_ids:
                continue
            chunks.extend(chunk_file(file, content_provider=content_provider))
        return chunks

    # ---- indexing --------------------------------------------------------- #
    def index_fixtures(self, fixtures_dir: Path, *, dry_run: bool = False,
                       batch_size: int = 64, content_provider: Any = None,
                       extra_files: list[File] | None = None) -> IndexReport:
        """Full pipeline. ``dry_run`` stops right before any SQL is sent.

        Resumable: chunks already embedded with the *same model* (same
        ``content_hash`` + ``embedding_model`` row) are skipped before the
        backend runs, so a long bge-m3 run interrupted midway can be replayed
        without re-embedding (or re-paying for) what is already stored.
        """
        chunks = self.build_chunks(fixtures_dir, content_provider=content_provider,
                                   extra_files=extra_files)
        report = IndexReport(repository_id=self._repository_id)
        report.chunk_count = len(chunks)
        for chunk in chunks:
            report.by_kind[chunk.kind] = report.by_kind.get(chunk.kind, 0) + 1
        if dry_run or not chunks:
            return report

        assert self._backend is not None, "a backend is required to write"
        pending = self._filter_pending(chunks)
        report.skipped_hashes = len(chunks) - len(pending)
        if not pending:
            return report

        import psycopg

        # Batch-granular commits: a CPU bge-m3 run can take longer than any
        # single shell timeout -- every committed batch is durable and the
        # next run resumes after it (see _filter_pending).
        with psycopg.connect(self._dsn) as connection:
            run_id = self._open_run(connection)
            report.run_id = str(run_id)
            try:
                for start in range(0, len(pending), batch_size):
                    window = pending[start:start + batch_size]
                    vectors = self._embed_all(window, batch_size)
                    report.embedded_count += len(vectors)
                    with connection.cursor() as cursor:
                        for chunk, vector in zip(window, vectors):
                            report.written_count += self._upsert_chunk(cursor, chunk, vector)
                    connection.commit()
            except Exception:
                self._fail_run(connection, run_id, report)
                raise
            self._close_run(connection, run_id, report)
            connection.commit()
        return report

    def _fail_run(self, connection: Any, run_id: Any, report: IndexReport) -> None:
        """Mark the ledger row as failed (best effort: the run must not lie)."""
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE ingestion_runs SET status = 'failed', "
                    "finished_at = now(), stats = %s WHERE run_id = %s",
                    (json.dumps({"written": report.written_count}), run_id),
                )
            connection.commit()
        except Exception:  # noqa: BLE001 - already in a failure path
            pass

    def _filter_pending(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Drop chunks already embedded with the current model (resumability)."""
        import psycopg

        model_name = self._backend.name if self._backend else EMBEDDING_MODEL
        hashes = [chunk.content_hash for chunk in chunks]
        done: set[str] = set()
        with psycopg.connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                for start in range(0, len(hashes), 500):
                    window = hashes[start:start + 500]
                    cursor.execute(
                        "SELECT content_hash FROM code_chunks "
                        "WHERE embedding IS NOT NULL AND embedding_model = %s "
                        "AND content_hash = ANY(%s)",
                        (model_name, window),
                    )
                    done.update(row[0] for row in cursor.fetchall())
        return [chunk for chunk in chunks if chunk.content_hash not in done]

    def _embed_all(self, chunks: list[CodeChunk], batch_size: int) -> list[list[float]]:
        vectors: list[list[float]] = []
        backend = self._backend
        assert backend is not None
        for start in range(0, len(chunks), batch_size):
            window = chunks[start:start + batch_size]
            vectors.extend(backend.embed([chunk.text_for_embedding() for chunk in window]))
        return vectors

    # ---- SQL --------------------------------------------------------------- #
    def _upsert_chunk(self, cursor: Any, chunk: CodeChunk, vector: list[float]) -> int:
        """One ``UPSERT``; returns 1 when a row was written, 0 on a no-op hash.

        Idempotency is two-layered: the UNIQUE ``(file_id, start_line,
        end_line)`` constraint absorbs replays, and the ``WHERE`` clause skips
        re-embedding rows whose content (and therefore vector) is unchanged.
        """
        metadata = {
            "kind": chunk.kind,
            "source_node_id": chunk.source_node_id,
            "prov_source": chunk.provenance.source.value,
            "prov_confidence": chunk.provenance.confidence,
            "prov_extractor": EXTRACTOR,
            "prov_extractor_version": EXTRACTOR_VERSION,
            "embedding_model": self._backend.name if self._backend else EMBEDDING_MODEL,
            "content_origin": chunk.content_origin,
        }
        cursor.execute(
            """
            INSERT INTO code_chunks
                (repository_id, file_id, path, module_path, language, symbol,
                 start_line, end_line, content, content_hash, commit_id,
                 metadata, embedding, embedding_model, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (file_id, start_line, end_line) DO UPDATE
                SET content = EXCLUDED.content,
                    content_hash = EXCLUDED.content_hash,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    symbol = EXCLUDED.symbol,
                    metadata = EXCLUDED.metadata
            WHERE code_chunks.content_hash IS DISTINCT FROM EXCLUDED.content_hash
               OR code_chunks.embedding_model IS DISTINCT FROM EXCLUDED.embedding_model
            """,
            (
                chunk.repository_id,
                chunk.file_id,
                chunk.path,
                chunk.module_path,
                chunk.language,
                chunk.symbol,
                chunk.start_line,
                chunk.end_line,
                chunk.content,
                chunk.content_hash,
                chunk.source_node_id if chunk.kind == "commit_message" else None,
                json.dumps(metadata),
                _vector_literal(vector),
                self._backend.name if self._backend else EMBEDDING_MODEL,
            ),
        )
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0

    def _open_run(self, connection: Any) -> Any:
        """Open an ``ingestion_runs`` ledger row (the pgvector-side provenance)."""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ingestion_runs (repository_id, extractor,
                                            extractor_version, status)
                VALUES (%s, %s, %s, 'running')
                RETURNING run_id
                """,
                (self._repository_id, EXTRACTOR, EXTRACTOR_VERSION),
            )
            return cursor.fetchone()[0]

    def _close_run(self, connection: Any, run_id: Any, report: IndexReport) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ingestion_runs
                SET status = 'succeeded', finished_at = now(), stats = %s
                WHERE run_id = %s
                """,
                (json.dumps({
                    "chunks": report.chunk_count,
                    "embedded": report.embedded_count,
                    "written": report.written_count,
                    "by_kind": report.by_kind,
                }), run_id),
            )

    # ---- fixtures -----------------------------------------------------------#
    @staticmethod
    def _load_fixtures(fixtures_dir: Path, stem: str, model_class: type) -> list:
        path = fixtures_dir / f"{stem}.json"
        if not path.exists():
            return []
        items = []
        for payload in json.loads(path.read_text(encoding="utf-8")):
            items.append(model_class.model_validate(payload))
        return items


def _vector_literal(vector: Iterable[float]) -> str:
    """``'[0.1,0.2,...]'`` -- pgvector's text input format for ``vector(1024)``."""
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"


__all__ = [
    "BgeM3Backend",
    "CHUNK_LINES",
    "CHUNK_OVERLAP",
    "CodeChunk",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "EmbeddingBackend",
    "EmbeddingIndexer",
    "HashingTokenBackend",
    "IndexReport",
    "MockEmbeddingBackend",
    "chunk_commit_message",
    "chunk_file",
    "chunk_incident",
    "content_hash",
    "make_chunk_id",
]
