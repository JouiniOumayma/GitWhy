#!/usr/bin/env python3
"""CLI: index code chunks into pgvector (fixtures and/or the real repository).

Default backend: ``sentence-transformers/all-MiniLM-L6-v2`` (dim 384,
~90 MB download, ~1 min for 774 chunks on CPU)::

    python scripts/index_embeddings.py --dry-run           # chunking plan, no DB
    python scripts/index_embeddings.py                    # real MiniLM embeddings
    python scripts/index_embeddings.py --hashing-embeddings  # stdlib fallback (no torch)

Week-3 Phase 2 usage (real content, read-only)::

    # real bodies from a local clone (git blob_sha verified, confidence 1.0)
    python scripts/index_embeddings.py --real-content --repo-path ../httpie-cli

    # real bodies from the GitHub blobs API (stdlib GET-only client)
    python scripts/index_embeddings.py --real-content --from-api

    # also ingest the real HEAD tree (133 files) into Neo4j first, then index
    python scripts/index_embeddings.py --real-content --from-api --ingest-tree

All modes are idempotent (UPSERT on content_hash) and every chunk's metadata
records ``content_origin`` (synthetic / git_blob_verified / api_blob).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.content import (  # noqa: E402
    GitHubBlobContentProvider,
    LocalRepoContentProvider,
)
from ingestion.embedding_indexer import (  # noqa: E402
    EmbeddingIndexer,
    HashingTokenBackend,
)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        without_comment = line.split("#", 1)[0].strip()
        if not without_comment or "=" not in without_comment:
            continue
        key, _, value = without_comment.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ingest_real_tree(repository_id: str, repository: str) -> int:
    """Ingest the real HEAD tree into Neo4j via the Phase-1 connector.

    Read-only GET (git/trees recursive), then the existing idempotent
    ``GraphWriter``. Returns the number of ``(:File)`` nodes written.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("error: neo4j driver missing (pip install -r requirements.txt)",
              file=sys.stderr)
        return -1
    from ingestion import GitHubClient, GraphWriter, file_from_api
    from models import Repository

    client = GitHubClient()
    tree = client.git_tree(repository, ref="HEAD", recursive=True)
    files = [file_from_api(entry, repository_id, ref="HEAD")
             for entry in tree.get("tree", []) if entry.get("type") == "blob"]
    uri = os.environ.get("NEO4J_URI") or "bolt://localhost:7687"
    with GraphDatabase.driver(uri, auth=(
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "user"),
    )) as driver:
        writer = GraphWriter(driver, os.environ.get("NEO4J_DATABASE", "neo4j"))
        repo_node = Repository(id=repository_id, owner=repository.split("/")[0],
                               name=repository.split("/")[1])
        written = 0
        for file_model in files:
            writer.write_node(file_model)
            for edge in file_model.edges():
                if edge.type.value == "CONTAINS":
                    writer.write_node(repo_node)
                    if writer.write_edge(edge):
                        written += 1
    # Return the in-memory models: re-reading them from Neo4j would need a
    # neo4j->pydantic roundtrip (temporal types, prov_* flattening) for nothing.
    return written, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", default=str(PROJECT_ROOT / "fixtures"))
    parser.add_argument("--dsn", default=None,
                        help="PostgreSQL DSN (default: POSTGRES_DSN from .env)")
    parser.add_argument("--dry-run", action="store_true",
                        help="chunk and validate, but send nothing to PostgreSQL")
    parser.add_argument("--hashing-embeddings", action="store_true",
                        help="use the stdlib hashing-token backend (dim 384, no download, "
                             "no torch: offline fallback with real token signal)")
    parser.add_argument("--real-embeddings", action="store_true",
                        help="deprecated alias: MiniLM is now the default "
                             "(kept for compatibility)")
    parser.add_argument("--prune", action="store_true",
                        help="delete this repository's rows that the current "
                             "chunk set no longer produces (corpus == run)")
    parser.add_argument("--real-content", action="store_true",
                        help="index real file bodies instead of synthetic text")
    parser.add_argument("--repo-path", default=None,
                        help="local clone path for --real-content (blob_sha verified)")
    parser.add_argument("--from-api", action="store_true",
                        help="fetch real bodies from the GitHub blobs API "
                             "(read-only, GITHUB_TOKEN)")
    parser.add_argument("--ingest-tree", action="store_true",
                        help="also ingest the real HEAD file tree into Neo4j "
                             "(read-only GET + idempotent MERGE)")
    parser.add_argument("--repository-id", default=None)
    args = parser.parse_args()

    _load_dotenv(PROJECT_ROOT / ".env")
    dsn = args.dsn or os.environ.get("POSTGRES_DSN") or \
        "postgresql://nexus:nexus@localhost:5432/nexus"
    repository_id = args.repository_id or os.environ.get("NEXUS_REPO_ID", "httpie/cli")

    provider = None
    blob_cache_dir = PROJECT_ROOT / ".blob_cache"
    if args.real_content or args.ingest_tree:
        if args.from_api:
            provider = GitHubBlobContentProvider(None, repository_id,
                                                 cache_dir=blob_cache_dir)
            print(f"content provider: GitHub blobs API (read-only GET, disk cache {blob_cache_dir})")
        elif args.repo_path:
            provider = LocalRepoContentProvider(args.repo_path)
            print(f"content provider: local clone {args.repo_path}")
        else:
            print("error: --real-content requires --repo-path or --from-api",
                  file=sys.stderr)
            return 2

    indexer = EmbeddingIndexer(dsn, repository_id=repository_id)

    if args.dry_run:
        if args.ingest_tree:
            print("  (dry run: the real HEAD tree would be ingested into Neo4j "
                  "and its files indexed)")
        chunks = indexer.build_chunks(Path(args.fixtures_dir),
                                      content_provider=provider)
        by_kind: dict[str, int] = {}
        for chunk in chunks:
            by_kind[chunk.kind] = by_kind.get(chunk.kind, 0) + 1
        origins: dict[str, int] = {}
        for chunk in chunks:
            origins[chunk.content_origin] = origins.get(chunk.content_origin, 0) + 1
        print(f"== Dry run: {len(chunks)} chunks would be indexed ==")
        for kind, count in sorted(by_kind.items()):
            print(f"  {kind:<18} {count}")
        for origin, count in sorted(origins.items()):
            print(f"  origin={origin:<18} {count}")
        print("  no database was touched.")
        return 0

    extra_files = None
    if args.ingest_tree:
        result = _ingest_real_tree(repository_id, repository_id)
        if result is None:
            return 1
        written, extra_files = result
        print(f"  real tree ingested into Neo4j: {written} new (:File) edges, "
              f"{len(extra_files)} files available for indexing")

    if args.hashing_embeddings:
        indexer._backend = HashingTokenBackend()
        print(f"embedding backend: {indexer._backend.name} (stdlib-only fallback)")
    else:
        # Default (and --real-embeddings): real MiniLM-L6-v2, ~90 MB.
        # Constructed lazily here -- AFTER chunking and blob fetching, so an
        # interrupted run keeps its disk cache warm.
        from ingestion.embedding_indexer import MiniLMBackend

        print("loading sentence-transformers/all-MiniLM-L6-v2 (~90 MB first run)...")
        indexer._backend = MiniLMBackend()

    try:
        report = indexer.index_fixtures(
            Path(args.fixtures_dir),
            content_provider=provider,
            extra_files=extra_files,
            prune=args.prune,
        )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"error: {error}", file=sys.stderr)
        print("is the postgres container up?", file=sys.stderr)
        return 1

    print(f"== Indexed {report.repository_id} ==")
    print(f"  chunks            : {report.chunk_count}")
    print(f"  already embedded  : {report.skipped_hashes} (resumable skip, same model)")
    print(f"  embedded now      : {report.embedded_count}")
    print(f"  written (upserts) : {report.written_count}")
    if args.prune:
        print(f"  pruned (stale)    : {report.pruned_count}")
    for kind, count in sorted(report.by_kind.items()):
        print(f"  {kind:<18} {count}")
    if report.run_id:
        print(f"  ingestion run     : {report.run_id}")
    return 0





if __name__ == "__main__":
    raise SystemExit(main())
