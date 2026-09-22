#!/usr/bin/env python3
"""CLI: index the fixtures into pgvector (code_chunks + ingestion_runs).

Week-1 Phase 2 usage::

    # 1. validate the chunking plan without any database (offline)
    python scripts/index_embeddings.py --dry-run

    # 2. smoke test against PostgreSQL with deterministic offline vectors
    python scripts/index_embeddings.py --mock-embeddings

    # 3. the real thing (needs sentence-transformers + BAAI/bge-m3 on disk)
    python scripts/index_embeddings.py

Everything is idempotent: re-running updates nothing whose ``content_hash``
is unchanged, and every run is recorded in ``ingestion_runs``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.embedding_indexer import (  # noqa: E402
    EmbeddingIndexer,
    MockEmbeddingBackend,
)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        import os
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", default=str(PROJECT_ROOT / "fixtures"))
    parser.add_argument("--dsn", default=None,
                        help="PostgreSQL DSN (default: POSTGRES_DSN from .env)")
    parser.add_argument("--dry-run", action="store_true",
                        help="chunk and validate, but send nothing to PostgreSQL")
    parser.add_argument("--mock-embeddings", action="store_true",
                        help="use deterministic hash vectors instead of bge-m3 "
                             "(offline smoke test of the SQL pipeline)")
    parser.add_argument("--repository-id", default=None,
                        help="scope of the run (default: NEXUS_REPO_ID / httpie/cli)")
    args = parser.parse_args()

    _load_dotenv(PROJECT_ROOT / ".env")
    import os
    dsn = args.dsn or os.environ.get("POSTGRES_DSN") or \
        "postgresql://nexus:nexus@localhost:5432/nexus"
    repository_id = args.repository_id or os.environ.get("NEXUS_REPO_ID", "httpie/cli")

    indexer = EmbeddingIndexer(dsn, repository_id=repository_id)

    if args.dry_run:
        chunks = indexer.build_chunks(Path(args.fixtures_dir))
        by_kind: dict[str, int] = {}
        for chunk in chunks:
            by_kind[chunk.kind] = by_kind.get(chunk.kind, 0) + 1
        print(f"== Dry run: {len(chunks)} chunks would be indexed ==")
        for kind, count in sorted(by_kind.items()):
            print(f"  {kind:<18} {count}")
        print("  sample chunk:")
        if chunks:
            sample = chunks[0]
            print(f"    id      : {sample.chunk_id}")
            print(f"    lines   : {sample.start_line}-{sample.end_line}")
            print(f"    symbol  : {sample.symbol}")
            print(f"    prov    : {sample.provenance.source.value} "
                  f"(confidence={sample.provenance.confidence})")
        print("  no database was touched.")
        return 0

    if args.mock_embeddings:
        indexer._backend = MockEmbeddingBackend()

    try:
        report = indexer.index_fixtures(Path(args.fixtures_dir))
    except RuntimeError as error:  # backend import failure, dry diagnostics
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"error: {error}", file=sys.stderr)
        print("is the postgres container up? (docker compose up -d postgres)",
              file=sys.stderr)
        return 1

    print(f"== Indexed {report.repository_id} ==")
    print(f"  chunks            : {report.chunk_count}")
    print(f"  embedded          : {report.embedded_count}")
    print(f"  written (upserts) : {report.written_count}")
    for kind, count in sorted(report.by_kind.items()):
        print(f"  {kind:<18} {count}")
    if report.run_id:
        print(f"  ingestion run     : {report.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
