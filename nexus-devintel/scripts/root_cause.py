#!/usr/bin/env python3
"""CLI: build an EvidencePath for "why did this incident happen?".

Phase 2 usage (stack running + fixtures loaded)::

    python scripts/root_cause.py httpie/cli#issue-1583
    python scripts/root_cause.py httpie/cli#issue-1583 --max-depth 4 --json out.json
    python scripts/root_cause.py httpie/cli#issue-1583 --write
    python scripts/root_cause.py httpie/cli#issue-1583 --hybrid --hybrid-query \
        "SSL certificate verify failed after requests upgrade"

Read-only against Neo4j and PostgreSQL; ``--write`` additionally persists the
Evidence nodes (``MERGE``d, so re-running is safe).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import RootCauseAnalyzer  # noqa: E402


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _print_hybrid(report: "object", limit: int = 5) -> None:
    print(f"  backend: {report.backend_name}")
    for evidence in report.evidences[:limit]:
        citation = evidence.node_references[0]
        print(f"    {evidence.score:>6.3f}  {citation.node_kind.value:<10} "
              f"{citation.node_id}  [{citation.line_start}-{citation.line_end}]"
              if citation.line_start else
              f"    {evidence.score:>6.3f}  {citation.node_kind.value:<10} {citation.node_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incident_id", help="incident id, e.g. httpie/cli#issue-1583")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="maximum traversal depth (default 6)")
    parser.add_argument("--limit", type=int, default=200,
                        help="maximum number of traversal rows (default 200)")
    parser.add_argument("--json", help="write the full report to this file")
    parser.add_argument("--write", action="store_true",
                        help="persist the Evidence nodes into Neo4j (idempotent)")
    parser.add_argument("--hybrid", action="store_true",
                        help="also run the hybrid (lexical+vector) retrieval and "
                             "merge its evidence into the report")
    parser.add_argument("--hybrid-query", default=None,
                        help="query text for --hybrid (default: the incident title)")
    parser.add_argument("--hybrid-mock-embeddings", action="store_true",
                        help="use deterministic hash vectors for --hybrid "
                             "(offline tests only, dim 384)")
    parser.add_argument("--hybrid-hashing-embeddings", action="store_true",
                        help="use the stdlib hashing-token backend for --hybrid "
                             "(dim 384, no torch)")
    args = parser.parse_args()

    _load_dotenv(PROJECT_ROOT / ".env")
    uri = os.environ.get("NEO4J_URI") or "bolt://localhost:7687"
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("error: the neo4j driver is not installed (pip install -r requirements.txt)",
              file=sys.stderr)
        return 2

    with GraphDatabase.driver(uri, auth=(
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "user"),
    )) as driver:
        analyzer = RootCauseAnalyzer(driver, database)
        try:
            report = analyzer.analyze(args.incident_id, max_depth=args.max_depth,
                                      limit=args.limit)
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

        hybrid_report = None
        if args.hybrid:
            try:
                from ingestion.embedding_indexer import (
                    HashingTokenBackend,
                    MiniLMBackend,
                    MockEmbeddingBackend,
                )
                from retrieval import HybridRetriever

                dsn = os.environ.get("POSTGRES_DSN") or \
                    "postgresql://nexus:nexus@localhost:5432/nexus"
                if args.hybrid_mock_embeddings:
                    # Only correct against an index built with --mock-embeddings.
                    backend = MockEmbeddingBackend()
                elif args.hybrid_hashing_embeddings:
                    backend = HashingTokenBackend()
                else:
                    # Default: real MiniLM, matching the indexer's default.
                    backend = MiniLMBackend()
                retriever = HybridRetriever(dsn, backend)
                hybrid_report = retriever.retrieve(
                    args.hybrid_query or report.path.hops[0].node_id,
                    match_count=10,
                )
            except Exception as error:  # noqa: BLE001 - retrieval is best-effort
                print(f"warning: hybrid retrieval skipped ({error})", file=sys.stderr)

        if args.write and report.evidences:
            from ingestion import GraphWriter

            writer = GraphWriter(driver, database)
            models_to_write = list(report.evidences)
            if hybrid_report:
                models_to_write.extend(hybrid_report.evidences)
            stats = writer.write_models(models_to_write)
            for key, value in sorted(stats.items()):
                print(f"  wrote {key:<28} {value}")

    path = report.path
    print(f"== Root Cause: {report.incident_id} ==")
    print(f"  fix commits     : {len(report.fix_commits)}")
    print(f"  fixing PRs      : {len(report.fixing_prs)}")
    print(f"  affected files  : {len(report.affected_files)}")
    print(f"  deployments     : {len(report.deployments)}")
    print(f"  path score      : {path.score} (valid={path.is_valid})")
    if report.best_chain:
        print("  best chain:")
        for element in report.best_chain:
            print(f"    {element}")
    print("  evidence hops:")
    for hop in path.hops[:12]:
        print(f"    {hop.score:>6.3f}  #{hop.step:<3} {hop.node_kind.value:<11} "
              f"{hop.node_id}  via {hop.relation_in or '-'}")
        if hop.rationale and hop.rationale.startswith("reached"):
            print(f"           {hop.rationale}")
    if not path.hops[1:]:
        print("    (none: this incident has no linked PR/commit/file in the graph)")
    if hybrid_report:
        print("  hybrid (lexical+vector) evidence:")
        _print_hybrid(hybrid_report)

    if args.json:
        Path(args.json).write_text(
            json.dumps({
                "incident_id": report.incident_id,
                "computed_at": report.computed_at.isoformat(),
                "fix_commits": report.fix_commits,
                "fixing_prs": report.fixing_prs,
                "affected_files": report.affected_files,
                "deployments": report.deployments,
                "best_chain": report.best_chain,
                "evidence_path": path.model_dump(mode="json"),
                "evidences": [e.model_dump(mode="json") for e in report.evidences],
                "hybrid": {
                    "query": hybrid_report.query,
                    "backend": hybrid_report.backend_name,
                    "matches": [
                        {
                            "chunk_id": m.chunk_id,
                            "file_id": m.file_id,
                            "rrf_score": m.rrf_score,
                            "vector_rank": m.vector_rank,
                            "lexical_rank": m.lexical_rank,
                        }
                        for m in hybrid_report.matches
                    ],
                } if hybrid_report else None,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nfull report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
