#!/usr/bin/env python3
"""CLI: build an EvidencePath for "what breaks if I change this file?".

Phase 1 usage (stack running + fixtures loaded)::

    python scripts/impact.py httpie/cli::httpie/context.py
    python scripts/impact.py httpie/cli::httpie/ssl_.py --max-depth 3 --json out.json

Read-only against Neo4j; ``--write`` additionally persists the Evidence nodes
and the path (``MERGE``d, so re-running is safe).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import ChangeImpactAnalyzer  # noqa: E402


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_id", help="target file id, e.g. httpie/cli::httpie/context.py")
    parser.add_argument("--max-depth", type=int, default=7,
                        help="maximum IMPORTS traversal depth (default 7)")
    parser.add_argument("--limit", type=int, default=1000,
                        help="maximum number of dependent PATHS returned (paths, not "
                             "distinct files: keep high enough to not truncate the "
                             "blast radius; default 1000)")
    parser.add_argument("--json", help="write the full report (path + evidence) to this file")
    parser.add_argument("--write", action="store_true",
                        help="persist the Evidence nodes into Neo4j (idempotent)")
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
        analyzer = ChangeImpactAnalyzer(driver, database)
        try:
            report = analyzer.analyze(args.file_id, max_depth=args.max_depth,
                                      limit=args.limit)
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

        if args.write and report.evidences:
            from ingestion import GraphWriter
            writer = GraphWriter(driver, database)
            stats = writer.write_models(report.evidences)
            for key, value in sorted(stats.items()):
                print(f"  wrote {key:<28} {value}")

    path = report.path
    print(f"== Change Impact: {report.target_file_id} ==")
    print(f"  direct dependents     : {len(report.direct_dependents)}")
    print(f"  transitive dependents : {report.transitive_dependents}")
    print(f"  max depth seen        : {report.max_depth_seen}")
    print(f"  path score            : {path.score} (valid={path.is_valid})")
    print("  top impacted files:")
    for hop in path.hops[1:11]:
        print(f"    {hop.score:>6.3f}  d{hop.step - 1 if hop.step else 0}  {hop.node_id}")
    if not path.hops[1:]:
        print("    (none: this file has an empty blast radius in the current graph)")

    if args.json:
        Path(args.json).write_text(
            json.dumps({
                "target_file_id": report.target_file_id,
                "computed_at": report.computed_at.isoformat(),
                "direct_dependents": report.direct_dependents,
                "transitive_dependents": report.transitive_dependents,
                "max_depth_seen": report.max_depth_seen,
                "evidence_path": path.model_dump(mode="json"),
                "evidences": [e.model_dump(mode="json") for e in report.evidences],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nfull report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
