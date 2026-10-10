#!/usr/bin/env python3
"""CLI: enrich the graph with ``(:PR)-[:CLOSES]->(:Incident)`` edges from GraphQL.

Week-2 Phase 2 usage (stack running, fixtures loaded, GITHUB_TOKEN set)::

    # full pass over httpie/cli (~3 GraphQL requests for ~280 PRs)
    python scripts/enrich_graphql.py httpie/cli

    # idempotent targeted re-run on the demo chain's PR
    python scripts/enrich_graphql.py httpie/cli --pr 1596

    # see what would be written, without writing anything (needs a token)
    python scripts/enrich_graphql.py httpie/cli --dry-run

Read-only guarantee: the GraphQL client refuses mutations before transport;
the only writes are idempotent ``MERGE``s into the local Neo4j.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.github_graphql import (  # noqa: E402
    ClosingIssuesEnricher,
    EnrichmentStats,
    GitHubGraphQLClient,
    build_close_edges,
)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _print_stats(stats: EnrichmentStats) -> None:
    print(f"  PRs seen                  : {stats.prs_seen}")
    print(f"  PRs with closing issues   : {stats.prs_with_closing_issues}")
    print(f"  (:PR) stubs created       : {stats.prs_created}")
    print(f"  (:PR)-[:MERGED_INTO]      : {stats.merge_edges_built}")
    print(f"  (:PR)-[:CLOSES] written   : {stats.edges_built}")
    print(f"  incidents discovered      : {len(stats.incidents_discovered or set())}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository_id", help="owner/name, e.g. httpie/cli")
    parser.add_argument("--pr", type=int, action="append", default=None,
                        help="restrict the pass to one PR number (repeatable)")
    parser.add_argument("--no-create-incidents", action="store_true",
                        help="do not create stub (:Incident) nodes for unknown issues")
    parser.add_argument("--no-create-prs", action="store_true",
                        help="do not create stub (:PR) nodes for unknown PRs "
                             "(edges whose PR is missing will be skipped)")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and report, but write nothing (needs a token)")
    args = parser.parse_args()

    _load_dotenv(PROJECT_ROOT / ".env")

    try:
        client = GitHubGraphQLClient()
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        count = 0
        for pr_links in client.iter_pull_requests_with_closing_issues(args.repository_id):
            if args.pr and pr_links.number not in args.pr:
                continue
            edges = build_close_edges(pr_links, repository_id=args.repository_id)
            for edge in edges:
                print(f"  would write: ({edge.source_id})-[:CLOSES]->({edge.target_id})")
                count += 1
        print(f"  {count} edge(s) would be written; nothing was written.")
        return 0

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("error: the neo4j driver is not installed (pip install -r requirements.txt)",
              file=sys.stderr)
        return 2

    uri = os.environ.get("NEO4J_URI") or "bolt://localhost:7687"
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    with GraphDatabase.driver(uri, auth=(
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "user"),
    )) as driver:
        from ingestion import GraphWriter

        enricher = ClosingIssuesEnricher(client, GraphWriter(driver, database))
        stats = enricher.enrich(
            args.repository_id,
            create_incidents=not args.no_create_incidents,
            create_prs=not args.no_create_prs,
            pr_numbers=args.pr,
        )
    print(f"== GraphQL enrichment of {args.repository_id} ==")
    _print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
