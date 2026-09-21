#!/usr/bin/env python3
"""Load the NEXUS-DevIntel fixtures into Neo4j (and optionally pgvector).

Idempotent by construction: nodes are ``MERGE``d on their ``id`` and
relationships are ``MERGE``d on their endpoints, so running the loader twice
converges instead of duplicating.

Usage::

    # 1. see what would happen, without a database
    python scripts/load_neo4j.py --dry-run

    # 2. apply the schema, then load
    python scripts/load_neo4j.py --apply-schema

    # 3. skip the schema (already applied)
    python scripts/load_neo4j.py

    # 4. also push the Evidence rows into pgvector
    python scripts/load_neo4j.py --with-pgvector

Connection settings come from the environment (see .env.example).

Between steps 1 and 2, start the stack with ``docker compose up -d``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    Answer,
    Commit,
    Deployment,
    Evidence,
    File,
    Incident,
    NodeKind,
    PersonRef,
    PullRequest,
    RelationType,
    Repository,
)

NODE_MODELS = {
    "repositories": Repository,
    "files": File,
    "commits": Commit,
    "pull_requests": PullRequest,
    "deployments": Deployment,
    "incidents": Incident,
    "evidence": Evidence,
    "answers": Answer,
}

#: Community-compatible schema, applied in order.
SCHEMA_FILES = [
    "schema/cypher/01_constraints_uniqueness.cypher",
    "schema/cypher/03_indexes.cypher",
    "schema/cypher/04_bootstrap.cypher",
]

#: Existence constraints: Neo4j **Enterprise** only (
#: ``Property existence constraint requires Neo4j Enterprise Edition``).
#: Skipped unless --enterprise is passed, because applying them on Community
#: aborts the script halfway and leaves a half-configured database.
ENTERPRISE_SCHEMA_FILES = [
    "schema/cypher/02_constraints_existence_enterprise.cypher",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_dotenv(path: Path) -> None:
    """Minimal ``.env`` reader (avoids a python-dotenv dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    uri: str
    user: str
    password: str
    database: str
    dsn: str | None
    batch_size: int

    @classmethod
    def from_env(cls) -> Settings:
        # Must match the default in docker-compose.yml / .env.example.
        password = os.environ.get("NEO4J_PASSWORD", "user")
        uri = os.environ.get("NEO4J_URI") or "bolt://localhost:7687"
        if "NEO4J_BOLT_PORT" in os.environ and "NEO4J_URI" not in os.environ:
            uri = f"bolt://localhost:{os.environ['NEO4J_BOLT_PORT']}"
        return cls(
            uri=uri,
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=password,
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
            dsn=os.environ.get("POSTGRES_DSN"),
            batch_size=int(os.environ.get("NEXUS_LOAD_BATCH_SIZE", "200")),
        )


# --------------------------------------------------------------------------- #
# Fixture loading
# --------------------------------------------------------------------------- #
def read_fixtures(fixtures_dir: Path) -> tuple[dict[str, list], list[PersonRef]]:
    nodes: dict[str, list] = {}
    for stem, model in NODE_MODELS.items():
        path = fixtures_dir / f"{stem}.json"
        if not path.exists():
            raise SystemExit(f"missing fixture file: {path}")
        nodes[stem] = [model.model_validate(payload) for payload in json.loads(
            path.read_text(encoding="utf-8")
        )]

    people: dict[str, PersonRef] = {}
    for stem in ("commits", "pull_requests", "incidents"):
        for record in nodes[stem]:
            for attr in ("author", "committer", "reporter", "merged_by"):
                person = getattr(record, attr, None)
                if isinstance(person, PersonRef):
                    people.setdefault(person.id, person)
    return nodes, list(people.values())


def collect_edges(nodes: dict[str, list]) -> list:
    edges = []
    for records in nodes.values():
        for record in records:
            edges.extend(record.edges())
    return edges


# --------------------------------------------------------------------------- #
# Cypher builders (labels/types are whitelisted before interpolation)
# --------------------------------------------------------------------------- #
def _safe_label(label: str) -> str:
    if label not in {kind.value for kind in NodeKind}:
        raise ValueError(f"refusing to interpolate unknown label: {label}")
    return label


def node_query(label: str) -> str:
    return (
        f"UNWIND $rows AS row\n"
        f"MERGE (n:`{_safe_label(label)}` {{id: row.id}})\n"
        f"SET n += row.properties"
    )


def person_query() -> str:
    return (
        "UNWIND $rows AS row\n"
        "MERGE (p:Person {id: row.id})\n"
        "SET p += row.properties"
    )


def edge_query(source_label: str, relation: str, target_label: str) -> str:
    if relation not in {rel.value for rel in RelationType}:
        raise ValueError(f"refusing to interpolate unknown relation type: {relation}")
    return (
        f"UNWIND $rows AS row\n"
        f"MATCH (a:`{_safe_label(source_label)}` {{id: row.source_id}})\n"
        f"MATCH (b:`{_safe_label(target_label)}` {{id: row.target_id}})\n"
        f"MERGE (a)-[r:`{relation}`]->(b)\n"
        f"SET r += row.properties\n"
        f"RETURN count(r) AS linked"
    )


def build_payloads(
    nodes: dict[str, list], edges: list, people: list[PersonRef]
) -> tuple[list[tuple[str, list[dict]]], dict[tuple[str, str, str], list[dict]], list[dict]]:
    node_payloads: list[tuple[str, list[dict]]] = []
    for stem, model in NODE_MODELS.items():
        rows = [
            {"id": record.id, "properties": record.to_neo4j_properties()}
            for record in nodes[stem]
        ]
        if rows:
            node_payloads.append((model.NEO4J_LABEL, rows))

    person_rows = [
        {
            "id": person.id,
            "properties": {
                "id": person.id,
                "login": person.login,
                "name": person.name,
                "email": person.email,
                "prov_source": "derived",
                "prov_extractor": "load_neo4j.person_ref",
                "prov_confidence": 1.0,
            },
        }
        for person in people
    ]
    if person_rows:
        # Ids with a None value are dropped: Neo4j rejects null properties.
        person_rows = [
            {"id": row["id"], "properties": {k: v for k, v in row["properties"].items() if v is not None}}
            for row in person_rows
        ]

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for edge in edges:
        grouped[
            (edge.source_label.value, edge.type.value, edge.target_label.value)
        ].append(
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "properties": edge.to_parameters(),
            }
        )
    return node_payloads, grouped, person_rows


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def split_statements(path: Path) -> list[str]:
    """Split a ``.cypher`` file into executable statements.

    Comment lines are removed *before* splitting on ``;``: these files document
    themselves, and a ``;`` inside a prose comment ("since Neo4j 5.7; the
    Enterprise-only equivalent...") would otherwise cut a statement in half and
    send a fragment of English to the Cypher parser.
    """
    raw = path.read_text(encoding="utf-8")
    without_comments = "\n".join(
        line for line in raw.splitlines() if not line.strip().startswith("//")
    )
    return [statement.strip() for statement in without_comments.split(";") if statement.strip()]


def apply_schema(driver, database: str, dry_run: bool, enterprise: bool = False) -> None:
    files = [*SCHEMA_FILES[:1], *ENTERPRISE_SCHEMA_FILES, *SCHEMA_FILES[1:]] if enterprise else SCHEMA_FILES
    for relative in files:
        path = PROJECT_ROOT / relative
        if not path.exists():
            print(f"  ! missing schema file {relative}")
            continue
        statements = split_statements(path)
        print(f"  applying {relative} ({len(statements)} statements)")
        if dry_run:
            continue
        with driver.session(database=database) as session:
            for statement in statements:
                session.run(statement)


def verify_provenance(driver, database: str) -> Counter:
    """Community-edition substitute for the provenance existence constraints.

    Returns the number of nodes missing ``prov_source`` per label. Anything above
    zero means the graph contains an unauditable node and the answer confidence
    score cannot be trusted for it.
    """
    # SchemaMeta is created by the schema itself (04_bootstrap.cypher), not by an
    # ingestion run: it has no source to declare, so the provenance rule does not
    # apply to it.
    query = (
        "MATCH (n) WHERE n.prov_source IS NULL AND NOT n:SchemaMeta\n"
        "RETURN labels(n)[0] AS label, count(*) AS missing"
    )
    with driver.session(database=database) as session:
        return Counter(
            {record["label"]: record["missing"] for record in session.run(query)}
        )


def load_graph(driver, database: str, nodes, grouped_edges, people, batch_size: int) -> Counter:
    stats: Counter[str] = Counter()
    with driver.session(database=database) as session:
        for label, rows in nodes:
            for chunk in _chunks(rows, batch_size):
                session.run(node_query(label), rows=chunk)
                stats[f"node:{label}"] += len(chunk)
        for chunk in _chunks(people, batch_size):
            session.run(person_query(), rows=chunk)
            stats["node:Person"] += len(chunk)

        for (source_label, relation, target_label), rows in grouped_edges.items():
            for chunk in _chunks(rows, batch_size):
                result = session.run(
                    edge_query(source_label, relation, target_label), rows=chunk
                ).single()
                linked = result["linked"] if result else 0
                stats[f"rel:{relation}"] += linked
                if linked != len(chunk):
                    stats["rel_skipped_unresolved"] += len(chunk) - linked
    return stats


def load_pgvector(dsn: str, nodes: dict[str, list]) -> Counter:
    """Insert the Evidence rows into pgvector. Embeddings are left NULL on purpose."""
    stats: Counter[str] = Counter()
    try:
        import psycopg  # type: ignore
    except ImportError:
        print("  ! psycopg is not installed: skipping the pgvector step "
              "(pip install 'psycopg[binary]')")
        return stats

    rows = []
    for evidence in nodes.get("evidence", []):
        ref = evidence.node_references[0] if evidence.node_references else None
        rows.append(
            (
                evidence.id,
                evidence.repository_id,
                evidence.answer_id,
                evidence.path_id,
                evidence.kind.value,
                ref.node_kind.value if ref else None,
                ref.node_id if ref else None,
                evidence.text or "",
                json.dumps({"score": evidence.score, "strategy": evidence.retrieval_strategy.value}),
            )
        )
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO evidence_embeddings
                    (evidence_id, repository_id, answer_id, path_id, kind,
                     node_kind, node_id, text, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (evidence_id) DO UPDATE
                    SET text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        answer_id = EXCLUDED.answer_id
                """,
                rows,
            )
            connection.commit()
            stats["pgvector:evidence_embeddings"] = len(rows)
    return stats


def _chunks(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", default=str(PROJECT_ROOT / "fixtures"))
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and print the load plan without touching any database")
    parser.add_argument("--apply-schema", action="store_true",
                        help="run schema/cypher/*.cypher before loading")
    parser.add_argument("--with-pgvector", action="store_true",
                        help="also upsert the Evidence rows into PostgreSQL/pgvector")
    parser.add_argument("--enterprise", action="store_true",
                        help="also apply the Neo4j Enterprise-only existence constraints")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings.from_env()

    fixtures_dir = Path(args.fixtures_dir).resolve()
    print(f"Fixtures : {fixtures_dir}")
    nodes, people = read_fixtures(fixtures_dir)
    edges = collect_edges(nodes)
    node_payloads, grouped_edges, person_rows = build_payloads(nodes, edges, people)

    print("\n== Load plan ==")
    for label, rows in node_payloads:
        print(f"  nodes {label:<12} {len(rows):>4}")
    print(f"  nodes {'Person':<12} {len(person_rows):>4}  (derived from embedded refs)")
    print(f"  relationships     {len(edges):>4} across {len(grouped_edges)} type(s):")
    for (source, relation, target), rows in sorted(grouped_edges.items(), key=lambda kv: -len(kv[1])):
        print(f"    {relation:<16}{len(rows):>4}   {source} -> {target}")

    if args.dry_run:
        print("\n[dry-run] nothing was written to any database.")
        print(f"[dry-run] would connect to {settings.uri} (user={settings.user}, "
              f"db={settings.database})")
        print("\nSample statements that would be executed:")
        print("  " + node_query("File").replace("\n", "\n  "))
        print("  " + edge_query("File", "IMPORTS", "File").replace("\n", "\n  "))
        if args.apply_schema:
            print("\n  schema files: " + ", ".join(SCHEMA_FILES))
        return 0

    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError:
        print(
            "\nerror: the neo4j driver is not installed.\n"
            "       pip install -r requirements.txt\n"
            "       (or run with --dry-run to validate the fixtures without a database)",
            file=sys.stderr,
        )
        return 2

    if args.apply_schema:
        print("\n== Applying schema ==")
        with GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password)) as driver:
            driver.verify_connectivity()
            apply_schema(driver, settings.database, dry_run=False, enterprise=args.enterprise)
            if not args.enterprise:
                print("  (existence constraints skipped: Neo4j Enterprise only, use --enterprise)")

    print("\n== Loading graph ==")
    with GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password)) as driver:
        driver.verify_connectivity()
        stats = load_graph(
            driver, settings.database, node_payloads, grouped_edges, person_rows,
            settings.batch_size,
        )

    for key in sorted(stats):
        print(f"  {key:<28} {stats[key]}")

    print("\n== Verifying provenance (Community substitute for the existence constraints) ==")
    with GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password)) as driver:
        missing = verify_provenance(driver, settings.database)
    if missing:
        print(f"  ! unauditable nodes detected: {dict(missing)}")
    else:
        print("  OK: every node carries prov_source")

    if stats.get("rel_skipped_unresolved"):
        print(
            f"\n  ! {stats['rel_skipped_unresolved']} relationship(s) were skipped because "
            "an endpoint node was missing (check with scripts/validate_fixtures.py)"
        )

    if args.with_pgvector:
        print("\n== Loading pgvector ==")
        if not settings.dsn:
            print("  ! POSTGRES_DSN is not set (see .env.example): skipping")
        else:
            for key, value in load_pgvector(settings.dsn, nodes).items():
                print(f"  {key:<28} {value}")

    print("\nDone. Open http://localhost:7474 and run:")
    print("  MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC;")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
