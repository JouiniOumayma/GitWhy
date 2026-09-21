"""Write NEXUS-DevIntel contract models into Neo4j.

The loader ``scripts/load_neo4j.py`` batch-loads whole fixture files; this writer
is its *incremental* counterpart for the live path (webhook / API ingestion).
Same invariants:

* nodes are ``MERGE``d on ``id`` -> re-ingesting the same commit is a no-op;
* relationships are ``MERGE``d on their endpoints -> no duplicate edges;
* labels and relation types are whitelisted before interpolation, so a bad
  contract cannot inject Cypher;
* every node must carry ``prov_source`` (the Community-edition substitute for
  the Enterprise existence constraints, mirroring ``verify_provenance()``).

A :class:`GraphWriter` accepts any object exposing ``NEO4J_LABEL``, ``id``,
``to_neo4j_properties()`` and ``edges()`` -- i.e. every ``NexusBaseModel``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from models import NodeKind, RelationType


class GraphWriter:
    """Thin, idempotent Neo4j writer for single model instances."""

    def __init__(self, driver: Any, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    # ---- helpers ------------------------------------------------------------ #
    @staticmethod
    def _safe_label(label: str) -> str:
        if label not in {kind.value for kind in NodeKind}:
            raise ValueError(f"refusing to interpolate unknown label: {label}")
        return label

    @staticmethod
    def _safe_relation(relation: str) -> str:
        if relation not in {rel.value for rel in RelationType}:
            raise ValueError(f"refusing to interpolate unknown relation type: {relation}")
        return relation

    # ---- nodes ---------------------------------------------------------------#
    def write_node(self, model: Any) -> str:
        """``MERGE`` one node on its ``id``. Returns the node ``id``."""
        label = self._safe_label(model.NEO4J_LABEL)
        properties = model.to_neo4j_properties()
        query = (
            f"MERGE (n:`{label}` {{id: $id}})\n"
            f"SET n += $properties\n"
            f"RETURN n.id AS id"
        )
        with self._driver.session(database=self._database) as session:
            record = session.run(query, id=model.id, properties=properties).single()
        return record["id"] if record else model.id

    def write_person(self, person: Any) -> str:
        """``MERGE`` a ``(:Person)`` from an embedded ``PersonRef``."""
        properties = {
            key: value
            for key, value in {
                "id": person.id,
                "login": person.login,
                "name": person.name,
                "email": person.email,
                "prov_source": "github_api",
                "prov_extractor": "ingestion.graph_writer.person",
                "prov_confidence": 1.0,
            }.items()
            if value is not None
        }
        with self._driver.session(database=self._database) as session:
            session.run(
                "MERGE (p:Person {id: $id}) SET p += $properties",
                id=person.id,
                properties=properties,
            )
        return person.id

    # ---- relationships --------------------------------------------------------#
    def write_edge(self, edge: Any) -> bool:
        """``MERGE`` one relationship; ``False`` when an endpoint is missing."""
        relation_type = edge.type.value if isinstance(edge.type, RelationType) else edge.type
        relation = self._safe_relation(relation_type)
        query = (
            f"MATCH (a:`{self._safe_label(edge.source_label.value)}` {{id: $source_id}})\n"
            f"MATCH (b:`{self._safe_label(edge.target_label.value)}` {{id: $target_id}})\n"
            f"MERGE (a)-[r:`{relation}`]->(b)\n"
            f"SET r += $properties\n"
            f"RETURN count(r) AS linked"
        )
        with self._driver.session(database=self._database) as session:
            record = session.run(
                query,
                source_id=edge.source_id,
                target_id=edge.target_id,
                properties=edge.to_parameters(),
            ).single()
        return bool(record and record["linked"])

    # ---- composite ------------------------------------------------------------#
    def write_model(self, model: Any, *, with_persons: bool = True) -> Counter:
        """Write one model: its node, then every relationship it declares.

        Relationship endpoints that do not exist yet (e.g. a ``:CLOSES`` towards
        an issue nobody ingested) are **skipped**, not created as dangling
        nodes -- the same policy as the batch loader.
        """
        stats: Counter[str] = Counter()
        self.write_node(model)
        stats[f"node:{model.NEO4J_LABEL}"] += 1

        if with_persons:
            for attribute in ("author", "committer", "reporter", "merged_by"):
                person = getattr(model, attribute, None)
                if person is not None:
                    self.write_person(person)
                    stats["node:Person"] += 1

        for edge in model.edges():
            linked = self.write_edge(edge)
            if linked:
                stats[f"rel:{edge.type.value}"] += 1
            else:
                stats["rel_skipped_unresolved"] += 1
        return stats

    def write_models(self, models: Iterable[Any]) -> Counter:
        total: Counter[str] = Counter()
        for model in models:
            total.update(self.write_model(model))
        return total
