"""Change Impact Analysis: from a file id to an auditable ``EvidencePath``.

This is the Phase 1 deliverable the fixtures only *simulate*: here the path is
built by actually traversing Neo4j. Given ``repo::path``, the analyzer

1. walks the reverse import graph ``(:File)-[:IMPORTS*1..N]->(target)`` with a
   bounded depth (default 7, the depth the Week-0 audit measured on httpie/cli);
2. turns every reached file into an :class:`~models.answer.EvidenceHop` whose
   score decays with traversal distance and is weighted by the ``confidence``
   stored on the ``:IMPORTS`` edge (static-analysis edges carry 0.9-0.95);
3. assembles an :class:`~models.answer.EvidencePath` plus one
   :class:`~models.evidence.Evidence` node per hop, ready to be written back
   with :class:`ingestion.graph_writer.GraphWriter`.

The Cypher is bounded (``1..max_depth``) so a pathological graph cannot turn the
query into an unbounded traversal, and ``is_valid``/``validation_notes`` report
what the traversal actually saw -- an empty graph yields
``status=insufficient_evidence`` semantics via ``hops == []``, not a fake answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from models import (
    AnswerIntent,
    Evidence,
    EvidenceHop,
    EvidenceKind,
    EvidencePath,
    EvidenceRef,
    EvidenceRole,
    NodeKind,
    Provenance,
    RetrievalStrategy,
    utcnow,
)

EXTRACTOR = "nexus-devintel.retrieval.impact"

#: One Cypher query, depth-bounded. Cypher does not allow parameters inside a
#: variable-length pattern bound (``*1..$max_depth`` is a syntax error), so the
#: bound is interpolated as a validated integer (see ``_validated_bound``).
#: Ordering makes hops stable across runs -- an EvidencePath must be reproducible.
TRAVERSAL_QUERY = """
MATCH (target:File {id: $file_id})
OPTIONAL MATCH p = (dependent:File)-[r:IMPORTS*1..%d]->(target)
WITH target, p,
     [rel IN relationships(p) | coalesce(rel.confidence, 1.0)] AS confidences,
     length(p) AS hops
UNWIND CASE WHEN p IS NULL THEN [] ELSE [p] END AS path
WITH target, path, hops, confidences,
     head(nodes(path)) AS dependent
RETURN dependent.id AS dependent_id,
       dependent.path AS dependent_path,
       hops AS distance,
       confidences AS edge_confidences,
       [n IN nodes(path) | n.id] AS chain
ORDER BY distance ASC, dependent_id ASC
LIMIT $limit
"""


@dataclass
class ImpactReport:
    """Everything the caller needs: the path, the evidence nodes, the metrics."""

    target_file_id: str
    path: EvidencePath
    evidences: list[Evidence] = field(default_factory=list)
    direct_dependents: list[str] = field(default_factory=list)
    transitive_dependents: int = 0
    max_depth_seen: int = 0
    computed_at: datetime = field(default_factory=utcnow)


class ChangeImpactAnalyzer:
    """Build ``EvidencePath``s for "what breaks if I change this file?"."""

    def __init__(self, driver: Any, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    # ---- graph access ------------------------------------------------------ #
    @staticmethod
    def _validated_bound(max_depth: int) -> int:
        """Sanitize the interpolated traversal bound (defence in depth)."""
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or not 1 <= max_depth <= 20:
            raise ValueError(f"max_depth must be an int in [1, 20], got {max_depth!r}")
        return max_depth

    def _rows(self, file_id: str, max_depth: int, limit: int) -> list[dict[str, Any]]:
        bound = self._validated_bound(max_depth)
        query = TRAVERSAL_QUERY % bound
        with self._driver.session(database=self._database) as session:
            result = session.run(
                query,
                file_id=file_id,
                limit=limit,
            )
            return [record.data() for record in result]

    def file_exists(self, file_id: str) -> bool:
        query = "MATCH (f:File {id: $file_id}) RETURN count(f) AS found"
        with self._driver.session(database=self._database) as session:
            record = session.run(query, file_id=file_id).single()
        return bool(record and record["found"])

    # ---- scoring ------------------------------------------------------------ #
    @staticmethod
    def _hop_score(distance: int, edge_confidences: list[float]) -> float:
        """Product of edge confidences, decayed 15% per hop beyond the first."""
        confidence = 1.0
        for value in edge_confidences or []:
            confidence *= float(value)
        decayed = confidence * (0.85 ** max(0, distance - 1))
        return round(min(1.0, max(0.0, decayed)), 4)

    # ---- path assembly -------------------------------------------------------#
    def analyze(
        self,
        file_id: str,
        *,
        max_depth: int = 7,
        limit: int = 50,
        path_id: str | None = None,
        intent: AnswerIntent = AnswerIntent.CHANGE_IMPACT,
    ) -> ImpactReport:
        """Traverse the graph and assemble the evidence chain."""
        if not self.file_exists(file_id):
            raise FileNotFoundError(
                f"{file_id} is not in the graph; ingest the repository first"
            )

        rows = self._rows(file_id, max_depth, limit)
        path_id = path_id or f"evidence-path-impact-{file_id.replace('::', '--')}"

        hops: list[EvidenceHop] = []
        evidences: list[Evidence] = []
        direct: list[str] = []

        # Hop 0: the target file itself, as the anchor of the chain.
        hops.append(
            EvidenceHop(
                step=0,
                node_kind=NodeKind.FILE,
                node_id=file_id,
                role=EvidenceRole.CONTEXT,
                score=1.0,
                rationale="file whose change impact is being analysed",
            )
        )
        evidences.append(
            self._evidence_for(file_id, path_id, 0, 1.0,
                               "analysed file (impact source)")
        )

        for row in rows:
            distance = int(row["distance"])
            dependent_id = row["dependent_id"]
            if dependent_id == file_id:
                continue
            score = self._hop_score(distance, row["edge_confidences"])
            role = EvidenceRole.BLAST_RADIUS
            hops.append(
                EvidenceHop(
                    step=len(hops),
                    node_kind=NodeKind.FILE,
                    node_id=dependent_id,
                    relation_in="IMPORTS",
                    role=role,
                    score=score,
                    rationale=f"imports {file_id} at distance {distance}",
                )
            )
            evidences.append(
                self._evidence_for(dependent_id, path_id, len(hops) - 1, score,
                                   row.get("dependent_path") or dependent_id,
                                   distance=distance, chain=row.get("chain"))
            )
            if distance == 1:
                direct.append(dependent_id)

        notes = [f"traversal bounded at depth {max_depth}, limit {limit}"]
        if not rows:
            notes.append("no dependent file found: the target has an empty blast radius")

        path = EvidencePath(
            path_id=path_id,
            intent=intent,
            hops=hops,
            score=round(sum(hop.score for hop in hops[1:]) / len(hops[1:]), 4) if hops[1:] else 0.0,
            is_valid=bool(rows),
            validation_notes=notes,
        )
        return ImpactReport(
            target_file_id=file_id,
            path=path,
            evidences=evidences,
            direct_dependents=sorted(direct),
            transitive_dependents=len({hop.node_id for hop in hops[1:]}),
            max_depth_seen=max((int(row["distance"]) for row in rows), default=0),
            computed_at=utcnow(),
        )

    @staticmethod
    def _evidence_for(
        node_id: str,
        path_id: str,
        step: int,
        score: float,
        label: str,
        *,
        distance: int | None = None,
        chain: list[str] | None = None,
    ) -> Evidence:
        rationale = (
            f"reaches the target through {distance} IMPORTS hop(s)"
            if distance and distance > 1
            else "imports the target directly"
            if distance == 1
            else "impact source"
        )
        if chain and distance and distance > 1:
            rationale += f" via {' -> '.join(chain)}"
        return Evidence(
            id=f"ev-impact-{path_id}-{step:03d}",
            kind=EvidenceKind.GRAPH_PATH if distance and distance > 1 else EvidenceKind.IMPORT_EDGE
            if distance == 1 else EvidenceKind.NODE,
            retrieval_strategy=RetrievalStrategy.GRAPH_TRAVERSAL,
            score=score,
            rank=step + 1,
            text=label,
            rationale=rationale,
            hop_index=step,
            path_id=path_id,
            node_references=[
                EvidenceRef(
                    node_kind=NodeKind.FILE,
                    node_id=node_id,
                    role=EvidenceRole.BLAST_RADIUS if distance else EvidenceRole.CONTEXT,
                    hop_index=step,
                    relation="IMPORTS" if distance else None,
                )
            ],
            retrieved_at=utcnow(),
            provenance=Provenance(
                source="derived",
                source_uri=f"cypher:IMPORTS*1..{max(1, distance or 1)}:{node_id}",
                extractor=EXTRACTOR,
                confidence=score or 0.5,
            ),
        )
