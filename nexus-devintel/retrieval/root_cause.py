"""Root Cause Analysis: from an incident id to an auditable ``EvidencePath``.

Phase 2 deliverable, built on the exact pattern of
:mod:`retrieval.impact` (same traversal/assembly structure, same conventions):

* Change Impact walks the *forward* import graph
  ``(:File)-[:IMPORTS*1..7]->(target)`` and answers "what breaks if I change
  this?";
* Root Cause walks the *history* around an incident
  ``(:Incident)-[:CLOSES|MERGED_INTO|MODIFIES|DEPLOYED_AT|OBSERVED_IN|AFFECTS|
  TOUCHES|DEPLOYED_AS*1..N]-(n)`` and answers "what caused this, where was it
  fixed, and what shipped it?". The relation set is the schema README's
  canonical RCA query, extended with ``OBSERVED_IN``/``AFFECTS``/``TOUCHES``/
  ``DEPLOYED_AS`` so the walk can actually reach deployments, PRs and files
  from the incident (the README chain ``CLOSES|MERGED_INTO|MODIFIES|
  DEPLOYED_AT`` only connects to deployments through a commit's
  ``DEPLOYED_AS`` edge).

The traversal is **undirected on purpose**: ``CLOSES`` points PR→Incident
while ``OBSERVED_IN`` points Incident→Deployment, so no single directed
pattern covers the chain; bounding the relation whitelist and the depth keeps
the query safe. Every reached node becomes an :class:`EvidenceHop` whose score
decays with distance and is weighted by the ``confidence`` stored on each
traversed edge (a ``graphql_closing_issues_reference`` ``:CLOSES`` at 1.0
weighs more than a regex-derived one at 0.6) -- the same
"product of edge confidences, 15% decay per hop" scoring as Change Impact.

The Cypher is depth-bounded and relation-whitelisted; an empty graph yields
``hops == []`` with ``is_valid=False``, never a fake answer. Provenance of the
produced Evidence nodes is ``source=derived`` with the traversal encoded in
``source_uri``, exactly like :mod:`retrieval.impact`.
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

EXTRACTOR = "nexus-devintel.retrieval.root_cause"

#: Relations that may appear in an RCA walk, whitelisted (cf. schema README §3):
#: incident layer (CLOSES, OBSERVED_IN, AFFECTS), PR layer (MERGED_INTO, TOUCHES),
#: commit layer (MODIFIES), deployment layer (DEPLOYED_AT, DEPLOYED_AS).
RCA_RELATIONS = (
    "CLOSES|MERGED_INTO|MODIFIES|DEPLOYED_AT|OBSERVED_IN|AFFECTS|TOUCHES|DEPLOYED_AS"
)

#: One depth-bounded Cypher query. The bound is interpolated as a validated
#: integer (Cypher forbids parameters inside ``*1..$n``), the ids stay bound.
#: Ordering makes the result stable across runs -- an EvidencePath must be
#: reproducible. ``labels(n)[0]`` drives the role mapping; ``coalesce`` gives
#: each node kind its display name (path / title / tag / subject / id).
TRAVERSAL_QUERY = """
MATCH (incident:Incident {id: $incident_id})
OPTIONAL MATCH p = (incident)-[r:%s*1..%d]-(n)
WITH incident, p, n,
     [rel IN relationships(p) | coalesce(rel.confidence, 1.0)] AS confidences,
     [rel IN relationships(p) | type(rel)] AS rel_types,
     length(p) AS hops
UNWIND CASE WHEN p IS NULL THEN [] ELSE [p] END AS path
WITH incident, path, n, hops, confidences, rel_types
RETURN n.id AS node_id,
       labels(n)[0] AS node_label,
       coalesce(n.path, n.title, n.tag, n.subject, n.id) AS node_name,
       hops AS distance,
       confidences AS edge_confidences,
       rel_types AS rel_types,
       [x IN nodes(path) | x.id] AS chain_ids,
       [x IN nodes(path) | labels(x)[0]] AS chain_labels
ORDER BY distance ASC, node_id ASC
LIMIT $limit
"""


@dataclass
class RootCauseReport:
    """Everything the caller needs: the path, the evidence nodes, the summary."""

    incident_id: str
    path: EvidencePath
    evidences: list[Evidence] = field(default_factory=list)
    #: Nodes grouped by kind, sorted by distance then id (deduplicated).
    fix_commits: list[str] = field(default_factory=list)
    fixing_prs: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    deployments: list[str] = field(default_factory=list)
    #: The shortest evidence chain reaching a file (the demo money-path).
    best_chain: list[str] = field(default_factory=list)
    max_depth_seen: int = 0
    computed_at: datetime = field(default_factory=utcnow)


class RootCauseAnalyzer:
    """Build ``EvidencePath``s for "why did this incident happen?"."""

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

    def _rows(self, incident_id: str, max_depth: int, limit: int) -> list[dict[str, Any]]:
        bound = self._validated_bound(max_depth)
        query = TRAVERSAL_QUERY % (RCA_RELATIONS, bound)
        with self._driver.session(database=self._database) as session:
            result = session.run(query, incident_id=incident_id, limit=limit)
            return [record.data() for record in result]

    def incident_exists(self, incident_id: str) -> bool:
        query = "MATCH (i:Incident {id: $incident_id}) RETURN count(i) AS found"
        with self._driver.session(database=self._database) as session:
            record = session.run(query, incident_id=incident_id).single()
        return bool(record and record["found"])

    # ---- scoring ------------------------------------------------------------ #
    @staticmethod
    def _hop_score(distance: int, edge_confidences: list[float]) -> float:
        """Product of edge confidences, decayed 15% per hop beyond the first.

        Same function as ``ChangeImpactAnalyzer._hop_score`` on purpose: both
        analyzers must score identical graphs identically.
        """
        confidence = 1.0
        for value in edge_confidences or []:
            confidence *= float(value)
        decayed = confidence * (0.85 ** max(0, distance - 1))
        return round(min(1.0, max(0.0, decayed)), 4)

    @staticmethod
    def _role_for(node_label: str) -> EvidenceRole:
        """Map a node kind to its role in the causal story."""
        return {
            NodeKind.INCIDENT.value: EvidenceRole.SYMPTOM,
            NodeKind.COMMIT.value: EvidenceRole.FIX,
            NodeKind.PR.value: EvidenceRole.FIX,
            NodeKind.FILE.value: EvidenceRole.ROOT_CAUSE,
            NodeKind.DEPLOYMENT.value: EvidenceRole.TIMELINE,
        }.get(node_label, EvidenceRole.CONTEXT)

    # ---- path assembly -------------------------------------------------------#
    def analyze(
        self,
        incident_id: str,
        *,
        max_depth: int = 6,
        limit: int = 200,
        path_id: str | None = None,
        intent: AnswerIntent = AnswerIntent.ROOT_CAUSE,
    ) -> RootCauseReport:
        """Traverse the graph backwards in history and assemble the evidence chain."""
        if not self.incident_exists(incident_id):
            raise FileNotFoundError(
                f"{incident_id} is not in the graph; ingest the repository first"
            )

        rows = self._rows(incident_id, max_depth, limit)
        path_id = path_id or f"evidence-path-rootcause-{incident_id.replace('::', '--')}"

        hops: list[EvidenceHop] = []
        evidences: list[Evidence] = []
        seen: set[str] = set()
        by_label: dict[str, list[tuple[int, str]]] = {}
        notes: list[str] = [f"traversal bounded at depth {max_depth}, limit {limit}"]

        # Hop 0: the incident itself, symptom side of the chain.
        hops.append(
            EvidenceHop(
                step=0,
                node_kind=NodeKind.INCIDENT,
                node_id=incident_id,
                role=EvidenceRole.SYMPTOM,
                score=1.0,
                rationale="incident whose root cause is being analysed",
            )
        )
        evidences.append(
            self._evidence_for(incident_id, NodeKind.INCIDENT, path_id, 0, 1.0,
                               "analysed incident (symptom)", None, [], [])
        )

        best_chain: list[str] = []
        for row in rows:
            node_id = row["node_id"]
            if node_id == incident_id or node_id in seen:
                continue
            label = row["node_label"]
            if label not in {kind.value for kind in NodeKind}:
                notes.append(f"skipped node {node_id}: unknown label {label!r}")
                continue
            seen.add(node_id)
            distance = int(row["distance"])
            score = self._hop_score(distance, row["edge_confidences"])
            rel_types = list(row["rel_types"] or [])
            hops.append(
                EvidenceHop(
                    step=len(hops),
                    node_kind=NodeKind(label),
                    node_id=node_id,
                    relation_in=rel_types[-1] if rel_types else None,
                    role=self._role_for(label),
                    score=score,
                    rationale=self._rationale(row, distance),
                )
            )
            evidences.append(
                self._evidence_for(node_id, NodeKind(label), path_id, len(hops) - 1,
                                   score, row.get("node_name") or node_id,
                                   distance, row.get("chain_ids") or [],
                                   row.get("chain_labels") or [])
            )
            by_label.setdefault(label, []).append((distance, node_id))
            # The money-path: the first (shortest) chain that reaches a File --
            # symptom -> ... -> fix -> file where the cause lived.
            if not best_chain and label == NodeKind.FILE.value:
                best_chain = self._format_chain(row)

        def _ids(label: str) -> list[str]:
            return [node_id for _, node_id in sorted(by_label.get(label, []))]

        if not rows:
            notes.append(
                "no connected node found: the incident has no CLOSES/MODIFIES/"
                "OBSERVED_IN links in the current graph"
            )

        path = EvidencePath(
            path_id=path_id,
            intent=intent,
            hops=hops,
            score=round(sum(hop.score for hop in hops[1:]) / len(hops[1:]), 4) if hops[1:] else 0.0,
            is_valid=bool(rows),
            validation_notes=notes,
        )
        return RootCauseReport(
            incident_id=incident_id,
            path=path,
            evidences=evidences,
            fix_commits=_ids(NodeKind.COMMIT.value),
            fixing_prs=_ids(NodeKind.PR.value),
            affected_files=_ids(NodeKind.FILE.value),
            deployments=_ids(NodeKind.DEPLOYMENT.value),
            best_chain=best_chain,
            max_depth_seen=max((int(row["distance"]) for row in rows), default=0),
            computed_at=utcnow(),
        )

    @staticmethod
    def _rationale(row: dict[str, Any], distance: int) -> str:
        """Explain the hop through the actual relation chain, for auditability."""
        rel_types = list(row["rel_types"] or [])
        if not rel_types:
            return "directly linked to the incident"
        return f"reached in {distance} hop(s) via {' -> '.join(rel_types)}"

    @staticmethod
    def _format_chain(row: dict[str, Any]) -> list[str]:
        """``['Incident:httpie/cli#issue-1583', 'PR:httpie/cli#1596', ...]``."""
        ids = row.get("chain_ids") or []
        labels = row.get("chain_labels") or []
        return [f"{labels[i]}:{ids[i]}" for i in range(min(len(ids), len(labels)))]

    @staticmethod
    def _evidence_for(
        node_id: str,
        node_kind: NodeKind,
        path_id: str,
        step: int,
        score: float,
        label: str,
        distance: int | None,
        chain_ids: list[str],
        chain_labels: list[str],
    ) -> Evidence:
        if distance is None or distance <= 1:
            kind = EvidenceKind.NODE if not distance else EvidenceKind.GRAPH_PATH
            rationale = "symptom node" if step == 0 else "directly linked to the incident"
        else:
            kind = EvidenceKind.GRAPH_PATH
            rationale = f"reaches the incident through {distance} hop(s)"
        if distance and distance > 1 and chain_ids and chain_labels:
            rationale += f" via {' -> '.join(f'{chain_labels[i]}:{chain_ids[i]}' for i in range(min(len(chain_ids), len(chain_labels))))}"
        return Evidence(
            id=f"ev-rootcause-{path_id}-{step:03d}",
            kind=kind,
            retrieval_strategy=RetrievalStrategy.GRAPH_TRAVERSAL,
            score=score,
            rank=step + 1,
            text=label,
            rationale=rationale,
            hop_index=step,
            path_id=path_id,
            node_references=[
                EvidenceRef(
                    node_kind=node_kind,
                    node_id=node_id,
                    role=RootCauseAnalyzer._role_for(node_kind.value),
                    hop_index=step,
                    relation="CLOSES" if node_kind in (NodeKind.PR, NodeKind.COMMIT)
                    and distance == 1 else None,
                )
            ],
            retrieved_at=utcnow(),
            provenance=Provenance(
                source="derived",
                source_uri=f"cypher:{RCA_RELATIONS}*1..{max(1, distance or 1)}:{node_id}",
                extractor=EXTRACTOR,
                confidence=score or 0.5,
            ),
        )


__all__ = [
    "RCA_RELATIONS",
    "RootCauseAnalyzer",
    "RootCauseReport",
]
