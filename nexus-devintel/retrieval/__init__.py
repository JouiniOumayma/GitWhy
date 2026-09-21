"""Retrieval layer: graph traversals that produce auditable ``EvidencePath``s.

Phase 1 exposes one analyzer: :class:`retrieval.impact.ChangeImpactAnalyzer`,
which answers "what is the blast radius of changing file F?" by walking
``(:File)-[:IMPORTS*]->(:File)`` backwards and packaging every hop as
``EvidenceHop`` / ``Evidence`` nodes with provenance-derived scores.
"""

from .impact import ChangeImpactAnalyzer, ImpactReport

__all__ = ["ChangeImpactAnalyzer", "ImpactReport"]
