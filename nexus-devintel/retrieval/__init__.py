"""Retrieval layer: graph traversals that produce auditable ``EvidencePath``s.

Phase 1 exposed :class:`retrieval.impact.ChangeImpactAnalyzer` ("what is the
blast radius of changing file F?"). Phase 2 adds:

* :class:`retrieval.root_cause.RootCauseAnalyzer` -- walk the history around an
  incident (``Incident -> PR -> Commit -> File -> Deployment``) and package the
  causal chain as ``EvidenceHop`` / ``Evidence`` nodes, on the exact pattern of
  the impact analyzer;
* :class:`retrieval.hybrid.HybridRetriever` -- the vector half: question in,
  hybrid-ranked (lexical + pgvector) chunks out, packaged as citable Evidence
  pointing back at the graph.
"""

from .hybrid import HybridMatch, HybridReport, HybridRetriever
from .impact import ChangeImpactAnalyzer, ImpactReport
from .root_cause import RootCauseAnalyzer, RootCauseReport

__all__ = [
    "ChangeImpactAnalyzer",
    "HybridMatch",
    "HybridReport",
    "HybridRetriever",
    "ImpactReport",
    "RootCauseAnalyzer",
    "RootCauseReport",
]
