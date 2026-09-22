"""Shared enumerations for the NEXUS-DevIntel contracts.

Every enum inherits from ``str`` so that values serialize to plain strings in
JSON fixtures, in pgvector text columns and in Neo4j properties.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String enum with a readable repr (mirrors ``enum.StrEnum``, Python 3.11+)."""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return str(self.value)


# --------------------------------------------------------------------------- #
# Provenance / ingestion
# --------------------------------------------------------------------------- #
class SourceKind(StrEnum):
    """Where a fact came from. Drives the trust we give to an evidence path."""

    GIT = "git"
    GITHUB_API = "github_api"  # REST v3
    GITHUB_GRAPHQL = "github_graphql"  # GraphQL v4 (closingIssuesReferences)
    CI = "ci"
    ISSUE_TRACKER = "issue_tracker"
    MANUAL = "manual"
    DERIVED = "derived"  # computed by the ingestion pipeline (e.g. IMPORTS edges)
    SYNTHETIC = "synthetic_fixture"  # hand-written fixture text, never trust for RCA


class NodeKind(StrEnum):
    """Node labels, usable as a discriminator on Evidence/Answer references."""

    REPOSITORY = "Repository"
    FILE = "File"
    COMMIT = "Commit"
    PR = "PR"
    PERSON = "Person"
    DEPLOYMENT = "Deployment"
    INCIDENT = "Incident"
    EVIDENCE = "Evidence"
    ANSWER = "Answer"
    CODE_CHUNK = "CodeChunk"


class RelationType(StrEnum):
    """Whitelist of graph relationship types.

    The Neo4j loader refuses any type absent from this list, which keeps the
    schema closed while still allowing the ingestion pipeline to add relations
    incrementally.
    """

    # structure
    CONTAINS = "CONTAINS"  # (:Repository)-[:CONTAINS]->(:File)
    HAS_COMMIT = "HAS_COMMIT"  # (:Repository)-[:HAS_COMMIT]->(:Commit)
    HAS_DEPLOYMENT = "HAS_DEPLOYMENT"  # (:Repository)-[:HAS_DEPLOYMENT]->(:Deployment)
    IMPORTS = "IMPORTS"  # (:File)-[:IMPORTS]->(:File)
    # history
    PARENT_OF = "PARENT_OF"  # (:Commit)-[:PARENT_OF]->(:Commit)
    MODIFIES = "MODIFIES"  # (:Commit)-[:MODIFIES]->(:File)
    AUTHORED = "AUTHORED"  # (:Person)-[:AUTHORED]->(:Commit)
    COMMITTED = "COMMITTED"  # (:Person)-[:COMMITTED]->(:Commit)
    # pull requests
    OPENED = "OPENED"  # (:Person)-[:OPENED]->(:PR)
    MERGED_INTO = "MERGED_INTO"  # (:PR)-[:MERGED_INTO]->(:Commit)
    TOUCHES = "TOUCHES"  # (:PR)-[:TOUCHES]->(:File)
    # incidents
    CLOSES = "CLOSES"  # (:PR|:Commit)-[:CLOSES]->(:Incident)
    REFERENCES = "REFERENCES"  # (:PR|:Commit|:Evidence)-[:REFERENCES]->(any)
    REPORTED = "REPORTED"  # (:Person)-[:REPORTED]->(:Incident)
    AFFECTS = "AFFECTS"  # (:Incident)-[:AFFECTS]->(:File)
    OBSERVED_IN = "OBSERVED_IN"  # (:Incident)-[:OBSERVED_IN]->(:Deployment)
    CAUSED_BY = "CAUSED_BY"  # (:Incident)-[:CAUSED_BY]->(:Incident)
    # deployments
    DEPLOYED_AT = "DEPLOYED_AT"  # (:Deployment)-[:DEPLOYED_AT]->(:Commit)
    DEPLOYED_AS = "DEPLOYED_AS"  # (:Commit)-[:DEPLOYED_AS]->(:Deployment)
    PRECEDES = "PRECEDES"  # (:Deployment)-[:PRECEDES]->(:Deployment)
    # agent layer
    SUPPORTED_BY = "SUPPORTED_BY"  # (:Answer)-[:SUPPORTED_BY]->(:Evidence)
    DERIVED_FROM = "DERIVED_FROM"  # (:Evidence)-[:DERIVED_FROM]->(:Evidence)
    RETRIEVED_BY = "RETRIEVED_BY"  # (:Evidence)-[:RETRIEVED_BY]->(:Answer)
    HAS_CHUNK = "HAS_CHUNK"  # (:File)-[:HAS_CHUNK]->(:CodeChunk)


# --------------------------------------------------------------------------- #
# Code / VCS
# --------------------------------------------------------------------------- #
class ChangeType(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"


class ImportKind(StrEnum):
    IMPORT = "import"  # import x / import x.y
    FROM = "from"  # from x import y
    RELATIVE = "relative"  # from .x import y


class PRState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


class MergeStrategy(StrEnum):
    MERGE = "merge"  # merge commit
    SQUASH = "squash"  # httpie/cli's dominant strategy (281 squash refs)
    REBASE = "rebase"
    UNKNOWN = "unknown"


class LinkMethod(StrEnum):
    """How an Incident <-> PR/Commit link was established."""

    GRAPHQL_CLOSING_REFERENCE = "graphql_closing_issues_reference"  # most reliable
    PR_BODY_REGEX = "pr_body_regex"  # "Closes #123" parsed from the PR body
    COMMIT_MESSAGE = "commit_message"  # parsed from the commit message
    MERGE_COMMIT_MESSAGE = "merge_commit_message"
    MANUAL = "manual"
    HEURISTIC = "heuristic"  # e.g. same author + same time window


# --------------------------------------------------------------------------- #
# Deployment
# --------------------------------------------------------------------------- #
class DeploymentKind(StrEnum):
    TAG = "tag"
    RELEASE = "release"
    ENVIRONMENT = "environment"


class Environment(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"


# --------------------------------------------------------------------------- #
# Incidents
# --------------------------------------------------------------------------- #
class IncidentStatus(StrEnum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IncidentSeverity(StrEnum):
    SEV1 = "sev1"
    SEV2 = "sev2"
    SEV3 = "sev3"
    SEV4 = "sev4"
    UNKNOWN = "unknown"


class DetectionSource(StrEnum):
    USER_REPORT = "user_report"
    TEST_FAILURE = "test_failure"
    MONITORING = "monitoring"
    MAINTAINER = "maintainer"
    UPSTREAM_DEPENDENCY = "upstream_dependency"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Agent layer (retrieval / answers)
# --------------------------------------------------------------------------- #
class EvidenceKind(StrEnum):
    NODE = "node"  # a single graph node pulled as proof
    IMPORT_EDGE = "import_edge"  # a (:File)-[:IMPORTS]->(:File) hop
    GRAPH_PATH = "graph_path"  # a multi-hop path
    CODE_CHUNK = "code_chunk"  # a pgvector chunk
    COMMIT_DIFF = "commit_diff"
    DEPLOYMENT_WINDOW = "deployment_window"


class RetrievalStrategy(StrEnum):
    GRAPH_TRAVERSAL = "graph_traversal"
    VECTOR_SIMILARITY = "vector_similarity"
    HYBRID = "hybrid"
    LEXICAL = "lexical"
    TEMPORAL = "temporal"
    HUMAN = "human"


class AnswerIntent(StrEnum):
    CHANGE_IMPACT = "change_impact"
    ROOT_CAUSE = "root_cause"
    GENERAL = "general"


class AnswerStatus(StrEnum):
    OK = "ok"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ERROR = "error"


class ConfidenceBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

    @classmethod
    def from_score(cls, score: float) -> ConfidenceBand:
        if score >= 0.75:
            return cls.HIGH
        if score >= 0.45:
            return cls.MEDIUM
        if score > 0.0:
            return cls.LOW
        return cls.UNKNOWN


class EvidenceRole(StrEnum):
    """Role a piece of evidence plays inside an EvidencePath."""

    SYMPTOM = "symptom"
    ROOT_CAUSE = "root_cause"
    FIX = "fix"
    BLAST_RADIUS = "blast_radius"
    TIMELINE = "timeline"
    CONTEXT = "context"
