"""NEXUS-DevIntel shared data contracts (Pydantic v2).

Import everything from here::

    from models import Answer, Commit, File, Incident, PullRequest, Repository

``NODE_MODELS`` is the registry used by ``fixtures/load_fixtures.py`` to map a
JSON fixture file to its Python class and Neo4j label.
"""

from .answer import Answer, Citation, EvidenceHop, EvidencePath
from .base import Edge, PersonRef, Provenance, SCHEMA_VERSION, scoped_id, utcnow
from .commit import Commit
from .deployment import Deployment, make_deployment_id, parse_semver
from .enums import (
    AnswerIntent,
    AnswerStatus,
    ChangeType,
    ConfidenceBand,
    DeploymentKind,
    DetectionSource,
    Environment,
    EvidenceKind,
    EvidenceRole,
    ImportKind,
    IncidentSeverity,
    IncidentStatus,
    LinkMethod,
    MergeStrategy,
    NodeKind,
    PRState,
    RelationType,
    RetrievalStrategy,
    SourceKind,
)
from .evidence import Evidence, EvidenceRef, VectorRef
from .file import File, FileChange, FileImpact, FileImport, make_file_id
from .incident import (
    Incident,
    IncidentCause,
    IncidentFileLink,
    IssueLink,
    IssueRef,
    make_incident_id,
)
from .pr import PullRequest, make_pr_id
from .repository import Repository, RepositoryAuditMetrics

#: Node models in ingestion order (a node is created before its relationships).
NODE_MODELS = (
    Repository,
    File,
    PersonRef,
    Commit,
    PullRequest,
    Deployment,
    Incident,
    Evidence,
    Answer,
)

#: ``Neo4j label -> model`` lookup (``PersonRef`` is not a full node model).
MODELS_BY_LABEL = {
    model.NEO4J_LABEL: model for model in NODE_MODELS if model is not PersonRef
}

#: JSON fixture file stem -> node model.
FIXTURE_MODELS = {
    "repositories": Repository,
    "files": File,
    "commits": Commit,
    "pull_requests": PullRequest,
    "deployments": Deployment,
    "incidents": Incident,
    "evidence": Evidence,
    "answers": Answer,
}

__all__ = [
    "NODE_MODELS",
    "MODELS_BY_LABEL",
    "FIXTURE_MODELS",
    "SCHEMA_VERSION",
    # nodes
    "Answer",
    "Commit",
    "Deployment",
    "Evidence",
    "File",
    "Incident",
    "PersonRef",
    "PullRequest",
    "Repository",
    # payloads
    "Citation",
    "Edge",
    "EvidenceHop",
    "EvidencePath",
    "EvidenceRef",
    "FileChange",
    "FileImpact",
    "FileImport",
    "IncidentCause",
    "IncidentFileLink",
    "IssueLink",
    "IssueRef",
    "Provenance",
    "RepositoryAuditMetrics",
    "VectorRef",
    # enums
    "AnswerIntent",
    "AnswerStatus",
    "ChangeType",
    "ConfidenceBand",
    "DeploymentKind",
    "DetectionSource",
    "Environment",
    "EvidenceKind",
    "EvidenceRole",
    "ImportKind",
    "IncidentSeverity",
    "IncidentStatus",
    "LinkMethod",
    "MergeStrategy",
    "NodeKind",
    "PRState",
    "RelationType",
    "RetrievalStrategy",
    "SourceKind",
    # helpers
    "make_deployment_id",
    "make_file_id",
    "make_incident_id",
    "make_pr_id",
    "parse_semver",
    "scoped_id",
    "utcnow",
]
