"""Ingestion layer: GitHub connector + webhook -> Neo4j (Phase 1 deliverable).

Two entry points:

* :class:`ingestion.github_client.GitHubClient` -- read-only REST v3 client that
  maps repository facts (commits, files, PRs, issues) onto the Pydantic
  contracts from :mod:`models`, with full ``Provenance`` (``github_api`` source).
* :class:`ingestion.webhook.WebhookServer` -- a tiny ``http.server`` endpoint
  that verifies GitHub's HMAC signature and pushes each push event into Neo4j
  through :class:`ingestion.graph_writer.GraphWriter`.

Both are stdlib-only (``urllib`` + ``http.server``): no new runtime dependency.
"""

from .github_client import GitHubClient, GitHubClientError, GitHubRateLimited
from .graph_writer import GraphWriter
from .models_adapter import (
    commit_from_api,
    file_from_api,
    incident_from_api,
    pull_request_from_api,
    repository_from_api,
)
from .webhook import WebhookServer, verify_signature

__all__ = [
    "GitHubClient",
    "GitHubClientError",
    "GitHubRateLimited",
    "GraphWriter",
    "WebhookServer",
    "verify_signature",
    "commit_from_api",
    "file_from_api",
    "incident_from_api",
    "pull_request_from_api",
    "repository_from_api",
]
