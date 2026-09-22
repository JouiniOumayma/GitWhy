"""Read-only GitHub GraphQL v4 client + ``closingIssuesReferences`` enrichment.

Why this module exists (Phase 2, Week 2): git does not store "this PR closed
that issue"; that relation lives server-side in the ``closingIssuesReferences``
field, and the local ``git log`` audit only recovered 5 complete
Incident -> PR -> Commit chains on httpie/cli -- too few for a convincing Root
Cause demo. This module asks GitHub GraphQL for the authoritative links and
turns them into ``(:PR)-[:CLOSES]->(:Incident)`` edges.

Security posture (mirrors ``ingestion/github_client.py``):

* **read-only** -- the transport *hard-codes* ``method="POST"`` (GraphQL
  requires it) but the query text is assembled exclusively from a module-level
  constant: no caller can inject a ``mutation``. ``_guard_read_only()`` rejects
  any query containing the token ``mutation`` before it leaves the process, and
  a whitelist of top-level fields pins what the client may ever ask for;
* **stdlib-only** -- ``urllib.request``, like the REST client;
* **provenance-preserving** -- edges carry ``source=github_graphql``, the
  GraphQL endpoint as ``source_uri``, ``link_method =
  graphql_closing_issues_reference`` and ``confidence = 1.0`` (server-side
  fact, the most reliable link method in ``models.enums.LinkMethod``);
* **token-aware** -- ``GITHUB_TOKEN`` is required by the real API for GraphQL
  (no anonymous tier); the client fails fast with a clear message.

Pagination: one query per PR is wasteful for ~280 PRs; the module pages through
``pullRequests(first: 100, after: ...)`` on the repository object itself, so
httpie/cli costs ~3 requests total.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from models import (
    Edge,
    IssueLink,
    LinkMethod,
    NodeKind,
    Provenance,
    PullRequest,
    RelationType,
    SourceKind,
    make_incident_id,
    utcnow,
)

GRAPHQL_ENDPOINT = "https://api.github.com/graphql"
EXTRACTOR = "nexus-devintel.ingestion.github_graphql"
EXTRACTOR_VERSION = "0.1.0"
#: Server-side fact: GitHub itself maintains closingIssuesReferences.
CLOSING_REFERENCE_CONFIDENCE = 1.0
#: Safety net on runaways; httpie/cli has ~280 PRs in total.
MAX_PRS_PER_RUN = 1000

#: The query is a module-level constant: callers can influence *parameters*
#: (owner, name, cursor -- bound values, never interpolated), never its shape.
PR_CLOSING_ISSUES_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    id
    pullRequests(first: 100, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      totalCount
      nodes {
        number
        title
        url
        closedAt
        closingIssuesReferences(first: 50) {
          totalCount
          nodes { number title url closedAt }
        }
      }
    }
  }
}
"""

#: Top-level query fields this client is ever allowed to send. Anything else
#: (mutation, another query shape) is rejected before transport.
_ALLOWED_QUERY_ROOTS = frozenset({"query"})


class GraphQLClientError(RuntimeError):
    """Non-rate-limit GraphQL failure (network, auth, errors in response)."""


class GraphQLMutationBlocked(GraphQLClientError):
    """A query resembling a mutation never left the process."""


class GraphQLRateLimited(GraphQLClientError):
    """The GraphQL quota (5000 points/h, 100 points/min) is exhausted."""

    def __init__(self, reset_epoch: float | None) -> None:
        super().__init__(
            f"GitHub GraphQL rate limit exhausted (reset={reset_epoch}). "
            "Set/refresh GITHUB_TOKEN."
        )
        self.reset_epoch = reset_epoch


def _guard_read_only(query: str) -> str:
    """Reject anything that is not a plain ``query`` before transport.

    Defence in depth, in order:
    1. the query must start with ``query`` (whitespace/comments stripped);
    2. the word ``mutation`` must not appear anywhere in the text;
    3. interpolation is impossible by construction -- the callers pass one of
       the two module-level constants and bound parameters separately.
    """
    stripped = query.lstrip()
    if not stripped.startswith(("query", "{")):
        raise GraphQLMutationBlocked("only read-only 'query' operations are allowed")
    if re.search(r"\bmutation\b", query, re.IGNORECASE):
        raise GraphQLMutationBlocked("mutations are blocked: this client is read-only")
    if re.search(r"\boperationName\s*:\s*[\"']?mutation", query, re.IGNORECASE):
        raise GraphQLMutationBlocked("named mutation operations are blocked")
    return query


@dataclass
class ClosingIssue:
    """One ``closingIssuesReferences.nodes[]`` entry."""

    number: int
    title: str | None
    url: str | None
    closed_at: str | None


@dataclass
class PullRequestLinks:
    """One PR node plus the issues GitHub says it closes."""

    number: int
    title: str | None
    url: str | None
    closed_at: str | None
    closing_issues: list[ClosingIssue]


def build_issue_links(pr: PullRequestLinks, repository_id: str) -> list[IssueLink]:
    """GraphQL facts -> ``IssueLink`` payloads for ``PullRequest.closes_issues``."""
    links: list[IssueLink] = []
    for issue in pr.closing_issues:
        links.append(
            IssueLink(
                incident_id=make_incident_id(repository_id, issue.number),
                issue_number=issue.number,
                method=LinkMethod.GRAPHQL_CLOSING_REFERENCE,
                confidence=CLOSING_REFERENCE_CONFIDENCE,
                referenced_in="pr_body",
                evidence_text=issue.title or f"#{issue.number}",
            )
        )
    return links


def build_close_edges(
    pr_links: PullRequestLinks,
    *,
    repository_id: str | None = None,
    ingested_at: Any = None,
) -> list[Edge]:
    """``(:PR)-[:CLOSES]->(:Incident)`` edges, ready for ``GraphWriter``.

    Provenance: ``source=github_graphql`` (a distinct ``SourceKind`` that
    already exists in ``models.enums``), the GraphQL endpoint as URI, and the
    ``graphql_closing_issues_reference`` link method at confidence 1.0 -- the
    schema README explicitly reserves this combination for the Week-2 pass.
    """
    repo_id = repository_id or pr_links.repository_id
    provenance = Provenance(
        source=SourceKind.GITHUB_GRAPHQL,
        source_uri=f"{GRAPHQL_ENDPOINT}#closingIssuesReferences",
        extractor=EXTRACTOR,
        extractor_version=EXTRACTOR_VERSION,
        ingested_at=ingested_at or utcnow(),
        confidence=CLOSING_REFERENCE_CONFIDENCE,
    )
    edges: list[Edge] = []
    for issue in pr_links.closing_issues:
        edges.append(
            Edge.link(
                source_id=f"{repo_id}#{pr_links.number}",
                source_label=NodeKind.PR,
                type=RelationType.CLOSES,
                target_id=make_incident_id(repo_id, issue.number),
                target_label=NodeKind.INCIDENT,
                provenance=provenance,
                confidence=CLOSING_REFERENCE_CONFIDENCE,
                method=LinkMethod.GRAPHQL_CLOSING_REFERENCE,
                referenced_in="pr_body",
                evidence_text=issue.title or f"#{issue.number}",
            )
        )
    return edges


@dataclass
class EnrichmentStats:
    """Counters of one enrichment pass (printed by the CLI, asserted in tests)."""

    prs_seen: int = 0
    prs_with_closing_issues: int = 0
    edges_built: int = 0
    incidents_discovered: set[int] | None = None

    def __post_init__(self) -> None:
        if self.incidents_discovered is None:
            self.incidents_discovered = set()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prs_seen": self.prs_seen,
            "prs_with_closing_issues": self.prs_with_closing_issues,
            "edges_built": self.edges_built,
            "incidents_discovered": sorted(self.incidents_discovered or set()),
        }


class GitHubGraphQLClient:
    """Minimal GraphQL v4 wrapper. All operations are read-only (``query``)."""

    def __init__(self, token: str | None = None, endpoint: str = GRAPHQL_ENDPOINT) -> None:
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self.endpoint = endpoint
        if not self.token:
            raise GraphQLClientError(
                "GITHUB_TOKEN is required for the GitHub GraphQL API "
                "(no anonymous access). Export it or put it in .env."
            )

    # ---- transport --------------------------------------------------------- #
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": "nexus-devintel-ingestion",
        }

    def execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one read-only ``query``; raise on transport/GraphQL errors.

        The method is ``POST`` because the GraphQL protocol requires it -- that
        is a transport detail, not a write: the *payload* is pinned to queries
        by :func:`_guard_read_only`.
        """
        _guard_read_only(query)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps({"query": query, "variables": variables or {}}).encode("utf-8"),
            headers=self._headers(),
            method="POST",  # protocol requirement; payload is query-only
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in (403, 429):
                reset = error.headers.get("X-RateLimit-Reset") if error.headers else None
                raise GraphQLRateLimited(float(reset) if reset else None) from error
            raise GraphQLClientError(f"POST {self.endpoint} failed: HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise GraphQLClientError(f"POST {self.endpoint} failed: {error.reason}") from error

        if "errors" in payload:
            raise GraphQLClientError(f"GraphQL errors: {payload['errors']}")
        return payload.get("data") or {}

    # ---- read-only operations ------------------------------------------------#
    def iter_pull_requests_with_closing_issues(
        self, repository_id: str, *, max_prs: int = MAX_PRS_PER_RUN
    ) -> Iterator[PullRequestLinks]:
        """Page through every PR of ``owner/name`` yielding its closing issues."""
        owner, _, name = repository_id.partition("/")
        if not owner or not name:
            raise ValueError(f"repository_id must be 'owner/name', got {repository_id!r}")
        cursor: str | None = None
        seen = 0
        while True:
            data = self.execute(
                PR_CLOSING_ISSUES_QUERY,
                {"owner": owner, "name": name, "cursor": cursor},
            )
            repo = (data or {}).get("repository") or {}
            page = repo.get("pullRequests") or {}
            for node in page.get("nodes") or []:
                seen += 1
                if seen > max_prs:
                    return
                yield PullRequestLinks(
                    number=int(node["number"]),
                    title=node.get("title"),
                    url=node.get("url"),
                    closed_at=node.get("closedAt"),
                    closing_issues=[
                        ClosingIssue(
                            number=int(issue["number"]),
                            title=issue.get("title"),
                            url=issue.get("url"),
                            closed_at=issue.get("closedAt"),
                        )
                        for issue in (node.get("closingIssuesReferences") or {}).get("nodes") or []
                    ],
                )
            page_info = page.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                return
            cursor = page_info.get("endCursor")
            if cursor is None:
                return


class ClosingIssuesEnricher:
    """Fetch -> map -> write pipeline: GraphQL facts into the existing graph.

    Deliberately thin: it reuses ``GraphWriter`` (idempotent ``MERGE``) instead
    of duplicating Cypher, and it can create the stub ``(:Incident)`` nodes the
    ``:CLOSES`` edges need when the issue was never ingested -- otherwise the
    writer would silently skip the edge (``rel_skipped_unresolved``).
    """

    def __init__(self, client: GitHubGraphQLClient, writer: Any) -> None:
        self._client = client
        self._writer = writer

    def enrich(
        self,
        repository_id: str,
        *,
        create_incidents: bool = True,
        max_prs: int = MAX_PRS_PER_RUN,
        pr_numbers: Sequence[int] | None = None,
    ) -> EnrichmentStats:
        """Run one enrichment pass; returns counters (never raises on missing nodes).

        Args:
            create_incidents: when True (default), issues not yet in the graph
                get a minimal ``(:Incident)`` stub node so the ``:CLOSES`` edge
                resolves. Stubs keep the issue's own title/url and carry
                ``source=github_graphql`` provenance.
            pr_numbers: optional filter to enrich a single PR (idempotent
                targeted re-run).
        """
        stats = EnrichmentStats()
        from collections import Counter

        for pr_links in self._client.iter_pull_requests_with_closing_issues(
            repository_id, max_prs=max_prs
        ):
            stats.prs_seen += 1
            if not pr_links.closing_issues:
                continue
            stats.prs_with_closing_issues += 1

            if create_incidents:
                for issue in pr_links.closing_issues:
                    incident_model = self._incident_stub(pr_links, issue, repository_id)
                    self._writer.write_node(incident_model)
                    stats.incidents_discovered.add(issue.number)

            edges = build_close_edges(pr_links, repository_id=repository_id)
            for edge in edges:
                linked = self._writer.write_edge(edge)
                if linked:
                    stats.edges_built += 1
        return stats

    @staticmethod
    def _incident_stub(pr_links: PullRequestLinks, issue: ClosingIssue,
                       repository_id: str) -> Any:
        from models import (
            DetectionSource,
            Incident,
            IncidentStatus,
        )

        return Incident(
            id=make_incident_id(repository_id, issue.number),
            repository_id=repository_id,
            number=issue.number,
            title=issue.title or f"#{issue.number}",
            url=issue.url,
            status=IncidentStatus.CLOSED if issue.closed_at else IncidentStatus.OPEN,
            closed_at=None,
            detection_source=DetectionSource.USER_REPORT,
            provenance=Provenance(
                source=SourceKind.GITHUB_GRAPHQL,
                source_uri=issue.url or f"{GRAPHQL_ENDPOINT}#issue-{issue.number}",
                extractor=EXTRACTOR,
                extractor_version=EXTRACTOR_VERSION,
                confidence=CLOSING_REFERENCE_CONFIDENCE,
            ),
        )


__all__ = [
    "CLOSING_REFERENCE_CONFIDENCE",
    "ClosingIssuesEnricher",
    "ClosingIssue",
    "EnrichmentStats",
    "GraphQLClientError",
    "GraphQLMutationBlocked",
    "GraphQLRateLimited",
    "GitHubGraphQLClient",
    "PR_CLOSING_ISSUES_QUERY",
    "PullRequestLinks",
    "build_close_edges",
    "build_issue_links",
]
