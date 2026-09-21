"""Read-only GitHub REST v3 client.

Design rules (Phase 1):

* **read-only** -- only ``GET`` requests are ever sent; there is no method that
  could mutate the repository state.
* **stdlib-only** -- ``urllib.request`` instead of ``requests``/``httpx`` so the
  runtime dependencies stay unchanged.
* **provenance-preserving** -- every payload is tagged ``source=github_api`` with
  the concrete ``api.github.com`` URL as ``source_uri``, which is what makes the
  resulting graph nodes auditable.
* **token-optional** -- ``GITHUB_TOKEN`` raises the rate limit from 60 to
  5 000 req/h but is not required for public repositories.

Rate-limit responses (HTTP 403/429 with ``X-RateLimit-Remaining: 0``) raise
:class:`GitHubRateLimited` carrying the reset timestamp, so callers can back off
instead of burning the remaining quota.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator

API_ROOT = "https://api.github.com"
PER_PAGE = 100
VERSION = "2022-11-28"


class GitHubClientError(RuntimeError):
    """Non-rate-limit API failure (network, auth, unexpected status)."""


class GitHubRateLimited(GitHubClientError):
    """The API refused the call because the hourly quota is exhausted."""

    def __init__(self, reset_epoch: float | None, resource: str = "core") -> None:
        super().__init__(
            f"GitHub rate limit exhausted (resource={resource}, reset={reset_epoch}). "
            "Set GITHUB_TOKEN to raise the quota from 60 to 5000 req/h."
        )
        self.reset_epoch = reset_epoch
        self.resource = resource


@dataclass
class GitHubClient:
    """Minimal REST v3 wrapper. All methods are read-only (``GET``)."""

    token: str | None = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN"))
    api_root: str = API_ROOT
    per_page: int = PER_PAGE

    # ---- transport --------------------------------------------------------- #
    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": VERSION,
            "User-Agent": "nexus-devintel-ingestion",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def get(self, path_or_url: str) -> tuple[dict[str, Any] | list[Any], str]:
        """``GET`` one resource; returns ``(payload, final_url)``."""
        url = path_or_url if path_or_url.startswith("http") else f"{self.api_root}{path_or_url}"
        request = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
                return json.loads(body), response.geturl()
        except urllib.error.HTTPError as error:
            if error.code in (403, 429):
                remaining = error.headers.get("X-RateLimit-Remaining")
                if remaining == "0":
                    reset = error.headers.get("X-RateLimit-Reset")
                    raise GitHubRateLimited(
                        float(reset) if reset else None
                    ) from error
            raise GitHubClientError(f"GET {url} failed: HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise GitHubClientError(f"GET {url} failed: {error.reason}") from error

    def get_paged(self, path: str) -> Iterator[dict[str, Any]]:
        """Yield every item of a paginated collection."""
        separator = "&" if "?" in path else "?"
        page = 1
        while True:
            payload, _ = self.get(f"{path}{separator}per_page={self.per_page}&page={page}")
            if not isinstance(payload, list):
                raise GitHubClientError(f"expected a JSON array from {path}, got object")
            for item in payload:
                yield item
            if len(payload) < self.per_page:
                return
            page += 1

    # ---- read-only endpoints ----------------------------------------------- #
    def repository(self, repo_id: str) -> tuple[dict[str, Any], str]:
        """``owner/name`` -> repository payload + final URL (for provenance)."""
        payload, url = self.get(f"/repos/{repo_id}")
        assert isinstance(payload, dict)
        return payload, url

    def commits(self, repo_id: str, *, sha: str | None = None,
                since: str | None = None) -> Iterator[dict[str, Any]]:
        """Every commit of the default (or ``sha``) branch, newest first."""
        query = ""
        if sha:
            query += f"&sha={sha}"
        if since:
            query += f"&since={since}"
        yield from self.get_paged(f"/repos/{repo_id}/commits{query}")

    def commit(self, repo_id: str, sha: str) -> tuple[dict[str, Any], str]:
        """One commit with its ``files`` array (the ``:MODIFIES`` payload)."""
        payload, url = self.get(f"/repos/{repo_id}/commits/{sha}")
        assert isinstance(payload, dict)
        return payload, url

    def git_tree(self, repo_id: str, ref: str = "HEAD", *, recursive: bool = True) -> dict[str, Any]:
        """The file listing of a ref, as a flat tree (source of ``:File`` nodes)."""
        payload, _ = self.get(f"/repos/{repo_id}/git/trees/{ref}?recursive={'1' if recursive else '0'}")
        assert isinstance(payload, dict)
        return payload

    def pull_requests(self, repo_id: str, *, state: str = "all") -> Iterator[dict[str, Any]]:
        yield from self.get_paged(f"/repos/{repo_id}/pulls?state={state}")

    def pull_request(self, repo_id: str, number: int) -> tuple[dict[str, Any], str]:
        payload, url = self.get(f"/repos/{repo_id}/pulls/{number}")
        assert isinstance(payload, dict)
        return payload, url

    def issue(self, repo_id: str, number: int) -> tuple[dict[str, Any], str]:
        """One issue. PRs share the numbering space: check ``payload['pr']``."""
        payload, url = self.get(f"/repos/{repo_id}/issues/{number}")
        assert isinstance(payload, dict)
        return payload, url

    def tags(self, repo_id: str) -> Iterator[dict[str, Any]]:
        yield from self.get_paged(f"/repos/{repo_id}/tags")
