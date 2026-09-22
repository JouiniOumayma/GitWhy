"""File-body providers for the embedding indexer (real content, read-only).

Fixture ``File`` nodes carry metadata but not bodies, so the indexer defaults
to deterministic synthetic text. Phase 2 Week 3 requires indexing **real**
content; two read-only providers fill that gap without waiting for the full
Person-A pipeline:

* :class:`LocalRepoContentProvider` -- reads the file from a local clone and
  *verifies* it against the fixture's ``blob_sha`` (git blob SHA-1:
  ``sha1("blob <len>\\0" + content)``). A match is a git-verified fact
  (confidence 1.0); a mismatch falls back to synthetic text rather than
  embedding wrong content under a real file id.
* :class:`GitHubBlobContentProvider` -- fetches the blob from the GitHub REST
  API (``GET /repos/{repo}/git/blobs/{sha}``, base64). Uses
  :class:`ingestion.github_client.GitHubClient` as-is: stdlib, GET-only,
  rate-limit aware, ``GITHUB_TOKEN`` raised the quota. Same ``blob_sha``
  verification applies.

Both return ``(content, origin)`` where ``origin`` lands in the chunk
metadata (``content_origin``), so the pgvector rows stay honest about what
was embedded -- the same provenance discipline as the graph.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any


from models import File

#: Marker values for ``metadata['content_origin']``.
ORIGIN_SYNTHETIC = "synthetic"
ORIGIN_GIT_VERIFIED = "git_blob_verified"
ORIGIN_API_BLOB = "api_blob"


def git_blob_sha(content: str) -> str:
    """Git's object SHA-1 for a blob: ``sha1(b"blob <len>\\0" + bytes)``."""
    data = content.encode("utf-8")
    digest = hashlib.sha1(b"blob %d\x00" % len(data) + data)
    return digest.hexdigest()


class ContentResult:
    """What a provider returns: the text, or ``None`` to fall back to synthetic."""

    __slots__ = ("content", "origin", "confidence")

    def __init__(self, content: str, origin: str, confidence: float) -> None:
        self.content = content
        self.origin = origin
        self.confidence = confidence


class LocalRepoContentProvider:
    """Real bodies from a local clone, verified against the fixture's blob_sha."""

    def __init__(self, repo_path: str | Path) -> None:
        self._root = Path(repo_path).resolve()
        if not self._root.is_dir():
            raise FileNotFoundError(f"local clone not found: {self._root}")

    def __call__(self, file: File) -> str | None:
        candidate = self._root / file.path
        try:
            content = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None  # binary/absent file -> synthetic fallback
        if file.blob_sha and git_blob_sha(content) != file.blob_sha:
            return None  # checked-out content differs from the fixture revision
        return content


class GitHubBlobContentProvider:
    """Real bodies from the GitHub blobs API (read-only GET), blob_sha verified.

    Two cache layers so a long indexing run can be interrupted and replayed
    without re-spending the quota:

    * in-memory per blob SHA (fixtures often share blobs);
    * on-disk JSON files (``cache_dir/<sha>.json``) surviving process death --
      the CPU MiniLM encoding can outlast a shell timeout, and the ~250 blob
      GETs must not be replayed on every retry.
    """

    def __init__(self, client: Any, repository_id: str,
                 cache_dir: str | Path | None = None) -> None:
        from ingestion import GitHubClient

        self._client = client if client is not None else GitHubClient()
        self._repository_id = repository_id
        self._memory: dict[str, str | None] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, file: File) -> str | None:
        if not file.blob_sha:
            return None
        if file.blob_sha in self._memory:
            return self._memory[file.blob_sha]
        cached = self._read_disk_cache(file.blob_sha)
        if cached is not None:
            self._memory[file.blob_sha] = cached
            return cached
        content = self._fetch(file.blob_sha)
        self._memory[file.blob_sha] = content
        self._write_disk_cache(file.blob_sha, content)
        return content

    # ---- disk cache ---------------------------------------------------------#
    def _disk_path(self, blob_sha: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{blob_sha}.json"

    def _read_disk_cache(self, blob_sha: str) -> str | None:
        path = self._disk_path(blob_sha)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        # ``null`` means "fetched, binary/undecodable" -- a valid cached answer.
        return payload.get("content")

    def _write_disk_cache(self, blob_sha: str, content: str | None) -> None:
        path = self._disk_path(blob_sha)
        if path is None:
            return
        try:
            path.write_text(json.dumps({"content": content}), encoding="utf-8")
        except OSError:
            pass  # cache is best-effort

    def _fetch(self, blob_sha: str) -> str | None:
        payload, _ = self._client.get(f"/repos/{self._repository_id}/git/blobs/{blob_sha}")
        if payload.get("encoding") != "base64":
            return None
        try:
            return base64.b64decode(payload.get("content") or "").decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None  # binary blob -> synthetic fallback


def content_with_origin(provider: Any, file: File) -> tuple[str, str]:
    """Normalize a provider result into ``(content, origin)``.

    Any ``None``/empty result degrades to the deterministic synthetic body
    (origin ``synthetic``), so indexing never dies on one unreadable file.
    The origin value itself carries the trust signal (``git_blob_verified``
    means the body matched the fixture's git object SHA).
    """
    try:
        content = provider(file) if provider is not None else None
    except Exception:  # noqa: BLE001 - one broken file must not stop the run
        content = None
    if content and content.strip():
        return content, ORIGIN_GIT_VERIFIED
    from ingestion.embedding_indexer import _synthetic_file_body

    return _synthetic_file_body(file), ORIGIN_SYNTHETIC


__all__ = [
    "ORIGIN_API_BLOB",
    "ORIGIN_GIT_VERIFIED",
    "ORIGIN_SYNTHETIC",
    "GitHubBlobContentProvider",
    "LocalRepoContentProvider",
    "content_with_origin",
    "git_blob_sha",
]
