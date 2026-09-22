"""Read-only GitHub MCP server (stdio, JSON-RPC 2.0, stdlib-only).

Week-4 Phase 2 deliverable: expose the Phase-1 :class:`GitHubClient` to an
MCP-capable agent (Claude Desktop, LangGraph MCP client, ...) with the same
security posture as the rest of the project.

Transport
---------
stdio, newline-delimited JSON-RPC 2.0 (the MCP stdio convention): one request
per line on ``stdin``, one response per line on ``stdout``. No third-party
dependency -- consistent with ``github_client.py`` and ``webhook.py``, and it
keeps the security review surface minimal.

Read-only guarantee (three independent layers)
----------------------------------------------
1. **method pinning** -- :class:`GitHubClient` only ever sends ``GET``
   (pinned in its ``get()`` transport);
2. **closed tool registry** -- ``MCP_TOOLS`` is a module-level constant; a
   tool name outside the registry is rejected before any dispatch, so no
   caller can reach a code path that was not security-reviewed;
3. **no write surface at all** -- the registry exposes no mutating GitHub
   operation, and a mutation-shaped *argument* (e.g. a ``method: POST``)
   cannot change the verb because the client never reads one.

Protocol coverage (enough for tool discovery and use):

* ``initialize`` -> protocol version + capabilities
* ``notifications/initialized`` -> acknowledged silently
* ``ping`` -> ``{}``
* ``tools/list`` -> the registry, JSON-schema'd
* ``tools/call`` -> dispatched with argument validation
* unknown method -> JSON-RPC error ``-32601``
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

JSONRPC_PARSE_ERROR = -32700
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "nexus-devintel-github"
SERVER_VERSION = "0.1.0"

#: Maximum result size before truncation (agents do not need 1 MB payloads).
MAX_RESULT_CHARS = 20_000


# --------------------------------------------------------------------------- #
# Tool implementations (thin wrappers over the Phase-1 read-only client)
# --------------------------------------------------------------------------- #
def _tool_get_file(repo: str, path: str, ref: str | None = None,
                   client: Any = None) -> dict[str, Any]:
    """Read ONE file content from a GitHub repository (read-only)."""
    from ingestion import GitHubClient
    from ingestion.content import (
        GitHubBlobContentProvider,
        content_with_origin,
        git_blob_sha,
    )
    from models import File

    real_client = client if client is not None else GitHubClient()
    tree = real_client.git_tree(repo, ref=ref or "HEAD", recursive=True)
    entry = next((e for e in tree.get("tree", []) if e.get("path") == path), None)
    if entry is None or entry.get("type") != "blob":
        return {"error": f"file not found in {repo}@{ref or 'HEAD'}: {path}"}
    provider = GitHubBlobContentProvider(real_client, repo)
    file_model = File(
        id=f"{repo}::{path}",
        repository_id=repo,
        path=path,
        blob_sha=entry.get("sha"),
    )
    content, origin = content_with_origin(provider, file_model)
    if origin == "synthetic":
        return {"error": f"content unreadable or binary: {path}", "blob_sha": entry.get("sha")}
    return {
        "repository": repo,
        "path": path,
        "ref": ref or "HEAD",
        "blob_sha": entry.get("sha"),
        "verified": git_blob_sha(content) == entry.get("sha"),
        "size_bytes": len(content.encode("utf-8")),
        "content": content,
    }


def _tool_list_files(repo: str, ref: str | None = None, path_prefix: str = "",
                     client: Any = None) -> dict[str, Any]:
    """List the files of a repository tree (read-only)."""
    from ingestion import GitHubClient

    real_client = client if client is not None else GitHubClient()
    tree = real_client.git_tree(repo, ref=ref or "HEAD", recursive=True)
    files = [
        {"path": e["path"], "size": e.get("size"), "blob_sha": e.get("sha")}
        for e in tree.get("tree", [])
        if e.get("type") == "blob" and e.get("path", "").startswith(path_prefix)
    ]
    return {"repository": repo, "ref": ref or "HEAD", "file_count": len(files),
            "files": files[:200], "truncated": len(files) > 200}


def _tool_get_commit(repo: str, sha: str, client: Any = None) -> dict[str, Any]:
    """One commit with its changed files (read-only)."""
    from ingestion import GitHubClient, commit_from_api

    real_client = client if client is not None else GitHubClient()
    payload, url = real_client.commit(repo, sha)
    commit = commit_from_api(payload, repo, url)
    return {
        "repository": repo,
        "sha": commit.id,
        "subject": commit.subject,
        "author": commit.author.name if commit.author else None,
        "committed_at": commit.committed_at.isoformat() if commit.committed_at else None,
        "files_changed": [c.path for c in commit.files_changed],
        "parents": commit.parents,
    }


def _tool_get_pull_request(repo: str, number: int, client: Any = None) -> dict[str, Any]:
    """One pull request, including the issues GitHub says it closes (read-only)."""
    from ingestion import GitHubClient, pull_request_from_api

    real_client = client if client is not None else GitHubClient()
    payload, url = real_client.pull_request(repo, number)
    pr = pull_request_from_api(payload, repo, url)
    return {
        "repository": repo,
        "number": pr.number,
        "title": pr.title,
        "state": pr.state.value,
        "author": pr.author.login if pr.author else None,
        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
        "merge_commit_id": pr.merge_commit_id,
        "changed_files_count": pr.changed_files_count,
    }


def _tool_search_issues(repo: str, query: str, limit: int = 10,
                        client: Any = None) -> dict[str, Any]:
    """Search issues/PRs of one repository (read-only)."""
    real_client = client
    if real_client is None:
        from ingestion import GitHubClient
        real_client = GitHubClient()
    payload, _ = real_client.get(f"/search/issues?q={_urlquote(query + ' repo:' + repo)}&per_page={min(int(limit), 50)}")
    items = payload.get("items") or []
    return {
        "total_count": payload.get("total_count"),
        "results": [
            {"number": i.get("number"), "title": i.get("title"),
             "state": i.get("state"),
             "is_pr": "pull_request" in i,
             "url": i.get("html_url")}
            for i in items[:min(int(limit), 50)]
        ],
    }


def _urlquote(value: str) -> str:
    import urllib.parse

    return urllib.parse.quote(value, safe="")


# --------------------------------------------------------------------------- #
# Closed tool registry: the security boundary of this server
# --------------------------------------------------------------------------- #
MCP_TOOLS: dict[str, dict[str, Any]] = {
    "get_file": {
        "description": "Read one file's content from a GitHub repository "
                       "(verified against its git blob SHA). Read-only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/name"},
                "path": {"type": "string", "description": "repository-relative path"},
                "ref": {"type": "string", "description": "branch/tag/sha (default HEAD)"},
            },
            "required": ["repo", "path"],
        },
        "handler": _tool_get_file,
    },
    "list_files": {
        "description": "List the file tree of a repository (read-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "ref": {"type": "string"},
                "path_prefix": {"type": "string"},
            },
            "required": ["repo"],
        },
        "handler": _tool_list_files,
    },
    "get_commit": {
        "description": "Read one commit: message, author, changed files (read-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "sha": {"type": "string"},
            },
            "required": ["repo", "sha"],
        },
        "handler": _tool_get_commit,
    },
    "get_pull_request": {
        "description": "Read one pull request's metadata (read-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "number": {"type": "integer"},
            },
            "required": ["repo", "number"],
        },
        "handler": _tool_get_pull_request,
    },
    "search_issues": {
        "description": "Search issues and pull requests of one repository (read-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["repo", "query"],
        },
        "handler": _tool_search_issues,
    },
}

#: Frozen at import time: nothing can register a tool after the server starts.
_TOOL_NAMES = frozenset(MCP_TOOLS)


def tools_list_payload() -> list[dict[str, Any]]:
    """The ``tools/list`` result, schema-complete."""
    return [
        {"name": name, "description": tool["description"],
         "inputSchema": tool["inputSchema"]}
        for name, tool in MCP_TOOLS.items()
    ]


def call_tool(name: str, arguments: dict[str, Any] | None,
              *, client: Any = None) -> dict[str, Any]:
    """Dispatch a ``tools/call`` against the closed registry.

    Raises :class:`KeyError` on an unknown tool (the caller turns it into a
    JSON-RPC error); missing required arguments raise :class:`ValueError`.
    """
    if name not in _TOOL_NAMES:
        raise KeyError(f"unknown tool: {name}")
    tool = MCP_TOOLS[name]
    arguments = arguments or {}
    required = tool["inputSchema"].get("required") or []
    for argument in required:
        if argument not in arguments:
            raise ValueError(f"missing required argument: {argument}")
    return tool["handler"](client=client, **arguments)


# --------------------------------------------------------------------------- #
# JSON-RPC framing
# --------------------------------------------------------------------------- #
def _result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


def _truncate(result: Any) -> Any:
    if isinstance(result, dict) and isinstance(result.get("content"), str) \
            and len(result["content"]) > MAX_RESULT_CHARS:
        result = dict(result)
        result["content"] = result["content"][:MAX_RESULT_CHARS] + "\n...[truncated]"
        result["truncated"] = True
    return result


def handle_message(message: dict[str, Any], *, client: Any = None) -> dict[str, Any] | None:
    """One JSON-RPC message -> one response (``None`` for notifications)."""
    method = message.get("method", "")
    request_id = message.get("id")

    if method.startswith("notifications/"):
        return None  # notifications are never answered

    try:
        if method == "initialize":
            return _result(request_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })
        if method == "ping":
            return _result(request_id, {})
        if method == "tools/list":
            return _result(request_id, {"tools": tools_list_payload()})
        if method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            try:
                outcome = call_tool(name, params.get("arguments"), client=client)
            except KeyError:
                return _error(request_id, JSONRPC_INVALID_PARAMS,
                              f"unknown tool: {name}")
            except ValueError as error:
                return _error(request_id, JSONRPC_INVALID_PARAMS, str(error))
            except Exception as error:  # noqa: BLE001 - tool boundary
                return _error(request_id, JSONRPC_INTERNAL_ERROR,
                              f"tool failed: {type(error).__name__}: {error}")
            return _result(request_id, {
                "content": [{"type": "text",
                             "text": json.dumps(_truncate(outcome), ensure_ascii=False)}],
            })
        return _error(request_id, JSONRPC_METHOD_NOT_FOUND, f"method not found: {method}")
    except Exception as error:  # noqa: BLE001 - server boundary
        return _error(request_id, JSONRPC_INTERNAL_ERROR,
                      f"{type(error).__name__}: {error}")


def serve(stdin: Any = None, stdout: Any = None, *, client: Any = None) -> None:
    """Main loop: one JSON request per line, one response per line."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            response = _error(None, JSONRPC_PARSE_ERROR, f"parse error: {error}")
        else:
            response = handle_message(message, client=client)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()


def main() -> int:
    print(f"{SERVER_NAME} {SERVER_VERSION}: read-only GitHub MCP server ready "
          f"({len(_TOOL_NAMES)} tools)", file=sys.stderr)
    serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
