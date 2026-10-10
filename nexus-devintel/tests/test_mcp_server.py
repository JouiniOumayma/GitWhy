"""MCP server tests -- offline, against a scripted fake GitHub client.

Pinned invariants:

* protocol: ``initialize`` / ``tools/list`` / ``tools/call`` / ``ping`` answer
  correctly; notifications are never answered; unknown methods get -32601;
* the tool registry is **closed**: an unknown tool name is rejected before any
  dispatch, and the registry exposes zero mutating operations;
* tool dispatch flows through the Phase-1 read-only client only (faked here);
* malformed JSON gets a parse error, not a crash of the loop.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from ingestion import mcp_server as mcp
from ingestion.mcp_server import (
    MCP_TOOLS,
    call_tool,
    handle_message,
    serve,
)

REPO = "httpie/cli"


class _FakeGitHubClient:
    """Scripted stand-in for the Phase-1 REST client (no network)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def get(self, path: str) -> tuple[dict[str, Any], str]:
        self.calls.append(("get", (path,)))
        if path.startswith("/search/"):
            return {"total_count": 1, "items": [
                {"number": 428, "title": "Simultaneous requests",
                 "state": "closed", "html_url": "u"}]}, "url"
        blob = {"encoding": "base64"}
        import base64
        blob["content"] = base64.b64encode(b"print('real code')").decode()
        return blob, "url"

    def git_tree(self, repo: str, ref: str = "HEAD", recursive: bool = True) -> dict:
        self.calls.append(("git_tree", (repo, ref)))
        return {"tree": [
            {"type": "blob", "path": "httpie/client.py", "size": 16,
             "sha": self._blob_sha_of(b"print('real code')")},
            {"type": "tree", "path": "httpie"},
        ]}

    def commit(self, repo: str, sha: str) -> tuple[dict, str]:
        self.calls.append(("commit", (repo, sha)))
        return {
            "sha": sha,
            "commit": {"message": "Close #1583\n\nbody",
                       "author": {"name": "A", "email": "a@b.c",
                                  "date": "2024-06-05T10:00:00Z"},
                       "committer": {"name": "A", "email": "a@b.c",
                                     "date": "2024-06-05T10:00:00Z"}},
            "author": {"login": "a"},
            "committer": {"login": "a"},
            "parents": [{"sha": "0" * 40}],
            "files": [{"filename": "httpie/ssl_.py", "status": "modified"}],
        }, "url"

    def pull_request(self, repo: str, number: int) -> tuple[dict, str]:
        self.calls.append(("pull_request", (repo, number)))
        return {
            "number": number, "title": "Fix SSL", "state": "closed", "merged": True,
            "user": {"login": "a"}, "merged_at": "2024-07-01T00:00:00Z",
            "merge_commit_sha": "b" * 40, "changed_files": 1,
            "html_url": "u", "head": {}, "base": {}, "labels": [],
        }, "url"

    @staticmethod
    def _blob_sha_of(data: bytes) -> str:
        import hashlib

        return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


# --------------------------------------------------------------------------- #
# Protocol level
# --------------------------------------------------------------------------- #
def test_initialize_handshake() -> None:
    response = handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                               "params": {}})
    assert response["result"]["protocolVersion"] == mcp.PROTOCOL_VERSION
    assert response["result"]["serverInfo"]["name"] == mcp.SERVER_NAME


def test_ping_and_notifications() -> None:
    assert handle_message({"jsonrpc": "2.0", "id": 2, "method": "ping"}) == \
        {"jsonrpc": "2.0", "id": 2, "result": {}}
    assert handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_method_is_method_not_found() -> None:
    response = handle_message({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
    assert response["error"]["code"] == mcp.JSONRPC_METHOD_NOT_FOUND


def test_malformed_json_gets_parse_error() -> None:
    stdout = io.StringIO()
    serve(io.StringIO("{not json}\n"), stdout)
    response = json.loads(stdout.getvalue())
    assert response["error"]["code"] == mcp.JSONRPC_PARSE_ERROR


def test_loop_survives_bad_lines() -> None:
    messages = ["{bad}", "", json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping"})]
    stdout = io.StringIO()
    serve(io.StringIO("\n".join(messages) + "\n"), stdout)
    lines = stdout.getvalue().strip().splitlines()
    assert len(lines) == 2  # parse error + ping answer, loop not dead
    assert json.loads(lines[1])["id"] == 9


# --------------------------------------------------------------------------- #
# Registry closure (the security core)
# --------------------------------------------------------------------------- #
def test_registry_is_closed_and_has_no_write_tool() -> None:
    expected = {"get_file", "list_files", "get_commit", "get_pull_request",
                "search_issues"}
    assert set(MCP_TOOLS) == expected
    for name in MCP_TOOLS:
        assert not any(word in name for word in
                       ("create", "update", "delete", "merge", "close", "edit",
                        "push", "post"))


def test_unknown_tool_is_rejected_before_dispatch() -> None:
    with pytest.raises(KeyError, match="unknown tool"):
        call_tool("delete_repository", {"repo": REPO})


def test_tools_list_payload_matches_registry() -> None:
    payload = tools = mcp.tools_list_payload()
    assert {tool["name"] for tool in payload} == set(MCP_TOOLS)
    assert all("inputSchema" in tool for tool in payload)


def test_missing_required_argument_is_value_error() -> None:
    with pytest.raises(ValueError, match="missing required argument: repo"):
        call_tool("get_file", {"path": "x"})


# --------------------------------------------------------------------------- #
# Tool dispatch through the fake client
# --------------------------------------------------------------------------- #
def test_tool_get_file_returns_verified_content() -> None:
    client = _FakeGitHubClient()
    outcome = call_tool("get_file", {"repo": REPO, "path": "httpie/client.py"},
                        client=client)
    assert outcome["verified"] is True
    assert "real code" in outcome["content"]
    assert outcome["blob_sha"] == client._blob_sha_of(b"print('real code')")


def test_tool_get_file_missing_path_is_an_error_result() -> None:
    outcome = call_tool("get_file", {"repo": REPO, "path": "no/such.py"},
                        client=_FakeGitHubClient())
    assert "error" in outcome


def test_tool_list_files_filters_by_prefix() -> None:
    outcome = call_tool("list_files", {"repo": REPO, "path_prefix": "httpie/"},
                        client=_FakeGitHubClient())
    assert outcome["file_count"] == 1
    assert outcome["files"][0]["path"] == "httpie/client.py"


def test_tool_get_commit_shapes_the_payload() -> None:
    outcome = call_tool("get_commit", {"repo": REPO, "sha": "a" * 40},
                        client=_FakeGitHubClient())
    assert outcome["subject"] == "Close #1583"
    assert outcome["files_changed"] == ["httpie/ssl_.py"]


def test_tool_get_pull_request_shapes_the_payload() -> None:
    outcome = call_tool("get_pull_request", {"repo": REPO, "number": 1596},
                        client=_FakeGitHubClient())
    assert outcome["state"] == "merged"
    assert outcome["merge_commit_id"] == "b" * 40


def test_tool_search_issues() -> None:
    outcome = call_tool("search_issues", {"repo": REPO, "query": "ssl"},
                        client=_FakeGitHubClient())
    assert outcome["total_count"] == 1
    assert outcome["results"][0]["number"] == 428


def test_tools_call_via_jsonrpc_roundtrip() -> None:
    response = handle_message({
        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": {"name": "list_files", "arguments": {"repo": REPO}},
    }, client=_FakeGitHubClient())
    inner = json.loads(response["result"]["content"][0]["text"])
    assert inner["repository"] == REPO


def test_serve_loop_end_to_end() -> None:
    requests = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "get_file",
                               "arguments": {"repo": REPO, "path": "httpie/client.py"}}}),
    ]) + "\n"
    stdout = io.StringIO()
    serve(io.StringIO(requests), stdout, client=_FakeGitHubClient())
    lines = [json.loads(line) for line in stdout.getvalue().strip().splitlines()]
    assert len(lines) == 3  # the notification produced no response
    assert lines[0]["result"]["serverInfo"]["name"] == mcp.SERVER_NAME
    tools = lines[1]["result"]["tools"]
    assert {tool["name"] for tool in tools} == set(MCP_TOOLS)
    inner = json.loads(lines[2]["result"]["content"][0]["text"])
    assert inner["verified"] is True
