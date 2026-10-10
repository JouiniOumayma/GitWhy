#!/usr/bin/env python3
"""Launch the read-only GitHub MCP server (stdio) or list its tools.

Configured in an MCP client (Claude Desktop, ...):

    {
      "mcpServers": {
        "nexus-github": {
          "command": "python",
          "args": ["scripts/mcp_github.py", "--serve"],
          "env": {"GITHUB_TOKEN": "..."}
        }
      }
    }

The server is read-only by construction: closed tool registry over the
Phase-1 GET-only client; no GitHub mutation is reachable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    import os

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true",
                        help="run the stdio JSON-RPC loop (for MCP clients)")
    parser.add_argument("--list-tools", action="store_true",
                        help="print the tool registry as JSON and exit")
    args = parser.parse_args()

    _load_dotenv(PROJECT_ROOT / ".env")

    from ingestion.mcp_server import _TOOL_NAMES, serve, tools_list_payload

    if args.list_tools:
        print(json.dumps({"tool_count": len(_TOOL_NAMES),
                          "tools": tools_list_payload()}, indent=2))
        return 0
    if args.serve:
        return serve()
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
