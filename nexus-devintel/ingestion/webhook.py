"""GitHub webhook endpoint: verify HMAC, then push the pushed commits into Neo4j.

Phase 1 scope: ``push`` events only. The flow is::

    GitHub POST /webhook  ->  verify X-Hub-Signature-256 (HMAC-SHA256)
                          ->  fetch each pushed commit (read-only REST call)
                          ->  adapter -> contract models
                          ->  GraphWriter (MERGE on id, idempotent)

The server is :class:`http.server.BaseHTTPRequestHandler`-based: stdlib-only,
single-threaded, and meant for local development / demo. It is **not** hardened
for the open internet (no TLS, no rate limiting) -- see README.md.

The handler takes the fetcher and the writer as callables, so tests can inject
fakes and never touch the network or Neo4j.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

SECRET_ENV = "GITHUB_WEBHOOK_SECRET"
MAX_BODY_BYTES = 10 * 1024 * 1024  # GitHub caps push payloads well below this


def verify_signature(secret: str, signature_header: str | None, body: bytes) -> bool:
    """Constant-time check of ``X-Hub-Signature-256: sha256=<hex>``.

    ``hmac.compare_digest`` is used on the *bytes* of the digest to avoid both
    timing attacks and the ``==``-on-hex pitfalls. ``None``/malformed headers
    simply fail; callers decide whether that is fatal (production) or a warning
    (local demo without a secret configured).
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    received = signature_header[len("sha256="):]
    return hmac.compare_digest(expected.encode(), received.encode())


@dataclass
class PushEvent:
    """The subset of a GitHub ``push`` event this pipeline consumes."""

    repository_id: str
    ref: str
    after_sha: str
    commits: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PushEvent:
        full_name = payload.get("repository", {}).get("full_name")
        if not full_name:
            raise ValueError("push event without repository.full_name")
        return cls(
            repository_id=full_name,
            ref=payload.get("ref") or "",
            after_sha=payload.get("after") or "",
            commits=list(payload.get("commits") or []),
        )


class WebhookServer:
    """HTTP server around a ``push -> models -> Neo4j`` pipeline.

    Args:
        fetch_commit: ``f(repository_id, sha) -> API commit payload``. Injected
            so tests replace the GitHub REST call with a fake.
        write_models: ``f(iterable_of_models) -> stats``. The real one wraps a
            :class:`ingestion.graph_writer.GraphWriter`.
        secret: HMAC secret; defaults to ``$GITHUB_WEBHOOK_SECRET``. Empty ->
            signatures are logged but **not** enforced (dev-only).
        port: listen port (default 8765).
    """

    def __init__(
        self,
        fetch_commit: Callable[[str, str], dict[str, Any]],
        write_models: Callable[[list[Any]], Any],
        *,
        secret: str | None = None,
        port: int = 8765,
    ) -> None:
        self._fetch_commit = fetch_commit
        self._write_models = write_models
        self._secret = secret if secret is not None else os.environ.get(SECRET_ENV, "")
        self._port = port
        self.received: list[PushEvent] = []
        self.processed: list[dict[str, Any]] = []

    # ---- event handling --------------------------------------------------- #
    def handle_push(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Turn one push event into contract models and write them."""
        event = PushEvent.from_payload(payload)
        self.received.append(event)
        from .models_adapter import commit_from_api

        written = 0
        for head in event.commits:
            # The push payload only carries a summary: the files/parents detail
            # lives on /repos/{repo}/commits/{sha}, fetched read-only.
            api_payload = self._fetch_commit(event.repository_id, head["id"])
            model = commit_from_api(api_payload, event.repository_id,
                                    f"https://api.github.com/repos/{event.repository_id}/commits/{head['id']}")
            self._write_models([model])
            written += 1
        result = {
            "repository": event.repository_id,
            "ref": event.ref,
            "commits_written": written,
        }
        self.processed.append(result)
        return result

    # ---- HTTP plumbing ----------------------------------------------------- #
    def _handler_class(self):  # noqa: ANN202 - BaseHTTPRequestHandler subclass
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY_BYTES:
                    self._reply(413, {"error": "payload too large"})
                    return
                body = self.rfile.read(length) if length else b""

                if server._secret:
                    if not verify_signature(
                        server._secret, self.headers.get("X-Hub-Signature-256"), body
                    ):
                        self._reply(401, {"error": "invalid signature"})
                        return
                else:
                    print("[webhook] WARNING: no secret configured; signature NOT verified")

                event_name = self.headers.get("X-GitHub-Event")
                if event_name != "push":
                    self._reply(202, {"ignored": f"event {event_name}"})
                    return

                try:
                    payload = json.loads(body.decode("utf-8"))
                    result = server.handle_push(payload)
                    self._reply(200, result)
                except Exception as error:  # noqa: BLE001 - convert to a 500 JSON body
                    self._reply(500, {"error": str(error)})

            def _reply(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                print(f"[webhook] {self.address_string()} {format % args}")

        return Handler

    def serve_forever(self) -> None:  # pragma: no cover - requires a live socket
        httpd = ThreadingHTTPServer(("0.0.0.0", self._port), self._handler_class())
        print(f"[webhook] listening on :{self._port} (push events)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("[webhook] shutting down")
        finally:
            httpd.server_close()
