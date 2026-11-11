#!/usr/bin/env python3
"""Ingestion complète d'un dépôt depuis son lien GitHub (CLI).

Même moteur que ``POST /api/ingest`` (voir ``nexus/ingest_service.py``) : la
logique vit dans un seul module, cette commande ne fait que l'appeler.

Usage
-----
    python scripts/ingest_repo.py https://github.com/owner/name
    python scripts/ingest_repo.py owner/name --skip-vectors
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexus.ingest_service import repo_id_from_input, run_ingestion  # noqa: E402


def _load_dotenv(path: Path) -> None:
    """Complète ``os.environ`` depuis ``.env`` sans écraser l'existant."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if not entry or "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingère un dépôt GitHub de bout en bout (graphe + vecteurs).",
    )
    parser.add_argument("repository", help="https://github.com/owner/name ou owner/name")
    parser.add_argument("--skip-vectors", action="store_true",
                        help="graphe seulement (sans MiniLM ni PostgreSQL)")
    args = parser.parse_args()

    _load_dotenv(PROJECT_ROOT / ".env")

    try:
        repo_id = repo_id_from_input(args.repository)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    print(f"== Ingestion de {repo_id} ==")
    print(f"  GitHub token : {'oui (5000 req/h)' if token else 'non (60 req/h)'}")

    def on_step(name: str, status: str, detail: str) -> None:
        mark = {"running": "…", "done": "✓", "failed": "✗"}.get(status, "-")
        print(f"  {mark} {name:<8} {detail}")

    try:
        result = run_ingestion(
            repo_id, skip_vectors=args.skip_vectors, on_step=on_step
        )
    except urllib.error.HTTPError as exc:
        print(f"error: GitHub a refusé la requête ({exc.code}) : {exc.reason}",
              file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 - frontière CLI
        print(f"error: ingestion interrompue : {exc}", file=sys.stderr)
        return 4

    print(f"  durée : {result['duration_s']}s")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
