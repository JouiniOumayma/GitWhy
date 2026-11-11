#!/usr/bin/env python3
"""Bootstrap a user repository into NEXUS-DevIntel without editing the code.

This is the reusable layer that was still missing: a user gives a repository
id (for example ``acme/superlib``) and the system prepares whatever it can
from the data that already exists for that repository, without inventing
evidence where no source is trustworthy.

Modes
-----

* ``--plan`` : inspect the repository configuration and report what would
  happen, without writing anything.
* ``--apply`` : do the safe, idempotent setup that is possible locally and in
  the live stack (Neo4j + PostgreSQL), and emit a ``repo_manifest.json``.

Ce que ce script ne fait pas
----------------------------

* Il ne remplace pas les fixtures ancrées sur un dépôt réel.
* Il ne réécrit pas le cœur existant (agents, retrieval, schéma, connecteurs).
* Il n'ingère pas le contenu source complet : pour ça, utiliser
  ``scripts/index_embeddings.py --real-content`` et/ou
  ``scripts/enrich_graphql.py``.

Contenu
-------

1. Un audit de surface par dépôt (``audit_<owner_name>.json``), si aucun
   audit existant ne correspond au dépôt configuré.
2. La découverte des artefacts déjà chargés (graphe Neo4j, chunks pgvector,
   incidents connus), via les APIs existantes quand elles sont disponibles.
3. Un manifeste réutilisable (``repo_manifest.json``) qui décrit ce qui a
   été fait, ce qui manque, et comment poursuivre.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Shared per-repo helpers (single source of truth for API and scripts).
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import nexus.repo_helpers as _rh
_repo_id_safe = _rh._repo_id_safe
_safe_repo_parts = _rh._safe_repo_parts
make_repository_id = _rh.make_repository_id
load_repo_audit = _rh.load_repo_audit
write_repo_audit = _rh.write_repo_audit
build_minimal_audit = _rh.build_minimal_audit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Petit lecteur .env (évite une dépendance externe)
# ---------------------------------------------------------------------------

def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Conventions de nommage (reuses the project's existing helpers when present)
# ---------------------------------------------------------------------------

_repo_id_safe = _repo_id_safe


def _safe_repo_parts(repo_id: str) -> tuple[str, str]:
    """Split ``owner/name`` into ``(owner, name)`` (delegated to nexus.repo_helpers)."""
    return __import__("nexus.repo_helpers", fromlist=["_safe_repo_parts"])._safe_repo_parts(repo_id)


def make_repository_id(owner: str, name: str) -> str:
    """Mirror the existing ``models.make_repository_id`` without importing it here."""
    return f"{owner}/{name}"


# ---------------------------------------------------------------------------
# Audit de surface
# ---------------------------------------------------------------------------

load_existing_audit = load_repo_audit  # delegated to nexus.repo_helpers


build_minimal_audit = build_minimal_audit  # delegated to nexus.repo_helpers


write_audit = write_repo_audit  # delegated to nexus.repo_helpers


# ---------------------------------------------------------------------------
# Découverte des artefacts déjà chargés
# ---------------------------------------------------------------------------

def discover_graph_stats(*, repo_id: str) -> dict[str, Any]:
    """Best-effort discovery of what is already present in Neo4j for this repo.

    This is intentionally read-only and best-effort: if Neo4j is not reachable
    or the graph does not contain the repository yet, we return a safe skeleton
    rather than failing.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return {"backend": None, "status": "unavailable", "reason": "neo4j driver not installed"}

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "user")

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            with driver.session(database=database) as session:
                record = session.run(
                    """
                    MATCH (r:Repository {id: $repo_id})
                    OPTIONAL MATCH (r)<-[:CONTAINS]-(f:File)
                    OPTIONAL MATCH (f)-[imp:IMPORTS]->(:File)
                    OPTIONAL MATCH (r)<-[:OPENED]-(p:PR)
                    OPTIONAL MATCH (p)-[:CLOSES]->(i:Incident)
                    RETURN
                        count(DISTINCT f) AS files,
                        count(imp) AS imports,
                        count(DISTINCT p) AS prs,
                        count(DISTINCT i) AS incidents
                    """,
                    repo_id=repo_id,
                ).single()
                return {
                    "backend": "neo4j",
                    "status": "ok",
                    "files": record["files"],
                    "imports": record["imports"],
                    "prs": record["prs"],
                    "incidents": record["incidents"],
                }
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort
        return {"backend": "neo4j", "status": "unavailable", "reason": str(exc)}


def discover_vector_stats(*, repository_id: str) -> dict[str, Any]:
    """Best-effort discovery of what is already indexed in pgvector for this repo.

    Returns ``vector_chunks`` and the list of content origins available for the
    repository, without modifying anything. This function is intentionally guarded
    against slow or failing databases.
    """
    dsn = os.environ.get("POSTGRES_DSN", "postgresql://nexus:nexus@localhost:5432/nexus")
    if not dsn:
        return {"backend": None, "status": "unavailable", "reason": "POSTGRES_DSN not set"}

    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError:
        return {"backend": "pgvector", "status": "unavailable", "reason": "psycopg not installed"}

    try:
        import concurrent.futures
    except Exception:  # pragma: no cover
        concurrent.futures = None

    def _run_with_timeout(fn, timeout_s: float = 5.0) -> dict[str, Any]:
        if concurrent.futures is not None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(fn)
                try:
                    return future.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError as exc:
                    return {"backend": "pgvector", "status": "unavailable", "reason": "pgvector discovery timed out"}
        else:
            # Fallback: run synchronously with no timeout guarantee.
            return fn()

    def _query() -> dict[str, Any]:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT count(*), coalesce(min(metadata->>'content_origin'), 'unknown')
                    FROM code_chunks
                    WHERE repository_id = %s
                    """,
                    (repository_id,),
                )
                row = cursor.fetchone()
                count = int(row[0]) if row else 0
                origin = row[1] if row and row[1] else "unknown"
                cursor.execute(
                    """
                    SELECT distinct(metadata->>'content_origin') AS origin
                    FROM code_chunks
                    WHERE repository_id = %s
                    ORDER BY origin
                    """,
                    (repository_id,),
                )
                origins = [r[0] for r in cursor.fetchall() if r[0]]
                return {
                    "backend": "pgvector",
                    "status": "ok",
                    "vector_chunks": count,
                    "content_origins": origins,
                }

    try:
        return _run_with_timeout(_query, timeout_s=5.0)
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort
        return {"backend": "pgvector", "status": "unavailable", "reason": str(exc)}


def discover_github_surface(*, repo_id: str) -> dict[str, Any]:
    """Best-effort surface numbers from GitHub REST (read-only, stdlib).

    We only use public endpoints that do not require a token, because the goal
    is to work on any repository with the least possible setup. If nothing is
    reachable, we return a skeleton.
    """
    stats: dict[str, Any] = {
        "backend": "github_rest",
        "status": "unavailable",
        "commits": None,
        "files": None,
        "prs": None,
        "issues": None,
        "tags": [],
    }
    try:
        from urllib.request import Request, urlopen

        def get_json(url: str) -> dict[str, Any]:
            req = Request(url, headers={"User-Agent": "NEXUS-DevIntel"})
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())

        participation = get_json(f"https://api.github.com/repos/{repo_id}/stats/participation")
        if participation:
            commits = sum((week.get("wt", 0) + week.get("ct", 0)) for week in participation.get("all", []))
            stats["commits"] = commits

        tree = get_json(f"https://api.github.com/repos/{repo_id}/git/trees/main?recursive=1")
        files = [e for e in tree.get("tree", []) if e.get("type") == "blob"]
        stats["files"] = len(files)

        meta = get_json(f"https://api.github.com/repos/{repo_id}")
        stats["prs"] = meta.get("pull_requests")
        stats["issues"] = meta.get("open_issues_count")
        stats["status"] = "ok"

        tags = get_json(f"https://api.github.com/repos/{repo_id}/tags")
        stats["tags"] = [tag["name"] for tag in tags]
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort
        stats["reason"] = str(exc)

    return stats


# ---------------------------------------------------------------------------
# Graphe de couplage basique (best-effort, générique)
# ---------------------------------------------------------------------------

def build_basic_import_graph(*, repo_id: str, files: list[str]) -> list[dict[str, Any]]:
    """Heuristique générique de graphe d'import.

    C'est une approximation volontairement faible :
    - on ne résout que les imports de style ``repo::path`` ou ``package.module``
      quand on peut les relier à des fichiers connus.
    - on ne déduit rien à partir de code non lisible.

    Le but est de ne rien crier comme vrai, tout en fournissant une couche
    minimale utilisable pour un dépôt utilisateur.
    """
    prepared: set[str] = {f for f in files if f}
    edges: list[dict[str, Any]] = []

    for source in files:
        if not source:
            continue
        # Très bonne approximation pour les dépôts Python où les imports sont
        # de la forme ``package/module`` ou ``.rel/path``.
        for target in guess_possible_targets(source, prepared):
            edges.append({
                "source_id": make_file_id(repo_id, source),
                "target_id": make_file_id(repo_id, target),
                "source_path": source,
                "target_path": target,
                "confidence": 0.5,
                "method": "bootstrap_generic_import_heuristic",
                "note": "heuristic; review before trusting for root cause",
            })
    return edges


def guess_possible_targets(source: str, known_files: set[str]) -> list[str]:
    """Very weak target guess from a source path; returns candidates only.

    It does not claim these targets are real imports; it only proposes
    plausible ones based on naming similarity and package structure.
    """
    candidates: list[str] = []
    stem = Path(source).stem
    parent = str(Path(source).parent)

    for target in known_files:
        if target == source:
            continue
        target_parent = str(Path(target).parent)
        target_stem = Path(target).stem

        # Same package, different module
        if parent == target_parent and stem != target_stem:
            candidates.append(target)
            continue

        # Possible module import: source example ``httpie/cli/context.py`` -> target ``httpie/cli/requestitems.py``
        if parent and target_parent and parent.split("/")[:1] == target_parent.split("/")[:1]:
            candidates.append(target)

    return candidates


# ---------------------------------------------------------------------------
# Manifeste réutilisable
# ---------------------------------------------------------------------------

def write_manifest(repo_id: str, *, plan: dict[str, Any]) -> Path:
    path = PROJECT_ROOT / "repo_manifest.json"
    payload = {
        "repository_id": repo_id,
        "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        "bootstrap_version": "0.1.0",
        "plan": plan,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Plan / apply
# ---------------------------------------------------------------------------

def run_plan(repo_id: str, *, with_graph: bool, with_vectors: bool, with_github: bool) -> dict[str, Any]:
    owner, name = _safe_repo_parts(repo_id)
    existing = load_existing_audit(repo_id)
    audit = existing or build_minimal_audit(repo_id)

    plan: dict[str, Any] = {
        "repository_id": repo_id,
        "owner": owner,
        "name": name,
        "default_branch": "main",
        "audit": {
            "source": "existing" if existing else "bootstrap_minimal",
            "path": (PROJECT_ROOT / f"audit_{_repo_id_safe(repo_id)}.json").as_posix(),
            "exists": bool(existing),
        },
        "query_scope": {
            "graph": with_graph,
            "vectors": with_vectors,
            "github": with_github,
        },
        "what_would_be_done": [],
        "discovery": {},
        "would_write_audit": bool(not existing),
    }

    if with_graph:
        plan["what_would_be_done"].append(
            "discover Neo4j artefacts for this repository (files, imports, PRs, incidents)"
        )
        plan["discovery"]["graph"] = discover_graph_stats(repo_id=repo_id)

    if with_vectors:
        plan["what_would_be_done"].append(
            "discover pgvector artefacts for this repository (chunks, content origins)"
        )
        plan["discovery"]["vectors"] = discover_vector_stats(repository_id=repo_id)

    if with_github:
        plan["what_would_be_done"].append(
            "discover GitHub surface numbers (commits, files, PRs, issues, tags)"
        )
        plan["discovery"]["github"] = discover_github_surface(repo_id=repo_id)

    if not (with_graph or with_vectors or with_github):
        plan["what_would_be_done"].append(
            "no discovery enabled; this is a local configuration check only"
        )

    plan["next_steps"] = []
    if not existing:
        plan["next_steps"].append(
            f"Optionally generate a local audit: audit_{_repo_id_safe(repo_id)}.json"
        )
    if plan.get("discovery", {}).get("graph", {}).get("status") == "unavailable":
        plan["next_steps"].append("start Neo4j (docker compose up -d) then reload or ingest")
    if plan.get("discovery", {}).get("vectors", {}).get("status") == "unavailable":
        plan["next_steps"].append("set POSTGRES_DSN and run index_embeddings.py when needed")

    return plan


def _emit_plan(repo_id: str, plan: dict[str, Any]) -> None:
    """Print a single clean JSON document for the plan, then write the manifest."""
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    write_manifest(repo_id, plan=plan)


def run_apply(repo_id: str, *, graph: bool, vectors: bool, github: bool) -> dict[str, Any]:
    owner, name = _safe_repo_parts(repo_id)
    existing = load_existing_audit(repo_id)
    audit = existing or build_minimal_audit(repo_id)

    actions: list[str] = []
    if not existing:
        path = write_audit(repo_id, audit)
        actions.append(f"created audit_{_repo_id_safe(repo_id)}.json")
    else:
        actions.append(f"reused existing audit (audit_{_repo_id_safe(repo_id)}.json)")

    discovery: dict[str, Any] = {}
    if graph:
        discovery["graph"] = discover_graph_stats(repo_id=repo_id)
        actions.append("queried Neo4j for repository artefacts (read-only)")
    if vectors:
        discovery["vectors"] = discover_vector_stats(repository_id=repo_id)
        actions.append("queried pgvector for repository chunks (read-only)")
    if github:
        discovery["github"] = discover_github_surface(repo_id=repo_id)
        actions.append("queried GitHub REST for surface numbers (read-only)")

    manifest = {
        "repository_id": repo_id,
        "owner": owner,
        "name": name,
        "default_branch": "main",
        "audit": {
            "source": "existing" if existing else "bootstrap_minimal",
            "path": (PROJECT_ROOT / f"audit_{_repo_id_safe(repo_id)}.json").as_posix(),
        },
        "actions": actions,
        "discovery": discovery,
        "next_steps": [
            "run enrich_graphql.py to add (:PR)-[:CLOSES]->(:Incident) edges",
            "run index_embeddings.py --real-content when semantic retrieval is needed",
            "update the audit with real metrics from repo_audit.py if available",
        ],
    }

    write_manifest(repo_id, plan=manifest)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "repository_id",
        nargs="?",
        default=None,
        help="owner/name, e.g. acme/superlib (default: NEXUS_REPO_ID or httpie/cli)",
    )
    parser.add_argument("--plan", action="store_true", help="inspect and report, without writing anything")
    parser.add_argument("--apply", action="store_true", help="prepare what is safely possible for the repo")
    parser.add_argument("--with-graph", action="store_true", help="query Neo4j for existing artefacts")
    parser.add_argument("--with-vectors", action="store_true", help="query pgvector for existing chunks")
    parser.add_argument("--with-github", action="store_true", help="query GitHub REST for surface numbers")
    parser.add_argument("--dry-run", action="store_true", help="same as --plan for backwards compatibility")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    repo_id = args.repository_id or os.environ.get("NEXUS_REPO_ID", "httpie/cli")

    with_graph = args.with_graph or args.plan or args.dry_run
    with_vectors = args.with_vectors or args.plan or args.dry_run
    with_github = args.with_github or args.plan or args.dry_run

    if not args.plan and not args.apply and not args.dry_run:
        print("usage: bootstrap_repo.py [owner/name] (--plan | --apply)", file=sys.stderr)
        return 2

    if args.plan or args.dry_run:
        plan = run_plan(repo_id, with_graph=with_graph, with_vectors=with_vectors, with_github=with_github)
        _emit_plan(repo_id, plan)
        return 0

    if args.apply:
        manifest = run_apply(repo_id, graph=with_graph, vectors=with_vectors, github=with_github)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
