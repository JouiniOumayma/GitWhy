"""Helpers partagés entre l'API et le bootstrap réutilisable pour n'importe quel dépôt."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Conventions de dépôt
# ---------------------------------------------------------------------------

def _repo_id_safe(repo_id: str) -> str:
    """Nom de fichier d'audit dérivé du dépôt : ``owner/name`` -> ``owner_name``."""
    return repo_id.replace("/", "_").strip()


def _safe_repo_parts(repo_id: str) -> tuple[str, str]:
    """Découper ``owner/name`` en ``(owner, name)``."""
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid repository id: {repo_id!r} (expected owner/name)")
    return parts[0], parts[1]


def make_repository_id(owner: str, name: str) -> str:
    """``httpie`` + ``cli`` -> ``httpie/cli`` (garde la convention existante)."""
    return f"{owner}/{name}"


# ---------------------------------------------------------------------------
# Audit local par dépôt
# ---------------------------------------------------------------------------

def _audit_candidates(repo_id: str, base: str) -> list[str]:
    """Chemins possibles pour un audit par dépôt, dans l'ordre de préférence.

    On cherche d'abord ``audit_<owner_name>.json``, puis on garde les audits
    existants (exemples : ``audit_httpie.json``) comme dernier resort local.
    """
    safe = _repo_id_safe(repo_id)
    # Uniquement l'audit de CE dépôt : servir l'audit d'un autre dépôt
    # afficherait des chiffres réels mais attribués au mauvais projet.
    names = (f"audit_{safe}.json",)
    paths: list[str] = []
    for name in names:
        paths.append(os.path.join(base, "..", name))
    paths += [
        os.path.join(base, name) for name in names
    ]
    paths += [
        os.path.join(os.getcwd(), name) for name in names
    ]
    return paths


def load_repo_audit(repo_id: str, *, base: str | None = None) -> dict[str, Any] | None:
    """Charger un audit de surface par dépôt si présent.

    Un audit utilisateur est prioritaire : ``audit_<owner_name>.json``.
    Les audits ancrés existants ne sont utilisés que comme fallback local.
    """
    base = base or os.path.dirname(__file__) or "."
    for path in _audit_candidates(repo_id, base):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:  # noqa: BLE001
            logger.warning("échec du chargement de l'audit %s : %s", path, exc)
    return None


def write_repo_audit(repo_id: str, audit: dict[str, Any]) -> Path:
    """Écrire un audit par dépôt dans le projet (nonce ``audit_<owner_name>.json``)."""
    name = f"audit_{_repo_id_safe(repo_id)}.json"
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / name
    path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Audit minimal (sans inventer de données)
# ---------------------------------------------------------------------------

def build_minimal_audit(repo_id: str, *, source: str = "bootstrap") -> dict[str, Any]:
    """Squelette d'audit minimal pour un dépôt utilisateur.

    Il ne ment pas : tous les compteurs sont à zéro et la note explique ce
    qu'il reste à fournir (repo_audit.py, ingestion, enrichissement, embeddings).
    """
    owner, name = _safe_repo_parts(repo_id)
    return {
        "repo": repo_id,
        "owner": owner,
        "name": name,
        "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": source,
        "python_files": 0,
        "total_files": 0,
        "total_commits": 0,
        "total_pull_requests": 0,
        "total_issues": 0,
        "import_graph": {
            "internal_edges": 0,
            "files_importing": 0,
            "files_depended_upon": 0,
        },
        "tags": [],
        "known_incidents": [],
        "notes": [
            "no local audit found for this repository; "
            "run repo_audit.py or ingest real data"
        ],
    }


# ---------------------------------------------------------------------------
# Compteurs de surface pour l'API
# ---------------------------------------------------------------------------

def audit_to_stats(audit: dict[str, Any]) -> dict[str, Any]:
    """Projeter un audit local mesuré dans le format attendu par `/api/stats`.

    Aucune valeur n'est déduite d'une autre : un compteur absent de l'audit
    vaut zéro (mieux vaut un zéro honnête qu'une estimation crédible).
    """
    return {
        "graph_nodes": int(audit.get("graph_nodes", 0) or 0),
        "graph_relations": int(
            audit.get("import_graph", {}).get("internal_edges", 0) or 0
        ),
        "vector_chunks": int(audit.get("vector_chunks", 0) or 0),
        "files": int(audit.get("python_files", 0) or 0),
        "commits": int(audit.get("total_commits", 0) or 0),
        "prs": int(audit.get("total_pull_requests", 0) or 0),
        "incidents": int(audit.get("total_issues", 0) or 0),
    }


def github_stats_to_stats(data: dict[str, Any]) -> dict[str, Any]:
    """Projeter un payload GitHub REST : surface mesurée seulement.

    Le nombre de nœuds, de relations et de chunks n'est pas déductible de
    GitHub : ces compteurs restent à zéro tant que le dépôt n'est pas ingéré.
    """
    return {
        "graph_nodes": 0,
        "graph_relations": 0,
        "vector_chunks": 0,
        "files": int(data.get("file_count", 0) or 0),
        "commits": int(data.get("total", 0) or 0),
        "prs": 0,
        "incidents": 0,
    }


# ---------------------------------------------------------------------------
# Journalisation légère (évite un import lourd d'`logging`)
# ---------------------------------------------------------------------------

def _warning(msg: str, *args: Any) -> None:
    try:
        import importlib
        logging = importlib.import_module("logging")
    except Exception:
        return
    logging.getLogger("nexus.repo_helpers").warning(msg, *args)


def logger() -> Any:
    try:
        import importlib
        return importlib.import_module("logging").getLogger("nexus.repo_helpers")
    except Exception:
        class _OnlyWarning:
            def warning(self, msg: str, *args: Any) -> None:
                _warning(msg, *args)
        return _OnlyWarning()


# Rendre `_warning` utilisable comme la classe ci-dessus dans les cas où le
# logger dynamique n'est pas disponible au moment de l'import.
logger = logger()
