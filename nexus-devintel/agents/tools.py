"""Outils whitelisted (lecture seule) — T2, le seul pont outils <-> agents.

Règle non négociable du brief : les agents ne peuvent appeler **que** des
outils whitelistés read-only. Ce module applique la règle en deux étages:

1. la whitelist est une ``frozenset`` figée à l'initialisation : un nom non
   déclaré (ou ressemblant à une écriture : ``merge_nodes``, ``delete_*``...)
   est refusé *avant tout dispatch*, pattern des tests d'attaque de la
   Phase 2 ;
2. chacun des outils n'enveloppe que des points d'entrée **read-only**
   existants (Phase 2) : les analyseurs (traversées Cypher SELECT-like),
   le retriever hybride (SQL SELECT-only) et le serveur MCP GitHub
   GET-only. Aucun ``GraphWriter`` n'est ici, donc aucun chemin d'écriture
   n'existe côté agent.

Un outil qui échoue (fichier absent, table indisponible, réseau) est
renvoyé comme :class:`ToolResult` avec ``ok=False`` : il ne leve pas vers
le graphe, il *documente* l'absence de preuve (le Critic escaladera).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from retrieval import ChangeImpactAnalyzer, HybridRetriever, RootCauseAnalyzer

__all__ = [
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolResult",
]


class ToolNotFoundError(LookupError):
    """Un nom d'outil hors whitelist : refuse avant tout dispatch."""


@dataclass(frozen=True)
class ToolResult:
    """Sortie normalisée d'un appel d'outil (succès OU échec documenté)."""

    name: str
    ok: bool
    value: Any = None
    error: str | None = None

    @property
    def evidence_available(self) -> bool:
        """Raccourci lisible du nœud Investigation."""
        return self.ok and self.value is not None


@dataclass(frozen=True)
class _ToolDef:
    """Une entrée de whitelist : nom, description, fonction read-only."""

    name: str
    description: str
    fn: Callable[..., Any]


class ToolRegistry:
    """Whitelist fermée d'outils read-only injectable dans les agents.

    Les dépendances sont optionnelles (``driver``/``dsn`` peuvent être
    ``None``, p.ex. en tests) : l'outil est alors *appelable* et renvoie
    simplement ``ok=False`` si la stack n'est pas disponible — ce qui
    prouve qu'aucun outil ne peut écrire par accident de configuration.
    """

    def __init__(
        self,
        *,
        driver: Any = None,
        database: str = "neo4j",
        dsn: str | None = None,
        repository_id: str = "httpie/cli",
        github_client: Any = None,
    ) -> None:
        self._repository_id = repository_id
        self._github_client = github_client
        if driver is not None:
            self._impact = ChangeImpactAnalyzer(driver, database)
            self._root_cause = RootCauseAnalyzer(driver, database)
        else:
            self._impact = None
            self._root_cause = None
        if dsn is not None:
            self._hybrid = HybridRetriever(dsn, repository_id=repository_id)
        else:
            self._hybrid = None
        # ---- la whitelist ------------------------------------------------- #
        self._tools: dict[str, _ToolDef] = {
            "find_file_id": _ToolDef(
                "find_file_id",
                "Resoudre un chemin de fichier en id de noeud File du graphe. Read-only.",
                self._find_file_id,
            ),
            "find_incident_id": _ToolDef(
                "find_incident_id",
                "Resoudre un numero d'issue en id de noeud Incident du graphe. Read-only.",
                self._find_incident_id,
            ),
            "analyze_impact": _ToolDef(
                "analyze_impact",
                "Change Impact d'un fichier: EvidencePath par traversée IMPORTS bornee. Read-only.",
                self._analyze_impact,
            ),
            "analyze_root_cause": _ToolDef(
                "analyze_root_cause",
                "Root cause d'un incident: EvidencePath Incident -> Deployment -> PR -> Commit. Read-only.",
                self._analyze_root_cause,
            ),
            "hybrid_search": _ToolDef(
                "hybrid_search",
                "Recherche hybride (pgvector + lexical) sur les chunks de code. Read-only.",
                self._hybrid_search,
            ),
            "get_file": _ToolDef(
                "get_file",
                "Contenu d'un fichier GitHub, verifie par blob SHA (MCP, GET only).",
                self._mcp_call("get_file"),
            ),
            "list_files": _ToolDef(
                "list_files",
                "Arborescence du depot GitHub (MCP, GET only).",
                self._mcp_call("list_files"),
            ),
            "get_commit": _ToolDef(
                "get_commit",
                "Un commit et ses fichiers modifies (MCP, GET only).",
                self._mcp_call("get_commit"),
            ),
            "get_pull_request": _ToolDef(
                "get_pull_request",
                "Une pull request et les issues qu'elle ferme (MCP, GET only).",
                self._mcp_call("get_pull_request"),
            ),
            "search_issues": _ToolDef(
                "search_issues",
                "Recherche issues/PRs d'un depot (MCP, GET only).",
                self._mcp_call("search_issues"),
            ),
        }
        self._names: frozenset[str] = frozenset(self._tools)
        #: observateur optionnel des appels (logs structurés T7) ; un hook
        #: ne peut JAMAIS modifier la whitelist ni le résultat.
        self._on_call: Callable[[str, ToolResult], None] | None = None

    # ------------------------------------------------------------------ #
    # Dispatch                                                            #
    # ------------------------------------------------------------------ #
    def call(self, name: str, **args: Any) -> ToolResult:
        """Appel whitelist. Un nom inconnu est refuse **avant** tout dispatch."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                f"outil '{name}' hors whitelist "
                f"(disponibles: {', '.join(sorted(self._names))})"
            )
        try:
            value = tool.fn(**args)
            result = ToolResult(name=name, ok=True, value=value)
        except Exception as exc:  # noqa: BLE001 - l'echec EST le resultat utile
            result = ToolResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}")
        self._trace(name, result)
        return result

    # ------------------------------------------------------------------ #
    # Observabilité (T7) : la collecte n'altère jamais whitelist/résultat #
    # ------------------------------------------------------------------ #
    def set_call_callback(self, callback: Callable[[str, ToolResult], None] | None) -> None:
        """Installe/retire un observateur d'appels (``None`` pour retirer)."""
        self._on_call = callback

    def _trace(self, name: str, result: ToolResult) -> None:
        """Un hook en erreur est signalé, jamais propagé à l'agent."""
        if self._on_call is None:
            return
        try:
            self._on_call(name, result)
        except Exception:  # noqa: BLE001
            pass

    @property
    def names(self) -> frozenset[str]:
        """La whitelist, figee."""
        return self._names

    # ------------------------------------------------------------------ #
    # Adaptateurs graphe (Phase 2 analyzers, aucune ecriture)              #
    # ------------------------------------------------------------------ #
    def _find_file_id(self, path_or_id: str) -> dict[str, Any]:
        if self._impact is None:
            raise RuntimeError("driver Neo4j non configure (stack live requise)")
        if path_or_id is None:
            return {"file_id": None, "error": "aucun candidat fichier"}
        candidate = str(path_or_id).strip()
        if not candidate:
            return {"file_id": None, "error": "aucun candidat fichier"}
        if "::" not in candidate:
            candidate = f"{self._repository_id}::{candidate}"
        if self._impact.file_exists(candidate):
            return {"file_id": candidate}
        return {"file_id": None, "error": f"'{candidate}' n'est pas dans le graphe"}

    def _find_incident_id(self, issue_number: int | str) -> dict[str, Any]:
        if self._root_cause is None:
            raise RuntimeError("driver Neo4j non configure (stack live requise)")
        if issue_number is None:
            return {"incident_id": None, "error": "aucun numéro d'incident"}
        try:
            num = int(issue_number)
        except (TypeError, ValueError):
            return {"incident_id": None, "error": f"numéro d'incident invalide : {issue_number!r}"}
        incident_id = f"{self._repository_id}#issue-{num}"
        if self._root_cause.incident_exists(incident_id):
            return {"incident_id": incident_id}
        return {"incident_id": None, "error": f"'{incident_id}' n'est pas dans le graphe"}

    def _analyze_impact(self, file_id: str, *, max_depth: int = 7, limit: int = 50) -> Any:
        if self._impact is None:
            raise RuntimeError("driver Neo4j non configure (stack live requise)")
        return self._impact.analyze(file_id, max_depth=max_depth, limit=limit)

    def _analyze_root_cause(self, incident_id: str, *, max_depth: int = 6, limit: int = 200) -> Any:
        if self._root_cause is None:
            raise RuntimeError("driver Neo4j non configure (stack live requise)")
        return self._root_cause.analyze(incident_id, max_depth=max_depth, limit=limit)

    def _hybrid_search(self, query: str, *, match_count: int = 5) -> Any:
        if self._hybrid is None:
            raise RuntimeError("DSN PostgreSQL non configure (stack live requise)")
        return self._hybrid.retrieve(query, match_count=match_count)

    # ------------------------------------------------------------------ #
    # Adaptateur MCP GitHub (registre lui-meme deja clos, GET-only)        #
    # ------------------------------------------------------------------ #
    def _mcp_call(self, name: str) -> Callable[..., Any]:
        def call(**args: Any) -> Any:
            from ingestion.mcp_server import call_tool as mcp_call_tool

            return mcp_call_tool(name, args, client=self._github_client)

        return call
