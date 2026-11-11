"""API FastAPI NEXUS-DevIntel — T7 : le seul point d'entrée HTTP.

Endpoints :

* ``POST /api/answer`` — corps ``{"question": "..."}`` (Pydantic strict) ;
  renvoie l'``Answer`` complète : texte, ``EvidencePath``, citations,
  score + bande, et le **flag d'escalade** ;
* ``GET /api/health`` — visibilité de la stack (graphe / LLM optionnels).
* ``GET /api/stats`` — retourne les compteurs d'artefacts pour l'en-tête UI.

Invariants du brief appliqués ici et nulle part ailleurs :

* **l'écriture vit ici** : la ``Answer`` et ses ``Evidence`` sont
  persistées via ``GraphWriter`` (Phase 1, MERGE idempotent) *après*
  production — les agents ne reçoivent jamais d'outil d'écriture, le test
  d'attaque de la Phase 2 continue donc de passer ;
* **logs structurés jsonl** (règle du brief) : question, route choisie,
  outils appelés, score, décision du Critic — une ligne json par requête.

Tout est injectable pour les tests (``tools`` / ``llm`` / ``writer``) :
aucun test ne touche la vraie base ni la vraie clé API.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from agents.llm import GeminiClient, MistralClient
from agents.planner import route
from agents.tools import ToolRegistry
from models import AnswerIntent, AnswerStatus, Citation, EvidencePath

from nexus.repo_helpers import (
    _repo_id_safe,
    _safe_repo_parts,
    load_repo_audit,
    audit_to_stats,
    github_stats_to_stats,
)

__all__ = ["AnswerResponse", "QuestionRequest", "create_app", "create_app_from_env"]

logger = logging.getLogger("nexus.api")


# --------------------------------------------------------------------------- #
# Contrats d'entrée / sortie (Pydantic strict, extra=forbid)
# --------------------------------------------------------------------------- #

class IngestRequest(BaseModel):
    """Corps de ``POST /api/ingest`` : un lien GitHub ou ``owner/name``."""

    model_config = ConfigDict(extra="forbid")

    repository: str = Field(min_length=1)
    skip_vectors: bool = False
    #: L'historique git coûte une requête GitHub par commit : la fenêtre est
    #: bornée et explicite plutôt que devinée.
    skip_history: bool = False
    max_commits: int = Field(default=150, ge=0, le=1000)
    max_prs: int = Field(default=100, ge=0, le=500)


class QuestionRequest(BaseModel):
    """Corps de ``POST /api/answer``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, max_length=1000)
    # repo_id optionnel : le dépôt par défaut vient de NEXUS_REPO_ID / create_app.
    repo_id: str | None = None


class SuggestedRepository(BaseModel):
    """Dépôt ingéré qui contient le fichier cité par la question."""

    repo_id: str
    paths: list[str] = Field(default_factory=list)


class RepoHint(BaseModel):
    """Diagnostic d'une escalade : le fichier demandé vit dans un autre dépôt."""

    missing: list[str] = Field(default_factory=list)
    repositories: list[SuggestedRepository] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    """Ligne de réponse unique (le contrat de l'UI, item 5 du brief)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    intent: AnswerIntent
    answer_text: str
    status: AnswerStatus
    #: indicateur d'escalade : toute réponse non OK (pas de preuve, retries
    #: épuisés, route hors périmètre) — l'UI l'affiche.
    escalated: bool
    confidence_score: float
    confidence_band: str
    evidence_path: EvidencePath | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    model: str | None = None
    latency_ms: int = 0
    #: True quand la Answer + ses Evidence ont été écrites dans Neo4j.
    persisted: bool = False
    #: Renseigné quand la question cite un fichier absent du dépôt visé mais
    #: présent dans un autre dépôt ingéré : évite une escalade incompréhensible.
    repo_hint: RepoHint | None = None


# --------------------------------------------------------------------------- #
# Usine d'app : tout injectable, aucun état global
# --------------------------------------------------------------------------- #

def create_app(
    *,
    tools: ToolRegistry,
    llm: GeminiClient | MistralClient | None,
    writer: Any | None = None,
    repo_id: str = "httpie/cli",
) -> FastAPI:
    """L'app montée sur une stack injectable (tests) **ou** réelle (T9).

    ``writer`` est le ``GraphWriter`` Phase 1 : ``None`` = lecture seule
    pure (pas de persistance, réponse inchangée).

    ``repo_id`` est le dépôt utilisateur par défaut ; il peut être
    outrepassé par le paramètre ``repo_id`` du corps `POST /api/answer`.
    """
    app = FastAPI(title="NEXUS-DevIntel", version="0.1.0")
    app.state.tools = tools
    app.state.llm = llm
    #: jobs d'ingestion en cours/terminés (en mémoire : un seul process).
    app.state.ingest_jobs: dict[str, dict[str, Any]] = {}
    app.state.ingest_active: dict[str, str] = {}
    app.state.ingest_lock = threading.Lock()
    #: dépôt configuré, capturé ici : la route ``stats`` reçoit un paramètre
    #: nommé ``repo_id`` qui masquerait le paramètre de fermeture.
    default_repo_id = repo_id

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Ping sans écriture : ce qui est branché, ce qui ne l'est pas."""
        return {
            "status": "ok",
            "graph": getattr(tools, "_impact", None) is not None,
            "hybrid": getattr(tools, "_hybrid", None) is not None,
            "llm": llm is not None,
            "persistence": writer is not None,
        }

    @app.get("/api/stats")
    def stats(repo_id: str | None = None) -> dict[str, Any]:
        """Compteurs d'artefacts mesurés (Neo4j + pgvector), jamais estimés.

        ``?repo_id=owner/name`` cible un autre dépôt que celui configuré :
        l'UI s'en sert pour afficher les compteurs du dépôt interrogé.
        """
        target = (
            _normalize_repo(repo_id) or (repo_id or "").strip() or default_repo_id
        )
        tools_state = getattr(app.state, "tools", None)
        return {
            "status": "ok",
            "repository": target,
            **_fetch_repo_stats(tools_state, target),
        }

    @app.post("/api/ingest", status_code=202)
    def ingest(req: IngestRequest) -> dict[str, Any]:
        """Lance l'ingestion d'un dépôt donné par un utilisateur final.

        Le travail est long (réseau + AST + embeddings) : il part dans un
        thread et la réponse immédiate contient un ``job_id`` à suivre sur
        ``GET /api/ingest/{job_id}``. Re-soumettre le même dépôt pendant qu'il
        tourne renvoie le job existant au lieu d'en démarrer un second.
        """
        from nexus.ingest_service import repo_id_from_input

        try:
            repository = repo_id_from_input(req.repository)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        jobs = app.state.ingest_jobs
        with app.state.ingest_lock:
            running_id = app.state.ingest_active.get(repository)
            if running_id is not None:
                running = jobs.get(running_id) or {}
                if running.get("status") in {"queued", "running"}:
                    return {
                        "job_id": running_id,
                        "repository": repository,
                        "status": running.get("status"),
                        "deduplicated": True,
                    }
            job_id = uuid.uuid4().hex[:12]
            jobs[job_id] = {
                "id": job_id,
                "repository": repository,
                "skip_vectors": bool(req.skip_vectors),
                "skip_history": bool(req.skip_history),
                "max_commits": int(req.max_commits),
                "max_prs": int(req.max_prs),
                "status": "queued",
                "steps": [],
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "finished_at": None,
                "error": None,
                "result": None,
            }
            app.state.ingest_active[repository] = job_id

        threading.Thread(
            target=_ingest_worker,
            args=(
                app,
                job_id,
                repository,
                bool(req.skip_vectors),
                bool(req.skip_history),
                int(req.max_commits),
                int(req.max_prs),
            ),
            name=f"ingest-{repository}",
            daemon=True,
        ).start()
        return {"job_id": job_id, "repository": repository, "status": "queued"}

    @app.get("/api/ingest/{job_id}")
    def ingest_status(job_id: str) -> dict[str, Any]:
        """État d'un job d'ingestion : statut, étapes, compteurs relus."""
        job = app.state.ingest_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job inconnu : {job_id}")
        return job

    @app.get("/api/graph")
    def graph(
        repo_id: str | None = None,
        node_id: str | None = None,
        limit: int = 40,
    ) -> dict[str, Any]:
        """Voisinage réel d'un nœud : importeurs, dépendances, commits, PRs.

        Sert la visualisation de l'UI. Tout vient du graphe (aucun exemple
        fabriqué) : un fichier absent renvoie 404 plutôt qu'un dessin vide.
        """
        repo = _normalize_repo(repo_id) or (repo_id or "").strip() or default_repo_id
        tools_state = getattr(app.state, "tools", None)
        driver = getattr(tools_state, "_driver", None)
        if driver is None:
            raise HTTPException(status_code=503, detail="graphe Neo4j indisponible")

        focus = (node_id or "").strip()
        if not focus:
            raise HTTPException(status_code=400, detail="node_id requis")
        if "::" not in focus:
            focus = f"{repo}::{focus.lstrip('/')}"
        elif not focus.startswith(f"{repo}::"):
            raise HTTPException(
                status_code=400, detail=f"nœud hors du dépôt {repo} : {focus}"
            )

        bounded = max(1, min(int(limit), 120))
        try:
            payload = _graph_neighbourhood(
                driver,
                getattr(tools_state, "_database", None) or "neo4j",
                focus,
                bounded,
            )
        except Exception as exc:  # noqa: BLE001 - Neo4j down = 503, pas 500
            raise HTTPException(
                status_code=503, detail=f"lecture du graphe impossible : {exc}"
            ) from exc
        if payload is None:
            raise HTTPException(
                status_code=404, detail=f"fichier absent du graphe : {focus}"
            )
        payload["repository"] = repo
        return payload

    @app.get("/api/top-files")
    def top_files(repo_id: str | None = None, limit: int = 6) -> dict[str, Any]:
        """Fichiers les plus structurants d'un dépôt (par nombre de dépendants).

        Sert à proposer des questions d'exemple pertinentes pour le dépôt
        choisi. Tout vient du graphe ; si Neo4j est indisponible on renvoie
        une liste vide plutôt qu'une erreur (non bloquant pour l'UI).
        """
        repo = _normalize_repo(repo_id) or (repo_id or "").strip() or default_repo_id
        tools_state = getattr(app.state, "tools", None)
        driver = getattr(tools_state, "_driver", None)
        bounded = max(1, min(int(limit), 20))
        if driver is None:
            return {"repository": repo, "files": []}
        database = getattr(tools_state, "_database", None) or "neo4j"
        try:
            files = _top_files(driver, database, repo, bounded)
        except Exception as exc:  # noqa: BLE001
            logger.warning("top-files impossible pour %s : %s", repo, exc)
            files = []
        return {"repository": repo, "files": files}

    @app.post("/api/answer", response_model=AnswerResponse)
    def answer(req: QuestionRequest) -> AnswerResponse:
        question = req.question.strip()
        # Un lien collé depuis l'UI (`https://github.com/pallets/flask`) doit
        # viser le même dépôt que la forme canonique (`pallets/flask`) sous
        # laquelle le graphe a été écrit : sinon aucun fichier n'est trouvé et
        # la réponse escalade à tort.
        repo = _normalize_repo(req.repo_id) or req.repo_id or repo_id

        trace: list[str] = []
        repo_tools = _tools_for_repo(app, tools, repo)
        repo_tools.set_call_callback(
            lambda name, result: trace.append(f"{name}:{'ok' if result.ok else 'fail'}")
        )
        start = time.monotonic()
        try:
            a, decision = route(question, repo_tools, llm, repo_id=repo)
            latency_ms = int((time.monotonic() - start) * 1000)
            persisted = _persist(writer, a, decision)
            payload = AnswerResponse(
                id=a.id,
                question=question,
                intent=a.intent,
                answer_text=a.answer_text,
                status=a.status,
                escalated=a.status != AnswerStatus.OK,
                confidence_score=a.confidence_score,
                confidence_band=a.confidence_band.value,
                evidence_path=a.evidence_path,
                evidence_ids=list(a.evidence_ids),
                citations=list(a.citations),
                model=a.model,
                latency_ms=latency_ms,
                persisted=persisted,
            )
            if payload.escalated and not payload.evidence_ids:
                tools_state = getattr(app.state, "tools", None)
                payload.repo_hint = _repo_hint(
                    getattr(tools_state, "_driver", None),
                    getattr(tools_state, "_database", None) or "neo4j",
                    repo,
                    question,
                )
            _log_request(payload, decision, trace)
            return payload
        finally:
            repo_tools.set_call_callback(None)  # jamais de callback orphelin

    return app


# --------------------------------------------------------------------------- #
# Persistance : ICI (et seulement ici) l'écriture Neo4j
# --------------------------------------------------------------------------- #

def _ingest_worker(
    app: FastAPI,
    job_id: str,
    repository: str,
    skip_vectors: bool,
    skip_history: bool,
    max_commits: int,
    max_prs: int,
) -> None:
    """Exécute l'ingestion hors boucle HTTP et tient le job à jour."""
    from nexus.ingest_service import run_ingestion

    jobs = app.state.ingest_jobs
    job = jobs[job_id]
    job["status"] = "running"

    def on_step(name: str, status: str, detail: str) -> None:
        with app.state.ingest_lock:
            for recorded in job["steps"]:
                if recorded["name"] == name:
                    recorded["status"] = status
                    recorded["detail"] = detail
                    return
            job["steps"].append({"name": name, "status": status, "detail": detail})

    try:
        job["result"] = run_ingestion(
            repository,
            skip_vectors=skip_vectors,
            skip_history=skip_history,
            max_commits=max_commits,
            max_prs=max_prs,
            on_step=on_step,
        )
        job["status"] = "completed"
        logger.info(json.dumps({
            "event": "ingest_completed", "repository": repository, "job_id": job_id,
        }))
    except Exception as exc:  # noqa: BLE001 - l'échec est l'état du job
        job["status"] = "failed"
        job["error"] = str(exc)
        logger.warning(json.dumps({
            "event": "ingest_failed", "repository": repository,
            "job_id": job_id, "error": str(exc),
        }, ensure_ascii=False))
    finally:
        job["finished_at"] = datetime.now(tz=timezone.utc).isoformat()
        with app.state.ingest_lock:
            app.state.ingest_active.pop(repository, None)


def _graph_neighbourhood(
    driver: Any,
    database: str,
    focus_id: str,
    limit: int,
) -> dict[str, Any] | None:
    """Sous-graphe autour d'un fichier, lu dans Neo4j (``None`` s'il n'existe pas)."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    counts = {"importers": 0, "dependencies": 0, "commits": 0, "prs": 0}

    with driver.session(database=database) as session:
        focus = session.run(
            """
            MATCH (f:File {id: $id})
            RETURN f.id AS id, f.path AS path, f.language AS language
            """,
            id=focus_id,
        ).single()
        if focus is None:
            return None
        focus_path = focus["path"] or focus_id.split("::", 1)[-1]
        nodes.append({
            "id": focus_id,
            "label": focus_path,
            "kind": "focus",
            "path": focus_path,
            "meta": focus["language"] or "",
        })

        # Qui importe ce fichier : c'est le sens de l'impact.
        for row in session.run(
            """
            MATCH (s:File)-[r:IMPORTS]->(:File {id: $id})
            RETURN s.id AS id, s.path AS path, r.confidence AS confidence
            ORDER BY s.path LIMIT $limit
            """,
            id=focus_id,
            limit=limit,
        ).data():
            counts["importers"] += 1
            nodes.append({
                "id": row["id"], "label": row["path"], "kind": "importer",
                "path": row["path"], "meta": "dépend de ce fichier",
            })
            edges.append({
                "source": row["id"], "target": focus_id, "type": "IMPORTS",
                "confidence": row["confidence"],
            })

        # Ce que ce fichier importe.
        for row in session.run(
            """
            MATCH (:File {id: $id})-[r:IMPORTS]->(t:File)
            RETURN t.id AS id, t.path AS path, r.confidence AS confidence
            ORDER BY t.path LIMIT $limit
            """,
            id=focus_id,
            limit=limit,
        ).data():
            counts["dependencies"] += 1
            nodes.append({
                "id": row["id"], "label": row["path"], "kind": "dependency",
                "path": row["path"], "meta": "importé par ce fichier",
            })
            edges.append({
                "source": focus_id, "target": row["id"], "type": "IMPORTS",
                "confidence": row["confidence"],
            })

        # Historique : commits qui ont touché le fichier (et les PRs qui les ont mergés).
        for row in session.run(
            """
            MATCH (c:Commit)-[r:MODIFIES]->(:File {id: $id})
            RETURN c.id AS id, c.subject AS subject, toString(c.committed_at) AS at
            ORDER BY c.committed_at DESC LIMIT 8
            """,
            id=focus_id,
        ).data():
            counts["commits"] += 1
            nodes.append({
                "id": row["id"],
                "label": (row["subject"] or row["id"])[:64],
                "kind": "commit",
                "path": None,
                "meta": (row["at"] or "")[:10],
            })
            edges.append({
                "source": row["id"], "target": focus_id, "type": "MODIFIES",
                "confidence": None,
            })

        for row in session.run(
            """
            MATCH (p:PR)-[:MERGED_INTO]->(c:Commit)-[:MODIFIES]->(:File {id: $id})
            RETURN DISTINCT p.id AS id, p.number AS number, p.title AS title,
                   c.id AS commit_id
            LIMIT 5
            """,
            id=focus_id,
        ).data():
            counts["prs"] += 1
            nodes.append({
                "id": row["id"],
                "label": f"PR #{row['number']}",
                "kind": "pr",
                "path": None,
                "meta": (row["title"] or "")[:64],
            })
            edges.append({
                "source": row["id"], "target": row["commit_id"],
                "type": "MERGED_INTO", "confidence": None,
            })

    return {
        "focus": {"id": focus_id, "path": focus_path},
        "nodes": nodes,
        "edges": edges,
        "counts": counts,
    }


#: Un chemin de fichier cité dans une question libre (``src/click/core.py``).
_PATH_IN_QUESTION = re.compile(
    r"[A-Za-z0-9_][A-Za-z0-9_./-]*\."
    r"(?:py|pyi|js|jsx|ts|tsx|go|rb|java|kt|rs|php|cs|c|h|cpp|hpp|md|rst|"
    r"toml|ya?ml|json|sh|cfg|ini|html|css|sql)"
)


def _mentioned_paths(question: str) -> list[str]:
    """Chemins plausibles cités dans la question, sans doublon ni bruit."""
    paths: list[str] = []
    for match in _PATH_IN_QUESTION.finditer(question or ""):
        path = match.group(0).lstrip("./")
        # Un simple « app.py » sans dossier reste ambigu : on l'ignore, sauf
        # s'il contient déjà un dossier.
        if "/" not in path:
            continue
        if path not in paths:
            paths.append(path)
    return paths


def _repo_hint(
    driver: Any,
    database: str,
    repo: str | None,
    question: str,
) -> RepoHint | None:
    """Explique une escalade due à un fichier d'un **autre** dépôt ingéré.

    Sans ce diagnostic, l'UI affiche « aucun fichier dans le graphe » alors que
    le fichier existe — simplement ailleurs. ``None`` dès que le cas ne
    s'applique pas ; une panne Neo4j ne doit jamais casser la réponse.
    """
    if driver is None or not repo:
        return None
    paths = _mentioned_paths(question)
    if not paths:
        return None
    try:
        with driver.session(database=database) as session:
            found = {
                record["p"]
                for record in session.run(
                    """
                    MATCH (f:File)
                    WHERE f.repository_id = $repo AND f.path IN $paths
                    RETURN f.path AS p
                    """,
                    repo=repo,
                    paths=paths,
                )
            }
            missing = [p for p in paths if p not in found]
            if not missing:
                return None
            rows = session.run(
                """
                MATCH (f:File)
                WHERE f.path IN $paths AND f.repository_id <> $repo
                RETURN f.repository_id AS repo_id, collect(DISTINCT f.path)[0..3] AS paths
                ORDER BY repo_id
                """,
                paths=missing,
                repo=repo,
            ).data()
    except Exception as exc:  # noqa: BLE001 - un diagnostic ne casse rien
        logger.warning(json.dumps({"event": "repo_hint_failed", "error": str(exc)}))
        return None
    if not rows:
        return None
    return RepoHint(
        missing=missing,
        repositories=[
            SuggestedRepository(repo_id=row["repo_id"], paths=list(row["paths"]))
            for row in rows
        ],
    )


def _normalize_repo(value: str | None) -> str | None:
    """``https://github.com/psf/requests`` → ``psf/requests`` (None si illisible)."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        from nexus.ingest_service import repo_id_from_input

        return repo_id_from_input(text)
    except Exception:  # noqa: BLE001 - une forme inconnue n'est pas une erreur
        return None


def _tools_for_repo(state: Any, tools: Any, repo_id: str) -> Any:
    """Registre d'outils du dépôt demandé (un par dépôt, mis en cache).

    ``ToolRegistry`` préfixe les identifiants avec son ``repository_id``
    (``psf/requests::src/requests/models.py``). Sans registre dédié, un
    ``repo_id`` fourni dans la requête serait ignoré : les fichiers du second
    dépôt seraient cherchés sous le préfixe du premier et la réponse
    escaladerait à tort.
    """
    declared = getattr(tools, "_repository_id", None)
    # ``None`` : registre injecté (tests, stub) qui ne scope rien — on le
    # laisse tel quel plutôt que d'en fabriquer un vide.
    if declared is None or declared == repo_id:
        return tools
    cache = getattr(state, "tools_by_repo", None)
    if cache is None:
        cache = {}
        state.tools_by_repo = cache
    scoped = cache.get(repo_id)
    if scoped is None:
        scoped = ToolRegistry(
            driver=getattr(tools, "_driver", None),
            database=getattr(tools, "_database", "neo4j"),
            dsn=getattr(tools, "_dsn", None),
            repository_id=repo_id,
            github_client=getattr(tools, "_github_client", None),
        )
        cache[repo_id] = scoped
    return scoped


def _persist(writer: Any, answer: Any, decision: Any) -> bool:
    """MERGE idempotent des ``Evidence`` puis de l'``Answer``.

    Un échec de persistance n'annule JAMAIS la réponse (dégradation
    idempotente ; journalisée, jamais levée vers le client).
    """
    if writer is None:
        return False
    try:
        writer.write_models([*decision.evidences, answer])
        return True
    except Exception as exc:  # noqa: BLE001 - la réponse prime sur la base
        logger.warning(json.dumps({"event": "persist_failed", "error": str(exc)}))
        return False


def _log_request(payload: AnswerResponse, decision: Any, trace: list[str]) -> None:
    """Une ligne jsonl par requête (question, route, outils, score, critic)."""
    logger.info(json.dumps({
        "event": "answer_request",
        "question": payload.question,
        "route": payload.intent.value,
        "route_method": decision.method,
        "planner_llm_calls": decision.llm_calls,
        "critic": decision.critic_verdict,
        "tools": trace,
        "status": payload.status.value,
        "escalated": payload.escalated,
        "score": payload.confidence_score,
        "latency_ms": payload.latency_ms,
        "persisted": payload.persisted,
    }, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Montage réel (T9) : lu depuis l'environnement, jamais en tests
# --------------------------------------------------------------------------- #


_repo_id_safe = _repo_id_safe  # delegated to nexus.repo_helpers


_load_repo_audit = load_repo_audit  # delegated to nexus.repo_helpers


_STATS_ZERO: dict[str, Any] = {
    "graph_nodes": 0,
    "graph_relations": 0,
    "vector_chunks": 0,
    "files": 0,
    "commits": 0,
    "prs": 0,
    "incidents": 0,
}

#: Compteurs du sous-graphe d'un dépôt : 4 sauts suffisent (Repository ->
#: File/Commit/Deployment -> PR -> Incident) et bornent le coût de traversée.
_REPO_SUBGRAPH_CYPHER = """
MATCH (r:Repository {owner: $owner, name: $name})
CALL {
    WITH r OPTIONAL MATCH (r)-[*1..4]-(x)
    RETURN count(DISTINCT x) AS graph_nodes
}
CALL {
    WITH r OPTIONAL MATCH (r)-[*1..4]-(x)
    WITH DISTINCT x MATCH (x)-[rel]-()
    RETURN count(DISTINCT rel) AS graph_relations
}
CALL {
    WITH r OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
    RETURN count(f) AS files
}
CALL {
    WITH r OPTIONAL MATCH (r)-[:HAS_COMMIT]->(c:Commit)
    RETURN count(c) AS commits
}
CALL {
    // Compté par propriété, pas par traversée : une PR mergée hors de la
    // fenêtre de commits ingérés reste une PR du dépôt, et une traversée
    // bornée la manquerait (constaté : 44 au lieu de 100).
    WITH r OPTIONAL MATCH (p:PR {repository_id: $owner + '/' + $name})
    RETURN count(p) AS prs
}
CALL {
    WITH r OPTIONAL MATCH (r)-[*1..4]-(i:Incident)
    RETURN count(DISTINCT i) AS incidents
}
RETURN graph_nodes, graph_relations, files, commits, prs, incidents
"""


def _live_stats(tools_state: Any, repo_id: str) -> dict[str, Any] | None:
    """Compteurs réels lus dans Neo4j et pgvector pour ``repo_id``.

    Renvoie ``None`` quand la stack est absente ou que le dépôt n'existe pas
    encore dans le graphe : l'appelant retombe alors sur un audit local ou
    sur zéro, jamais sur une estimation.
    """
    driver = getattr(tools_state, "_driver", None)
    if driver is None:
        return None
    try:
        owner, name = _safe_repo_parts(repo_id)
    except ValueError:
        return None

    database = getattr(tools_state, "_database", None) or "neo4j"
    try:
        with driver.session(database=database) as session:
            record = session.run(
                _REPO_SUBGRAPH_CYPHER, owner=owner, name=name
            ).single()
    except Exception as exc:  # noqa: BLE001 - Neo4j down = repli, pas 500
        logger.warning(json.dumps({
            "event": "stats_neo4j_failed",
            "repo": repo_id,
            "error": str(exc),
        }, ensure_ascii=False))
        return None
    if record is None:
        return None

    chunks = _chunk_count(getattr(tools_state, "_dsn", None), repo_id)
    return {
        "graph_nodes": int(record["graph_nodes"] or 0),
        "graph_relations": int(record["graph_relations"] or 0),
        "vector_chunks": chunks or 0,
        "files": int(record["files"] or 0),
        "commits": int(record["commits"] or 0),
        "prs": int(record["prs"] or 0),
        "incidents": int(record["incidents"] or 0),
        "source": "neo4j+postgres" if chunks is not None else "neo4j",
    }


def _chunk_count(dsn: str | None, repo_id: str) -> int | None:
    """Nombre réel de chunks vectorisés pour ``repo_id`` (``None`` si KO)."""
    if not dsn:
        return None
    try:
        import psycopg
    except ImportError:
        logger.warning("psycopg absent : vector_chunks non mesurable")
        return None
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM code_chunks WHERE repository_id = %s",
                    (repo_id,),
                )
                row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:  # noqa: BLE001 - Postgres down = repli, pas 500
        logger.warning(json.dumps({
            "event": "stats_postgres_failed",
            "repo": repo_id,
            "error": str(exc),
        }, ensure_ascii=False))
        return None


def _top_files(driver: Any, database: str, repo_id: str, limit: int) -> list[dict[str, Any]]:
    """Fichiers d'un dépôt classés par nombre de dépendants (incoming IMPORTS)."""
    rows = []
    with driver.session(database=database) as session:
        for row in session.run(
            """
            MATCH (f:File {repository_id: $repo})
            OPTIONAL MATCH (g:File {repository_id: $repo})-[:IMPORTS]->(f)
            WITH f, count(g) AS dependents
            ORDER BY dependents DESC, f.path
            LIMIT $limit
            RETURN f.path AS path, dependents AS dependents
            """,
            repo=repo_id,
            limit=limit,
        ).data():
            rows.append({"path": row["path"], "dependents": int(row["dependents"] or 0)})
    return rows


def _fetch_repo_stats(tools_state: Any, repo_id: str) -> dict[str, Any]:
    """Compteurs d'artefacts d'un dépôt quelconque, mesurés et non estimés.

    Ordre de préférence :
    1. la stack vivante (Neo4j + pgvector) — source de vérité ;
    2. un audit local ``audit_<owner_name>.json`` mesuré par repo_audit.py ;
    3. la surface GitHub (fichiers/commits réels, graphe à zéro) ;
    4. zéro explicite : un dépôt non ingéré n'affiche pas de chiffres
       plausibles mais faux.
    """
    live = _live_stats(tools_state, repo_id)
    if live is not None:
        return live

    audit = load_repo_audit(repo_id)
    if audit is not None:
        return {**audit_to_stats(audit), "source": f"audit:{_repo_id_safe(repo_id)}"}

    surface = _fetch_github_stats(tools_state, repo_id)
    if surface is not None:
        return {**surface, "source": "github"}

    return {**_STATS_ZERO, "source": "none"}


def _fetch_github_stats(tools_state: Any, repo_id: str) -> dict[str, Any] | None:
    """Surface GitHub d'un dépôt non ingéré (fichiers/commits réels).

    Aucun compteur de graphe ni de vecteur n'est déduit d'ici : sans
    ingestion, ces chiffres valent zéro. ``None`` si l'API ne répond pas.
    """
    try:
        url = f"https://api.github.com/repos/{repo_id}/stats/participation"
        req = urllib.request.Request(url, headers={"User-Agent": "NEXUS-DevIntel"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        contents_url = f"https://api.github.com/repos/{repo_id}/git/trees/master?recursive=1"
        req2 = urllib.request.Request(
            contents_url,
            headers={"User-Agent": "NEXUS-DevIntel"},
        )
        with urllib.request.urlopen(req2, timeout=10) as response2:
            tree_data = json.loads(response2.read().decode())
            file_count = len(tree_data.get("tree", []))

        return github_stats_to_stats({
            "total": (data or {}).get("total", 0),
            "file_count": file_count,
        })
    except Exception as exc:
        logger.warning("échec du fallback GitHub stats pour %s : %s", repo_id, exc)
        return None

def create_app_from_env() -> FastAPI:
    """Fabrique uvicorn : ``uvicorn api.app:create_app_from_env``.

    Lit NEO4J_*, POSTGRES_DSN, GEMINI_API_KEY (prioritaire), MISTRAL_API_KEY,
    _NEXUS_REPO_ID_ (dépôt utilisateur) ; sans clé LLM ni stack, l'app
    démarre quand même (lecture seule + escalade).

    Le dépôt utilisateur est paramétré par ``NEXUS_REPO_ID`` (exemple :
    ``acme/superlib``). Quand il n'est pas positionné, le système démarre
    avec le dépôt démonstration ``httpie/cli``.

    Priorité LLM : Gemini > Mistral > None (mode dégradé).
    """
    # Import localisé : le driver Neo4j et l'outil d'écriture ne sont
    # importés que par l'API réelle, jamais par les agents (testés).
    from ingestion import GraphWriter
    from neo4j import GraphDatabase

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "user")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    tools = ToolRegistry(
        driver=driver,
        database=database,
        dsn=os.environ.get("POSTGRES_DSN"),
        repository_id=os.environ.get("NEXUS_REPO_ID") or "httpie/cli",
    )
    writer = GraphWriter(driver, database)
    # Gemini prioritaire, Mistral en repli, None = mode dégradé
    llm: GeminiClient | MistralClient | None = (
        GeminiClient.from_env() or MistralClient.from_env()
    )
    if llm is not None:
        logger.info("LLM backend : %s (%s)", type(llm).__name__, getattr(llm, "_model", "?"))
    else:
        logger.warning("Aucune clé LLM (GEMINI_API_KEY / MISTRAL_API_KEY) — mode dégradé.")
    return create_app(
        tools=tools,
        llm=llm,
        writer=writer,
        repo_id=os.environ.get("NEXUS_REPO_ID") or "httpie/cli",
    )