"""API FastAPI NEXUS-DevIntel — T7 : le seul point d'entrée HTTP.

Endpoints :

* ``POST /api/answer`` — corps ``{"question": "..."}`` (Pydantic strict) ;
  renvoie l'``Answer`` complète : texte, ``EvidencePath``, citations,
  score + bande, et le **flag d'escalade** ;
* ``GET /api/health`` — visibilité de la stack (graphe / LLM optionnels).

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
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from agents.llm import MistralClient
from agents.planner import route
from agents.tools import ToolRegistry
from models import AnswerIntent, AnswerStatus, Citation, EvidencePath

__all__ = ["AnswerResponse", "QuestionRequest", "create_app", "create_app_from_env"]

logger = logging.getLogger("nexus.api")


# --------------------------------------------------------------------------- #
# Contrats d'entrée / sortie (Pydantic strict, extra=forbid)
# --------------------------------------------------------------------------- #
class QuestionRequest(BaseModel):
    """Corps de ``POST /api/answer``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, max_length=1000)
    # repo_id optionnel : la démo vise httpie/cli par défaut.
    repo_id: str | None = None


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


# --------------------------------------------------------------------------- #
# Usine d'app : tout injectable, aucun état global
# --------------------------------------------------------------------------- #
def create_app(
    *,
    tools: ToolRegistry,
    llm: MistralClient | None,
    writer: Any | None = None,
    repo_id: str = "httpie/cli",
) -> FastAPI:
    """L'app montée sur une stack injectable (tests) **ou** réelle (T9).

    ``writer`` est le ``GraphWriter`` Phase 1 : ``None`` = lecture seule
    pure (pas de persistance, réponse inchangée).
    """
    app = FastAPI(title="NEXUS-DevIntel", version="0.1.0")
    app.state.tools = tools
    app.state.llm = llm

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

    @app.post("/api/answer", response_model=AnswerResponse)
    def answer(req: QuestionRequest) -> AnswerResponse:
        question = req.question.strip()
        repo = req.repo_id or repo_id

        trace: list[str] = []
        tools.set_call_callback(
            lambda name, result: trace.append(f"{name}:{'ok' if result.ok else 'fail'}")
        )
        start = time.monotonic()
        try:
            a, decision = route(question, tools, llm, repo_id=repo)
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
            _log_request(payload, decision, trace)
            return payload
        finally:
            tools.set_call_callback(None)  # jamais de callback orphelin

    return app


# --------------------------------------------------------------------------- #
# Persistance : ICI (et seulement ici) l'écriture Neo4j                          #
# --------------------------------------------------------------------------- #
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
# Montage réel (T9) : lu depuis l'environnement, jamais en tests              #
# --------------------------------------------------------------------------- #
def create_app_from_env() -> FastAPI:
    """Fabrique uvicorn : ``uvicorn api.app:create_app_from_env``.

    Lit NEO4J_*, POSTGRES_DSN, MISTRAL_API_KEY, NEXUS_* ; sans clé LLM ni
    stack, l'app démarre quand même (lecture seule + escalade).
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
        repository_id=os.environ.get("NEXUS_REPO_ID", "httpie/cli"),
    )
    writer = GraphWriter(driver, database)
    llm = MistralClient.from_env()
    return create_app(
        tools=tools,
        llm=llm,
        writer=writer,
        repo_id=os.environ.get("NEXUS_REPO_ID", "httpie/cli"),
    )
