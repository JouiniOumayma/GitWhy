"""État partagé des agents LangGraph — T1 de la Phase 3.

Un état LangGraph est un ``TypedDict`` : aucune dépendance à ``langgraph``
n'est nécessaire ici. Ce module définit

* :class:`AgentBudget`  — les limites imposées à chaque exécution (appels LLM,
  retries, timeout). Garde-fou du brief : "quotas/budgets sur les appels LLM".
* :class:`CriticVerdict` — la décision du Critic (``accept`` / ``retry`` /
  ``escalate``), rejouable telle quelle dans les logs structurés.
* :class:`AgentState`   — l'état des deux agents, alimenté par le Planner
  (``route``), par les outils (``evidence_path``, ``evidences``) et finalisé
  en :class:`~models.answer.Answer`.

Les helpers sont des fonctions **pures** : elles reçoivent l'état et
retournent un *patch* (dict partiel) que le nœud LangGraph fusionne. Rien
n'est muté in place, ce qui rend les agents triviaux à tester sans LLM.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from models import Answer, AnswerIntent, Evidence, EvidencePath

__all__ = [
    "AgentBudget",
    "AgentState",
    "CriticVerdict",
    "can_call_llm",
    "can_retry",
    "initial_state",
    "llm_call_patch",
    "llm_budget_left",
    "retry_patch",
]


class AgentBudget(BaseModel):
    """Limites d'exécution d'un agent. Le Critic doit rester dans ces bornes."""

    model_config = ConfigDict(extra="forbid")

    #: Appels LLM max par question (rédaction + critiques + repli du planner).
    max_llm_calls: int = Field(default=6, ge=1, le=50)
    #: Retours du Critic acceptés avant escalade (brief : max 1–2).
    max_retries: int = Field(default=2, ge=0, le=5)
    #: Timeout par appel LLM (imposé par le wrapper Mistral de ``agents.llm``).
    llm_timeout_s: int = Field(default=45, ge=1, le=300)


class CriticVerdict(StrEnum):
    """Décision du nœud Critics, tracée telle quelle dans les logs."""

    ACCEPT = "accept"
    RETRY = "retry"
    ESCALATE = "escalate"


class AgentState(TypedDict):
    """État des agents Change Impact et Root Cause (schéma LangGraph)."""

    # entrée (remplie par le Planner / l'API)
    question: str
    repo_id: str
    route: AnswerIntent
    budget: AgentBudget
    # compteurs (immuables : un nœud ne fait que retourner un patch)
    llm_calls: int
    retries: int
    # sorties des outils (nœud Investigation)
    evidence_path: EvidencePath | None
    evidences: list[Evidence]
    # rédaction + Critic
    draft_answer: str
    critic_verdict: CriticVerdict
    critic_notes: list[str]
    # objet final validé par Pydantic (règle : pas de preuve -> PAS d'ok)
    answer: Answer | None


# --------------------------------------------------------------------------- #
# Construction / budget : fonctions pures, testables sans LangGraph ni LLM
# --------------------------------------------------------------------------- #

_STATE_DEFAULTS: dict[str, Any] = {
    "llm_calls": 0,
    "retries": 0,
    "evidence_path": None,
    "evidences": [],
    "draft_answer": "",
    "critic_verdict": None,
    "critic_notes": [],
    "answer": None,
}


def initial_state(
    question: str,
    repo_id: str,
    *,
    route: AnswerIntent,
    budget: AgentBudget | None = None,
) -> AgentState:
    """État initial prêt à être passé au premier nœud."""
    return AgentState(
        question=question,
        repo_id=repo_id,
        route=route,
        budget=budget or AgentBudget(),
        **_STATE_DEFAULTS,
    )


def llm_budget_left(state: AgentState) -> int:
    """Appels LLM restants avant épuisement du budget."""
    left = state["budget"].max_llm_calls - state["llm_calls"]
    return max(0, left)


def can_call_llm(state: AgentState) -> bool:
    """False => le nœud doit renoncer et laisser le Critic escalader."""
    return llm_budget_left(state) > 0


def llm_call_patch(state: AgentState) -> dict[str, Any]:
    """Patch à fusionner après un appel LLM effectif."""
    if not can_call_llm(state):
        raise RuntimeError("budget LLM épuisé : un appel de plus écrirait en dehors des bornes")
    return {"llm_calls": state["llm_calls"] + 1}


def can_retry(state: AgentState) -> bool:
    """Le Critic peut-il demander le maximum de retours demandés ?"""
    return state["retries"] < state["budget"].max_retries


def retry_patch(state: AgentState) -> dict[str, Any]:
    """Patch à fusionner quand le Critic demande un nouveau tour."""
    if not can_retry(state):
        raise RuntimeError("retries épuisés")
    return {"retries": state["retries"] + 1}
