"""Planner — T6 : routage simple entre les deux agents (pas un vrai agent).

Conforme au brief : « simple fonction de routage (pas un vrai agent) ».
Trois niveaux, sans re-planning :

1. **règles pures** (0 LLM) : mots-clés + signaux structurés — un chemin
   ``fichier.py`` penche Change Impact, un ``#numéro`` d'issue penche
   Root Cause ; trois sous-cas read-only dans ``agents.tools`` ;
2. **1 seul appel LLM en repli** (``json_mode``) si la question est ambiguë
   (ex : ``pourquoi httpie/context.py ?`` -> les deux camps à égalité) ;
3. **route par défaut CHANGE_IMPACT** si le repli échoue (pas de clé API,
   LLM en panne, JSON incohérent) : l'agent produira alors une ``Answer``
   ``INSUFFICIENT_EVIDENCE`` — la règle du brief « pas de preuve -> pas de
   réponse affirmative » reste opérante.

Deux invariants : le budget est honnête (les appels consommés par le
repli sont **soustraits** du budget de l'agent, cf.
``_budget_after_planning``, jamais gratuits), et tout passe par la
whitelist read-only ``agents.tools`` — le planner n'appelle jamais ni
Neo4j, ni PostgreSQL, ni GitHub directement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agents.impact_agent import build_impact_agent, extract_file_candidate
from agents.root_cause_agent import build_root_cause_agent
from agents.state import AgentBudget, initial_state
from agents.tools import ToolRegistry
from models import Answer, AnswerIntent

__all__ = [
    "RouteDecision",
    "classify",
    "classify_with_fallback",
    "route",
]

_ROOT_HINTS = (
    "pourquoi", "cause", "racine", "root cause", "postmortem",
    "incident", "panne", "outage", "régression", "regression",
    "symptôme", "symptom", "provoqué", "déclenché", "cassé par",
)
_IMPACT_HINTS = (
    "casse", "casser", "impact", "dépend", "depend", "dépendance",
    "dépendances", "refactor", "supprimer", "remplacer", "toucher",
    "modifier", "blast radius", "quels fichiers", "utilise",
)

_ISSUE_RE = re.compile(r"#\d+")


@dataclass(frozen=True)
class RouteDecision:
    """Décision de routage, à tracer telle quelle dans les logs (T7)."""

    question: str
    intent: AnswerIntent
    #: "keyword" | "llm" | "default" (repli impossible, escalade attendue)
    method: str
    #: appels LLM déjà dépensés pour classer (réduits du budget agent)
    llm_calls: int


def classify(question: str) -> AnswerIntent | None:
    """Règles pures : ``ChangeImpact``/``RootCause``, ou ``None`` si ambigu.

    Un tie (un signal de chaque camp, ex: ``pourquoi httpie/context.py ?``)
    est déclaré ambigu : c'est le rôle de l'agent LLM de départager.
    """
    q = " ".join((question or "").lower().split())
    root = sum(h in q for h in _ROOT_HINTS) + (1 if _ISSUE_RE.search(q) else 0)
    impact = sum(h in q for h in _IMPACT_HINTS) \
        + (1 if extract_file_candidate(question or "") else 0)
    if root > impact:
        return AnswerIntent.ROOT_CAUSE
    if impact > root:
        return AnswerIntent.CHANGE_IMPACT
    return None


_LLM_SYSTEM = (
    "Tu classes une question d'analyse logicielle en deux intentions. "
    "Réponds STRICTEMENT en JSON: "
    "{\"intent\": \"change_impact\"} (que casse ce changement ?) ou "
    "{\"intent\": \"root_cause\"} (pourquoi cet incident ?). "
    "Si la question ne relève clairement d'aucune des deux, "
    "réponds {\"intent\": null}. N'invente pas une intention."
)

_VALID_LLM_INTENTS: dict[str, AnswerIntent] = {
    "change_impact": AnswerIntent.CHANGE_IMPACT,
    "root_cause": AnswerIntent.ROOT_CAUSE,
}


def classify_with_fallback(
    question: str, llm: Any, *, budget: AgentBudget | None = None
) -> tuple[AnswerIntent, str, int]:
    """(intention, méthode, appels LLM consommés) — au plus 1 appel de repli.

    ``method`` : ``"keyword"`` (règles suffisantes), ``"llm"`` (repli), ou
    ``"default"`` (pas de clb / LLM en panne / réponse incohérente).
    """
    intent = classify(question)
    if intent is not None:
        return intent, "keyword", 0
    if llm is None or (budget is not None and budget.max_llm_calls < 1):
        return AnswerIntent.CHANGE_IMPACT, "default", 0
    try:
        data = llm.chat(_LLM_SYSTEM, question, json_mode=True).data or {}
        value = data.get("intent")
        if isinstance(value, str) and value in _VALID_LLM_INTENTS:
            return _VALID_LLM_INTENTS[value], "llm", 1
    except Exception:
        pass  # repli déterministe, documenté dans la méthode
    return AnswerIntent.CHANGE_IMPACT, "default", 1


def _budget_after_planning(budget: AgentBudget | None, used: int) -> AgentBudget:
    """Le budget soldé des appels JSON déjà consommés par le repli."""
    base = budget or AgentBudget()
    return AgentBudget(
        max_llm_calls=max(1, base.max_llm_calls - used),
        max_retries=base.max_retries,
        llm_timeout_s=base.llm_timeout_s,
    )


def route(
    question: str,
    tools: ToolRegistry,
    llm: Any,
    *,
    repo_id: str = "httpie/cli",
    budget: AgentBudget | None = None,
) -> tuple[Answer, RouteDecision]:
    """Une question -> (``Answer`` complète, ``RouteDecision`` tracée)."""
    intent, method, used = classify_with_fallback(question, llm, budget=budget)
    agent = (
        build_impact_agent(tools, llm)
        if intent == AnswerIntent.CHANGE_IMPACT
        else build_root_cause_agent(tools, llm)
    )
    state = initial_state(
        question, repo_id,
        route=intent,
        budget=_budget_after_planning(budget, used),
    )
    answer = agent.invoke(state)["answer"]
    return answer, RouteDecision(
        question=question, intent=intent, method=method, llm_calls=used,
    )
