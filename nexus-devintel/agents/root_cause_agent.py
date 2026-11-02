"""Agent LangGraph « Root Cause » — T5.

Question type : *« pourquoi l'incident httpie/cli#issue-1583 ? »*

IDENTIQUE à l'agent Change Impact (T4) sauf le nœud Investigation : les
nœuds Rédaction, Critic, Finalisation et l'assemblage du graphe sont
**hérités** tels quels de ``ImpactAgent`` / ``build_agent_graph`` (règle
du brief : ne pas dupliquer ce qui marche).

Specifique Root Cause (nœud Investigation) :

1. extraction du numéro d'issue : regex pure ``#1583`` / ``issue 1583``,
   LLM ``json_mode`` (``{"incident_number": ...}``) en repli ;
2. ``find_incident_id`` -> ``analyze_root_cause`` via la whitelist read-only
   (la chaîne Incident -> Deployment -> PR -> Commit -> File, enrichie
   GraphQL ``closingIssuesReferences`` à l'ingestion, Phase 2) ;
3. ``hybrid_search`` (pgvector + lexical) ajoute les chunks de code du
   correctif comme preuves symptomatiques ; un échec (stack absente, sim
   MiniLM indisponible) est documenté en note, jamais bloquant : la preuve
   de chaîne vient du graphe.

Pas de preuve (issue inconnue du graphe, hops vides) => escalade, comme en
T4 : ``Answer.status = INSUFFICIENT_EVIDENCE``, jamais de réponse
affirmative extrapolée.
"""

from __future__ import annotations

import re
from typing import Any

from agents.impact_agent import ImpactAgent, build_agent_graph
from agents.state import (
    AgentBudget,
    AgentState,
    CriticVerdict,
    can_call_llm,
    initial_state,
    llm_call_patch,
)
from agents.tools import ToolRegistry
from models import Answer, AnswerIntent, utcnow

__all__ = [
    "RootCauseAgent",
    "build_root_cause_agent",
    "extract_issue_candidate",
    "run_root_cause_agent",
]

#: ``#1583``, ``issue 1583``, ``bug 1583``...
_ISSUE_RE = re.compile(r"(?:#|issue|incident|bug)[ \-]?(\d{1,7})", re.IGNORECASE)

_MAX_HYBRID_MATCHES = 5


def extract_issue_candidate(question: str) -> int | None:
    """Le numéro d'issue (regex pure, 0 LLM)."""
    match = _ISSUE_RE.search(question or "")
    if not match:
        return None
    return int(match.group(1))


class RootCauseAgent(ImpactAgent):
    """Sous-classe de T4 : seule l'intention, les prompts et le nœud Investigation changent."""

    INTENT = AnswerIntent.ROOT_CAUSE

    EXTRACTION_SYSTEM = (
        "Tu extrais une entité d'une question d'incident logiciel. "
        "Réponds STRICTEMENT en JSON: {\"incident_number\": <entier>} "
        "si la question référence un numéro d'issue GitHub, "
        "sinon {\"incident_number\": null}. "
        "N'invente aucun numéro absent de la question."
    )

    DRAFT_SYSTEM = (
        "Tu es l'agent NEXUS-DevIntel, analyste root cause. "
        "Tu rédiges une réponse courte (5-8 phrases, en français) qui explique "
        "l'incident: symptôme observé, cause racine, correctifs (PR/commits), "
        "déploiements touchés. Tu utilises UNIQUEMENT plutôt les preuves "
        "listées dans la question utilisateur. "
        "Interdictions: inventer un commit, une PR ou un déploiement non portés "
        "par un hop de l'EvidencePath; attribuer une cause sans arête graphe. "
        "Si la preuve est faible, dis-le franchement."
    )

    # ------------------------------------------------------------------ #
    # Noeud 1 : Investigation (issue -> chaîne graphique -> chunks hybrides) #
    # ------------------------------------------------------------------ #
    def investigate(self, state: AgentState) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        candidate = extract_issue_candidate(state["question"])

        if candidate is None and self._llm is not None and can_call_llm(state):
            patch = llm_call_patch(state)
            try:
                out = self._llm.chat(self.EXTRACTION_SYSTEM, state["question"], json_mode=True)
                value = (out.data or {}).get("incident_number")
                candidate = int(value) if isinstance(value, int) else None
            except Exception:
                patch["critic_notes"] = ["extraction LLM indisponible : numéro d'issue non identifié"]

        resolved = self._tools.call("find_incident_id", issue_number=candidate)
        incident_id = (resolved.value or {}).get("incident_id")
        report = self._tools.call("analyze_root_cause", incident_id=incident_id) \
            if incident_id else None

        if report is None or not report.ok:
            patch["critic_notes"] = patch.get("critic_notes", []) + [
                f"{resolved.name} -> {resolved.error or 'aucun incident dans le graphe'}"
            ]
            patch["critic_verdict"] = CriticVerdict.ESCALATE
            patch["evidence_path"] = None
            patch["evidences"] = []
            return patch

        value = report.value  # RootCauseReport
        patch["evidence_path"] = value.path
        patch["evidences"] = list(value.evidences)
        patch["critic_notes"] = patch.get("critic_notes", []) + [
            f"fix commits={len(value.fix_commits)} PRs={len(value.fixing_prs)} "
            f"déploiements={len(value.deployments)} fichiers={len(value.affected_files)}"
        ]

        # + symptômes en chunks réels (pgvector), facultatif : jamais bloquant
        hybrid = self._tools.call("hybrid_search", query=state["question"],
                                  match_count=_MAX_HYBRID_MATCHES)
        if hybrid.ok:
            extra = list(hybrid.value.evidences)
            patch["evidences"] = patch["evidences"] + extra
            patch["critic_notes"] = patch.get("critic_notes", []) + [
                f"chunks hybrides ajoutés={len(extra)}"
            ]
        else:
            patch["critic_notes"] = patch.get("critic_notes", []) + [
                "recherche hybride indisponible : preuve de chaîne seule"
            ]

        if not value.path.hops[1:]:
            # l'incident existe mais la chaîne de cause est vide => escalade
            patch["critic_verdict"] = CriticVerdict.ESCALATE
        return patch


# --------------------------------------------------------------------------- #
# Assemblage (hérité de T4) et confort d'exécution                              #
# --------------------------------------------------------------------------- #
def build_root_cause_agent(tools: ToolRegistry, llm: Any = None) -> Any:
    """Graphe Root Cause compilé : même topologie que T4."""
    return build_agent_graph(RootCauseAgent(tools, llm))


def run_root_cause_agent(
    tools: ToolRegistry,
    llm: Any,
    question: str,
    *,
    repo_id: str = "httpie/cli",
    budget: AgentBudget | None = None,
) -> Answer:
    """Confort : état initial + invocation + l'``Answer`` finale."""
    state = initial_state(
        question, repo_id,
        route=AnswerIntent.ROOT_CAUSE,
        budget=budget or AgentBudget(),
    )
    answer = build_root_cause_agent(tools, llm).invoke(state)["answer"]
    return answer
