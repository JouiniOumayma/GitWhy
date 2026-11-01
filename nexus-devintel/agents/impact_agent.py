"""Agent LangGraph « Change Impact » — T4.

Question type : *« que casse si je change ``httpie/cli::httpie/context.py`` ? »*

Graphe (StateGraph langgraph)::

        START -> investigate -> draft -> critic
        critic ACCEPT -> finalize -> END
        critic RETRY -> draft (borne par max_retries)
        critic ESCALATE -> finalize

Règles du brief appliquées ici :

* **preuves par les outils uniquement** : le nœud Investigation appelle la
  whitelist read-only (``agents.tools``) qui enveloppe
  ``ChangeImpactAnalyzer`` — le LLM n'invente **aucun** hop ;
* **pas de preuve -> pas de réponse affirmative** : aucun fichier trouvé
  dans le graphe, ou rayon d'impact vide => ``Answer.status =
  INSUFFICIENT_EVIDENCE`` + escalade ;
* **Critic** : vérifie en JSON structuré que la rédaction est appuyée par
  l'``EvidencePath`` ; ``retry`` (nouvelle rédaction) est borné par
  ``AgentBudget.max_retries``, aucun re-planning ;
* **budget** : chaque appel LLM réel est encadré par ``can_call_llm`` /
  ``llm_call_patch`` — un mode déterministe (sans LLM ou budget épuisé)
  ne consomme **pas** de budget, il ne fait que calquer l'``Answer`` sur
  les hops du graphe, ce qui reste une preuve valide.Dégradation propre (démo sans clé API) : sans ``llm``, la rédaction et la
critique retombent en mode déterministe — l'``Answer`` reste alors adossée
aux hops réels du graphe, jamais inventée par un LLM.
"""

from __future__ import annotations

import re
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.state import (
    AgentBudget,
    AgentState,
    CriticVerdict,
    can_call_llm,
    can_retry,
    initial_state,
    llm_call_patch,
    retry_patch,
)
from agents.tools import ToolRegistry
from models import (
    Answer,
    AnswerIntent,
    AnswerStatus,
    Citation,
    EvidencePath,
    utcnow,
)

__all__ = ["build_impact_agent", "run_impact_agent", "ImpactAgent", "extract_file_candidate"]

#: Nombre max de hops cités tels quels dans le prompt (les plus gros scores).
MAX_HOPS_IN_PROMPT = 15

#: Chemins plausibles d'un repo Python (démo httpie/cli : ``.py`` uniquement).
_PATH_RE = re.compile(r"\b(?:[\w\-./]+::)?[\w\-./]+\.py\b")


def extract_file_candidate(question: str) -> str | None:
    """Le candidat fichier dans la question (regex pure, 0 LLM)."""
    match = _PATH_RE.search(question or "")
    if not match:
        return None
    return match.group(0).strip(".,?;:()")


class ImpactAgent:
    """Les quatre nœuds, montés sur la whitelist d'outils + un LLM optionnel.

    ``llm=None`` (ou clé API absente) => rédaction et Critic déterministes :
    l'``Answer`` reste adossée aux hops réels du graphe, jamais inventée.

    Sous-classé par ``RootCauseAgent`` (T5) qui ne change que l'intention,
    les prompts et le nœud Investigation ; les nœuds Rédaction, Critic et
    Finalisation sont partagés tels quels.
    """

    #: intention de la ``Answer`` produite (CHANGE_IMPACT ici).
    INTENT: AnswerIntent = AnswerIntent.CHANGE_IMPACT
    #: prompt d'extraction d'entité (json_mode, ``{<clé>: valeur|null}``).
    EXTRACTION_SYSTEM = (
        "Tu extrais une entité d'une question Python. "
        "Réponds STRICTEMENT en JSON: {\"file_path\": \"chemin/relatif.py\"} "
        "si la question mentionne un fichier, sinon {\"file_path\": null}. "
        "N'invente aucun chemin absent de la question."
    )
    DRAFT_SYSTEM = (
        "Tu es l'agent NEXUS-DevIntel, analyste d'impact logiciel. "
        "Tu rédiges une réponse courte (5-8 phrases, en français) UNIQUEMENT à "
        "partir des preuves listées dans la question utilisateur. "
        "Interdictions: inventer un fichier, un commit ou une prédiction non "
        "portée par un hop de l'EvidencePath; citer une ligne qui n'y figure pas. "
        "Si la preuve est faible, dis-le franchement."
    )
    CRITIC_SYSTEM = (
        "Tu es le Critic de NEXUS-DevIntel. On te donne une réponse candidate et "
        "les hops d'un EvidencePath réel issu du graphe Neo4j. "
        "Vérifie une seule chose: CHAQUE affirmation de la réponse est appuyée "
        "par au moins un hop (node_id, relation, role). "
        "Réponds STRICTEMENT en JSON: "
        "{\"supported\": true|false, \"reason\": \"<une phrase>\"}."
    )

    def __init__(self, tools: ToolRegistry, llm: Any = None) -> None:
        self._tools = tools
        self._llm = llm

    # ------------------------------------------------------------------ #
    # Noeud 1 : Investigation (outils read-only, 0..1 appel LLM)          #
    # ------------------------------------------------------------------ #
    def investigate(self, state: AgentState) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        candidate = extract_file_candidate(state["question"])

        if candidate is None and self._llm is not None and can_call_llm(state):
            patch = llm_call_patch(state)
            try:
                out = self._llm.chat(self.EXTRACTION_SYSTEM, state["question"], json_mode=True)
                value = (out.data or {}).get("file_path")
                candidate = value.strip() if isinstance(value, str) and value.strip() else None
            except Exception:
                patch["critic_notes"] = ["extraction LLM indisponible : chemin non identifié"]

        resolved = self._tools.call("find_file_id", path_or_id=candidate)
        report = self._tools.call("analyze_impact", file_id=(resolved.value or {}).get("file_id")) \
            if resolved.ok and (resolved.value or {}).get("file_id") else None

        if report is None or not report.ok:
            patch["critic_notes"] = patch.get("critic_notes", []) + [
                f"{resolved.name} -> {resolved.error or 'aucun fichier dans le graphe'}"
            ]
            patch["critic_verdict"] = CriticVerdict.ESCALATE
            patch["evidence_path"] = None
            patch["evidences"] = []
            return patch

        value = report.value  # ImpactReport
        patch["evidence_path"] = value.path
        patch["evidences"] = value.evidences
        patch["critic_notes"] = patch.get("critic_notes", []) + [
            f"impacts directs={len(value.direct_dependents)} "
            f"transitifs={value.transitive_dependents}"
        ]
        if not value.path.hops[1:]:
            # le fichier existe mais personne n'en dépend : preuve vide => escalade
            patch["critic_verdict"] = CriticVerdict.ESCALATE
        return patch

    # ------------------------------------------------------------------ #
    # Noeud 2 : Rédaction (1 appel LLM, fallback déterministe 0-budget)   #
    # ------------------------------------------------------------------ #
    def draft(self, state: AgentState) -> dict[str, Any]:
        path = state.get("evidence_path")
        if path is None or not path.hops[1:]:
            return {"draft_answer": ""}
        if self._llm is not None and can_call_llm(state):
            patch = llm_call_patch(state)
            user = (
                f"Question: {state['question']}\n\n"
                f"EvidencePath (score global {path.score}):\n{self._hops_prompt(path)}"
            )
            patch["draft_answer"] = self._llm.chat(self.DRAFT_SYSTEM, user).content.strip()
            return patch
        return {"draft_answer": self._fallback_draft(path)}

    # ------------------------------------------------------------------ #
    # Noeud 3 : Critic (1 appel LLM, fallback déterministe 0-budget)      #
    # ------------------------------------------------------------------ #
    def critic(self, state: AgentState) -> dict[str, Any]:
        path = state.get("evidence_path")
        if path is None or not path.hops[1:]:
            return {
                "critic_verdict": CriticVerdict.ESCALATE,
                "critic_notes": state["critic_notes"] + ["aucune preuve : escalade"],
            }

        if self._llm is not None and can_call_llm(state):
            patch = llm_call_patch(state)
            user = (
                f"Réponse candidate:\n{state['draft_answer']}\n\n"
                f"Preuves disponibles:\n{self._hops_prompt(path)}"
            )
            try:
                data = self._llm.chat(self.CRITIC_SYSTEM, user, json_mode=True).data or {}
                supported = bool(data.get("supported"))
                reason = str(data.get("reason", "sans motif"))
            except Exception:
                # LML en panne : le Critic ne peut se rabattre que sur la
                # validité FORMELLE de la preuve déjà construite par le graphe
                supported = path.is_valid
                reason = "Critic LLM indisponible : vérification formelle de la preuve"
        else:
            patch = {}
            supported = path.is_valid
            reason = "mode déterministe : la réponse est calquée sur les hops du graphe"

        if supported:
            return {**patch, "critic_verdict": CriticVerdict.ACCEPT,
                    "critic_notes": state["critic_notes"] + [f"critic: {reason}"]}

        note = f"critic : réponse non appuyée ({reason})"
        if can_retry(state):
            return {**patch, **retry_patch(state), "critic_verdict": CriticVerdict.RETRY,
                    "critic_notes": state["critic_notes"] + [note]}
        return {**patch, "critic_verdict": CriticVerdict.ESCALATE,
                "critic_notes": state["critic_notes"] + [note + " (retries épuisés)"]}

    # ------------------------------------------------------------------ #
    # Noeud 4 : Finalisation (objet Answer validé par Pydantic)           #
    # ------------------------------------------------------------------ #
    def finalize(self, state: AgentState) -> dict[str, Any]:
        path = state.get("evidence_path")
        notes = list(state["critic_notes"])
        accepted = state["critic_verdict"] == CriticVerdict.ACCEPT and bool(state["draft_answer"])
        hops = list(path.hops[1:]) if path else []

        answer = Answer(
            id=f"ans-{utcnow().strftime('%Y%m%d-%H%M%S-%f')}",
            question=state["question"],
            intent=self.INTENT,
            answer_text=state["draft_answer"] if accepted else "",
            status=AnswerStatus.OK if accepted else AnswerStatus.INSUFFICIENT_EVIDENCE,
            repository_id=state["repo_id"],
            confidence_score=path.score if path else 0.0,
            confidence_breakdown={"graph_path": path.score if path else 0.0},
            evidence_path=path,
            evidence_ids=[e.id for e in state["evidences"]],
            citations=self._citations(hops),
            suggested_actions=(
                []
                if accepted
                else ["reformuler la question avec un chemin de fichier du dépôt"]
            ),
            model=getattr(self._llm, "_model", None),
            created_at=utcnow(),
        )
        if not accepted:
            notes.append("pas de preuve exploitable -> pas de réponse affirmative")
            answer.answer_text = self._insufficient_text(notes)
        return {"answer": answer}

    # ------------------------------------------------------------------ #
    # Utilitaires                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hops_prompt(path: EvidencePath) -> str:
        lines = []
        for hop in path.hops[1:MAX_HOPS_IN_PROMPT + 1]:
            lines.append(
                f"- step {hop.step} node={hop.node_kind.value}:{hop.node_id} "
                f"rel={hop.relation_in} role={hop.role.value} score={hop.score} "
                f"why={hop.rationale}"
            )
        return "\n".join(lines) or "(aucun hop)"

    @staticmethod
    def _fallback_draft(path: EvidencePath) -> str:
        """Texte déterministe, 100 % dérivé des hops — rien d'inventé."""
        dependents = [h.node_id for h in path.hops[1:]]
        direct = [n for n in dependents if n]  # ordonnés par distance de traversée
        return (
            f"Changement analysé sur le graphe NEXUS : "
            f"{len(direct)} fichier(s) dépendant(s) trouvé(s), "
            f"score de confiance {path.score:.2f}.\n"
            "L'EvidencePath ci-dessous liste les fichiers impactés, "
            "ordonnés par distance d'IMPORTS et pondérés par le score de chaque hop."
        )

    @staticmethod
    def _citations(hops: list, max_items: int = 5) -> list:
        return [
            Citation(label=hop.node_id, node_id=hop.node_id, node_kind=hop.node_kind)
            for hop in hops[:max_items]
        ]

    @staticmethod
    def _insufficient_text(notes: list[str]) -> str:
        base = (
            "Je ne peux pas répondre avec une preuve suffisante : "
            "le graphe ne contient aucun chemin qui soutient cette question."
        )
        tail = "".join(f"\n- {note}" for note in notes[-3:])
        return f"{base}{tail}"


# --------------------------------------------------------------------------- #
# Assemblage du graphe LangGraph                                               #
# --------------------------------------------------------------------------- #
def build_agent_graph(agent: ImpactAgent) -> StateGraph:
    """Graphe compilé : START -> investigate -> draft -> critic -> finalize.

    Partagé par les deux agents (T4 et T5) : seule l'instance change.
    """
    graph = StateGraph(AgentState)
    graph.add_node("investigate", agent.investigate)
    graph.add_node("draft", agent.draft)
    graph.add_node("critic", agent.critic)
    graph.add_node("finalize", agent.finalize)

    graph.add_edge(START, "investigate")
    graph.add_edge("investigate", "draft")
    graph.add_edge("draft", "critic")

    def _after_critic(state: AgentState) -> str:
        if state["critic_verdict"] == CriticVerdict.RETRY:
            return "draft"
        return "finalize"

    graph.add_conditional_edges("critic", _after_critic, ["draft", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()


def build_impact_agent(tools: ToolRegistry, llm: Any = None) -> StateGraph:
    """Graphe Change Impact compilé (T4)."""
    return build_agent_graph(ImpactAgent(tools, llm))


def run_impact_agent(
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
        route=AnswerIntent.CHANGE_IMPACT,
        budget=budget or AgentBudget(),
    )
    return build_impact_agent(tools, llm).invoke(state)["answer"]
