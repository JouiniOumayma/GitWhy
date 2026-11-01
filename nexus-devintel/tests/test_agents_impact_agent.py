"""Tests de ``agents/impact_agent.py`` — T4 : graphe complet, zéro réseau.

Les critiques LLM sont mockés (FakeLlm), les outils sont des stubs canards
(FakeTools). Toute la mécanique de preuve reste celle du graphe: on
s'assure qu'aucune réponse OK n'existe sans hops réels.
"""

import json
from dataclasses import dataclass, field

import pytest

from agents.impact_agent import (
    ImpactAgent,
    build_impact_agent,
    extract_file_candidate,
    run_impact_agent,
)
from agents.llm import LlmOutput
from agents.state import AgentBudget, AgentState, CriticVerdict, initial_state
from models import (
    AnswerIntent,
    AnswerStatus,
    ConfidenceBand,
    Evidence,
    EvidenceHop,
    EvidenceKind,
    EvidencePath,
    EvidenceRef,
    EvidenceRole,
    NodeKind,
    Provenance,
    RetrievalStrategy,
    utcnow,
)


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
@dataclass
class FakeTools:
    """Canard-typpé minimal de ToolRegistry (la seule interface utilisée)."""

    file_id: str | None = "httpie/cli::httpie/context.py"
    impact_report: object | None = None
    errors: list[str] = field(default_factory=list)  # noms refusés (attaque)
    called: list[tuple] = field(default_factory=list)

    def call(self, name: str, **args):
        self.called.append((name, args))
        if name in ("find_file_id", "find_incident_id", "analyze_impact",
                    "analyze_root_cause", "hybrid_search", "get_file"):
            return _ToolResult(name, True, {"file_id": self.file_id} if name == "find_file_id" else self.impact_report)
        # simule la whitelist fermée du vrai registre
        self.errors.append(name)
        return _ToolResult(name, False, None, "hors whitelist")


@dataclass
class _ToolResult:
    name: str
    ok: bool
    value: object = None
    error: str | None = None


class FakeLlm:
    """Réponses en file : d'abord la rédaction, puis les verdicts JSON."""

    def __init__(self, responses: list[str], json_responses: list[dict] | None = None):
        self.text_queue = list(responses)
        self.json_queue = list(json_responses or [])
        self.calls: list[dict] = []

    @property
    def _model(self):  # convention attendue par ImpactAgent.finalize
        return "fake-model"

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> LlmOutput:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        if json_mode:
            data = self.json_queue.pop(0) if self.json_queue else {"supported": True, "reason": "ok"}
            return LlmOutput(model=self._model, content=json.dumps(data), data=dict(data))
        content = self.text_queue.pop(0) if self.text_queue else "réponse test"
        return LlmOutput(model=self._model, content=content)


# --------------------------------------------------------------------------- #
# Fabrique de preuve réelle (même forme que ChangeImpactAnalyzer)
# --------------------------------------------------------------------------- #
def make_path(score: float = 0.80, n_dependents: int = 2) -> EvidencePath:
    hops = [
        EvidenceHop(step=0, node_kind=NodeKind.FILE, node_id="httpie/cli::httpie/context.py",
                    role=EvidenceRole.CONTEXT, score=1.0, rationale="fichier analysé"),
    ]
    for i in range(n_dependents):
        hops.append(
            EvidenceHop(step=i + 1, node_kind=NodeKind.FILE,
                        node_id=f"httpie/cli::module_{i}.py",
                        relation_in="IMPORTS", role=EvidenceRole.BLAST_RADIUS,
                        score=0.7, rationale=f"imports la cible (distance {i + 1})")
        )
    return EvidencePath(path_id="path-test", intent=AnswerIntent.CHANGE_IMPACT,
                        hops=hops, score=score, is_valid=True)


def make_evidences(path: EvidencePath) -> list[Evidence]:
    out = []
    for hop in path.hops:
        out.append(
            Evidence(
                id=f"ev-test-{hop.step:03d}",
                kind=EvidenceKind.NODE if hop.step == 0 else EvidenceKind.IMPORT_EDGE,
                retrieval_strategy=RetrievalStrategy.GRAPH_TRAVERSAL,
                score=hop.score,
                rank=hop.step + 1,
                text=hop.node_id,
                rationale=hop.rationale or "",
                hop_index=hop.step,
                path_id=path.path_id,
                node_references=[EvidenceRef(node_kind=NodeKind.FILE,
                                             node_id=hop.node_id,
                                             role=hop.role,
                                             hop_index=hop.step)],
                retrieved_at=utcnow(),
                provenance=Provenance(source="derived", extractor="test", confidence=hop.score),
            )
        )
    return out


@dataclass(frozen=True)
class _Report:
    path: EvidencePath
    evidences: list
    direct_dependents: list
    transitive_dependents: int


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_extract_file_candidate_finds_plain_path() -> None:
    assert extract_file_candidate("que casse si je change httpie/context.py ?") == "httpie/context.py"


def test_extract_file_candidate_finds_scoped_id() -> None:
    assert extract_file_candidate("impact de httpie/cli::httpie/ssl_.py ?") == "httpie/cli::httpie/ssl_.py"


def test_extract_file_candidate_none_without_path() -> None:
    assert extract_file_candidate("pourquoi l'incident 1583 ?") is None
    assert extract_file_candidate("") is None


# --------------------------------------------------------------------------- #
# Cas heureux
# --------------------------------------------------------------------------- #
def test_no_llm_still_answers_from_graph_evidence() -> None:
    """Sans LLM : la réponse est calquée sur les hops; pas d'appel, budget 0."""
    path = make_path()
    tools = FakeTools(impact_report=_Report(path, make_evidences(path), ["m0.py"], 2))
    answer = run_impact_agent(tools, llm=None,
                              question="que casse si je change httpie/cli::httpie/context.py ?")
    assert answer.status == AnswerStatus.OK
    assert answer.confidence_score == path.score
    assert answer.confidence_band == ConfidenceBand.HIGH  # from_score(0.80) = high
    assert [c.label for c in answer.citations] == [h.node_id for h in path.hops[1:5]]
    assert answer.evidence_path is path
    assert answer.model is None
    # la whitelist n'a eu que des outils légitimes
    assert tools.errors == []
    assert {name for name, _ in tools.called} <= {"find_file_id", "analyze_impact"}


def test_llm_draft_and_critic_accept() -> None:
    path = make_path()
    tools = FakeTools(impact_report=_Report(path, make_evidences(path), [], 2))
    llm = FakeLlm(responses=["Impact du changement: voir l'EvidencePath adjacente."],
                  json_responses=[{"supported": True, "reason": "chaque affirmation a un hop"}])
    answer = run_impact_agent(tools, llm, "impact de httpie/cli::httpie/context.py ?")
    assert answer.status == AnswerStatus.OK
    assert answer.answer_text.startswith("Impact du changement")
    assert answer.model == "fake-model"


def test_two_llm_calls_only_draft_and_critic() -> None:
    path = make_path()
    tools = FakeTools(impact_report=_Report(path, make_evidences(path), [], 2))
    llm = FakeLlm(responses=["OK."], json_responses=[{"supported": True, "reason": "ok"}])
    state = initial_state("que casse httpie/context.py ?", "httpie/cli",
                          route=AnswerIntent.CHANGE_IMPACT)
    build_impact_agent(tools, llm).invoke(state)
    assert len(llm.calls) == 2
    # les system prompts sont bien ceux de la rédaction puis du Critic
    systems = [c["system"] for c in llm.calls]
    assert "analyste d'impact" in systems[0]
    assert "Critic" in systems[1]


# --------------------------------------------------------------------------- #
# Retry du Critic (borné)
# --------------------------------------------------------------------------- #
def test_critic_retry_once_then_accept() -> None:
    path = make_path()
    tools = FakeTools(impact_report=_Report(path, make_evidences(path), [], 2))
    llm = FakeLlm(responses=["brouillon 1", "brouillon 2"],
                  json_responses=[{"supported": False, "reason": "affirmation non appuyée"},
                                  {"supported": True, "reason": "ok cette fois"}])
    answer = run_impact_agent(tools, llm, "impact de httpie/cli::httpie/context.py ?")
    assert answer.status == AnswerStatus.OK
    assert answer.answer_text == "brouillon 2"
    # draft + critic(refus) + draft + critic(accept)
    assert len(llm.calls) == 4


def test_critic_refuses_everything_despite_max_retries() -> None:
    path = make_path()
    tools = FakeTools(impact_report=_Report(path, make_evidences(path), [], 2))
    llm = FakeLlm(responses=["b1", "b2", "b3"],
                  json_responses=[{"supported": False, "reason": "non"}] * 3)
    answer = run_impact_agent(tools, llm, "impact de httpie/cli::httpie/context.py ?")
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    # l'escalade explique le pourquoi; la rédaction refusée n'est JAMAIS exposée
    assert "preuve suffisante" in answer.answer_text
    assert "retries épuisés" in answer.answer_text
    assert "b1" not in answer.answer_text and "b2" not in answer.answer_text
    # le Critic a tourné 3 fois : 2 retries + l'escalade finale
    critic_calls = [c for c in llm.calls if "Critic" in c["system"]]
    assert len(critic_calls) == 3


# --------------------------------------------------------------------------- #
# Pas de preuve -> escalade
# --------------------------------------------------------------------------- #
def test_unknown_file_escalates_without_llm() -> None:
    tools = FakeTools(file_id=None)          # find_file_id -> rien dans le graphe
    answer = run_impact_agent(tools, llm=None, question="impact de httpie/inconnu.py ?")
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.evidence_path is None
    assert answer.confidence_score == 0.0
    assert answer.confidence_band == ConfidenceBand.UNKNOWN


def test_empty_blast_radius_escalates() -> None:
    """Fichier trouvé mais personne n'en dépend : pas de réponse affirmative."""
    path = make_path(n_dependents=0)
    tools = FakeTools(impact_report=_Report(path, [], [], 0))
    answer = run_impact_agent(tools, llm=None, question="impact de httpie/isole.py ?")
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE


def test_extraction_llm_used_when_regex_fails() -> None:
    """Question sans ``.py`` : l'extraction passe par le LLM (json_mode)."""
    path = make_path()
    tools = FakeTools(impact_report=_Report(path, make_evidences(path), [], 2))
    llm = FakeLlm(responses=["Le fichier request est ci-dessous."],
                  # 1re réponse JSON : extraction ; 2e : verdict du Critic
                  json_responses=[{"file_path": "httpie/context.py"},
                                  {"supported": True, "reason": "hop suffisant"}])
    answer = run_impact_agent(tools, llm, "qu'est-ce qui casse si le module context est touché ?")
    assert answer.status == AnswerStatus.OK
    # extraction(1) + draft(1) + critic(1)
    assert len(llm.calls) == 3
    extraction = llm.calls[0]
    assert extraction["json_mode"] is True
    assert "JSON" in extraction["system"]


def test_an_answer_without_any_guard_true_fails() -> None:
    """Filet de sécurité du contrat : preuves vides => jamais Answer OK."""
    path = make_path()
    tools = FakeTools(impact_report=_Report(path, [], [], 0))   # evidences vides
    answer = run_impact_agent(tools, llm=None, question="impact de httpie/cli::httpie/isole.py ?")
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.evidence_ids == []


# --------------------------------------------------------------------------- #
# Budget LLM
# --------------------------------------------------------------------------- #
def test_budget_exhaustion_falls_back_to_deterministic_accept() -> None:
    """Budget épuisé après la rédaction: le Critic déterministe calque le graphe."""
    path = make_path(score=0.90)
    tools = FakeTools(impact_report=_Report(path, make_evidences(path), [], 2))
    llm = FakeLlm(responses=["texte LLM."], json_responses=[])
    answer = run_impact_agent(tools, llm, "impact de httpie/cli::httpie/context.py ?",
                              budget=AgentBudget(max_llm_calls=1, max_retries=0))
    assert answer.status == AnswerStatus.OK
    assert len(llm.calls) == 1            # draft seulement
    assert answer.answer_text == "texte LLM."
    assert answer.confidence_score == 0.90


def test_budget_never_allows_more_calls_than_configured() -> None:
    path = make_path()
    tools = FakeTools(impact_report=_Report(path, make_evidences(path), [], 2))
    llm = FakeLlm(responses=["draft."], json_responses=[{"supported": True, "reason": "ok"}])
    run_impact_agent(tools, llm, "impact de httpie/cli::httpie/context.py ?",
                     budget=AgentBudget(max_llm_calls=2))
    assert len(llm.calls) <= 2


# --------------------------------------------------------------------------- #
# Sécurité : l'agent n'appelle rien hors whitelist
# --------------------------------------------------------------------------- #
def test_agent_never_touches_non_whitelisted_tools() -> None:
    path = make_path()
    tools = FakeTools(impact_report=_Report(path, make_evidences(path), [], 2))
    run_impact_agent(tools, llm=None, question="que casse httpie/cli::httpie/context.py ?")
    assert tools.errors == []             # aucun nom hors whitelist tenté
    names = [name for name, _ in tools.called]
    assert set(names) <= {"find_file_id", "analyze_impact"}


# --------------------------------------------------------------------------- #
# Divers : état initial intact, notes de Critic traçables
# --------------------------------------------------------------------------- #
def test_state_notes_are_traced_in_answer() -> None:
    """Les notes du Critic (brèves) se retrouvent dans l'Answer escaladée."""
    tools = FakeTools(file_id=None)
    answer = run_impact_agent(tools, llm=None, question="impact de httpie/rien.py ?")
    assert "aucun chemin" in answer.answer_text or "preuve" in answer.answer_text
