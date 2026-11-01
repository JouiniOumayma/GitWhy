"""Tests de ``agents/root_cause_agent.py`` — T5 : héritage T4, zéro réseau.

La chaîne de démo de référence : ``httpie/cli#issue-1583`` -> Deployment,
PRs, commits, ``httpie/ssl_.py`` (les fixtures Phase 1 la décrivent, ici on
la stubule aux mêmes formes que ``RootCauseReport``).
"""

from dataclasses import dataclass, field

import pytest

from agents.root_cause_agent import (
    RootCauseAgent,
    build_root_cause_agent,
    extract_issue_candidate,
    run_root_cause_agent,
)
from agents.state import AgentBudget, initial_state
from agents.llm import LlmOutput
from tests.test_agents_impact_agent import FakeLlm, _Report, make_evidences, make_path
from models import AnswerIntent, AnswerStatus, ConfidenceBand


# --------------------------------------------------------------------------- #
# Stubs spécifiques RCA
# --------------------------------------------------------------------------- #
@dataclass
class FakeRcaTools:
    """Canard-typpé de ToolRegistry orienté Root Cause."""

    incident_id: str | None = "httpie/cli#issue-1583"
    rca_report: object | None = None
    hybrid_report: object | None = None
    called: list[tuple] = field(default_factory=list)

    def call(self, name: str, **args):
        self.called.append((name, args))
        if name == "find_incident_id":
            return _Simple(name, True, {"incident_id": self.incident_id})
        if name == "analyze_root_cause":
            return _Simple(name, self.rca_report is not None, self.rca_report)
        if name == "hybrid_search":
            return _Simple(name, self.hybrid_report is not None, self.hybrid_report)
        return _Simple(name, False, None, "hors whitelist")


@dataclass
class _Simple:
    name: str
    ok: bool
    value: object = None
    error: str | None = None


@dataclass(frozen=True)
class _RcaReport:
    """Forme minimale de RootCauseReport (path + evidences + synthèse)."""

    path: object
    evidences: list
    fix_commits: list = field(default_factory=list)
    fixing_prs: list = field(default_factory=list)
    affected_files: list = field(default_factory=list)
    deployments: list = field(default_factory=list)
    best_chain: list = field(default_factory=list)


@dataclass(frozen=True)
class _HybridReport:
    evidences: list


QUESTION_DEMO = "pourquoi l'incident httpie/cli#issue-1583 a-t-il cassé les requêtes SSL ?"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_extract_issue_candidate_finds_hash_number() -> None:
    assert extract_issue_candidate(QUESTION_DEMO) == 1583


def test_extract_issue_candidate_finds_word_forms() -> None:
    assert extract_issue_candidate("explique moi l'issue 6730 de requests") == 6730
    assert extract_issue_candidate("bug 211 Ssl") == 211


def test_extract_issue_candidate_none_sans_numero() -> None:
    assert extract_issue_candidate("pourquoi ça plante ?") is None
    assert extract_issue_candidate("") is None


# --------------------------------------------------------------------------- #
# Cas heureux (sans LLM : la chaîne de preuve suffit)
# --------------------------------------------------------------------------- #
def _happy_tools(n_hops: int = 3, hybrid: object | None = "default"):
    path = make_path(score=0.63, n_dependents=n_hops)
    hybrid = _HybridReport(evidences=[]) if hybrid == "default" else hybrid
    return FakeRcaTools(rca_report=_RcaReport(path, make_evidences(path)), hybrid_report=hybrid)


def test_demo_chain_answers_ok_without_llm() -> None:
    tools = _happy_tools()
    answer = run_root_cause_agent(tools, llm=None, question=QUESTION_DEMO)
    assert answer.status == AnswerStatus.OK
    assert answer.intent == AnswerIntent.ROOT_CAUSE
    assert answer.confidence_band == ConfidenceBand.MEDIUM  # score 0.63
    assert {name for name, _ in tools.called} <= {
        "find_incident_id", "analyze_root_cause", "hybrid_search"}
    assert answer.model is None


def test_hybrid_evidence_extends_the_chain_and_survives_failure() -> None:
    """Chunks hybrides -> evidence_ids allongés ; hybride en panne -> non bloquant."""
    path = make_path(score=0.63)
    hybrid = _HybridReport(evidences=make_evidences(make_path())[:1])  # 1 chunk d'exemple
    tools = FakeRcaTools(rca_report=_RcaReport(path, make_evidences(path)),
                         hybrid_report=hybrid)
    answer = run_root_cause_agent(tools, llm=None, question=QUESTION_DEMO)
    assert answer.status == AnswerStatus.OK
    assert len(answer.evidence_ids) == len(path.hops) + 1

    tools_without_hybrid = FakeRcaTools(rca_report=_RcaReport(path, make_evidences(path)),
                                        hybrid_report=None)
    answer_fail = run_root_cause_agent(tools_without_hybrid, llm=None, question=QUESTION_DEMO)
    assert answer_fail.status == AnswerStatus.OK          # la chaîne graphique suffit
    assert len(answer_fail.evidence_ids) == len(path.hops)


# --------------------------------------------------------------------------- #
# Extraction LLM en repli
# --------------------------------------------------------------------------- #
def test_extraction_llm_fallback_without_hash() -> None:
    path = make_path()
    tools = FakeRcaTools(rca_report=_RcaReport(path, make_evidences(path)))
    llm = FakeLlm(responses=["redraft."],
                  json_responses=[{"incident_number": 1583},
                                  {"supported": True, "reason": "hop suffisant"}])
    answer = run_root_cause_agent(tools, llm, "pourquoi le serveur a-t-il rejeté les connexions TLS ?")
    assert answer.status == AnswerStatus.OK
    assert len(llm.calls) == 3            # extraction + draft + critic
    assert llm.calls[0]["json_mode"] is True
    # le numéro extrait a bien été transmis à l'outil
    rca_calls = [args for name, args in tools.called if name == "find_incident_id"]
    assert rca_calls == [{"issue_number": 1583}]


# --------------------------------------------------------------------------- #
# Pas de preuve -> escalade
# --------------------------------------------------------------------------- #
def test_unknown_incident_escalates() -> None:
    tools = FakeRcaTools(incident_id=None)     # find_incident_id -> rien
    answer = run_root_cause_agent(tools, llm=None, question=QUESTION_DEMO)
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.evidence_path is None
    assert answer.confidence_score == 0.0
    assert "incidents" in answer.answer_text or "preuve" in answer.answer_text


def test_empty_causal_chain_escalates() -> None:
    """Issue connue du graphe mais zéro hop : pas de réponse affirmative."""
    path = make_path(n_dependents=0)
    tools = FakeRcaTools(rca_report=_RcaReport(path, []))
    answer = run_root_cause_agent(tools, llm=None, question=QUESTION_DEMO)
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.evidence_ids == []


# --------------------------------------------------------------------------- #
# Critic : retry borné (hérité de T4, vérifié une fois ici)
# --------------------------------------------------------------------------- #
def test_critic_retry_then_accept_roundtrip() -> None:
    path = make_path()
    tools = FakeRcaTools(rca_report=_RcaReport(path, make_evidences(path)))
    llm = FakeLlm(responses=["b1", "b2"],
                  json_responses=[{"supported": False, "reason": "cause non appuyée"},
                                  {"supported": True, "reason": "ok"}])
    answer = run_root_cause_agent(tools, llm, QUESTION_DEMO)
    assert answer.status == AnswerStatus.OK
    assert answer.answer_text == "b2"
    assert len(llm.calls) == 4            # draft + reject + retry draft + accept


def test_budget_limited_to_max_llm_calls() -> None:
    path = make_path()
    tools = FakeRcaTools(rca_report=_RcaReport(path, make_evidences(path)))
    llm = FakeLlm(responses=["draft."], json_responses=[{"supported": True, "reason": "ok"}])
    run_root_cause_agent(tools, llm, QUESTION_DEMO, budget=AgentBudget(max_llm_calls=2))
    assert len(llm.calls) <= 2


# --------------------------------------------------------------------------- #
# Sécurité : rien hors whitelist
# --------------------------------------------------------------------------- #
def test_agent_never_calls_non_whitelisted_tools() -> None:
    tools = _happy_tools()
    run_root_cause_agent(tools, llm=None, question=QUESTION_DEMO)
    names = [name for name, _ in tools.called]
    assert set(names) <= {"find_incident_id", "analyze_root_cause", "hybrid_search"}


# --------------------------------------------------------------------------- #
# Cohérence de l'état initial (diffusion via build_root_cause_agent)
# --------------------------------------------------------------------------- #
def test_compiled_graph_keeps_the_shared_topology() -> None:
    tools = _happy_tools()
    graph = build_root_cause_agent(tools, llm=None)
    state = initial_state(QUESTION_DEMO, "httpie/cli", route=AnswerIntent.ROOT_CAUSE)
    out = graph.invoke(state)
    assert out["critic_verdict"].value == "accept"
    assert out["route"] == AnswerIntent.ROOT_CAUSE
