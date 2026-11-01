"""Tests de ``agents/planner.py`` — T6 : routage, repli LLM unique, budget."""

from dataclasses import dataclass, field

import pytest

from agents.planner import (
    RouteDecision,
    classify,
    classify_with_fallback,
    route,
)
from agents.state import AgentBudget
from agents.tools import ToolRegistry
from models import AnswerIntent, AnswerStatus
from tests.test_agents_impact_agent import FakeLlm, _Report, make_evidences, make_path
from tests.test_agents_root_cause_agent import _RcaReport, _HybridReport


# --------------------------------------------------------------------------- #
# Classification par règles pures
# --------------------------------------------------------------------------- #
def test_classify_change_impact_by_path_signal() -> None:
    assert classify("que casse si je change httpie/context.py ?") == AnswerIntent.CHANGE_IMPACT


def test_classify_root_cause_by_issue_number() -> None:
    assert classify("que s'est-il passé sur l'incident #1583 ?") == AnswerIntent.ROOT_CAUSE


def test_classify_by_keywords_only() -> None:
    assert classify("pourquoi ces requêtes ont-elles échoué hier ?") == AnswerIntent.ROOT_CAUSE
    assert classify("quels fichiers dépendent de utils.py ?") == AnswerIntent.CHANGE_IMPACT


def test_classify_tie_is_ambiguous() -> None:
    """Un signal de chaque camp à égalité -> None : au repli de départager."""
    assert classify("pourquoi httpie/context.py ?") is None


def test_classify_no_signal_is_ambiguous() -> None:
    assert classify("explique-moi le projet") is None
    assert classify("") is None
    assert classify(None) is None                  # garde : jamais de crash


# --------------------------------------------------------------------------- #
# Repli LLM unique
# --------------------------------------------------------------------------- #
def test_fallback_llm_routes_to_root_cause() -> None:
    llm = FakeLlm(responses=[], json_responses=[{"intent": "root_cause"}])
    intent, method, used = classify_with_fallback("pourquoi httpie/context.py ?", llm)
    assert intent == AnswerIntent.ROOT_CAUSE
    assert method == "llm"
    assert used == 1


def test_fallback_llm_unrecognized_intent_defaults_to_impact() -> None:
    llm = FakeLlm(responses=[], json_responses=[{"intent": "general"}])
    intent, method, used = classify_with_fallback("explique-moi le projet", llm)
    assert intent == AnswerIntent.CHANGE_IMPACT   # route par défaut
    assert method == "default"
    assert used == 1


def test_fallback_without_key_defaults_and_consumes_nothing() -> None:
    intent, method, used = classify_with_fallback("explique-moi le projet", llm=None)
    assert intent == AnswerIntent.CHANGE_IMPACT
    assert method == "default"
    assert used == 0


def test_fallback_llm_failure_defaults_and_spends_one() -> None:
    class BrokenLlm:
        def chat(self, *a, **k):
            raise RuntimeError(" réseau down")

    intent, method, used = classify_with_fallback("explique-moi le projet", BrokenLlm())
    assert intent == AnswerIntent.CHANGE_IMPACT
    assert method == "default"
    assert used == 1


def test_keyword_hit_never_calls_the_llm() -> None:
    llm = FakeLlm(responses=[], json_responses=[])
    intent, method, used = classify_with_fallback("que casse le refactoring de ssl_.py ?", llm)
    assert intent == AnswerIntent.CHANGE_IMPACT
    assert method == "keyword"
    assert used == 0
    assert llm.calls == []                        # zéro appel LLM


# --------------------------------------------------------------------------- #
# route() : bout-en-bout vers le bon agent
# --------------------------------------------------------------------------- #
@dataclass
class FakeBothTools:
    """Outils factices couvrant les deux agents."""

    file_id: str | None = "httpie/cli::httpie/context.py"
    incident_id: str | None = "httpie/cli#issue-1583"
    rca_report: object | None = None
    called: list[tuple] = field(default_factory=list)

    def call(self, name: str, **args):
        self.called.append((name, args))
        if name == "find_file_id":
            return _Any(name, True, {"file_id": self.file_id})
        if name == "analyze_impact":
            return _Any(name, self.impact is not None, self.impact)
        if name == "find_incident_id":
            return _Any(name, True, {"incident_id": self.incident_id})
        if name == "analyze_root_cause":
            return _Any(name, self.rca_report is not None, self.rca_report)
        if name == "hybrid_search":
            return _Any(name, True, self.hybrid if self.hybrid is not None else _HybridReport(evidences=[]))
        return _Any(name, False, None, "hors whitelist")

    impact: object | None = None
    hybrid: object | None = None


@dataclass
class _Any:
    name: str
    ok: bool
    value: object = None
    error: str | None = None


def test_route_impact_question_calls_impact_agent() -> None:
    path = make_path()
    tools = FakeBothTools(impact=_Report(path, make_evidences(path), [], 2))
    answer, decision = route("que casse si je change httpie/cli::httpie/context.py ?",
                             tools, llm=None)
    assert answer.status == AnswerStatus.OK
    assert answer.intent == AnswerIntent.CHANGE_IMPACT
    assert decision.method == "keyword"
    assert decision.llm_calls == 0
    names = [n for n, _ in tools.called]
    assert "analyze_impact" in names
    assert "analyze_root_cause" not in names       # le mauvais agent n'a pas tourné


def test_route_root_cause_question_calls_rca_agent() -> None:
    path = make_path(score=0.63)
    tools = FakeBothTools(rca_report=_RcaReport(path, make_evidences(path)))
    answer, decision = route("pourquoi l'incident #1583 a tout bloqué ?",
                             tools, llm=None)
    assert answer.status == AnswerStatus.OK
    assert answer.intent == AnswerIntent.ROOT_CAUSE
    assert decision.method == "keyword"
    names = [n for n, _ in tools.called]
    assert "analyze_root_cause" in names


def test_route_ambiguous_with_llm_consumes_one_call_and_still_answers() -> None:
    path = make_path()
    tools = FakeBothTools(impact=_Report(path, make_evidences(path), [], 2))
    llm = FakeLlm(responses=["réponse LLM."],
                  json_responses=[{"intent": "change_impact"},
                                  {"supported": True, "reason": "hop ok"}])
    answer, decision = route("pourquoi httpie/context.py ?", tools, llm)
    assert answer.status == AnswerStatus.OK
    assert decision.method == "llm"
    assert decision.llm_calls == 1
    assert len(llm.calls) == 3            # classement + draft + critic


def test_route_default_keeps_registry_read_only_and_escalates() -> None:
    """Question hors périmètre, pas de LLM : route par défaut -> absence d'entité -> escalade."""
    tools = FakeBothTools(file_id=None, incident_id=None)
    answer, decision = route("explique-moi le projet", tools, llm=None)
    assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE   # flag d'escalade
    assert decision.method == "default"
    assert answer.evidence_path is None


def test_route_budget_sold_after_llm_fallback() -> None:
    """Le repli a consommé 1 appel : l'agent n'en a plus que max-1."""
    path = make_path()
    tools = FakeBothTools(impact=_Report(path, make_evidences(path), [], 2))
    llm = FakeLlm(responses=["draft."],
                  json_responses=[{"intent": "change_impact"},
                                  {"supported": True, "reason": "ok"}])
    answer, decision = route("pourquoi httpie/context.py ?",
                             tools, llm,
                             budget=AgentBudget(max_llm_calls=2))
    assert answer.status == AnswerStatus.OK
    assert decision.llm_calls == 1
    # budget honnête : planner(1) + agent(1) = 2. Le Critic, budget épuisé,
    # retombe en mode déterministe au lieu de dépasser la borne.
    assert len(llm.calls) == 2


def test_route_never_touches_non_whitelisted_tools() -> None:
    path = make_path()
    tools = FakeBothTools(impact=_Report(path, make_evidences(path), [], 2))
    route("que casse httpie/cli::httpie/context.py ?", tools, llm=None)
    names = [n for n, _ in tools.called]
    assert set(names) <= {
        "find_file_id", "analyze_impact", "find_incident_id",
        "analyze_root_cause", "hybrid_search",
    }


def test_route_decision_is_traceable() -> None:
    """RouteDecision porte tout ce que les logs structurés (T7) attendront."""
    path = make_path()
    tools = FakeBothTools(rca_report=_RcaReport(path, make_evidences(path)))
    answer, decision = route("pourquoi l'incident #1583 ?", tools, llm=None)
    assert isinstance(decision, RouteDecision)
    assert decision.intent == AnswerIntent.ROOT_CAUSE
    assert decision.question.startswith("pourquoi")
    assert decision.method in ("keyword", "llm", "default")
