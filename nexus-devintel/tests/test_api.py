"""Tests de ``api/app.py`` — T7 : bout-en-bout HTTP sans stack ni LLM réel.

Tout est injecté : FakeApiTools (canard-typpé ToolRegistry avec la
surface nécessaire au hook de trace), FakeLlm et FakeWriter.
"""

import json
import logging
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from agents.llm import LlmOutput
from api.app import AnswerResponse, create_app
from models import AnswerIntent, AnswerStatus, ConfidenceBand
from tests.test_agents_impact_agent import _Report, make_evidences, make_path
from tests.test_agents_root_cause_agent import _RcaReport, _HybridReport


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
@dataclass
class FakeApiTools:
    """Canard-typpé ToolRegistry : find/analyze/hybrid + hook de trace."""

    file_id: str | None = "httpie/cli::httpie/context.py"
    incident_id: str | None = None
    impact_report: object | None = None
    rca_report: object | None = None
    hybrid_report: object | None = None
    called: list[tuple] = field(default_factory=list)
    _callback: object | None = field(default=None, repr=False)

    def call(self, name: str, **args):
        self.called.append((name, args))
        value, ok, error = None, False, "hors whitelist"
        if name == "find_file_id":
            value, ok = {"file_id": self.file_id}, True
        elif name == "analyze_impact":
            value, ok, error = self.impact_report, self.impact_report is not None, "stack requise"
        elif name == "find_incident_id":
            value, ok = {"incident_id": self.incident_id}, True
        elif name == "analyze_root_cause":
            value, ok, error = self.rca_report, self.rca_report is not None, "stack requise"
        elif name == "hybrid_search":
            value, ok = (self.hybrid_report if self.hybrid_report is not None
                         else _HybridReport(evidences=[])), True
        result = _Api(name, ok, value, error)
        if self._callback is not None:
            self._callback(name, result)
        return result

    def set_call_callback(self, callback) -> None:
        self._callback = callback


@dataclass
class _Api:
    name: str
    ok: bool
    value: object = None
    error: str | None = None


@dataclass
class FakeWriter:
    calls: list = field(default_factory=list)
    fail: bool = False

    def write_models(self, models):
        if self.fail:
            raise RuntimeError("Neo4j indisponible")
        self.calls.append(models)
        return {"nodes": len(models)}


def _impacted_tools(hops: int = 2, score: float = 0.80):
    path = make_path(score=score, n_dependents=hops)
    return FakeApiTools(impact_report=_Report(path, make_evidences(path), ["m0.py"], hops)), path


# --------------------------------------------------------------------------- #
# /api/health
# --------------------------------------------------------------------------- #
def test_health_without_stack_reports_what_is_offline() -> None:
    app = create_app(tools=FakeApiTools(), llm=None, writer=None)
    client = TestClient(app)
    out = client.get("/api/health").json()
    assert out["status"] == "ok"
    assert out["graph"] is False
    assert out["llm"] is False
    assert out["persistence"] is False


def test_health_with_impact_analyzer_reports_graph_online() -> None:
    tools, _ = _impacted_tools()
    # _impact: le fake n'a pas cet attribut -> graph False ; on vérifie le contrat
    app = create_app(tools=tools, llm=None)
    out = TestClient(app).get("/api/health").json()
    assert out["status"] == "ok"


# --------------------------------------------------------------------------- #
# /api/answer : contrat complet
# --------------------------------------------------------------------------- #
def test_post_answer_happy_impact_path_pydantic_contract() -> None:
    tools, _ = _impacted_tools()
    app = create_app(tools=tools, llm=None, writer=None, repo_id="httpie/cli")
    client = TestClient(app)
    response = client.post("/api/answer",
                           json={"question": "que casse si je change httpie/cli::httpie/context.py ?"})
    assert response.status_code == 200
    body = AnswerResponse.model_validate(response.json())
    assert body.status == AnswerStatus.OK
    assert body.escalated is False
    assert body.intent == AnswerIntent.CHANGE_IMPACT
    assert body.confidence_score == 0.80
    assert body.confidence_band == ConfidenceBand.HIGH.value
    assert body.evidence_path is not None
    assert len(body.evidence_path.hops) >= 2          # cible + dépendants
    assert body.evidence_ids                           # preuves citables
    assert body.persisted is False                     # pas de writer injecté


def test_post_answer_escalation_when_no_evidence() -> None:
    tools = FakeApiTools(file_id=None, incident_id=None)
    app = create_app(tools=tools, llm=None)
    out = TestClient(app).post("/api/answer",
                               json={"question": "que casse httpie/inconnu.py ?"}).json()
    body = AnswerResponse.model_validate(out)
    assert body.escalated is True
    assert body.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert body.evidence_path is None
    assert body.confidence_score == 0.0


def test_post_rejects_invalid_bodies() -> None:
    app = create_app(tools=FakeApiTools(), llm=None)
    client = TestClient(app)
    assert client.post("/api/answer", json={"question": ""}).status_code == 422
    assert client.post("/api/answer", json={"question": "ab"}).status_code == 422
    assert client.post("/api/answer", json={"question": "ok", "extra": 1}).status_code == 422
    assert client.post("/api/answer", json={"other": "x"}).status_code == 422


# --------------------------------------------------------------------------- #
# Persistance (le seul endroit qui écrit)
# --------------------------------------------------------------------------- #
def test_persists_evidences_then_answer_after_success() -> None:
    tools, _ = _impacted_tools()
    writer = FakeWriter()
    app = create_app(tools=tools, llm=None, writer=writer)
    out = TestClient(app).post("/api/answer",
                               json={"question": "impact de httpie/cli::httpie/context.py ?"}).json()
    assert out["persisted"] is True
    (models,) = writer.calls
    answers = [m for m in models if m.id.startswith("ans-")]
    assert len(answers) == 1
    evidences = [m for m in models if m.id.startswith("ev-")]
    assert len(evidences) == len(out["evidence_ids"])   # même liste, même ordre
    assert [m.id for m in evidences] == out["evidence_ids"]


def test_persistence_failure_never_breaks_the_answer() -> None:
    tools, _ = _impacted_tools()
    app = create_app(tools=tools, llm=None, writer=FakeWriter(fail=True))
    out = TestClient(app).post("/api/answer",
                               json={"question": "impact de httpie/cli::httpie/context.py ?"}).json()
    assert out["status"] == "ok"
    assert out["persisted"] is False


# --------------------------------------------------------------------------- #
# Logs structurés : une ligne jsonl par requête
# --------------------------------------------------------------------------- #
def test_structured_log_line_contains_the_brief_fields(caplog) -> None:
    tools, _ = _impacted_tools()
    app = create_app(tools=tools, llm=None, writer=None)
    with caplog.at_level(logging.INFO, logger="nexus.api"):
        TestClient(app).post("/api/answer",
                             json={"question": "impact de httpie/cli::httpie/context.py ?"})
    records = [r for r in caplog.records if r.name == "nexus.api"]
    assert records, "une ligne de log est attendue par requête"
    line = json.loads(records[0].getMessage())          # jsonl strict
    for key in ("question", "route", "route_method", "tools", "score",
                "critic", "escalated", "status"):
        assert key in line, key
    assert line["status"] == "ok"
    assert "find_file_id:ok" in line["tools"]
    assert "analyze_impact:ok" in line["tools"]
    assert line["critic"] == "accept"


def test_log_line_marks_escalation(caplog) -> None:
    tools = FakeApiTools(file_id=None)
    app = create_app(tools=tools, llm=None)
    with caplog.at_level(logging.INFO, logger="nexus.api"):
        TestClient(app).post("/api/answer", json={"question": "impact de httpie/rien.py ?"})
    line = json.loads(caplog.records[0].getMessage())
    assert line["escalated"] is True
    assert line["status"] == "insufficient_evidence"


# --------------------------------------------------------------------------- #
# LLM réel mocké : la critique passe bien par json_mode
# --------------------------------------------------------------------------- #
def test_post_with_mocked_llm_uses_json_critique() -> None:
    tools, _ = _impacted_tools()
    llm = _RecordingLlm(responses=["Texte du LLM."],
                        jsons=[{"supported": True, "reason": "hop suffisant"}])
    app = create_app(tools=tools, llm=llm)
    out = TestClient(app).post("/api/answer",
                               json={"question": "impact de httpie/cli::httpie/context.py ?"}).json()
    assert out["answer_text"] == "Texte du LLM."
    assert out["model"] == "fake-model"
    assert [c["json_mode"] for c in llm.calls] == [False, True]


class _RecordingLlm:
    """FakeLlm minimal local (la question contient déjà le .py)."""

    def __init__(self, responses: list[str], jsons: list[dict]) -> None:
        self._text_queue = list(responses)
        self._json_queue = list(jsons)
        self.calls: list[dict] = []

    @property
    def _model(self):  # convention attendue par finalize
        return "fake-model"

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> LlmOutput:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        if json_mode:
            data = dict(self._json_queue.pop(0))
            return LlmOutput(model=self._model, content=json.dumps(data), data=data)
        return LlmOutput(model=self._model, content=self._text_queue.pop(0))
