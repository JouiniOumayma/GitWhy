"""Root Cause analyzer tests -- offline, against a scripted fake session.

The fake reproduces exactly what ``TRAVERSAL_QUERY`` returns: one row per
(node, path) pair with ``distance``, ``edge_confidences``, ``rel_types`` and
chain arrays. The tests pin the invariants that make an RCA answer auditable:

* stable ordering and deduplication of hops;
* scores decaying with distance and weighted by edge confidence (a GraphQL
  ``:CLOSES`` at 1.0 outweighs a regex link at 0.6);
* role mapping (symptom / fix / root cause / timeline);
* honest ``is_valid=False`` on an empty graph -- never a fake answer;
* the money-path extraction (shortest chain reaching a File).
"""

from __future__ import annotations

from typing import Any

import pytest

from retrieval import RootCauseAnalyzer
from models import (
    AnswerIntent,
    EvidenceKind,
    EvidenceRole,
    NodeKind,
    RetrievalStrategy,
)

INCIDENT = "httpie/cli#issue-1583"


class _Single:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def single(self) -> "_Single":
        return self

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


class _List:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    def __iter__(self):
        return iter(_Record(item) for item in self._items)


class _Record:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class _ScriptedSession:
    def __init__(self, rows: list[dict[str, Any]], incidents: set[str] | None = None) -> None:
        self.rows = rows
        self.incidents = incidents or set()
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def run(self, query: str, **parameters: Any):
        self.queries.append((query, parameters))
        if "count(i) AS found" in query:
            return _Single({"found": 1 if parameters["incident_id"] in self.incidents else 0})
        return _List(self.rows)

    def __enter__(self) -> "_ScriptedSession":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeDriver:
    def __init__(self, rows: list[dict[str, Any]], incidents: set[str]) -> None:
        self._session = _ScriptedSession(rows, incidents)
        self.sessions: list[_ScriptedSession] = [self._session]

    def session(self, database: str) -> _ScriptedSession:
        return self._session


def _row(node_id: str, label: str, distance: int, confidences: list[float],
         rel_types: list[str], name: str | None = None,
         chain: bool = True) -> dict[str, Any]:
    chain_ids = [INCIDENT] + [f"n{i}" for i in range(1, distance)] + [node_id]
    chain_labels = ["Incident"] + ["X"] * (distance - 1) + [label]
    return {
        "node_id": node_id,
        "node_label": label,
        "node_name": name or node_id,
        "distance": distance,
        "edge_confidences": confidences,
        "rel_types": rel_types,
        "chain_ids": chain_ids if chain else [INCIDENT, node_id],
        "chain_labels": chain_labels if chain else ["Incident", label],
    }


# --------------------------------------------------------------------------- #
# Missing incident
# --------------------------------------------------------------------------- #
def test_missing_incident_raises_file_not_found() -> None:
    analyzer = RootCauseAnalyzer(_FakeDriver([], set()))
    with pytest.raises(FileNotFoundError, match="not in the graph"):
        analyzer.analyze(INCIDENT)


# --------------------------------------------------------------------------- #
# The demo chain: 1583 -> PR 1596 -> 7f03c52 -> ssl_.py -> @3.2.4
# --------------------------------------------------------------------------- #
DEMO_ROWS = [
    _row(f"httpie/cli#1596", "PR", 1, [1.0], ["CLOSES"], "Fix SSL context creation"),
    _row("7f03c52d2237440c5a672296ce6955aae4ed4f09", "Commit", 1, [0.9], ["CLOSES"],
         "Close #1583: pin requests<2.32.3"),
    _row("httpie/cli::httpie/ssl_.py", "File", 2, [1.0, 1.0],
         ["CLOSES", "MODIFIES"], "httpie/ssl_.py"),
    _row("httpie/cli@3.2.4", "Deployment", 3, [1.0, 1.0, 1.0],
         ["OBSERVED_IN", "DEPLOYED_AS", "CLOSES"], "3.2.4"),
]


def test_demo_chain_hops_and_roles() -> None:
    driver = _FakeDriver(DEMO_ROWS, {INCIDENT})
    report = analyzer_analyze(driver)
    path = report.path
    # hop 0 = incident, then one per distinct node, ordered by distance
    assert [hop.step for hop in path.hops] == list(range(len(path.hops)))
    assert path.hops[0].node_kind == NodeKind.INCIDENT
    assert path.hops[0].role == EvidenceRole.SYMPTOM
    pr_hop = next(h for h in path.hops if h.node_kind == NodeKind.PR)
    assert pr_hop.role == EvidenceRole.FIX
    assert pr_hop.relation_in == "CLOSES"
    file_hop = next(h for h in path.hops if h.node_kind == NodeKind.FILE)
    assert file_hop.role == EvidenceRole.ROOT_CAUSE
    deployment_hop = next(h for h in path.hops if h.node_kind == NodeKind.DEPLOYMENT)
    assert deployment_hop.role == EvidenceRole.TIMELINE
    assert path.intent == AnswerIntent.ROOT_CAUSE


def analyzer_analyze(driver: _FakeDriver, **kwargs: Any) -> Any:
    return RootCauseAnalyzer(driver).analyze(INCIDENT, **kwargs)


def test_demo_chain_summary_lists() -> None:
    report = analyzer_analyze(_FakeDriver(DEMO_ROWS, {INCIDENT}))
    assert report.fixing_prs == [f"httpie/cli#1596"]
    assert report.fix_commits == ["7f03c52d2237440c5a672296ce6955aae4ed4f09"]
    assert report.affected_files == ["httpie/cli::httpie/ssl_.py"]
    assert report.deployments == ["httpie/cli@3.2.4"]


def test_scores_decay_and_weight_by_confidence() -> None:
    report = analyzer_analyze(_FakeDriver(DEMO_ROWS, {INCIDENT}))
    path = report.path
    pr_hop = next(h for h in path.hops if h.node_kind == NodeKind.PR)
    file_hop = next(h for h in path.hops if h.node_kind == NodeKind.FILE)
    deployment_hop = next(h for h in path.hops if h.node_kind == NodeKind.DEPLOYMENT)
    assert pr_hop.score == 1.0                      # distance 1, confidence 1.0
    assert file_hop.score == pytest.approx(0.85)    # 1.0 * 0.85**1
    assert deployment_hop.score == pytest.approx(0.7225)  # 1.0 * 0.85**2 (rounded)
    assert file_hop.score > deployment_hop.score


def test_low_confidence_edge_lowers_hop_score() -> None:
    rows = [
        _row("httpie/cli#1596", "PR", 1, [0.6], ["CLOSES"], "regex-derived link"),
    ]
    report = analyzer_analyze(_FakeDriver(rows, {INCIDENT}))
    pr_hop = next(h for h in report.path.hops if h.node_kind == NodeKind.PR)
    assert pr_hop.score == pytest.approx(0.6)


def test_best_chain_prefers_the_first_file_reached() -> None:
    report = analyzer_analyze(_FakeDriver(DEMO_ROWS, {INCIDENT}))
    assert report.best_chain, "the demo chain contains a File"
    assert report.best_chain[0] == f"Incident:{INCIDENT}"
    assert any(element.startswith("File:") for element in report.best_chain)


def test_duplicate_rows_are_deduplicated() -> None:
    rows = DEMO_ROWS + [DEMO_ROWS[0]]  # same PR reached twice
    report = analyzer_analyze(_FakeDriver(rows, {INCIDENT}))
    pr_hops = [h for h in report.path.hops if h.node_id == "httpie/cli#1596"]
    assert len(pr_hops) == 1


def test_evidences_share_path_id_and_carry_derived_provenance() -> None:
    report = analyzer_analyze(_FakeDriver(DEMO_ROWS, {INCIDENT}))
    path_ids = {e.path_id for e in report.evidences}
    assert len(path_ids) == 1
    for evidence in report.evidences:
        assert evidence.provenance.source.value == "derived"
        assert evidence.provenance.extractor == "nexus-devintel.retrieval.root_cause"
        assert evidence.retrieval_strategy == RetrievalStrategy.GRAPH_TRAVERSAL
    kinds = {e.kind for e in report.evidences}
    assert EvidenceKind.GRAPH_PATH in kinds


# --------------------------------------------------------------------------- #
# Honest failure modes
# --------------------------------------------------------------------------- #
def test_empty_graph_yields_invalid_path_not_a_crash() -> None:
    report = analyzer_analyze(_FakeDriver([], {INCIDENT}))
    assert report.path.hops[0].node_kind == NodeKind.INCIDENT
    assert len(report.path.hops) == 1
    assert report.path.is_valid is False
    assert report.path.score == 0.0
    assert any("no connected node" in note for note in report.path.validation_notes)


def test_unknown_label_is_skipped_with_a_note() -> None:
    rows = [_row("weird", "Mystery", 1, [1.0], ["CLOSES"])]
    report = analyzer_analyze(_FakeDriver(rows, {INCIDENT}))
    assert all(hop.node_kind != "Mystery" for hop in report.path.hops)
    assert any("unknown label" in note for note in report.path.validation_notes)


def test_max_depth_is_validated() -> None:
    analyzer = RootCauseAnalyzer(_FakeDriver([], set()))
    with pytest.raises(ValueError, match="max_depth"):
        analyzer._validated_bound(21)
    with pytest.raises(ValueError, match="max_depth"):
        analyzer._validated_bound(True)  # bool is not an int here


def test_traversal_query_is_depth_bounded_and_whitelisted() -> None:
    from retrieval.root_cause import RCA_RELATIONS, TRAVERSAL_QUERY

    query = TRAVERSAL_QUERY % (RCA_RELATIONS, 6)
    assert "*1..6" in query
    for relation in ("CLOSES", "MERGED_INTO", "MODIFIES", "DEPLOYED_AT",
                     "OBSERVED_IN", "AFFECTS", "TOUCHES", "DEPLOYED_AS"):
        assert relation in query
    # parameters are bound, not interpolated
    assert "$incident_id" in query
    assert "$limit" in query
