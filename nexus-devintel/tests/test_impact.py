"""Change Impact analyzer tests -- offline, against a scripted fake session.

The fake reproduces exactly what ``TRAVERSAL_QUERY`` returns: one row per
(dependent, path) pair with ``distance``, ``edge_confidences`` and ``chain``.
The tests pin the EvidencePath invariants: stable ordering, decaying scores,
honest ``is_valid`` on an empty graph, and Evidence nodes grouped under the
same ``path_id``.
"""

from __future__ import annotations

from typing import Any

import pytest

from retrieval import ChangeImpactAnalyzer
from models import (
    EvidenceKind,
    EvidenceRole,
    NodeKind,
)

TARGET = "httpie/cli::httpie/context.py"


class _ScriptedSession:
    """Returns pre-baked rows for TRAVERSAL_QUERY, counts existence probes."""

    def __init__(self, rows: list[dict[str, Any]], files: set[str] | None = None) -> None:
        self.rows = rows
        self.files = files or set()
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def run(self, query: str, **parameters: Any):
        self.queries.append((query, parameters))
        if "count(f) AS found" in query:
            return _Single({"found": 1 if parameters["file_id"] in self.files else 0})
        return _List(self.rows)

    def __enter__(self) -> _ScriptedSession:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _Single:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def single(self) -> _Single:
        return self

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


class _List:
    """Iterable of records exposing ``.data()``, like neo4j's Result."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    def data(self) -> list[dict[str, Any]]:  # pragma: no cover - unused path
        raise AssertionError("data() must not be called on the list result")

    def __iter__(self):
        return iter(_Record(item) for item in self._items)


class _Record:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class _FakeDriver:
    def __init__(self, rows: list[dict[str, Any]], files: set[str]) -> None:
        self._session = _ScriptedSession(rows, files)

    def session(self, database: str) -> _ScriptedSession:
        return self._session


def _row(dependent: str, distance: int, confidences: list[float]) -> dict[str, Any]:
    return {
        "dependent_id": dependent,
        "dependent_path": dependent.split("::")[-1],
        "distance": distance,
        "edge_confidences": confidences,
        "chain": ["httpie/cli::other.py"] * distance,
    }


def test_analyze_builds_ordered_hops_with_decaying_scores() -> None:
    rows = [
        _row("httpie/cli::httpie/client.py", 1, [1.0]),
        _row("httpie/cli::httpie/core.py", 1, [0.9]),
        _row("httpie/cli::httpie/cli/argparser.py", 2, [0.95, 1.0]),
    ]
    analyzer = ChangeImpactAnalyzer(_FakeDriver(rows, {TARGET}))
    report = analyzer.analyze(TARGET, path_id="p-test")

    path = report.path
    assert path.hops[0].node_id == TARGET
    assert path.hops[0].step == 0
    # order: distance 1 first, then distance 2
    distances = [int(hop.rationale.split("distance ")[-1]) for hop in path.hops[1:]]
    assert distances == sorted(distances)
    # scores decay with distance and edge confidence
    direct_scores = [hop.score for hop in path.hops[1:3]]
    assert direct_scores[0] == 1.0
    assert 0.85 <= direct_scores[1] < 1.0
    distant = path.hops[3]
    assert distant.score < direct_scores[1]
    assert all(hop.role is EvidenceRole.BLAST_RADIUS for hop in path.hops[1:])
    assert path.is_valid is True
    assert path.total_hops == 4


def test_analyze_reports_direct_and_transitive_dependents() -> None:
    rows = [
        _row("httpie/cli::a.py", 1, [1.0]),
        _row("httpie/cli::b.py", 1, [1.0]),
        _row("httpie/cli::c.py", 3, [0.9, 0.9, 0.9]),
    ]
    report = ChangeImpactAnalyzer(_FakeDriver(rows, {TARGET})).analyze(TARGET)
    assert report.direct_dependents == ["httpie/cli::a.py", "httpie/cli::b.py"]
    assert report.transitive_dependents == 3
    assert report.max_depth_seen == 3


def test_analyze_on_empty_blast_radius_is_honest() -> None:
    analyzer = ChangeImpactAnalyzer(_FakeDriver([], {TARGET}))
    report = analyzer.analyze(TARGET)
    assert report.path.hops == [report.path.hops[0]]  # only the anchor
    assert report.transitive_dependents == 0
    assert report.path.is_valid is False
    assert any("empty blast radius" in note for note in report.path.validation_notes)


def test_analyze_rejects_unknown_files() -> None:
    analyzer = ChangeImpactAnalyzer(_FakeDriver([], set()))
    with pytest.raises(FileNotFoundError):
        analyzer.analyze("httpie/cli::missing.py")


def test_analyze_produces_evidence_nodes_sharing_the_path_id() -> None:
    rows = [_row("httpie/cli::a.py", 1, [1.0]),
            _row("httpie/cli::deep.py", 2, [0.9, 0.9])]
    report = ChangeImpactAnalyzer(_FakeDriver(rows, {TARGET})).analyze(TARGET,
                                                                       path_id="p-ev")
    assert len(report.evidences) == 3  # anchor + 2 dependents
    assert all(evidence.path_id == "p-ev" for evidence in report.evidences)
    kinds = [evidence.kind for evidence in report.evidences]
    assert kinds[0] is EvidenceKind.NODE
    assert kinds[1] is EvidenceKind.IMPORT_EDGE
    assert kinds[2] is EvidenceKind.GRAPH_PATH
    # hop indices line up with the path
    assert [evidence.hop_index for evidence in report.evidences] == [0, 1, 2]
    # every evidence references a File node
    for evidence in report.evidences:
        assert evidence.node_references[0].node_kind is NodeKind.FILE


def test_traversal_query_is_depth_bounded() -> None:
    from retrieval.impact import TRAVERSAL_QUERY

    assert "*1..%d" in TRAVERSAL_QUERY
    assert "LIMIT $limit" in TRAVERSAL_QUERY


def test_traversal_bound_is_validated_before_interpolation() -> None:
    analyzer = ChangeImpactAnalyzer(_FakeDriver([], {TARGET}))
    assert analyzer._validated_bound(7) == 7
    for bad in (0, 21, -1, "7", 7.0, True, None):
        with pytest.raises(ValueError, match="max_depth must be"):
            analyzer._validated_bound(bad)  # type: ignore[arg-type]
