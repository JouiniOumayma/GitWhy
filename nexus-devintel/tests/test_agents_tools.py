"""Tests de ``agents/tools.py`` — T2 : whitelist fermée et lecture seule.

Le test d'attaque niveau agent : un nom d'outil forgé (ou un nom scalaire
ressemblant à une écriture) doit être refusé **avant** tout dispatch, et
aucune requête d'écriture ne peut traverser le registre.
"""

import pytest

from agents import ToolNotFoundError, ToolRegistry, ToolResult


# --------------------------------------------------------------------------- #
# Fakes lecteurs (même forme que le driver neo4j dans les tests Phase 2)
# --------------------------------------------------------------------------- #
class _Record:
    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str):  # noqa: ANN001 - piggy-back du driver
        return self._data[key]

    def data(self) -> dict:
        return self._data


class _Result:
    def __init__(self, data: dict) -> None:
        self._single = _Record(data)

    def single(self) -> _Record:
        return self._single


class _Session:
    def __init__(self, queries: list[str], found: dict[str, int]) -> None:
        self._queries = queries
        self._found = found

    def run(self, query: str, **params):  # noqa: ANN001
        self._queries.append(query)
        file_id = params["file_id"] if "file_id" in params else params.get("incident_id")
        return _Result({"found": self._found.get(file_id, 0)})

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _FakeDriver:
    """Driver neo4j minimal : enregistre les requêtes, ne les *exécute* pas."""

    def __init__(self, found: dict[str, int] | None = None) -> None:
        self.queries: list[str] = []
        self._found = found or {}

    def session(self, database: str):  # noqa: ARG002, ANN001
        return _Session(self.queries, self._found)


# --------------------------------------------------------------------------- #
# Whitelist
# --------------------------------------------------------------------------- #
def test_whitelist_is_closed_and_read_only() -> None:
    registry = ToolRegistry()
    assert isinstance(registry.names, frozenset)
    expected = {
        "find_file_id", "find_incident_id",
        "analyze_impact", "analyze_root_cause", "hybrid_search",
        "get_file", "list_files", "get_commit", "get_pull_request", "search_issues",
    }
    assert registry.names == expected
    assert all(not name.startswith("write") for name in registry.names)


@pytest.mark.parametrize(
    "forged",
    [
        "merge_nodes",
        "delete_incident",
        "write_models",
        "create_evidence",
        "UnknownTool",
        "get_file ",          # padding SQL-like
        "get_file; DROP ALL",
        "",
    ],
)
def test_forged_tool_names_are_refused_before_any_dispatch(forged: str) -> None:
    """Aucun driver, aucun client GitHub : le refus doit preceder le dispatch."""
    registry = ToolRegistry(driver=_FakeDriver(), dsn="postgresql://fake")
    with pytest.raises(ToolNotFoundError):
        registry.call(forged)


def test_registry_with_no_stack_tool_result_is_documented_not_raised() -> None:
    """Sans driver ni DSN, l'outils renvoie ok=False: pas de preuve -> incident."""
    registry = ToolRegistry()
    for name, args in [
        ("analyze_impact", {"file_id": "httpie/cli::httpie/ssl_.py"}),
        ("analyze_root_cause", {"incident_id": "httpie/cli#issue-1583"}),
        ("hybrid_search", {"query": "ssl verify failed"}),
    ]:
        result = registry.call(name, **args)
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert "non configure" in (result.error or "")
        assert not result.evidence_available


# --------------------------------------------------------------------------- #
# Outils graphe
# --------------------------------------------------------------------------- #
def test_find_file_id_prefixes_the_repository(monkeypatch) -> None:
    driver = _FakeDriver(found={"httpie/cli::httpie/ssl_.py": 1})
    registry = ToolRegistry(driver=driver)
    result = registry.call("find_file_id", path_or_id="httpie/ssl_.py")
    assert result.ok is True
    assert result.value == {"file_id": "httpie/cli::httpie/ssl_.py"}
    # exactement des requetes de lecture: pas un MERGE/CREATE/SET/DELETE
    assert driver.queries
    for query in driver.queries:
        assert not any(kw in query.upper() for kw in ("MERGE", "CREATE", "DELETE", "SET ")), query


def test_find_file_id_unknown_file_reports_the_missing_evidence() -> None:
    registry = ToolRegistry(driver=_FakeDriver())  # rien n'existe
    result = registry.call("find_file_id", path_or_id="httpie/inconnu.py")
    assert result.ok is True
    assert result.value["file_id"] is None
    assert "n'est pas dans le graphe" in result.value["error"]


def test_analyze_impact_uses_the_phase2_analyzer_unchanged(monkeypatch) -> None:
    """L'outil enveloppe ChangeImpactAnalyzer, sans le reformaater."""
    registry = ToolRegistry(driver=_FakeDriver())

    captured: dict = {}

    class _FakeAnalyzer:
        def analyze(self, file_id, *, max_depth, limit):  # noqa: ANN001
            captured.update(file_id=file_id, max_depth=max_depth, limit=limit)
            return object()

    monkeypatch.setattr(registry, "_impact", _FakeAnalyzer())
    registry.call("analyze_impact", file_id="httpie/cli::httpie/ssl_.py", max_depth=5)
    assert captured == {"file_id": "httpie/cli::httpie/ssl_.py", "max_depth": 5, "limit": 50}


def test_analyze_root_cause_builds_the_incident_id() -> None:
    registry = ToolRegistry(driver=_FakeDriver(found={"httpie/cli#issue-1583": 1}))
    result = registry.call("find_incident_id", issue_number="1583")
    assert result.ok is True
    assert result.value == {"incident_id": "httpie/cli#issue-1583"}


# --------------------------------------------------------------------------- #
# Adaptateur MCP (le registre MCP lui-meme est deja clos, testee Phase 2)
# --------------------------------------------------------------------------- #
def test_mcp_tool_calls_pass_through_with_arguments(monkeypatch) -> None:
    seen: dict = {}

    def fake_call_tool(name, arguments, *, client=None):  # noqa: ANN001
        seen.update(name=name, arguments=arguments, client=client)
        return {"content": "<file contents>"}

    import ingestion.mcp_server as mcp

    monkeypatch.setattr(mcp, "call_tool", fake_call_tool)
    registry = ToolRegistry(github_client="UNLAZY-FAKE-CLIENT")
    result = registry.call("get_file", repo="httpie/cli", path="httpie/ssl_.py")
    assert result.ok is True
    assert result.value == {"content": "<file contents>"}
    assert seen == {
        "name": "get_file",
        "arguments": {"repo": "httpie/cli", "path": "httpie/ssl_.py"},
        "client": "UNLAZY-FAKE-CLIENT",
    }


def test_mcp_failure_surfaces_as_a_normal_tool_result(monkeypatch) -> None:
    def boom(name, arguments, *, client=None):  # noqa: ANN001
        raise RuntimeError("GitHub rate limited")

    import ingestion.mcp_server as mcp

    monkeypatch.setattr(mcp, "call_tool", boom)
    registry = ToolRegistry()
    result = registry.call("search_issues", repo="httpie/cli", query="x")
    assert result.ok is False
    assert "rate limited" in result.error
