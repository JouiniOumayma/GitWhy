"""Ingestion d'un dépôt GitHub, réutilisable par la CLI et par l'API.

Le même code sert deux entrées :

* ``scripts/ingest_repo.py`` — usage terminal ;
* ``POST /api/ingest`` — un utilisateur colle un lien dans l'UI.

Rien n'est inventé ici : l'arborescence vient de GitHub, les arêtes
``IMPORTS`` de l'analyse AST du vrai code, les chunks des corps réels
vérifiés par blob SHA.

``run_ingestion`` est bloquant (il fait du réseau, du CPU et de la base) :
l'appelant HTTP le lance dans un thread et suit sa progression via
``on_step``.
"""
from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent

StepCallback = Callable[[str, str, str], None]
"""``(nom_étape, statut, détail)`` — statut ∈ {running, done, failed}."""

#: Backend d'embeddings gardé en mémoire : MiniLM pèse ~90 Mo et met une
#: quarantaine de secondes à charger. Le recharger à chaque ingestion rendait
#: un dépôt minuscule plus lent qu'un gros. Le verrou protège deux jobs
#: concurrents (deux dépôts différents) sur le même modèle.
_EMBEDDING_BACKEND: Any = None
_EMBEDDING_LOCK = None


def _embedding_backend() -> Any:
    global _EMBEDDING_BACKEND, _EMBEDDING_LOCK
    import threading

    if _EMBEDDING_LOCK is None:
        _EMBEDDING_LOCK = threading.Lock()
    with _EMBEDDING_LOCK:
        if _EMBEDDING_BACKEND is None:
            from ingestion.embedding_indexer import MiniLMBackend

            _EMBEDDING_BACKEND = MiniLMBackend()
    return _EMBEDDING_BACKEND


# --------------------------------------------------------------------------- #
# Entrée utilisateur : lien GitHub ou owner/name
# --------------------------------------------------------------------------- #
def repo_id_from_input(raw: str) -> str:
    """``https://github.com/owner/name(.git)`` ou ``owner/name`` -> ``owner/name``."""
    value = (raw or "").strip().rstrip("/")
    if not value:
        raise ValueError("aucun dépôt fourni")
    lowered = value.lower()
    for prefix in ("https://github.com/", "http://github.com/", "github.com/"):
        if lowered.startswith(prefix):
            value = value[len(prefix):]
            break
    value = value.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"dépôt illisible : {raw!r} (attendu owner/name ou "
            "https://github.com/owner/name)"
        )
    return f"{parts[0]}/{parts[1]}"


def _parse_imports(relpath: str, source: str) -> list[Any]:
    """``fixtures/build_fixtures.py`` n'est pas un package : import explicite."""
    import sys

    fixtures_dir = str(PROJECT_ROOT / "fixtures")
    if fixtures_dir not in sys.path:
        sys.path.insert(0, fixtures_dir)
    import build_fixtures

    return build_fixtures.parse_imports(relpath, source)


def _module_name(relpath: str) -> str:
    import sys

    fixtures_dir = str(PROJECT_ROOT / "fixtures")
    if fixtures_dir not in sys.path:
        sys.path.insert(0, fixtures_dir)
    import build_fixtures

    return build_fixtures.module_name(relpath)


def _derived_provenance(detail: str, confidence: float) -> Any:
    import sys

    fixtures_dir = str(PROJECT_ROOT / "fixtures")
    if fixtures_dir not in sys.path:
        sys.path.insert(0, fixtures_dir)
    import build_fixtures

    return build_fixtures.derived_provenance(detail, confidence)


# --------------------------------------------------------------------------- #
# 1. Arborescence réelle -> (:File)
# --------------------------------------------------------------------------- #
def fetch_files(client: Any, repo_id: str) -> tuple[list[Any], bool]:
    """Tous les blobs du HEAD, convertis en modèles ``File``."""
    tree = client.git_tree(repo_id, ref="HEAD", recursive=True)
    entries = [e for e in tree.get("tree", []) if e.get("type") == "blob"]
    from ingestion import file_from_api

    files = [file_from_api(entry, repo_id, ref="HEAD") for entry in entries]
    return files, bool(tree.get("truncated"))


# --------------------------------------------------------------------------- #
# 2. IMPORTS par analyse AST du code réel
# --------------------------------------------------------------------------- #
def build_module_index(files: list[Any]) -> dict[str, str]:
    """``module pointé`` -> chemin dans le dépôt (``pkg/__init__.py`` -> ``pkg``).

    Les dépôts en « src layout » (``src/requests/models.py``) importent leur
    paquet sous son nom court (``requests.models``) : on enregistre donc aussi
    l'alias sans le préfixe ``src/``. C'est une convention de packaging
    standard, pas une supposition sur le contenu.
    """
    index: dict[str, str] = {}
    for file in files:
        if file.path.endswith(".py") and file.path.startswith("src/"):
            index.setdefault(_module_name(file.path[len("src/"):]), file.path)
    for file in files:
        if file.path.endswith(".py"):
            index.setdefault(_module_name(file.path), file.path)
    return index


def resolve_import(
    target_module: str, names: list[str], module_index: dict[str, str]
) -> str | None:
    """Résout un module importé vers un fichier du dépôt, sinon ``None``.

    ``from pkg import mod`` produit le module ``pkg`` : si ``pkg`` correspond à
    un ``__init__.py`` on garde cette arête, et sinon on cherche ``pkg.mod``
    (sous-module réellement référencé). Tout ce qui sort du dépôt est ignoré.
    """
    direct = module_index.get(target_module)
    if direct is not None:
        return direct
    for name in names:
        nested = module_index.get(f"{target_module}.{name}")
        if nested is not None:
            return nested
    return None


def build_imports(
    file: Any, source: str, module_index: dict[str, str]
) -> list[Any]:
    """Un ``FileImport`` par couple (importeur, importé), lignes conservées."""
    from models import make_file_id
    from models.enums import ImportKind
    from models.file import FileImport

    parsed = _parse_imports(file.path, source)
    grouped: dict[str, dict[str, Any]] = {}
    for target_module, line, kind, names, is_relative, type_checking_only in parsed:
        target_path = resolve_import(target_module, names, module_index)
        if target_path is None or target_path == file.path:
            continue
        entry = grouped.setdefault(
            target_path,
            {"names": set(), "lines": [], "kind": kind,
             "relative": False, "tc": True},
        )
        entry["names"].update(names)
        entry["lines"].append(line)
        entry["relative"] = entry["relative"] or is_relative
        # « type-checking seulement » uniquement si CHAQUE occurrence l'est
        entry["tc"] = entry["tc"] and type_checking_only

    imports: list[Any] = []
    for target_path in sorted(grouped):
        entry = grouped[target_path]
        lines = sorted(set(entry["lines"]))
        type_checking_only = bool(entry["tc"])
        confidence = 0.9 if type_checking_only else 1.0
        imports.append(
            FileImport(
                target_file_id=make_file_id(file.repository_id, target_path),
                target_path=target_path,
                imported_names=sorted(entry["names"]),
                line=lines[0],
                import_lines=lines,
                kind=ImportKind(entry["kind"]),
                is_relative=bool(entry["relative"]),
                is_type_checking_only=type_checking_only,
                confidence=confidence,
                provenance=_derived_provenance(
                    f"ast:{file.path}:{','.join(str(line) for line in lines)}",
                    confidence,
                ),
            )
        )
    return imports


def attach_imports(
    files: list[Any], provider: Any, module_index: dict[str, str]
) -> tuple[int, int]:
    """Remplit ``File.imports`` depuis le code réel.

    Renvoie ``(arêtes_créées, fichiers_python_illisibles)``.
    """
    total = 0
    unreadable = 0
    for file in files:
        if not file.path.endswith(".py"):
            continue
        source = provider(file)
        if not source or not source.strip():
            unreadable += 1
            continue
        imports = build_imports(file, source, module_index)
        file.imports = imports
        total += len(imports)
    return total, unreadable


# --------------------------------------------------------------------------- #
# 3. Écriture du graphe
# --------------------------------------------------------------------------- #
def write_graph(files: list[Any], repo_id: str) -> dict[str, int]:
    """MERGE l'arbre + les arêtes dans Neo4j (idempotent)."""
    from neo4j import GraphDatabase

    from ingestion import GraphWriter
    from models import Repository

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "user"),
    )
    owner, name = repo_id.split("/")
    repo_node = Repository(id=repo_id, owner=owner, name=name)

    counts: dict[str, int] = {"files": 0, "imports": 0, "contains": 0}
    with GraphDatabase.driver(uri, auth=auth) as driver:
        writer = GraphWriter(driver, os.environ.get("NEO4J_DATABASE", "neo4j"))
        # Passe 1 : tous les nœuds. Passe 2 : toutes les arêtes.
        # ``write_edge`` utilise MATCH sur ses deux extrémités : écrire les
        # arêtes fichier par fichier perdrait celles qui pointent vers un
        # fichier pas encore créé (constaté : 39 IMPORTS sur 105).
        writer.write_node(repo_node)
        for file in files:
            writer.write_node(file)
            counts["files"] += 1
        for file in files:
            for edge in file.edges():
                if not writer.write_edge(edge):
                    continue
                if edge.type.value == "IMPORTS":
                    counts["imports"] += 1
                else:
                    counts["contains"] += 1
    return counts


# --------------------------------------------------------------------------- #
# 3 bis. Historique git : commits, fichiers modifiés, pull requests
# --------------------------------------------------------------------------- #
def fetch_commits(
    client: Any,
    repo_id: str,
    max_commits: int = 150,
) -> tuple[list[Any], int]:
    """Les ``max_commits`` derniers commits, **avec leurs fichiers modifiés**.

    La liste ``/commits`` ne porte pas la liste des fichiers : il faut relire
    chaque commit (1 requête) pour obtenir le payload ``:MODIFIES``. C'est ce
    qui rend l'historique utile au GraphRAG (« qui a touché ce fichier ? »),
    donc on paie la requête par commit.
    """
    from ingestion import commit_from_api

    shas: list[str] = []
    for shallow in client.commits(repo_id):
        if len(shas) >= max_commits:
            break
        sha = shallow.get("sha")
        if sha:
            shas.append(sha)

    def detail(sha: str) -> Any:
        """Un commit complet, ou ``None`` s'il est illisible."""
        try:
            payload, url = client.commit(repo_id, sha)
            return commit_from_api(payload, repo_id, url)
        except Exception:  # noqa: BLE001 - un commit raté ne bloque pas le lot
            return None

    # 150 commits séquentiels coûtaient ~5 minutes : la latence GitHub dominait.
    # L'ordre de retour de ``map`` suit celui des shas, donc l'ingestion reste
    # reproductible.
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(detail, shas))
    commits = [commit for commit in results if commit is not None]
    return commits, len(shas) - len(commits)


def fetch_pull_requests(
    client: Any,
    repo_id: str,
    max_prs: int = 100,
) -> list[Any]:
    """Les ``max_prs`` dernières pull requests (tous états)."""
    from ingestion import pull_request_from_api

    prs: list[Any] = []
    for payload in client.pull_requests(repo_id, state="all"):
        if len(prs) >= max_prs:
            break
        number = payload.get("number")
        if number is None:
            continue
        prs.append(
            pull_request_from_api(
                payload,
                repo_id,
                f"https://api.github.com/repos/{repo_id}/pulls/{number}",
            )
        )
    return prs


def write_history(
    commits: list[Any],
    prs: list[Any],
    repo_id: str,
) -> dict[str, int]:
    """MERGE commits et PRs ; renvoie les arêtes réellement écrites.

    Une arête dont une extrémité manque (fichier supprimé, commit de merge
    hors fenêtre, issue jamais ingérée) est ignorée par ``write_edge`` : on
    compte ces sauts au lieu de les masquer.
    """
    from neo4j import GraphDatabase

    from ingestion import GraphWriter
    from models import Repository

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "user"),
    )
    owner, name = repo_id.split("/")
    counts = {
        "commits": 0,
        "modifies": 0,
        "prs": 0,
        "merged_into": 0,
        "opened": 0,
        "skipped": 0,
    }
    with GraphDatabase.driver(uri, auth=auth) as driver:
        writer = GraphWriter(driver, os.environ.get("NEO4J_DATABASE", "neo4j"))
        # ``HAS_COMMIT`` part du dépôt : il doit exister avant les commits.
        writer.write_node(Repository(id=repo_id, owner=owner, name=name))
        for commit in commits:
            stats = writer.write_model(commit)
            counts["commits"] += 1
            counts["modifies"] += int(stats.get("rel:MODIFIES", 0))
            counts["skipped"] += int(stats.get("rel_skipped_unresolved", 0))
        for pr in prs:
            stats = writer.write_model(pr)
            counts["prs"] += 1
            counts["merged_into"] += int(stats.get("rel:MERGED_INTO", 0))
            counts["opened"] += int(stats.get("rel:OPENED", 0))
            counts["skipped"] += int(stats.get("rel_skipped_unresolved", 0))
    return counts


# --------------------------------------------------------------------------- #
# 4. Chunks + embeddings
# --------------------------------------------------------------------------- #
def index_vectors(
    files: list[Any], repo_id: str, provider: Any, *, dsn: str | None
) -> dict[str, Any] | None:
    """Chunke et embedde le contenu réel dans pgvector."""
    if not dsn:
        return None
    from ingestion.embedding_indexer import EmbeddingIndexer

    indexer = EmbeddingIndexer(dsn, repository_id=repo_id)
    # Dossier de fixtures vide : on n'indexe QUE ce dépôt, jamais les
    # fixtures httpie/cli qui vivent dans le dépôt du projet.
    empty_fixtures = Path(tempfile.mkdtemp(prefix="nexus-ingest-"))
    indexer._backend = _embedding_backend()  # noqa: SLF001 - même backend que l'existant
    report = indexer.index_fixtures(
        empty_fixtures,
        content_provider=provider,
        extra_files=files,
    )
    return {
        "chunks": report.chunk_count,
        "embedded": report.embedded_count,
        "written": report.written_count,
        "skipped": report.skipped_hashes,
        "run_id": report.run_id,
    }


# --------------------------------------------------------------------------- #
# 5. Vérification : relire les compteurs dans les bases
# --------------------------------------------------------------------------- #
def verify(repo_id: str, *, dsn: str | None) -> dict[str, Any]:
    """Compteurs relus depuis Neo4j et pgvector (preuve de l'ingestion)."""
    from neo4j import GraphDatabase

    owner, name = repo_id.split("/")
    result: dict[str, Any] = {}
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "user"),
    )
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
            record = session.run(
                """
                MATCH (r:Repository {owner: $owner, name: $name})
                OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
                RETURN count(DISTINCT f) AS files
                """,
                owner=owner,
                name=name,
            ).single()
            result["files_in_graph"] = int(record["files"] or 0) if record else 0
            # Les IMPORTS sont comptés à part, sur le périmètre du dépôt : les
            # agréger dans la requête ci-dessus produirait un produit
            # cartésien (chaque fichier x chaque arête du graphe).
            scoped = session.run(
                """
                MATCH (r:Repository {owner: $owner, name: $name})-[:CONTAINS]->(f:File)
                MATCH (f)-[i:IMPORTS]->(g:File)
                RETURN count(i) AS imports, count(DISTINCT g) AS imported_targets
                """,
                owner=owner,
                name=name,
            ).single()
            result["imports_in_graph"] = int(scoped["imports"] or 0) if scoped else 0
            result["import_targets"] = (
                int(scoped["imported_targets"] or 0) if scoped else 0
            )
            # Historique : commits reliés au dépôt, et arêtes MODIFIES réellement
            # présentes (une arête sautée faute de fichier ne compte pas).
            history = session.run(
                """
                MATCH (r:Repository {owner: $owner, name: $name})
                OPTIONAL MATCH (r)-[:HAS_COMMIT]->(c:Commit)
                OPTIONAL MATCH (c)-[m:MODIFIES]->(:File)
                RETURN count(DISTINCT c) AS commits, count(m) AS modifies
                """,
                owner=owner,
                name=name,
            ).single()
            result["commits_in_graph"] = int(history["commits"] or 0) if history else 0
            result["modifies_in_graph"] = int(history["modifies"] or 0) if history else 0
            pulls = session.run(
                """
                MATCH (p:PR {repository_id: $owner + '/' + $name})
                RETURN count(p) AS prs
                """,
                owner=owner,
                name=name,
            ).single()
            result["prs_in_graph"] = int(pulls["prs"] or 0) if pulls else 0
    if dsn:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM code_chunks WHERE repository_id = %s",
                    (repo_id,),
                )
                row = cur.fetchone()
                result["chunks_in_postgres"] = int(row[0]) if row else 0
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_ingestion(
    repo_id: str,
    *,
    skip_vectors: bool = False,
    skip_history: bool = False,
    max_commits: int = 150,
    max_prs: int = 100,
    on_step: StepCallback | None = None,
) -> dict[str, Any]:
    """Ingère ``repo_id`` de bout en bout et renvoie les compteurs relus.

    Lève l'exception d'origine en cas d'échec ; ``on_step`` est appelé à
    chaque transition pour que l'UI puisse afficher où en est le travail.
    """
    from ingestion import GitHubClient
    from ingestion.content import GitHubBlobContentProvider

    def step(name: str, status: str, detail: str = "") -> None:
        if on_step is not None:
            on_step(name, status, detail)

    dsn = os.environ.get("POSTGRES_DSN")
    started = time.monotonic()

    step("tree", "running", "arborescence GitHub du HEAD")
    client = GitHubClient()
    files, truncated = fetch_files(client, repo_id)
    step("tree", "done", f"{len(files)} fichiers" + (" (arborescence tronquée)" if truncated else ""))

    provider = GitHubBlobContentProvider(
        client, repo_id, cache_dir=PROJECT_ROOT / ".blob_cache"
    )
    module_index = build_module_index(files)
    step("imports", "running", "analyse AST des imports")
    import_total, unreadable = attach_imports(files, provider, module_index)
    step(
        "imports",
        "done",
        f"{import_total} arêtes IMPORTS sur {len(module_index)} modules"
        + (f" ({unreadable} fichier(s) illisible(s))" if unreadable else ""),
    )

    step("graph", "running", "écriture Neo4j")
    graph = write_graph(files, repo_id)
    step("graph", "done", f"{graph['files']} File, {graph['imports']} IMPORTS")

    history: dict[str, int] | None = None
    pulls: dict[str, int] | None = None
    if skip_history:
        step("history", "done", "ignoré (skip_history)")
        step("pulls", "done", "ignoré (skip_history)")
    else:
        step("history", "running", f"commits GitHub (max {max_commits})")
        commits, unreadable_commits = fetch_commits(client, repo_id, max_commits)
        history = write_history(commits, [], repo_id)
        step(
            "history",
            "done",
            f"{history['commits']} commits, {history['modifies']} MODIFIES"
            + (f" ({unreadable_commits} illisible(s))" if unreadable_commits else "")
            + (f", {history['skipped']} arête(s) sans extrémité" if history['skipped'] else ""),
        )
        step("pulls", "running", f"pull requests (max {max_prs})")
        prs = fetch_pull_requests(client, repo_id, max_prs)
        pulls = write_history([], prs, repo_id)
        step(
            "pulls",
            "done",
            f"{pulls['prs']} PRs, {pulls['merged_into']} MERGED_INTO"
            + (f", {pulls['skipped']} arête(s) sans extrémité" if pulls['skipped'] else ""),
        )

    vectors: dict[str, Any] | None = None
    if skip_vectors:
        step("vectors", "done", "ignoré (skip_vectors)")
    else:
        step("vectors", "running", "chunks + embeddings MiniLM")
        vectors = index_vectors(files, repo_id, provider, dsn=dsn)
        step(
            "vectors",
            "done",
            f"{vectors['chunks']} chunks ({vectors['embedded']} embeddés, "
            f"{vectors['skipped']} déjà à jour)" if vectors
            else "PostgreSQL indisponible",
        )

    step("verify", "running", "relecture des compteurs")
    proof = verify(repo_id, dsn=None if skip_vectors else dsn)
    step("verify", "done", ", ".join(f"{k}={v}" for k, v in proof.items()))

    return {
        "repository": repo_id,
        "files": graph["files"],
        "imports": graph["imports"],
        "history": history,
        "pulls": pulls,
        "vectors": vectors,
        "proof": proof,
        "duration_s": round(time.monotonic() - started, 1),
    }
