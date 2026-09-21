#!/usr/bin/env python3
"""Build the NEXUS-DevIntel JSON fixtures.

The fixtures are *grounded on real data*: every File, Commit, Deployment and
issue reference is harvested from the actual ``httpie/cli`` clone, so the ids,
paths, line numbers, tags and SHAs are real. Only the narrative fields of the
Incidents/PRs (titles, bodies, severities, root-cause attribution) are
synthetic, and they are tagged with ``provenance.source == "synthetic_fixture"``
so nobody mistakes them for ingested GitHub data.

Regenerate with::

    python fixtures/build_fixtures.py --repo-path ../httpie-cli

Why a generator instead of hand-written JSON? Because a Knowledge Graph fixture
must be *referentially closed*: a ``FileChange.file_id`` that points at a
missing File is a silent bug that only shows up as a missing edge three weeks
later. Building the fixtures through the Pydantic models makes the id scheme and
the enums impossible to typo.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    Answer,
    Citation,
    Commit,
    ConfidenceBand,
    Deployment,
    DetectionSource,
    Environment,
    Evidence,
    EvidenceHop,
    EvidenceKind,
    EvidencePath,
    EvidenceRef,
    EvidenceRole,
    File,
    FileChange,
    FileImpact,
    FileImport,
    Incident,
    IncidentCause,
    IncidentFileLink,
    IncidentSeverity,
    IncidentStatus,
    ImportKind,
    IssueLink,
    IssueRef,
    LinkMethod,
    MergeStrategy,
    NodeKind,
    PRState,
    PersonRef,
    Provenance,
    PullRequest,
    Repository,
    RepositoryAuditMetrics,
    RetrievalStrategy,
    SourceKind,
    make_deployment_id,
    make_file_id,
    make_incident_id,
    make_pr_id,
)

REPO_ID = "httpie/cli"
UPSTREAM_REPO_ID = "psf/requests"
INGESTED_AT = datetime(2026, 9, 18, tzinfo=timezone.utc)

# --------------------------------------------------------------------------- #
# Curated inventory (all SHAs verified against the clone)
# --------------------------------------------------------------------------- #
CURATED_COMMITS = [
    "5b604c37c6c67e18e7c3e9aee6c88a8c22b98345",  # HEAD, squash of PR #1611
    "fd30c4ef6230a927f9dcfad6301c40e8bf846156",  # PR #1596, real fix for #1583
    "7f03c52d2237440c5a672296ce6955aae4ed4f09",  # tag 3.2.3, workaround for #1583
    "2105caa49bae87c5809c274e407619a0de2639d1",  # tag 3.2.4, release commit
    "3c07a2532647f64a4aaa7c729557fee5dcc2e182",  # PR #1029, closes #998
    "8f83bfe7679c56843452cbd255adfd4f6d582c89",  # PR #1040, fixes #1039
    "cae83b3f9e2077afee85992e8135cae70bb91cf3",  # closes #761
    "e73c3e6c249b89496b4f81fa20bb449911da79f1",  # closes #1461 + #1467
    "29de4ce1151936856cda417767f8c507fc2c201c",  # tag 3.2.2
    "2142ae60c339144697cf93f40d1e740be344224d",  # PR #1387, tag 3.2.0
    "266c6375c6d687c3aad07962cb3f5bef7d39bbbf",  # PR #1313, tag 3.1.0
    "88140422a9d6585a7edfb2c265ebed5d0736df2c",  # PR #1272, tag 3.0.0
]

#: Tags -> the commit they point at (harvested, not guessed).
CURATED_TAGS = ["3.2.4", "3.2.3", "3.2.2", "3.2.0", "3.1.0", "3.0.0"]

#: How many ancestors to pull in per curated commit, so that (:Commit)-[:PARENT_OF]
#: is not an empty relation in the fixture graph. 1 = a linear frontier.
ANCESTOR_DEPTH = 1

#: Deployment window whose **entire** first-parent chain is ingested.
#:
#: Without this, the fixture graph contains the two tagged commits but not the
#: commits between them, so `(fix)-[:PARENT_OF*]->(tagged_commit)` returns nothing
#: and "which release shipped this fix?" cannot be answered at all — the demo's
#: central question. 10 commits for 3.2.3 -> 3.2.4 is a cheap price for a
#: connected temporal axis. Add more windows here as the demos grow.
CONNECTED_WINDOW = ("3.2.3", "3.2.4")
MAX_WINDOW_HOPS = 60

#: Non-Python files touched by the curated commits, so every :MODIFIES resolves.
EXTRA_FILES = [
    "setup.cfg",
    "CHANGELOG.md",
    "extras/httpie-completion.fish",
]

#: ``(#N)`` at the end of a squash-merged subject.
SQUASH_RE = re.compile(r"\(#(\d+)\)\s*$")
BARE_NUMBER_RE = re.compile(r"\(#(\d+)\)\s*$")
CLOSING_RE = re.compile(
    r"\b(fix(?:e[sd])?|close[sd]?|resolve[sd]?)\b\s*:?\s*#(\d+)", re.IGNORECASE
)
# NOTE: bare "(#N)" cross-references in commit subjects are deliberately NOT turned
# into :REFERENCES edges. GitHub shares one numbering space between issues and pull
# requests, so a subject like "... (#1583) (#1596)" is ambiguous: #1596 is a PR.
# Week 2 resolves this properly with the GraphQL closingIssuesReferences field.
RELEASE_SUBJECT_RE = re.compile(r"^(v?\d+\.\d+\.\d+|final release prep|release prep|.*release prep.*)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()}")
    return proc.stdout


def git_provenance(source_uri: str, confidence: float = 1.0) -> Provenance:
    return Provenance(
        source=SourceKind.GIT,
        source_uri=source_uri,
        ingested_at=INGESTED_AT,
        extractor="nexus-devintel.fixtures.build_fixtures",
        confidence=confidence,
    )


def derived_provenance(detail: str, confidence: float = 0.95) -> Provenance:
    return Provenance(
        source=SourceKind.DERIVED,
        source_uri=detail,
        ingested_at=INGESTED_AT,
        extractor="nexus-devintel.fixtures.build_fixtures",
        confidence=confidence,
    )


def synthetic_provenance(detail: str, confidence: float = 0.5) -> Provenance:
    return Provenance(
        source=SourceKind.SYNTHETIC,
        source_uri=detail,
        ingested_at=INGESTED_AT,
        extractor="nexus-devintel.fixtures.build_fixtures",
        confidence=confidence,
    )


# --------------------------------------------------------------------------- #
# Files + import graph
# --------------------------------------------------------------------------- #
def module_name(relpath: str) -> str:
    mod = relpath[:-3]
    parts = mod.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def parse_imports(relpath: str, source: str) -> list[tuple[str, int, str, list[str], bool, bool]]:
    """Return ``(target_module, line, kind, names, is_relative, type_checking_only)``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    out: list[tuple[str, int, str, list[str], bool, bool]] = []
    type_checking_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            if "TYPE_CHECKING" in test:
                for inner in ast.walk(node):
                    if isinstance(inner, (ast.Import, ast.ImportFrom)):
                        type_checking_lines.add(inner.lineno)

    is_init = relpath.endswith("__init__.py")
    cur_mod = module_name(relpath)
    cur_pkg = cur_mod if is_init else (cur_mod.rsplit(".", 1)[0] if "." in cur_mod else "")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(
                    (alias.name, node.lineno, ImportKind.IMPORT.value, [], False,
                     node.lineno in type_checking_lines)
                )
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            tc = node.lineno in type_checking_lines
            if node.level:
                parts = [p for p in cur_pkg.split(".") if p]
                base = parts[: max(0, len(parts) - (node.level - 1))]
                prefix = ".".join(base)
                target = f"{prefix}.{node.module}" if node.module else prefix
                out.append((target, node.lineno, ImportKind.RELATIVE.value, names, True, tc))
            elif node.module:
                out.append((node.module, node.lineno, ImportKind.FROM.value, names, False, tc))
    return out


def harvest_files(repo: Path, commit_shas: list[str]) -> tuple[list[File], dict[str, set[str]]]:
    """Build every File node + return the module -> path index."""
    paths: set[str] = set()
    source_dir = repo / "httpie"
    for py in source_dir.rglob("*.py"):
        rel = py.relative_to(repo).as_posix()
        if "/tests/" in rel or rel.startswith("tests/"):
            continue
        paths.add(rel)
    for extra in EXTRA_FILES:
        if (repo / extra).exists():
            paths.add(extra)
        else:
            print(f"  ! missing extra file in clone: {extra}")
    # every file touched by a curated commit must exist as a node
    for sha in commit_shas:
        for line in git(repo, "show", "--name-only", "--format=", sha).splitlines():
            line = line.strip()
            if line and (repo / line).exists():
                paths.add(line)

    module_to_path = {module_name(p): p for p in paths if p.endswith(".py")}

    # last modification map in a single git pass
    last_modified: dict[str, tuple[str, str]] = {}
    raw = git(repo, "log", "--format=%x1e%H%x1f%cI", "--name-only")
    for record in raw.split("\x1e"):
        record = record.strip()
        if not record:
            continue
        head, _, body = record.partition("\n")
        sha, _, cdate = head.partition("\x1f")
        for name in body.splitlines():
            name = name.strip()
            if name and name not in last_modified:  # git log is newest-first
                last_modified[name] = (sha, cdate)

    files: list[File] = []
    for rel in sorted(paths):
        full = repo / rel
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        is_python = rel.endswith(".py")
        imports: list[FileImport] = []
        if is_python:
            imports = _aggregate_imports(rel, parse_imports(rel, text), module_to_path)
        sha, cdate = last_modified.get(rel, (None, None))
        parts = rel.split("/")
        files.append(
            File(
                id=make_file_id(REPO_ID, rel),
                repository_id=REPO_ID,
                path=rel,
                module_path=module_name(rel) if is_python else None,
                package="/".join(parts[:-1]) or None,
                top_package=parts[0] if len(parts) > 1 else None,
                extension=full.suffix or None,
                language="Python" if is_python else _language_for(full.suffix),
                is_python=is_python,
                is_test="/test" in rel or parts[0] == "tests",
                is_package_init=rel.endswith("__init__.py"),
                loc=len(text.splitlines()),
                size_bytes=full.stat().st_size,
                content_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
                blob_sha=git(repo, "rev-parse", f"HEAD:{rel}").strip(),
                last_modified_commit_id=sha,
                last_modified_at=datetime.fromisoformat(cdate) if cdate else None,
                imports=imports,
                provenance=git_provenance(f"git:HEAD:{rel}"),
            )
        )
    return files, module_to_path


def _aggregate_imports(
    relpath: str,
    parsed: list[tuple[str, int, str, list[str], bool, bool]],
    module_to_path: dict[str, str],
) -> list[FileImport]:
    """Collapse the parsed statements into one FileImport per (importer, imported) pair.

    The graph wants one ``:IMPORTS`` edge per file pair (that is the unit of a
    change-impact hop), while the source may import the same module on several
    lines. All lines are kept in ``import_lines``.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for target, line, kind, names, is_relative, tc in parsed:
        target_path = module_to_path.get(target)
        if target_path is None or target_path == relpath:
            continue  # outside the fixture subset / self import
        entry = grouped.setdefault(
            target_path,
            {"names": set(), "lines": [], "kind": kind, "relative": False, "tc": True},
        )
        entry["names"].update(names)
        entry["lines"].append(line)
        entry["relative"] = entry["relative"] or is_relative
        # a target is "TYPE_CHECKING only" only if every occurrence is guarded
        entry["tc"] = entry["tc"] and tc

    imports: list[FileImport] = []
    for target_path, entry in grouped.items():
        lines = sorted(set(entry["lines"]))
        tc = bool(entry["tc"])
        imports.append(
            FileImport(
                target_file_id=make_file_id(REPO_ID, target_path),
                target_path=target_path,
                imported_names=sorted(entry["names"]),
                line=lines[0],
                import_lines=lines,
                kind=ImportKind(entry["kind"]),
                is_relative=bool(entry["relative"]),
                is_type_checking_only=tc,
                confidence=0.9 if tc else 1.0,
                provenance=derived_provenance(
                    f"ast:{relpath}:{','.join(str(line) for line in lines)}",
                    0.9 if tc else 1.0,
                ),
            )
        )
    return sorted(imports, key=lambda imp: imp.target_path)


def _language_for(suffix: str) -> str:
    return {
        ".py": "Python", ".cfg": "INI", ".ini": "INI", ".md": "Markdown",
        ".fish": "Fish", ".1": "Roff", ".rst": "reStructuredText",
    }.get(suffix.lower(), "Text")


def compute_impact(files: list[File]) -> None:
    """Fill ``File.impact`` with reverse-reachability metrics over :IMPORTS."""
    reverse: dict[str, set[str]] = defaultdict(set)
    for f in files:
        for imp in f.imports:
            reverse[imp.target_file_id].add(f.id)

    for f in files:
        seen = {f.id}
        frontier = [f.id]
        depth = 0
        while frontier:
            nxt: list[str] = []
            for node in frontier:
                for dep in reverse.get(node, ()):
                    if dep not in seen:
                        seen.add(dep)
                        nxt.append(dep)
            if nxt:
                depth += 1
            frontier = nxt
        dependents = sorted(seen - {f.id})
        f.impact = FileImpact(
            direct_dependents=len(reverse.get(f.id, ())),
            transitive_dependents=len(dependents),
            max_depth=depth,
            dependent_file_ids=dependents[:25],
            is_truncated=len(dependents) > 25,
            computed_at=INGESTED_AT,
        )


# --------------------------------------------------------------------------- #
# Commits
# --------------------------------------------------------------------------- #
def expand_commit_set(repo: Path, seeds: list[str], depth: int = ANCESTOR_DEPTH) -> list[str]:
    """Curated commits plus ``depth`` generations of first-parent ancestors.

    The ancestors are *context* nodes: they exist so the temporal axis
    (``:PARENT_OF``) and the "what shipped between two tags" traversal are
explorable in the demo graph.
    """
    ordered = list(seeds)
    frontier = list(seeds)
    for _ in range(depth):
        next_frontier: list[str] = []
        for sha in frontier:
            # %P (capital) = full parent SHAs; %p would give abbreviated ones and
            # silently break the membership test in harvest_commits().
            parents = git(repo, "show", "-s", "--format=%P", sha).split()
            if not parents:
                continue  # root commit, stop
            if parents[0] not in ordered:
                ordered.append(parents[0])
                next_frontier.append(parents[0])
        frontier = next_frontier
    return ordered


def connect_window(repo: Path, oldest_tag: str, newest_tag: str) -> list[str]:
    """Every first-parent commit between two tags, newest first.

    Follows the first parent only: that is the line the tags actually live on, and
    it keeps side branches (where footnotes live) out of the fixture budget.
    """
    oldest = git(repo, "log", "-1", "--format=%H", f"{oldest_tag}^{{commit}}").strip()
    newest = git(repo, "log", "-1", "--format=%H", f"{newest_tag}^{{commit}}").strip()
    chain: list[str] = []
    sha: str | None = newest
    hops = 0
    while sha and hops <= MAX_WINDOW_HOPS:
        if sha == oldest:
            return chain  # the window is fully connected
        if sha not in chain:
            chain.append(sha)
        parents = git(repo, "show", "-s", "--format=%P", sha).split()
        sha = parents[0] if parents else None
        hops += 1
    print(
        f"  ! could not connect {oldest_tag} -> {newest_tag} within "
        f"{MAX_WINDOW_HOPS} hops: the window will not be traversable"
    )
    return chain


def harvest_commits(
    repo: Path, commit_set: list[str], known_file_paths: set[str]
) -> list[Commit]:
    commits: list[Commit] = []
    for sha in commit_set:
        meta = git(repo, "show", "-s", "--format=%H%x1f%an%x1f%ae%x1f%aI%x1f%cn%x1f%ce%x1f%cI%x1f%P%x1f%s", sha)
        (full, an, ae, adate, cn, ce, cdate, parents, subject) = meta.strip().split("\x1f")
        body = git(repo, "show", "-s", "--format=%b", sha).strip() or None

        changes: list[FileChange] = []
        for line in git(repo, "show", "--numstat", "--format=", sha).splitlines():
            cols = line.split("\t")
            if len(cols) != 3:
                continue
            added, deleted, path = cols
            if path not in known_file_paths:
                continue
            changes.append(
                FileChange(
                    file_id=make_file_id(REPO_ID, path),
                    path=path,
                    change_type="added" if added != "0" and deleted == "0" else "modified",
                    additions=int(added) if added.isdigit() else 0,
                    deletions=int(deleted) if deleted.isdigit() else 0,
                )
            )

        tags = [t for t in git(repo, "tag", "--points-at", sha).split() if t]
        squash = SQUASH_RE.search(subject)
        pr_number = int(squash.group(1)) if squash else None

        commits.append(
            Commit(
                id=full,
                repository_id=REPO_ID,
                subject=subject,
                body=body,
                is_merge=len(parents.split()) > 1,
                is_release_commit=bool(RELEASE_SUBJECT_RE.match(subject)),
                branch="master",
                authored_at=datetime.fromisoformat(adate),
                committed_at=datetime.fromisoformat(cdate),
                author=PersonRef(id=ae.strip().lower(), email=ae.strip().lower(), name=an),
                committer=PersonRef(id=ce.strip().lower(), email=ce.strip().lower(), name=cn),
                parents=[p for p in parents.split() if p in commit_set],
                files_changed=changes,
                changed_files_count=len(changes),
                additions=sum(c.additions for c in changes) or None,
                deletions=sum(c.deletions for c in changes) or None,
                tags=tags,
                issue_refs=_issue_refs(subject, body or "", f"git:{full[:7]}"),
                involves_pull_request=pr_number is not None,
                pull_request_number=pr_number,
                provenance=git_provenance(f"git:{full}"),
            )
        )
    return commits


def _issue_refs(subject: str, body: str, source_uri: str) -> list[IssueRef]:
    """Parse ``Closes/Fixes #N`` (body first) and cross-references in the subject."""
    refs: list[IssueRef] = []
    for keyword, number in CLOSING_RE.findall(body):
        refs.append(
            IssueRef(
                incident_id=make_incident_id(REPO_ID, int(number)),
                issue_number=int(number),
                relation="CLOSES",
                method=LinkMethod.COMMIT_MESSAGE,
                confidence=0.95,
                referenced_in="body",
                evidence_text=f"{keyword.title()} #{number}",
                provenance=git_provenance(source_uri, 0.95),
            )
        )
    for keyword, number in CLOSING_RE.findall(subject):
        refs.append(
            IssueRef(
                incident_id=make_incident_id(REPO_ID, int(number)),
                issue_number=int(number),
                relation="CLOSES",
                method=LinkMethod.COMMIT_MESSAGE,
                confidence=0.9,
                referenced_in="subject",
                evidence_text=f"{keyword.title()} #{number}",
                provenance=git_provenance(source_uri, 0.9),
            )
        )
    return refs


# --------------------------------------------------------------------------- #
# Deployments
# --------------------------------------------------------------------------- #
def harvest_deployments(repo: Path) -> list[Deployment]:
    deployments: list[Deployment] = []
    entries = []
    for tag in CURATED_TAGS:
        sha = git(repo, "log", "-1", "--format=%H", f"{tag}^{{commit}}").strip()
        date = git(repo, "log", "-1", "--format=%cI", f"{tag}^{{commit}}").strip()
        entries.append((tag, sha, datetime.fromisoformat(date)))
    entries.sort(key=lambda e: e[2])

    for index, (tag, sha, date) in enumerate(entries):
        previous = entries[index - 1] if index else None
        delta = None
        days = None
        if previous:
            delta = len(git(repo, "rev-list", f"{previous[0]}..{tag}", "--no-merges").splitlines())
            days = round((date - previous[2]).total_seconds() / 86400, 1)
        deployments.append(
            Deployment(
                id=make_deployment_id(REPO_ID, tag),
                repository_id=REPO_ID,
                tag=tag,
                name=f"HTTPie {tag}",
                environment=Environment.PRODUCTION,
                url=f"https://github.com/httpie/cli/releases/tag/{tag}",
                commit_id=sha,
                created_at=date,
                published_at=date,
                is_head=sha == CURATED_COMMITS[0],
                changelog_url="https://github.com/httpie/cli/blob/master/CHANGELOG.md",
                previous_deployment_id=make_deployment_id(REPO_ID, previous[0]) if previous else None,
                previous_tag=previous[0] if previous else None,
                commit_delta=delta,
                days_since_previous=days,
                provenance=git_provenance(f"git:refs/tags/{tag}"),
            )
        )
    # Upstream release that broke httpie. No commit id: the SHA belongs to a
    # repository we have not ingested, which is exactly the cross-repo gap the
    # agent will have to reason about in Week 3.
    deployments.append(
        Deployment(
            id=make_deployment_id(UPSTREAM_REPO_ID, "2.32.3"),
            repository_id=UPSTREAM_REPO_ID,
            tag="2.32.3",
            name="requests 2.32.3",
            environment=Environment.PRODUCTION,
            url="https://github.com/psf/requests/releases/tag/v2.32.3",
            created_at=datetime(2024, 5, 29, tzinfo=timezone.utc),
            published_at=datetime(2024, 5, 29, tzinfo=timezone.utc),
            release_notes=(
                "Dropped the automatic loading of the system CA bundle into "
                "user-provided SSL contexts. See psf/requests#6730."
            ),
            provenance=synthetic_provenance("manual:psf/requests@2.32.3", 0.8),
        )
    )
    return deployments


# --------------------------------------------------------------------------- #
# Pull requests (squash merges harvested from the clone)
# --------------------------------------------------------------------------- #
PR_SEEDS = [
    (1611, "5b604c37c6c67e18e7c3e9aee6c88a8c22b98345", "ash@sorrel.sh", "ash",
     "Fix `https` behaviour in fish",
     "Using the name `https` to invoke HTTPie should make HTTPie default to the "
     "https:// scheme, but a fish function replacing `https` by `http` defeated "
     "that behaviour."),
    (1596, "fd30c4ef6230a927f9dcfad6301c40e8bf846156", "adam@blueradius.ca", "Adam Williamson",
     "Explicitly load default certificates when creating SSL context (#1583)",
     "Requests >= 2.32.3 no longer loads the default (system-wide) set of trusted "
     "certificates into custom SSL contexts, which broke HTTPie users. We now "
     "explicitly load them, and drop the upper bound on the requests dependency "
     "that PR-less commit 7f03c52 had introduced as a stop-gap."),
    (1029, "3c07a2532647f64a4aaa7c729557fee5dcc2e182",
     "41421345+LuckyDenis@users.noreply.github.com", "Denis Belavin",
     "Add support for max-age=0 cookie expiry",
     "A cookie with `max-age=0` must be treated as expired immediately."),
    (1040, "8f83bfe7679c56843452cbd255adfd4f6d582c89", "almad@apible.io", "Almad",
     "Replace typography quotes",
     "Use straight quotes in generated documents instead of typographic ones."),
    (1387, "2142ae60c339144697cf93f40d1e740be344224d", "isidentical@gmail.com", "Batuhan Taskaya",
     "Final release prep for 3.2.0", "Version bump, man pages and changelog."),
    (1313, "266c6375c6d687c3aad07962cb3f5bef7d39bbbf", "isidentical@gmail.com", "Batuhan Taskaya",
     "Release prep for 3.1.0", "Version bump, man pages and changelog."),
]

#: PR -> issues the PR closes, with the link method that is actually available
#: today. Week 2 replaces the regex methods with GraphQL closingIssuesReferences.
PR_CLOSES = {
    1596: [(1583, LinkMethod.COMMIT_MESSAGE, 0.6,
            "Explicitly load default certificates when creating SSL context (#1583) (#1596)")],
}


def build_pull_requests(commits: list[Commit], files_by_path: dict[str, File]) -> list[PullRequest]:
    by_sha = {c.id: c for c in commits}
    prs: list[PullRequest] = []
    for number, sha, email, name, title, body in PR_SEEDS:
        commit = by_sha[sha]
        changes = [
            FileChange(
                file_id=change.file_id,
                path=change.path,
                change_type=change.change_type,
                additions=change.additions,
                deletions=change.deletions,
            )
            for change in commit.files_changed
        ]
        closes = [
            IssueLink(
                incident_id=make_incident_id(REPO_ID, issue_number),
                issue_number=issue_number,
                method=method,
                confidence=confidence,
                referenced_in="body",
                evidence_text=evidence,
                provenance=synthetic_provenance(f"git:{sha[:7]}:subject", confidence),
            )
            for issue_number, method, confidence, evidence in PR_CLOSES.get(number, [])
        ]
        prs.append(
            PullRequest(
                id=make_pr_id(REPO_ID, number),
                repository_id=REPO_ID,
                number=number,
                title=title,
                body=body,
                url=f"https://github.com/httpie/cli/pull/{number}",
                state=PRState.MERGED,
                author=PersonRef(id=email.lower(), email=email.lower(), name=name),
                base_branch="master",
                head_branch=f"{name.split()[0].lower()}/patch-{number}",
                created_at=commit.authored_at,
                merged_at=commit.committed_at,
                merge_commit_id=sha,
                merge_strategy=MergeStrategy.SQUASH,
                commits=[sha],
                files_changed=changes,
                changed_files_count=len(changes),
                additions=commit.additions,
                deletions=commit.deletions,
                review_count=1 if number < 1600 else 2,
                labels=[],
                closes_issues=closes,
                provenance=synthetic_provenance(f"git:{sha[:7]}", 0.85),
            )
        )
    return prs


# --------------------------------------------------------------------------- #
# Incidents
# --------------------------------------------------------------------------- #
def build_incidents(files_by_path: dict[str, File]) -> list[Incident]:
    def flink(path: str, role: EvidenceRole, confidence: float, method: str, rationale: str) -> IncidentFileLink:
        return IncidentFileLink(
            file_id=make_file_id(REPO_ID, path),
            path=path,
            role=role,
            confidence=confidence,
            method=method,
            rationale=rationale,
        )

    incidents = [
        Incident(
            id=make_incident_id(REPO_ID, 1583),
            repository_id=REPO_ID,
            number=1583,
            title="HTTPS requests fail after requests 2.32.3: certificate verify failed",
            body=(
                "Since requests 2.32.3, `http https://example.org` fails with "
                "`SSLError: certificate verify failed` when a custom CA bundle is "
                "used, because requests no longer loads the system default "
                "certificates into user-provided SSL contexts. Reported by several "
                "distribution packagers." 
            ),
            url="https://github.com/httpie/cli/issues/1583",
            status=IncidentStatus.CLOSED,
            severity=IncidentSeverity.SEV1,
            severity_score=0.95,
            labels=["bug", "ssl", "regression"],
            reporter=PersonRef(
                id="awilliam@redhat.com", email="awilliam@redhat.com", name="Adam Williamson"
            ),
            opened_at=datetime(2024, 6, 28, 9, 12, tzinfo=timezone.utc),
            closed_at=datetime(2024, 11, 1, 17, 45, tzinfo=timezone.utc),
            detection_source=DetectionSource.USER_REPORT,
            is_regression=True,
            resolution_commit_ids=[
                "7f03c52d2237440c5a672296ce6955aae4ed4f09",
                "fd30c4ef6230a927f9dcfad6301c40e8bf846156",
            ],
            resolution_pr_numbers=[1596],
            first_affected_version="3.2.2",
            fixed_in_version="3.2.4",
            affected_files=[
                flink("httpie/ssl_.py", EvidenceRole.ROOT_CAUSE, 0.9, "commit_overlap",
                      "The real fix (fd30c4e) added explicit loading of the default "
                      "certificates in create_ssl_context()."),
                flink("setup.cfg", EvidenceRole.FIX, 0.85, "commit_overlap",
                      "Workaround 7f03c52 pinned requests==2.31.0 here; fd30c4e "
                      "dropped that upper bound again."),
                flink("httpie/__init__.py", EvidenceRole.SYMPTOM, 0.5, "commit_overlap",
                      "Version bumped in every release commit, so it co-occurs with "
                      "the fix without being a cause."),
            ],
            observed_in=["httpie/cli@3.2.2", "httpie/cli@3.2.3"],
            caused_by=[
                IncidentCause(
                    incident_id=make_incident_id(UPSTREAM_REPO_ID, 6730),
                    repository_id=UPSTREAM_REPO_ID,
                    confidence=0.9,
                    method="issue_body_reference",
                    rationale=(
                        "The PR body of #1596 explicitly points at "
                        "psf/requests#6730: requests no longer loads the system CA "
                        "bundle into custom SSL contexts."
                    ),
                    evidence_url="https://github.com/psf/requests/issues/6730",
                )
            ],
            provenance=synthetic_provenance("manual:httpie/cli#1583", 0.6),
        ),
        Incident(
            id=make_incident_id(REPO_ID, 1581),
            repository_id=REPO_ID,
            number=1581,
            title="SSL: unable to get local issuer certificate with a custom CA bundle",
            body=(
                "Same root symptom as #1583, reported against 3.2.2. Closed by the "
                "same stop-gap commit that pinned requests to 2.31.0."
            ),
            url="https://github.com/httpie/cli/issues/1581",
            status=IncidentStatus.CLOSED,
            severity=IncidentSeverity.SEV2,
            severity_score=0.7,
            labels=["bug", "ssl"],
            opened_at=datetime(2024, 6, 24, 16, 3, tzinfo=timezone.utc),
            closed_at=datetime(2024, 7, 10, 16, 20, tzinfo=timezone.utc),
            detection_source=DetectionSource.USER_REPORT,
            resolution_commit_ids=["7f03c52d2237440c5a672296ce6955aae4ed4f09"],
            first_affected_version="3.2.2",
            fixed_in_version="3.2.3",
            affected_files=[
                flink("setup.cfg", EvidenceRole.FIX, 0.9, "commit_overlap",
                      "requests pinned to 2.31.0."),
            ],
            observed_in=["httpie/cli@3.2.2"],
            related_incident_ids=[make_incident_id(REPO_ID, 1583)],
            provenance=synthetic_provenance("manual:httpie/cli#1581", 0.5),
        ),
        Incident(
            id=make_incident_id(REPO_ID, 998),
            repository_id=REPO_ID,
            number=998,
            title="Cookies with max-age=0 are never treated as expired",
            body=(
                "RFC 6265 says `Max-Age=0` deletes the cookie immediately; HTTPie "
                "kept sending it because the expiry check only handled negative "
                "values."
            ),
            url="https://github.com/httpie/cli/issues/998",
            status=IncidentStatus.CLOSED,
            severity=IncidentSeverity.SEV3,
            severity_score=0.4,
            labels=["bug", "sessions"],
            reporter=PersonRef(
                id="reporter998@example.org", email="reporter998@example.org",
                name="Community reporter",
            ),
            opened_at=datetime(2020, 11, 30, 12, 0, tzinfo=timezone.utc),
            closed_at=datetime(2021, 2, 6, 10, 50, tzinfo=timezone.utc),
            detection_source=DetectionSource.USER_REPORT,
            resolution_commit_ids=["3c07a2532647f64a4aaa7c729557fee5dcc2e182"],
            resolution_pr_numbers=[1029],
            first_affected_version="2.4.0",
            fixed_in_version="3.0.0",
            affected_files=[
                flink("httpie/utils.py", EvidenceRole.ROOT_CAUSE, 0.9, "commit_overlap",
                      "The expiry helper lives in httpie/utils.py and grew +9 lines "
                      "in the fixing commit."),
                flink("httpie/models.py", EvidenceRole.CONTEXT, 0.4, "import_blast_radius",
                      "Imports httpie/utils.py, so it is inside the blast radius but "
                      "was not modified."),
            ],
            observed_in=["httpie/cli@3.0.0"],
            provenance=synthetic_provenance("manual:httpie/cli#998", 0.6),
        ),
        Incident(
            id=make_incident_id(REPO_ID, 1039),
            repository_id=REPO_ID,
            number=1039,
            title="Typographic quotes in generated documents break CLI copy/paste",
            body="Examples copied from the docs pasted the wrong quote characters.",
            url="https://github.com/httpie/cli/issues/1039",
            status=IncidentStatus.CLOSED,
            severity=IncidentSeverity.SEV4,
            severity_score=0.15,
            labels=["docs"],
            opened_at=datetime(2021, 2, 20, 8, 0, tzinfo=timezone.utc),
            closed_at=datetime(2021, 2, 24, 14, 38, tzinfo=timezone.utc),
            detection_source=DetectionSource.MAINTAINER,
            resolution_commit_ids=["8f83bfe7679c56843452cbd255adfd4f6d582c89"],
            resolution_pr_numbers=[1040],
            fixed_in_version="3.0.0",
            affected_files=[
                flink("setup.cfg", EvidenceRole.ROOT_CAUSE, 0.8, "commit_overlap",
                      "The only file changed by the fix."),
            ],
            provenance=synthetic_provenance("manual:httpie/cli#1039", 0.5),
        ),
        Incident(
            id=make_incident_id(REPO_ID, 1461),
            repository_id=REPO_ID,
            number=1461,
            title="Test suite fails with responses >= 0.22.0",
            body=(
                "The `responses` test double changed its public API, so HTTPie's "
                "session tests started failing in CI."
            ),
            url="https://github.com/httpie/cli/issues/1461",
            status=IncidentStatus.CLOSED,
            severity=IncidentSeverity.SEV3,
            severity_score=0.35,
            labels=["ci", "tests"],
            opened_at=datetime(2023, 1, 10, 11, 0, tzinfo=timezone.utc),
            closed_at=datetime(2023, 1, 15, 17, 43, tzinfo=timezone.utc),
            detection_source=DetectionSource.TEST_FAILURE,
            is_regression=True,
            resolution_commit_ids=["e73c3e6c249b89496b4f81fa20bb449911da79f1"],
            first_affected_version="3.2.0",
            fixed_in_version="3.2.2",
            affected_files=[
                flink("httpie/models.py", EvidenceRole.ROOT_CAUSE, 0.7, "commit_overlap",
                      "+27/-19 lines in the fixing commit."),
            ],
            provenance=synthetic_provenance("manual:httpie/cli#1461", 0.5),
        ),
        Incident(
            id=make_incident_id(REPO_ID, 1467),
            repository_id=REPO_ID,
            number=1467,
            title="CI: session tests flaky on Python 3.11",
            body="Same root cause as #1461; closed by the same commit.",
            url="https://github.com/httpie/cli/issues/1467",
            status=IncidentStatus.CLOSED,
            severity=IncidentSeverity.SEV4,
            severity_score=0.2,
            labels=["ci"],
            opened_at=datetime(2023, 1, 13, 9, 30, tzinfo=timezone.utc),
            closed_at=datetime(2023, 1, 15, 17, 43, tzinfo=timezone.utc),
            detection_source=DetectionSource.TEST_FAILURE,
            resolution_commit_ids=["e73c3e6c249b89496b4f81fa20bb449911da79f1"],
            fixed_in_version="3.2.2",
            related_incident_ids=[make_incident_id(REPO_ID, 1461)],
            provenance=synthetic_provenance("manual:httpie/cli#1467", 0.5),
        ),
        Incident(
            id=make_incident_id(REPO_ID, 761),
            repository_id=REPO_ID,
            number=761,
            title="Documentation: no FreeBSD installation instructions",
            body="Users on FreeBSD had no documented install path.",
            url="https://github.com/httpie/cli/issues/761",
            status=IncidentStatus.CLOSED,
            severity=IncidentSeverity.SEV4,
            severity_score=0.1,
            labels=["docs"],
            opened_at=datetime(2021, 9, 1, 10, 0, tzinfo=timezone.utc),
            closed_at=datetime(2021, 9, 23, 10, 46, tzinfo=timezone.utc),
            detection_source=DetectionSource.USER_REPORT,
            resolution_commit_ids=["cae83b3f9e2077afee85992e8135cae70bb91cf3"],
            fixed_in_version="3.0.0",
            provenance=synthetic_provenance("manual:httpie/cli#761", 0.5),
        ),
        # ---------------- upstream repository ----------------
        Incident(
            id=make_incident_id(UPSTREAM_REPO_ID, 6730),
            repository_id=UPSTREAM_REPO_ID,
            number=6730,
            title="Requests 2.32.3 no longer loads the system CA bundle into custom SSL contexts",
            body=(
                "`requests.Session.verify` with a custom context stopped trusting "
                "system certificates. Deliberate change, slow to be reverted "
                "upstream because of security considerations."
            ),
            url="https://github.com/psf/requests/issues/6730",
            status=IncidentStatus.MITIGATED,
            severity=IncidentSeverity.SEV2,
            severity_score=0.75,
            labels=["ssl", "breaking-change"],
            opened_at=datetime(2024, 5, 30, 7, 0, tzinfo=timezone.utc),
            detection_source=DetectionSource.UPSTREAM_DEPENDENCY,
            is_regression=True,
            first_affected_version="2.32.3",
            observed_in=["psf/requests@2.32.3"],
            provenance=synthetic_provenance("manual:psf/requests#6730", 0.55),
        ),
    ]
    # Referential smoke test: an AFFECTS edge must never dangle.
    known = set(files_by_path)
    for incident in incidents:
        for link in incident.affected_files:
            if link.path not in known:
                raise SystemExit(f"{incident.id}: AFFECTS -> unknown file {link.path}")
    return incidents


def add_stub_incidents(
    commits: list[Commit], prs: list[PullRequest], incidents: list[Incident]
) -> list[Incident]:
    """Create placeholder Incidents for referenced issue numbers we did not curate.

    Ancestor/context commits carry their own ``Closes #N`` references. Rather than
    dropping them (which would make the ``:CLOSES`` edge silently disappear), a
    stub Incident is created and flagged with a low-confidence synthetic
    provenance, mimicking what a real ingestion would fill from the GitHub API.
    """
    known = {incident.id for incident in incidents}
    referenced: dict[str, str] = {}
    for commit in commits:
        for ref in commit.issue_refs:
            referenced.setdefault(ref.incident_id, commit.subject)
    for pr in prs:
        for link in pr.closes_issues:
            referenced.setdefault(link.incident_id, pr.title)

    stubs: list[Incident] = []
    for incident_id, subject in referenced.items():
        if incident_id in known:
            continue
        repository_id, _, tail = incident_id.partition("#issue-")
        if not tail.isdigit():
            continue
        stubs.append(
            Incident(
                id=incident_id,
                repository_id=repository_id,
                number=int(tail),
                title=f"[stub] {subject[:120]}",
                body=(
                    "Placeholder created because a commit or PR references this issue "
                    "number but the issue itself was not curated in the fixtures. A real "
                    "ingestion fills this node from the GitHub API."
                ),
                url=f"https://github.com/{repository_id}/issues/{tail}",
                status=IncidentStatus.OPEN,
                severity=IncidentSeverity.UNKNOWN,
                detection_source=DetectionSource.UNKNOWN,
                provenance=synthetic_provenance(f"stub:git-reference:{incident_id}", 0.2),
            )
        )
    return stubs


# --------------------------------------------------------------------------- #
# Vector side-car: insertable rows for pgvector (no embedding computed here)
# --------------------------------------------------------------------------- #
EVIDENCE_ROWS = [
    # (id, answer_id, kind, node_kind, node_id, text, retrieval, score, hop)
    (
        "ev-1583-01", "ans-2026-09-18-0001", EvidenceKind.NODE, NodeKind.INCIDENT,
        make_incident_id(REPO_ID, 1583),
        "Incident httpie/cli#issue-1583: HTTPS requests fail after requests 2.32.3: "
        "certificate verify failed. sev1, regression, opened 2024-06-28.",
        RetrievalStrategy.LEXICAL, 0.9, 0,
    ),
    (
        "ev-1583-02", "ans-2026-09-18-0001", EvidenceKind.COMMIT_DIFF, NodeKind.COMMIT,
        "fd30c4ef6230a927f9dcfad6301c40e8bf846156",
        "Commit fd30c4e 'Explicitly load default certificates when creating SSL "
        "context (#1583) (#1596)' modifies httpie/ssl_.py (+7/-0) and setup.cfg (+1/-1).",
        RetrievalStrategy.GRAPH_TRAVERSAL, 0.95, 1,
    ),
    (
        "ev-1583-03", "ans-2026-09-18-0001", EvidenceKind.NODE, NodeKind.FILE,
        make_file_id(REPO_ID, "httpie/ssl_.py"),
        "httpie/ssl_.py: create_ssl_context() is the single place where HTTPie "
        "builds an SSLContext; it is the root cause of the certificate regression.",
        RetrievalStrategy.HYBRID, 0.93, 2,
    ),
    (
        "ev-1583-04", "ans-2026-09-18-0001", EvidenceKind.DEPLOYMENT_WINDOW,
        NodeKind.DEPLOYMENT, make_deployment_id(REPO_ID, "3.2.4"),
        "Deployment httpie/cli@3.2.4 (2024-11-01) points at 2105caa and fixes the "
        "incident; the previous deployment 3.2.3 was the workaround release.",
        RetrievalStrategy.TEMPORAL, 0.8, 3,
    ),
    (
        "ev-1583-05", "ans-2026-09-18-0001", EvidenceKind.NODE, NodeKind.INCIDENT,
        make_incident_id(UPSTREAM_REPO_ID, 6730),
        "Upstream incident psf/requests#issue-6730: requests 2.32.3 stopped loading "
        "the system CA bundle into custom SSL contexts.",
        RetrievalStrategy.GRAPH_TRAVERSAL, 0.88, 2,
    ),
    (
        "ev-ctx-01", "ans-2026-09-18-0002", EvidenceKind.NODE, NodeKind.FILE,
        make_file_id(REPO_ID, "httpie/context.py"),
        "httpie/context.py holds the Environment object: config loading, output "
        "formatting, and the request/response pipeline state.",
        RetrievalStrategy.VECTOR_SIMILARITY, 0.82, 0,
    ),
    (
        "ev-ctx-02", "ans-2026-09-18-0002", EvidenceKind.IMPORT_EDGE, NodeKind.FILE,
        make_file_id(REPO_ID, "httpie/cli/argparser.py"),
        "httpie/cli/argparser.py -> httpie/context.py (:IMPORTS). Direct dependent.",
        RetrievalStrategy.GRAPH_TRAVERSAL, 0.9, 1,
    ),
    (
        "ev-ctx-03", "ans-2026-09-18-0002", EvidenceKind.GRAPH_PATH, NodeKind.FILE,
        make_file_id(REPO_ID, "httpie/manager/cli.py"),
        "httpie/manager/cli.py -> httpie/cli/argparser.py -> httpie/context.py is the "
        "2-hop chain: the manager CLI ('httpie' command) reaches Environment through "
        "the argparser.",
        RetrievalStrategy.GRAPH_TRAVERSAL, 0.75, 2,
    ),
    (
        "ev-ui-01", "ans-2026-09-18-0003", EvidenceKind.NODE, NodeKind.FILE,
        make_file_id(REPO_ID, "httpie/output/ui/__init__.py"),
        "httpie/output/ui/__init__.py is the deepest node of the import graph "
        "max depth 7; it is imported lazily by httpie/cli/argparser.py.",
        RetrievalStrategy.GRAPH_TRAVERSAL, 0.7, 0,
    ),
    (
        "ev-ui-02", "ans-2026-09-18-0003", EvidenceKind.GRAPH_PATH, NodeKind.FILE,
        make_file_id(REPO_ID, "httpie/output/ui/palette.py"),
        "httpie/output/ui/palette.py -> httpie/context.py is part of the chain that "
        "gives httpie/output/ui its depth.",
        RetrievalStrategy.GRAPH_TRAVERSAL, 0.72, 1,
    ),
]

ANSWERS = [
    {
        "id": "ans-2026-09-18-0001",
        "question": "Pourquoi les connexions HTTPS échouaient-elles dans HTTPie 3.2.2 et qu'est-ce qui les a corrigées ?",
        "intent": "root_cause",
        "status": "ok",
        "answer_text": (
            "**Cause racine** : `requests >= 2.32.3` a cessé de charger le bundle "
            "de certificats système dans les `SSLContext` fournis par l'appelant "
            "(psf/requests#6730).\n\n"
            "**Chemin de preuve** : Incident `httpie/cli#issue-1583` → Commit "
            "`7f03c52` (contournement : épinglage de `requests==2.31.0`, livré dans "
            "3.2.3) → PR `#1596` → Commit `fd30c4e` (correctif réel dans "
            "`httpie/ssl_.py`) → Deployment `httpie/cli@3.2.4`.\n\n"
            "**Point de vigilance** : le commit `7f03c52` a corrigé le symptôme sans "
            "PR ; le lien Incident→PR n'existe que via le titre de la PR #1596, d'où "
            "une confiance de 0.6 sur cette arête."
        ),
        "confidence_score": 0.86,
        "breakdown": {
            "graph_coverage": 0.95,
            "path_length_penalty": -0.06,
            "link_confidence_weighted": 0.62,
            "deployment_corroboration": 0.9,
            "contradiction_penalty": 0.0,
        },
        "evidence_ids": ["ev-1583-01", "ev-1583-02", "ev-1583-03", "ev-1583-05", "ev-1583-04"],
        "hops": [
            (0, NodeKind.INCIDENT, make_incident_id(REPO_ID, 1583), None, EvidenceRole.SYMPTOM, 0.9),
            (1, NodeKind.COMMIT, "fd30c4ef6230a927f9dcfad6301c40e8bf846156", "CLOSES", EvidenceRole.FIX, 0.95),
            (2, NodeKind.FILE, make_file_id(REPO_ID, "httpie/ssl_.py"), "MODIFIES", EvidenceRole.ROOT_CAUSE, 0.93),
            (3, NodeKind.PR, make_pr_id(REPO_ID, 1596), "MERGED_INTO", EvidenceRole.FIX, 0.8),
            (4, NodeKind.DEPLOYMENT, make_deployment_id(REPO_ID, "3.2.4"), "DEPLOYED_AT", EvidenceRole.TIMELINE, 0.85),
            (5, NodeKind.INCIDENT, make_incident_id(UPSTREAM_REPO_ID, 6730), "CAUSED_BY", EvidenceRole.ROOT_CAUSE, 0.88),
        ],
        "citations": [
            ("httpie/ssl_.py", "File", make_file_id(REPO_ID, "httpie/ssl_.py"), "https://github.com/httpie/cli/pull/1596"),
            ("commit fd30c4e", "Commit", "fd30c4ef6230a927f9dcfad6301c40e8bf846156", "https://github.com/httpie/cli/commit/fd30c4e"),
            ("psf/requests#6730", "Incident", make_incident_id(UPSTREAM_REPO_ID, 6730), "https://github.com/psf/requests/issues/6730"),
        ],
        "actions": [
            "Épingler requests<2.32.3 en attendant la sortie de 3.2.4 (fait dans 7f03c52).",
            "Ajouter un test de régression sur create_ssl_context() avec un CA bundle custom.",
        ],
        "latency_ms": 4210,
        "tool_calls": 7,
    },
    {
        "id": "ans-2026-09-18-0002",
        "question": "Quel est l'impact d'une modification de httpie/context.py ?",
        "intent": "change_impact",
        "status": "ok",
        "answer_text": (
            "`httpie/context.py` est un nœud à fort couplage : **22 dépendants "
            "directs**, **37 fichiers atteints de façon transitive**, profondeur "
            "**3**. Les dépendants critiques sont `httpie/cli/argparser.py` (chaque "
            "invocation CLI le traverse), `httpie/client.py` et `httpie/core.py` "
            "(le pipeline requête/réponse), `httpie/output/writer.py` (le formatage "
            "de sortie) et `httpie/manager/tasks/plugins.py` (le CLI "
            "d'administration). `tests/test_sessions.py` est aussi impacté : "
            "un changement ici ne se limite pas au code de production."
        ),
        "confidence_score": 0.78,
        "breakdown": {
            "graph_coverage": 0.92,
            "path_length_penalty": -0.05,
            "metrics_freshness": 0.95,
            "unresolved_imports_penalty": -0.09,
        },
        "evidence_ids": ["ev-ctx-01", "ev-ctx-02", "ev-ctx-03"],
        "hops": [
            (0, NodeKind.FILE, make_file_id(REPO_ID, "httpie/context.py"), None, EvidenceRole.BLAST_RADIUS, 0.9),
            (1, NodeKind.FILE, make_file_id(REPO_ID, "httpie/cli/argparser.py"), "IMPORTS", EvidenceRole.BLAST_RADIUS, 0.9),
            (2, NodeKind.FILE, make_file_id(REPO_ID, "httpie/manager/cli.py"), "IMPORTS", EvidenceRole.BLAST_RADIUS, 0.75),
        ],
        "citations": [
            ("httpie/context.py", "File", make_file_id(REPO_ID, "httpie/context.py"), None),
            ("httpie/cli/argparser.py", "File", make_file_id(REPO_ID, "httpie/cli/argparser.py"), None),
        ],
        "actions": [
            "Lancer tests/test_sessions.py : c'est le seul fichier de tests présent dans le rayon d'impact ingéré.",
            "Vérifier les imports paresseux (httpie/context.py:194, httpie/cli/argparser.py:270, 562, 577) qui masquent le couplage réel.",
        ],
        "latency_ms": 2380,
        "tool_calls": 4,
    },
    {
        "id": "ans-2026-09-18-0003",
        "question": "Quel fichier du graphe d'imports est le plus profond, et pourquoi est-ce risqué de le modifier ?",
        "intent": "change_impact",
        "status": "ok",
        "answer_text": (
            "`httpie/output/ui/__init__.py` est à la profondeur maximale du graphe "
            "(7 sauts). Il est importé paresseusement par `httpie/cli/argparser.py` "
            "(lignes 562 et 577), donc une erreur d'import n'apparaît qu'au moment "
            "où l'utilisateur demande `--help`, pas au démarrage.")
            ,
        "confidence_score": 0.66,
        "breakdown": {
            "graph_coverage": 0.8,
            "path_length_penalty": -0.14,
            "lazy_import_penalty": -0.05,
        },
        "evidence_ids": ["ev-ui-01", "ev-ui-02"],
        "hops": [
            (0, NodeKind.FILE, make_file_id(REPO_ID, "httpie/output/ui/__init__.py"), None, EvidenceRole.CONTEXT, 0.7),
            (1, NodeKind.FILE, make_file_id(REPO_ID, "httpie/output/ui/palette.py"), "IMPORTS", EvidenceRole.CONTEXT, 0.72),
        ],
        "citations": [
            ("httpie/output/ui/__init__.py", "File", make_file_id(REPO_ID, "httpie/output/ui/__init__.py"), None),
        ],
        "actions": ["Ajouter un test d'import du module `--help` pour couvrir le chemin paresseux."],
        "latency_ms": 1720,
        "tool_calls": 3,
    },
    {
        "id": "ans-2026-09-18-0004",
        "question": "Pourquoi tests/test_sessions.py échoue-t-il sur la branche feature/xyz ?",
        "intent": "root_cause",
        "status": "insufficient_evidence",
        "answer_text": (
            "Le graphe ne contient pas la branche `feature/xyz` ni l'exécution CI "
            "concernée : aucune preuve exploitable. L'ingestion doit couvrir "
            "`refs/heads/*` et les runs CI avant que ce type de question soit "
            "répondable."
        ),
        "confidence_score": 0.12,
        "breakdown": {
            "graph_coverage": 0.1,
            "missing_scope_penalty": -0.8,
        },
        "evidence_ids": [],
        "hops": [],
        "citations": [],
        "actions": ["Étendre l'ingestion aux branches et aux runs CI (§ Semaine 2)."],
        "latency_ms": 640,
        "tool_calls": 2,
    },
]


def build_evidence_and_answers() -> tuple[list[Evidence], list[Answer]]:
    evidence: list[Evidence] = []
    for (ev_id, answer_id, kind, node_kind, node_id, text, strategy, score, hop) in EVIDENCE_ROWS:
        role = EvidenceRole.ROOT_CAUSE if hop >= 2 else (
            EvidenceRole.SYMPTOM if hop == 0 and node_kind == NodeKind.INCIDENT else EvidenceRole.CONTEXT
        )
        evidence.append(
            Evidence(
                id=ev_id,
                repository_id=REPO_ID if not node_id.startswith(UPSTREAM_REPO_ID) else UPSTREAM_REPO_ID,
                answer_id=answer_id,
                kind=kind,
                retrieval_strategy=strategy,
                score=score,
                rank=hop + 1,
                text=text,
                rationale=f"Retrieved with {strategy.value} at hop {hop}.",
                hop_index=hop,
                path_id=answer_id,
                node_references=[
                    EvidenceRef(
                        node_kind=node_kind,
                        node_id=node_id,
                        role=role,
                        hop_index=hop,
                        relation=None,
                        rationale=text.split(".")[0],
                    )
                ],
                vector={
                    "table": "evidence_embeddings",
                    "row_id": ev_id,
                    "model": "BAAI/bge-m3",
                    "dim": 1024,
                    "distance": None,
                },
                retrieved_at=INGESTED_AT,
                provenance=synthetic_provenance(f"agent:retrieval:{ev_id}", 0.7),
            )
        )

    answers: list[Answer] = []
    for spec in ANSWERS:
        hops = [
            EvidenceHop(
                step=step,
                node_kind=node_kind,
                node_id=node_id,
                relation_in=relation,
                role=role,
                score=score,
                rationale=f"hop {step}: {role.value}",
            )
            for (step, node_kind, node_id, relation, role, score) in spec["hops"]
        ]
        answers.append(
            Answer(
                id=spec["id"],
                question=spec["question"],
                intent=spec["intent"],
                answer_text=spec["answer_text"],
                status=spec["status"],
                repository_id=REPO_ID,
                confidence_score=spec["confidence_score"],
                confidence_band=ConfidenceBand.from_score(spec["confidence_score"]),
                confidence_breakdown=spec["breakdown"],
                evidence_path=EvidencePath(
                    path_id=spec["id"],
                    intent=spec["intent"],
                    hops=hops,
                    score=spec["confidence_score"],
                    is_valid=bool(hops),
                    cycle_detected=False,
                    validation_notes=(
                        ["Every hop resolved against the graph."]
                        if hops
                        else ["No hop could be resolved: nothing to validate."]
                    ),
                ),
                evidence_ids=spec["evidence_ids"],
                citations=[
                    Citation(label=label, node_kind=NodeKind(kind), node_id=node_id, url=url)
                    for (label, kind, node_id, url) in spec["citations"]
                ],
                suggested_actions=spec["actions"],
                model="gpt-4.1-mini",
                prompt_version="nexus-agent-v0.1.0",
                latency_ms=spec["latency_ms"],
                token_usage={"prompt": 3120, "completion": 480},
                tool_calls=spec["tool_calls"],
                created_at=datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc),
                graph_snapshot_at=INGESTED_AT,
                provenance=synthetic_provenance(f"agent:langgraph:{spec['id']}", 0.6),
            )
        )
    return evidence, answers


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def dump(name: str, payload: list) -> None:
    path = Path(__file__).resolve().parent / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {path.name:<24} {len(payload):>3} records")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-path", default=str(PROJECT_ROOT.parent / "httpie-cli"),
        help="local clone of httpie/cli (default: ../httpie-cli)",
    )
    args = parser.parse_args()
    repo = Path(args.repo_path).resolve()
    if not (repo / ".git").exists():
        print(f"error: {repo} is not a git clone of httpie/cli", file=sys.stderr)
        return 1

    print(f"Harvesting from {repo}")
    commit_set = expand_commit_set(repo, CURATED_COMMITS)
    window = connect_window(repo, *CONNECTED_WINDOW)
    for sha in window:
        if sha not in commit_set:
            commit_set.append(sha)
    files, _ = harvest_files(repo, commit_set)
    compute_impact(files)
    files_by_path = {f.path: f for f in files}
    commits = harvest_commits(repo, commit_set, set(files_by_path))
    print(f"  {len(CURATED_COMMITS)} curated commits + {len(commit_set) - len(CURATED_COMMITS)} "
          f"context commits (ancestors + the {CONNECTED_WINDOW[0]}->{CONNECTED_WINDOW[1]} window, "
          f"{len(window)} commits)")
    deployments = harvest_deployments(repo)
    prs = build_pull_requests(commits, files_by_path)
    incidents = build_incidents(files_by_path)
    incidents += add_stub_incidents(commits, prs, incidents)
    evidence, answers = build_evidence_and_answers()

    repositories = [
        Repository(
            id=REPO_ID,
            host="github.com",
            owner="httpie",
            name="cli",
            url="https://github.com/httpie/cli",
            description="Modern, user-friendly command-line HTTP client for the API era.",
            default_branch="master",
            primary_language="Python",
            license_spdx="BSD-3-Clause",
            stars=48800,
            forks=3700,
            is_archived=False,
            created_at=datetime(2012, 2, 25, tzinfo=timezone.utc),
            pushed_at=datetime(2024, 12, 17, 17, 30, tzinfo=timezone.utc),
            local_path=str(repo),
            metrics=RepositoryAuditMetrics(
                total_commits=1797,
                merge_commits=107,
                python_files=133,
                tag_count=50,
                size_worktree_mb=2.1,
                size_incl_git_mb=9.59,
                contributors=174,
                import_edges=214,
                files_importing=60,
                files_depended_upon=69,
                impact_max_depth=7,
                impact_avg_blast_radius=21.99,
                linked_commit_pct=29.4,
                closing_keyword_commit_pct=6.3,
                unique_issue_or_pr_refs=552,
                computed_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
            ),
            provenance=derived_provenance("audit:repo_audit.py", 1.0),
        ),
        Repository(
            id=UPSTREAM_REPO_ID,
            host="github.com",
            owner="psf",
            name="requests",
            url="https://github.com/psf/requests",
            description="A simple, yet elegant, HTTP library. Direct runtime dependency of HTTPie.",
            default_branch="main",
            primary_language="Python",
            license_spdx="Apache-2.0",
            stars=52700,
            forks=9400,
            created_at=datetime(2011, 2, 13, tzinfo=timezone.utc),
            metrics=RepositoryAuditMetrics(
                total_commits=6494,
                merge_commits=1612,
                python_files=37,
                tag_count=162,
                size_worktree_mb=4.45,
                size_incl_git_mb=19.18,
                contributors=804,
                import_edges=72,
                impact_max_depth=3,
                impact_avg_blast_radius=11.62,
                linked_commit_pct=28.2,
                closing_keyword_commit_pct=2.8,
                unique_issue_or_pr_refs=1761,
                computed_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
            ),
            provenance=derived_provenance("audit:repo_audit.py", 1.0),
        ),
    ]

    dump("repositories", [r.model_dump(mode="json", exclude_none=True) for r in repositories])
    dump("files", [f.model_dump(mode="json", exclude_none=True) for f in files])
    dump("commits", [c.model_dump(mode="json", exclude_none=True) for c in commits])
    dump("pull_requests", [p.model_dump(mode="json", exclude_none=True) for p in prs])
    dump("deployments", [d.model_dump(mode="json", exclude_none=True) for d in deployments])
    dump("incidents", [i.model_dump(mode="json", exclude_none=True) for i in incidents])
    dump("evidence", [e.model_dump(mode="json", exclude_none=True) for e in evidence])
    dump("answers", [a.model_dump(mode="json", exclude_none=True) for a in answers])

    edges = [e.model_dump(mode="json", exclude_none=True)
             for model_list in (files, commits, prs, deployments, incidents, evidence, answers)
             for model in model_list
             for e in model.edges()]
    path = Path(__file__).resolve().parent / "graph_edges.json"
    path.write_text(json.dumps(edges, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  wrote {path.name:<24} {len(edges):>3} derived edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
