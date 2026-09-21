#!/usr/bin/env python3
"""repo_audit.py - Audit a local git repository for NEXUS-DevIntel suitability.

Metrics produced:
  1. commit statistics (total, non-merge, merges)
  2. python file count + total repo size
  3. issue <-> PR <-> commit linkage (closing keywords, PR merges, squash refs)
  4. internal import graph (module/file granularity) for Change Impact Analysis
  5. tags / releases (used to model "Deployment" nodes)

Usage:
    python repo_audit.py <path-to-repo> [--json out.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".tox", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", "build", "dist", ".eggs",
    "site-packages", ".idea", ".vscode",
}

CLOSING_RE = re.compile(
    r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\b\s*:?\s*(#[0-9]+(?:[,\s]+#[0-9]+)*)",
    re.IGNORECASE,
)
ANY_REF_RE = re.compile(r"#[0-9]+")
PR_MERGE_RE = re.compile(r"merge pull request #([0-9]+)", re.IGNORECASE)
SQUASH_RE = re.compile(r"\(#([0-9]+)\)\s*$")


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def dir_size(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def discover_python_files(repo: Path) -> list[str]:
    out = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(".py"):
                out.append(str((Path(root) / name).relative_to(repo)).replace(os.sep, "/"))
    return sorted(out)


def is_test_file(rel: str) -> bool:
    parts = rel.split("/")
    name = parts[-1]
    return (
        any(p in {"tests", "test"} for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def package_roots(files: list[str]) -> set[str]:
    """Top-level importable package names (dir with __init__.py) in the repo."""
    roots = set()
    for f in files:
        if f.endswith("/__init__.py"):
            roots.add(f.split("/", 1)[0])
    return roots


def module_name(relpath: str) -> str:
    mod = relpath[:-3] if relpath.endswith(".py") else relpath
    parts = mod.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_edges(edges: set[tuple[str, str]]) -> list[tuple[str, str, int]]:
    """Aggregate file-level edges into directory (module) level couplings."""

    def pkg(path: str) -> str:
        return path.rsplit("/", 1)[0] + "/" if "/" in path else "(root)"

    agg: Counter[tuple[str, str]] = Counter()
    for src, dst in edges:
        a, b = pkg(src), pkg(dst)
        if a != b:
            agg[(a, b)] += 1
    return sorted(
        ((a, b, n) for (a, b), n in agg.items()), key=lambda t: (-t[2], t[0], t[1])
    )


def impact_radius(edges: set[tuple[str, str]]) -> dict:
    """Reverse-reachability stats: blast radius of changing a single file."""
    reverse: dict[str, set[str]] = defaultdict(set)
    for src, dst in edges:
        reverse[dst].add(src)

    radius = 0
    sizes: list[int] = []
    deepest: tuple[str, int] = ("", 0)
    for node in reverse:
        seen = {node}
        frontier = [node]
        depth = 0
        while frontier:
            nxt = []
            for n in frontier:
                for dep in reverse.get(n, ()):
                    if dep not in seen:
                        seen.add(dep)
                        nxt.append(dep)
            if nxt:
                depth += 1
            frontier = nxt
        blast = len(seen) - 1
        sizes.append(blast)
        radius = max(radius, depth)
        if depth > deepest[1]:
            deepest = (node, depth)
    sizes.sort()
    return {
        "max_depth": radius,
        "deepest_node": deepest[0],
        "max_blast_radius": sizes[-1] if sizes else 0,
        "avg_blast_radius": round(sum(sizes) / len(sizes), 2) if sizes else 0,
        "median_blast_radius": sizes[len(sizes) // 2] if sizes else 0,
    }


def build_import_graph(repo: Path, files: list[str]) -> dict:
    roots = package_roots(files)
    mod_of = {f: module_name(f) for f in files}
    by_module = {m: f for f, m in mod_of.items()}

    edges: set[tuple[str, str]] = set()          # file -> file
    unresolved: Counter[str] = Counter()
    external: Counter[str] = Counter()
    syntax_errors: list[str] = []

    for rel in files:
        src = repo / rel
        try:
            tree = ast.parse(src.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            syntax_errors.append(rel)
            continue

        cur_mod = mod_of[rel]
        if rel.endswith("__init__.py"):
            cur_pkg = cur_mod
        else:
            cur_pkg = cur_mod.rsplit(".", 1)[0] if "." in cur_mod else ""

        targets: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    targets.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import, resolved against the current package
                    parts = [p for p in cur_pkg.split(".") if p]
                    base = parts[: max(0, len(parts) - (node.level - 1))]
                    prefix = ".".join(base)
                    targets.append(f"{prefix}.{node.module}" if node.module else prefix)
                elif node.module:
                    targets.append(node.module)

        for target in targets:
            top = target.split(".")[0]
            if top in roots:
                tgt_file = by_module.get(target) or by_module.get(target + ".__init__")
                if tgt_file is None:
                    # first path component may be the module itself
                    tgt_file = by_module.get(top)
                if tgt_file is None:
                    unresolved[target] += 1
                elif tgt_file != rel:
                    edges.add((rel, tgt_file))
            else:
                external[top] += 1

    out_deg = Counter(src for src, _ in edges)
    in_deg = Counter(dst for _, dst in edges)
    bidirectional = {(a, b) for a, b in edges if (b, a) in edges}

    # Change Impact Analysis proxy: if file F changes, which files depend on it
    # transitively? Depth = how far the ripple travels, set size = blast radius.
    impact = impact_radius(edges)

    return {
        "package_roots": sorted(roots),
        "internal_edges": len(edges),
        "files_with_internal_imports": len(out_deg),
        "files_depended_upon": len(in_deg),
        "max_out_degree": out_deg.most_common(1)[0] if out_deg else (None, 0),
        "avg_out_degree": round(sum(out_deg.values()) / len(out_deg), 2) if out_deg else 0,
        "bidirectional_pairs": len(bidirectional) // 2,
        "impact_radius": impact,
        "package_edges": package_edges(edges),
        "most_depended_upon": in_deg.most_common(10),
        "most_importing": out_deg.most_common(10),
        "external_top": external.most_common(15),
        "unresolved_internal": unresolved.most_common(10),
        "syntax_errors": syntax_errors,
        "edges": sorted(edges),
    }


def commit_stats(repo: Path) -> dict:  # noqa: C901
    raw = git(repo, "log", "--pretty=format:%x1e%H%x1f%P%x1f%s%x1f%b", "--no-color")
    records = [r for r in raw.split("\x1e") if r.strip()]
    total = len(records)
    merges = non_merges = 0
    closing = any_ref = pr_merge = squash = 0
    merge_with_closing = merge_with_issue_ref = 0
    refs: set[int] = set()

    for rec in records:
        parts = rec.split("\x1f")
        if len(parts) < 4:
            continue
        sha, parents, subject, body = parts[0], parts[1], parts[2], "\x1f".join(parts[3:])
        is_merge = len(parents.split()) > 1
        merges += is_merge
        non_merges += not is_merge

        text = f"{subject}\n{body}"
        has_closing = bool(CLOSING_RE.search(text))
        if has_closing:
            closing += 1
        if is_merge and has_closing:
            merge_with_closing += 1
        if is_merge and ANY_REF_RE.search(body):
            merge_with_issue_ref += 1
        found = ANY_REF_RE.findall(text)
        if found:
            any_ref += 1
            refs.update(int(n[1:]) for n in found)
        if PR_MERGE_RE.search(subject):
            pr_merge += 1
        if SQUASH_RE.search(subject):
            squash += 1

    tags_raw = git(repo, "tag", "--sort=-creatordate").splitlines()
    head = git(repo, "log", "-1", "--format=%H%x1f%cI%x1f%s").split("\x1f")
    return {
        "total_commits": total,
        "merge_commits": merges,
        "non_merge_commits": non_merges,
        "commits_any_issue_or_pr_ref": any_ref,
        "pct_any_ref": round(100 * any_ref / total, 1) if total else 0,
        "commits_with_closing_keyword": closing,
        "pct_closing": round(100 * closing / total, 1) if total else 0,
        "merge_pull_request_subjects": pr_merge,
        "squash_merge_subjects": squash,
        "merges_with_closing_keyword": merge_with_closing,
        "merges_referencing_issue_in_body": merge_with_issue_ref,
        "total_linked_commits": any_ref,
        "pct_linked": round(100 * any_ref / total, 1) if total else 0,
        "unique_referenced_numbers": len(refs),
        "tags": len(tags_raw),
        "recent_tags": tags_raw[:10],
        "head_sha": head[0][:12],
        "head_date": head[1],
        "head_subject": head[2],
        "first_commit_date": git(repo, "log", "--reverse", "--format=%cI").splitlines()[0].strip(),
        "contributors": len(set(git(repo, "log", "--format=%aE").split())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="path to local git repository")
    ap.add_argument("--json", help="write full result (incl. edges) to this file")
    ap.add_argument(
        "--include-tests", action="store_true",
        help="keep test files in the import graph (default: source only)",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        print(f"error: {repo} is not a git repository", file=sys.stderr)
        return 1

    files = discover_python_files(repo)
    worktree_mb = round(dir_size(repo) / 1e6, 2)
    total_mb = round(sum(
        (p.stat().st_size for p in repo.rglob("*") if p.is_file()), start=0
    ) / 1e6, 2)
    result = {
        "repo": str(repo),
        "name": repo.name,
        "size_total_incl_git_mb": total_mb,
        "size_worktree_mb": worktree_mb,
        "python_files": len(files),
        **commit_stats(repo),
        "import_graph": build_import_graph(
            repo, files if args.include_tests else [f for f in files if not is_test_file(f)]
        ),
        "import_graph_scope": "all files" if args.include_tests else "source only (tests excluded)",
    }
    result["python_files_by_dir"] = dict(
        Counter(f.split("/")[0] if "/" in f else "(root)" for f in files).most_common()
    )

    graph = result["import_graph"]
    print(f"== {result['name']} ==")
    print(f"commits            : {result['total_commits']} "
          f"(merges {result['merge_commits']} / non-merge {result['non_merge_commits']})")
    print(f"python files       : {result['python_files']}  {result['python_files_by_dir']}")
    print(f"size               : {result['size_worktree_mb']} MB worktree / "
          f"{result['size_total_incl_git_mb']} MB incl. .git")
    print(f"tags / releases    : {result['tags']}")
    print(f"linked commits     : {result['total_linked_commits']} "
          f"({result['pct_linked']}%) | closing keywords {result['commits_with_closing_keyword']} "
          f"({result['pct_closing']}%) | PR merges {result['merge_pull_request_subjects']} | "
          f"squash refs {result['squash_merge_subjects']}")
    print(f"unique issue/PR #  : {result['unique_referenced_numbers']}")
    print(f"merge commits tied to an issue (Incident->PR->Commit chains): "
          f"{result['merges_with_closing_keyword']} closing / "
          f"{result['merges_referencing_issue_in_body']} body ref")
    print(f"contributors       : {result['contributors']}")
    print(f"history            : {result['first_commit_date'][:10]} -> {result['head_date'][:10]} "
          f"(HEAD {result['head_sha']})")
    print(f"import graph       : {graph['internal_edges']} edges | scope: {result['import_graph_scope']} "
          f"| roots {graph['package_roots']}")
    print(f"  files importing  : {graph['files_with_internal_imports']} | "
          f"files depended on: {graph['files_depended_upon']} | "
          f"avg out-degree: {graph['avg_out_degree']} | cyclic pairs: {graph['bidirectional_pairs']}")
    print(f"  impact radius  : max depth {graph['impact_radius']['max_depth']} "
          f"(deepest: {graph['impact_radius']['deepest_node']}) | avg blast radius "
          f"{graph['impact_radius']['avg_blast_radius']} files | max "
          f"{graph['impact_radius']['max_blast_radius']} files")
    print("  cross-module couplings (top 12):")
    for a, b, n in graph["package_edges"][:12]:
        print(f"    {n:>3}  {a} -> {b}")
    print("  most depended upon:")
    for f, d in graph["most_depended_upon"]:
        print(f"    {d:>3}  {f}")
    if graph["syntax_errors"]:
        print(f"  syntax errors: {len(graph['syntax_errors'])}")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nfull JSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
