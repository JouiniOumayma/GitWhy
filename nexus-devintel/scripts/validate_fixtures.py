#!/usr/bin/env python3
"""Validate the JSON fixtures before they reach the database.

Two layers of checks:

1. **Schema** -- every record must deserialize into its Pydantic model. This
   catches typos in enum values, bad id formats (a File id that is not
   ``repo::path``), unknown fields, etc.
2. **Referential integrity** -- every id referenced by a relationship or by an
   embedded payload (``FileChange.file_id``, ``IssueRef.incident_id``,
   ``EvidenceRef.node_id``, ...) must exist as a node in the fixture set. A
   dangling reference is invisible once the loader silently skips it, and shows
   up much later as a missing edge in an EvidencePath.

Usage::

    python scripts/validate_fixtures.py [--fixtures-dir fixtures]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    Answer,
    Commit,
    Deployment,
    Evidence,
    File,
    Incident,
    PersonRef,
    PullRequest,
    Repository,
    RelationType,
)

FIXTURE_MODELS = {
    "repositories": Repository,
    "files": File,
    "commits": Commit,
    "pull_requests": PullRequest,
    "deployments": Deployment,
    "incidents": Incident,
    "evidence": Evidence,
    "answers": Answer,
}

LABEL_OF = {
    "repositories": "Repository",
    "files": "File",
    "commits": "Commit",
    "pull_requests": "PR",
    "deployments": "Deployment",
    "incidents": "Incident",
    "evidence": "Evidence",
    "answers": "Answer",
}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def load(models: dict) -> tuple[dict[str, list], Report]:
    report = Report()
    loaded: dict[str, list] = {}
    for stem, model in models.items():
        path = FIXTURES_DIR / f"{stem}.json"
        if not path.exists():
            report.error(f"{path.name}: missing fixture file")
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = []
        for index, payload in enumerate(raw):
            try:
                records.append(model.model_validate(payload))
            except Exception as exc:  # pydantic ValidationError
                report.error(f"{path.name}[{index}] ({payload.get('id', '?')}): {exc}")
        loaded[stem] = records
        if len(records) != len(raw):
            report.error(f"{path.name}: {len(raw) - len(records)} record(s) failed to validate")
    return loaded, report


FIXTURES_DIR = PROJECT_ROOT / "fixtures"


def main() -> int:
    global FIXTURES_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", default=str(PROJECT_ROOT / "fixtures"))
    args = parser.parse_args()
    FIXTURES_DIR = Path(args.fixtures_dir).resolve()

    report = Report()
    data, load_report = load(FIXTURE_MODELS)
    report.errors.extend(load_report.errors)

    # ------------------------------------------------------------------ #
    # Node id universe, per label.
    # ------------------------------------------------------------------ #
    ids_by_label: dict[str, set[str]] = defaultdict(set)
    for stem, label in LABEL_OF.items():
        for record in data.get(stem, []):
            ids_by_label[label].add(record.id)

    # Person nodes are created on the fly by the loader from embedded PersonRef.
    for stem in ("commits", "pull_requests", "incidents"):
        for record in data.get(stem, []):
            for attr in ("author", "committer", "reporter", "merged_by"):
                person = getattr(record, attr, None)
                if isinstance(person, PersonRef):
                    ids_by_label["Person"].add(person.id)
                    if person.email and person.id != person.email:
                        report.warn(
                            f"{record.id}: Person.id ({person.id}) != email ({person.email}); "
                            "idents are matched on email, this will create a duplicate node"
                        )
                    if not person.email:
                        report.warn(f"{record.id}: Person {person.id} has no email, id is not stable across forges")

    ids_by_label["Repository"].add("psf/requests")  # seeded by 03_bootstrap.cypher

    # ------------------------------------------------------------------ #
    # Check every reference.
    # ------------------------------------------------------------------ #
    def check(label: str, node_id: str, where: str, severity: str = "error") -> None:
        if node_id in ids_by_label[label]:
            return
        message = f"{where}: {label} '{node_id}' does not exist in the fixtures"
        report.error(message) if severity == "error" else report.warn(message)

    # Files
    for record in data.get("files", []):
        for imp in record.imports:
            check("File", imp.target_file_id, f"{record.id} IMPORTS", "warn")

    # Commits
    for record in data.get("commits", []):
        for change in record.files_changed:
            check("File", change.file_id, f"{record.id} MODIFIES")
        for ref in record.issue_refs:
            check("Incident", ref.incident_id, f"{record.id} {ref.relation}")
        for parent in record.parents:
            check("Commit", parent, f"{record.id} PARENT_OF")

    # Pull requests
    for record in data.get("pull_requests", []):
        for change in record.files_changed:
            check("File", change.file_id, f"{record.id} TOUCHES")
        for link in record.closes_issues:
            check("Incident", link.incident_id, f"{record.id} CLOSES", "warn")
        if record.merge_commit_id:
            check("Commit", record.merge_commit_id, f"{record.id} MERGED_INTO")

    # Deployments
    for record in data.get("deployments", []):
        if record.commit_id:
            check("Commit", record.commit_id, f"{record.id} DEPLOYED_AT")
        elif record.repository_id == "httpie/cli":
            report.warn(f"{record.id}: no commit_id, DEPLOYED_AT will be skipped")
        if record.previous_deployment_id:
            check("Deployment", record.previous_deployment_id, f"{record.id} PRECEDES")

    # Incidents
    for record in data.get("incidents", []):
        for link in record.affected_files:
            check("File", link.file_id, f"{record.id} AFFECTS")
        for cause in record.caused_by:
            check("Incident", cause.incident_id, f"{record.id} CAUSED_BY")
        for deployment_id in record.observed_in:
            check("Deployment", deployment_id, f"{record.id} OBSERVED_IN")
        for related in record.related_incident_ids:
            check("Incident", related, f"{record.id} related")
        if record.resolution_commit_ids:
            for sha in record.resolution_commit_ids:
                check("Commit", sha, f"{record.id} resolution_commit", "warn")
        if record.status.value == "closed" and not (
            record.resolution_commit_ids or record.resolution_pr_numbers
        ):
            report.warn(f"{record.id}: closed without any resolution commit or PR")
        # A high-severity incident with no implicated file is only suspicious when
        # we actually ingested that repository's sources.
        if (
            record.severity.value in {"sev1", "sev2"}
            and not record.affected_files
            and record.repository_id == "httpie/cli"
        ):
            report.warn(f"{record.id}: {record.severity.value} incident with no AFFECTS file")

    # Evidence
    for record in data.get("evidence", []):
        for ref in record.node_references:
            check(ref.node_kind.value, ref.node_id, f"{record.id} REFERENCES")
        for parent in record.derived_from:
            check("Evidence", parent, f"{record.id} DERIVED_FROM")
        if record.answer_id:
            check("Answer", record.answer_id, f"{record.id} -> answer")

    # Answers
    for record in data.get("answers", []):
        for evidence_id in record.evidence_ids:
            check("Evidence", evidence_id, f"{record.id} SUPPORTED_BY")
        for citation in record.citations:
            check(citation.node_kind.value, citation.node_id, f"{record.id} citation")
        if record.evidence_path:
            for hop in record.evidence_path.hops:
                check(hop.node_kind.value, hop.node_id, f"{record.id} hop {hop.step}")
            if record.evidence_path.total_hops != len(record.evidence_path.hops):
                report.error(f"{record.id}: total_hops mismatch")
        if record.status.value == "ok" and not record.evidence_ids:
            report.error(f"{record.id}: status=ok with zero evidence")
        if record.status.value == "insufficient_evidence" and record.evidence_ids:
            report.warn(f"{record.id}: insufficient_evidence but evidence is attached")

    # ------------------------------------------------------------------ #
    # Derived edges (the same objects the loader will write).
    # ------------------------------------------------------------------ #
    edge_counts: Counter[str] = Counter()
    edge_reports: list[str] = []
    for stem in FIXTURE_MODELS:
        for record in data.get(stem, []):
            for edge in record.edges():
                edge_counts[edge.type.value] += 1
                if edge.type not in RelationType:
                    report.error(f"unknown relation type {edge.type} in {record.id}")
                check(edge.source_label.value, edge.source_id, f"{record.id} edge source", "warn")
                check(edge.target_label.value, edge.target_id, f"{record.id} {edge.type.value}", "warn")
                edge_reports.append(f"{record.id} -[{edge.type.value}]-> {edge.target_id}")

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    print("== Fixture inventory ==")
    for stem, label in LABEL_OF.items():
        print(f"  {label:<12} {len(data.get(stem, [])):>4}")
    print(f"  {'Person':<12} {len(ids_by_label['Person']):>4}  (derived from embedded refs)")
    print("\n== Derived relationships ==")
    for relation, count in sorted(edge_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {relation:<16} {count:>4}")
    print(f"  {'TOTAL':<16} {sum(edge_counts.values()):>4}")

    print("\n== Change Impact Analysis on the fixtures ==")
    ranked = sorted(
        (r for r in data.get("files", []) if r.impact and r.impact.transitive_dependents),
        key=lambda r: -r.impact.transitive_dependents,
    )
    for record in ranked[:5]:
        print(
            f"  {record.impact.transitive_dependents:>3} files impacted by changing "
            f"{record.path:<40} (depth {record.impact.max_depth}, "
            f"{record.impact.direct_dependents} direct)"
        )
    deepest = max(
        (r for r in data.get("files", []) if r.impact), key=lambda r: r.impact.max_depth,
        default=None,
    )
    if deepest is not None:
        print(f"  deepest chain: {deepest.path} (depth {deepest.impact.max_depth})")

    if report.warnings:
        print(f"\n== {len(report.warnings)} warning(s) ==")
        for message in report.warnings:
            print(f"  ! {message}")

    if report.errors:
        print(f"\n== {len(report.errors)} ERROR(S) ==")
        for message in report.errors:
            print(f"  x {message}")
        return 1

    print("\nOK: fixtures are valid and referentially closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
