"""``:File`` -- the unit of Change Impact Analysis.

A source file, its outgoing ``:IMPORTS`` edges and its pre-computed impact
metrics. ``FileChange`` also lives here because a "file changed by a commit or a
PR" belongs to the File vocabulary, and both :mod:`models.commit` and
:mod:`models.pr` reuse it without creating a circular import.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import Edge, NexusBaseModel, Provenance, scoped_id
from .enums import ChangeType, ImportKind, NodeKind, RelationType

__all__ = ["File", "FileChange", "FileImport", "FileImpact", "make_file_id"]


def make_file_id(repository_id: str, path: str) -> str:
    """``httpie/cli`` + ``httpie/utils.py`` -> ``httpie/cli::httpie/utils.py``."""
    return scoped_id(repository_id, path)


class FileImport(BaseModel):
    """One resolved internal import: a ``(:File)-[:IMPORTS]->(:File)`` edge.

    There is exactly **one edge per (importer, imported) pair** — that is the
    granularity Change Impact Analysis needs. When the same module is imported on
    several lines (e.g. two lazy imports of ``httpie/output/ui`` in the argparser),
    ``line`` keeps the first occurrence and ``import_lines`` keeps them all, so no
    code site is lost. Aggregating here matters: Neo4j relationships have no
    uniqueness constraint, so leaving both rows in would let ``MERGE`` silently
    collapse them.
    """

    model_config = ConfigDict(extra="forbid")

    target_file_id: str = Field(..., description="``repo::path`` of the imported file.")
    target_path: str = Field(..., description="Repository-relative path.")
    imported_names: list[str] = Field(
        default_factory=list, description="Named symbols, e.g. ``['get_content_type']``."
    )
    line: int | None = Field(
        default=None, ge=1, description="1-based line of the first import statement."
    )
    import_lines: list[int] = Field(
        default_factory=list, description="Every line importing that target, sorted."
    )
    kind: ImportKind = ImportKind.FROM
    is_relative: bool = False
    is_type_checking_only: bool = Field(
        default=False, description="True when guarded by ``if TYPE_CHECKING:``."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="< 1.0 for dynamic/conditional imports resolved by heuristics.",
    )
    provenance: Provenance | None = None


class FileImpact(BaseModel):
    """Pre-computed blast radius of changing this file.

    ``transitive_dependents`` is the reverse-reachability set over ``:IMPORTS``.
    On httpie/cli the average blast radius is ~22 files with a max depth of 7,
    which is what makes the repository interesting for this project.
    """

    model_config = ConfigDict(extra="forbid")

    direct_dependents: int = Field(default=0, ge=0)
    transitive_dependents: int = Field(default=0, ge=0)
    max_depth: int = Field(default=0, ge=0)
    dependent_file_ids: list[str] = Field(
        default_factory=list, description="Truncated list of impacted file ids."
    )
    is_truncated: bool = False
    computed_at: datetime | None = None


class FileChange(BaseModel):
    """A file touched by a commit or a PR (``:MODIFIES`` / ``:TOUCHES`` payload)."""

    model_config = ConfigDict(extra="forbid")

    file_id: str
    path: str
    change_type: ChangeType = ChangeType.MODIFIED
    additions: int = Field(default=0, ge=0)
    deletions: int = Field(default=0, ge=0)
    previous_path: str | None = Field(
        default=None, description="Source path when ``change_type == renamed``."
    )
    is_binary: bool = False

    @property
    def churn(self) -> int:
        return self.additions + self.deletions


class File(NexusBaseModel):
    """A file of a repository, keyed by ``repo::path``."""

    NEO4J_LABEL = "File"
    EDGE_FIELDS = ("imports",)  # exported as :IMPORTS edges, not as a property

    id: str = Field(..., description="``owner/name::path``.")
    repository_id: str = Field(..., min_length=1)
    path: str = Field(
        ..., min_length=1, description="Repository-relative POSIX path."
    )
    module_path: str | None = Field(
        default=None, description="Dotted import path, e.g. ``httpie.cli.requestitems``."
    )
    package: str | None = Field(
        default=None, description="Owning directory, e.g. ``httpie/cli``."
    )
    top_package: str | None = Field(
        default=None, description="First path segment, e.g. ``httpie``."
    )
    extension: str | None = Field(default=None, description="``.py``.")
    language: str | None = Field(default="Python")
    is_python: bool = True
    is_test: bool = False
    is_package_init: bool = False
    loc: int | None = Field(default=None, ge=0, description="Lines of code.")
    size_bytes: int | None = Field(default=None, ge=0)
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    blob_sha: str | None = Field(default=None, description="Git blob SHA at HEAD.")
    last_modified_commit_id: str | None = None
    last_modified_at: datetime | None = None
    imports: list[FileImport] = Field(default_factory=list)
    impact: FileImpact | None = None

    @model_validator(mode="after")
    def _check_id(self) -> File:
        expected = make_file_id(self.repository_id, self.path)
        if self.id != expected:
            raise ValueError(f"File.id must be '{expected}', got '{self.id}'")
        return self

    def edges(self) -> list:
        """``CONTAINS`` (incoming from the repository) + one ``IMPORTS`` per import."""
        edges = [
            Edge.link(
                source_id=self.repository_id,
                source_label=NodeKind.REPOSITORY,
                type=RelationType.CONTAINS,
                target_id=self.id,
                target_label=NodeKind.FILE,
            )
        ]
        for imp in self.imports:
            edges.append(
                Edge.link(
                    source_id=self.id,
                    source_label=NodeKind.FILE,
                    type=RelationType.IMPORTS,
                    target_id=imp.target_file_id,
                    target_label=NodeKind.FILE,
                    confidence=imp.confidence,
                    provenance=imp.provenance,
                    imported_names=imp.imported_names,
                    line=imp.line,
                    import_lines=imp.import_lines or ([imp.line] if imp.line else []),
                    kind=imp.kind.value,
                    is_relative=imp.is_relative,
                    is_type_checking_only=imp.is_type_checking_only,
                    target_path=imp.target_path,
                )
            )
        return edges
