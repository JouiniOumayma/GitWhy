"""Base machinery shared by every NEXUS-DevIntel contract.

Three concerns live here:

* :class:`Provenance` -- a uniform traceability block attached to every node and
  every edge (``source``, ``ingested_at``, ``confidence``, ...). It is what makes
  an ``EvidencePath`` auditable and what drives the answer confidence score.
* :class:`NexusBaseModel` -- a Pydantic v2 base class that knows its Neo4j label,
  flattens itself into Neo4j-compatible properties (Neo4j stores neither maps nor
  nested objects) and derives its relationships via :meth:`NexusBaseModel.edges`.
* :class:`Edge` -- a loader-neutral description of a relationship, so a model can
  emit ``(:Commit)-[:MODIFIES]->(:File)`` without knowing Cypher.

Design rules
------------
* Every entity carries a globally stable ``id`` (see ``schema/README.md`` for the
  id scheme). Ids never contain the ingestion timestamp, so re-ingesting the same
  repository is idempotent (``MERGE`` on ``id``).
* ``provenance`` is flattened with a ``prov_`` prefix (``prov_source``,
  ``prov_ingested_at``, ``prov_confidence``, ...) so it stays queryable in Cypher
  instead of being buried in a JSON string.
* Any other nested value is stored as a canonical JSON string: Neo4j cannot store
  maps, and round-tripping through JSON keeps the raw payload available.
"""

from __future__ import annotations

import json
from abc import ABC
from datetime import date, datetime, timezone
from typing import Any, ClassVar, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .enums import LinkMethod, NodeKind, RelationType, SourceKind

__all__ = [
    "Edge",
    "IngestionMeta",
    "NexusBaseModel",
    "PersonRef",
    "Provenance",
    "scoped_id",
    "utcnow",
]

SCHEMA_VERSION = "0.1.0"
ID_SEP = "::"  # separator between a repository scope and a local identifier
PROVENANCE_PREFIX = "prov_"


def utcnow() -> datetime:
    """Timezone-aware "now", the single clock used for ingestion timestamps."""
    return datetime.now(timezone.utc)


def scoped_id(scope: str, local: str) -> str:
    """Build an id that is unique across repositories: ``httpie/cli::httpie/utils.py``."""
    return f"{scope}{ID_SEP}{local}"


class Provenance(BaseModel):
    """Traceability metadata attached to every node and edge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: SourceKind = SourceKind.GIT
    source_uri: str | None = Field(
        default=None,
        description="Concrete locator: git object, API URL, file path, ticket URL...",
    )
    ingested_at: datetime = Field(default_factory=utcnow)
    extractor: str = Field(default="nexus-devintel.ingestion", min_length=1)
    extractor_version: str = Field(default=SCHEMA_VERSION, min_length=1)
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in this single fact. 1.0 = observed directly (git object), "
            "lower values = inferred (regex issue link, static-analysis import)."
        ),
    )

    def flattened(self) -> dict[str, Any]:
        """``prov_*`` properties, ready to be merged into a Neo4j node."""
        return {
            "prov_source": self.source.value,
            "prov_source_uri": self.source_uri,
            "prov_ingested_at": self.ingested_at,
            "prov_extractor": self.extractor,
            "prov_extractor_version": self.extractor_version,
            "prov_confidence": self.confidence,
        }


class IngestionMeta(BaseModel):
    """Free-form counters produced by a pipeline run (kept out of the graph)."""

    model_config = ConfigDict(extra="allow")

    run_id: str | None = None
    duration_ms: int | None = None


class PersonRef(BaseModel):
    """Lightweight author/committer reference carried inside other models.

    The loader ``MERGE``s a ``(:Person {id})`` node for each reference, so a
    Commit fixture can name its author without a separate ``persons.json`` entry.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Normalized email, else login.")
    login: str | None = None
    name: str | None = None
    email: str | None = None

    @classmethod
    def from_email(cls, email: str, name: str | None = None) -> PersonRef:
        normalized = email.strip().lower()
        return cls(id=normalized, email=normalized, name=name, login=email.split("@")[0].lower())


class Edge(BaseModel):
    """A relationship to materialize in Neo4j.

    ``source_id``/``target_id`` are the ``id`` of the already-ingested endpoint
    nodes; the loader resolves them with ``MATCH (a {id}) MATCH (b {id}) MERGE``.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1)
    source_label: NodeKind
    type: RelationType
    target_id: str = Field(..., min_length=1)
    target_label: NodeKind
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance | None = None

    def to_parameters(self, *, native_temporal: bool = True) -> dict[str, Any]:
        props = _clean(
            {k: _neo4j_value(v, native_temporal) for k, v in self.properties.items()}
        )
        if self.provenance is not None:
            props.update(_clean(self.provenance.flattened()))
        return props

    @classmethod
    def link(
        cls,
        *,
        source_id: str,
        source_label: NodeKind,
        type: RelationType,
        target_id: str,
        target_label: NodeKind,
        confidence: float | None = None,
        method: LinkMethod | None = None,
        provenance: Provenance | None = None,
        **properties: Any,
    ) -> Edge:
        """Convenience constructor used by ``Model.edges()`` implementations."""
        props = dict(properties)
        if confidence is not None:
            props["confidence"] = confidence
        if method is not None:
            props["link_method"] = (
                method.value if isinstance(method, LinkMethod) else str(method)
            )
        return cls(
            source_id=source_id,
            source_label=source_label,
            type=type,
            target_id=target_id,
            target_label=target_label,
            properties=props,
            provenance=provenance,
        )


# --------------------------------------------------------------------------- #
# Neo4j value conversion
# --------------------------------------------------------------------------- #
def _neo4j_value(value: Any, native_temporal: bool) -> Any:
    """Convert a Python value into something the Neo4j driver accepts."""
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, (str, int, float)):
        return value
    if isinstance(value, datetime):
        return value if native_temporal else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return json.dumps(value.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if all(isinstance(i, (str, bool, int, float)) for i in items):
            return items  # Neo4j native array of primitives
        if all(isinstance(i, datetime) for i in items):
            return items if native_temporal else [i.isoformat() for i in items]
        return json.dumps(
            [i.model_dump(mode="json") if isinstance(i, BaseModel) else i for i in items],
            sort_keys=True,
            ensure_ascii=False,
        )
    return json.dumps(str(value), ensure_ascii=False)


def _clean(mapping: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values: Neo4j cannot store them, and ``SET`` would erase data."""
    return {k: v for k, v in mapping.items() if v is not None}


# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #
class NexusBaseModel(BaseModel, ABC):
    """Parent of every node contract."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    #: Neo4j label(s) applied to the node. Subclasses must override.
    NEO4J_LABEL: ClassVar[str] = "Entity"

    #: Fields materialized as relationships rather than as properties.
    EDGE_FIELDS: ClassVar[Sequence[str]] = ()

    id: str = Field(..., min_length=1, description="Stable, globally unique identifier.")
    provenance: Provenance = Field(
        default_factory=Provenance,
        description="Where this fact comes from and how much we trust it.",
    )

    # ---- identity ---------------------------------------------------------- #
    @classmethod
    def label(cls) -> str:
        return cls.NEO4J_LABEL

    @property
    def node_kind(self) -> NodeKind:
        return NodeKind(self.NEO4J_LABEL)

    # ---- Neo4j serialization ---------------------------------------------- #
    def _property_fields(self) -> Iterable[str]:
        """Fields exported as node properties (everything but the edge payloads)."""
        skip = set(self.EDGE_FIELDS) | {"provenance"}
        for name in type(self).model_fields:
            if name in skip:
                continue
            yield name

    def to_neo4j_properties(
        self, *, native_temporal: bool = True, include_provenance: bool = True
    ) -> dict[str, Any]:
        """Flat, Neo4j-compatible property dictionary.

        Args:
            native_temporal: keep ``datetime`` objects so the driver stores real
                Neo4j ``DateTime`` values (queryable with ``datetime()``). Pass
                ``False`` to get ISO-8601 strings (cypher-shell friendly).
            include_provenance: add the ``prov_*`` block.
        """
        props: dict[str, Any] = {}
        for name in self._property_fields():
            props[name] = _neo4j_value(getattr(self, name), native_temporal)
        if include_provenance:
            props.update(
                {
                    k: _neo4j_value(v, native_temporal)
                    for k, v in self.provenance.flattened().items()
                }
            )
        return _clean(props)

    def to_neo4j_node(self, *, native_temporal: bool = True) -> dict[str, Any]:
        """``{"labels": [...], "id": ..., "properties": {...}}`` for the loader."""
        return {
            "labels": [self.NEO4J_LABEL],
            "id": self.id,
            "properties": self.to_neo4j_properties(native_temporal=native_temporal),
        }

    # ---- relationships ----------------------------------------------------- #
    def edges(self) -> list[Edge]:
        """Relationships this node owns, given only its own payload."""
        return []

    # ---- helpers ----------------------------------------------------------- #
    def model_dump_json_safe(self, **kwargs: Any) -> dict[str, Any]:
        """JSON-safe dict (datetimes as ISO strings) for fixtures and pgvector rows."""
        return self.model_dump(mode="json", **kwargs)
