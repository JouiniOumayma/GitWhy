// =====================================================================
// NEXUS-DevIntel — 02_constraints_existence_enterprise.cypher
//
//   ⚠  NEO4J ENTERPRISE EDITION ONLY  ⚠
//
// The docker-compose.yml of this project runs neo4j:5.26-community, on which
// `REQUIRE n.x IS NOT NULL` fails with:
//   Neo.DatabaseError.Schema.ConstraintCreationFailed:
//   Property existence constraint requires Neo4j Enterprise Edition.
//
// That is why these statements live in their own file instead of being part of
// 01_constraints_uniqueness.cypher: applying them on Community aborts the whole
// script halfway, leaving a half-configured database. The loader skips this file
// unless you pass --enterprise.
//
// On Community the same guarantees are enforced two other ways:
//   * at write time  : build_fixtures.py cannot emit a missing field (Pydantic
//                      raises before any JSON is written);
//   * at load time   : scripts/load_neo4j.py::verify_provenance() runs an
//                      assertion query after loading and reports any node whose
//                      prov_source is missing (fail loud, not fail silent).
// Switch to Neo4j Enterprise and apply this file if you want the database itself
// to refuse an unprovenanced node.
// =====================================================================

// ---------------------------------------------------------------------
// Identity properties the traversals rely on.
// ---------------------------------------------------------------------
CREATE CONSTRAINT repository_id_not_null IF NOT EXISTS
FOR (n:Repository) REQUIRE n.id IS NOT NULL;

CREATE CONSTRAINT file_repository_not_null IF NOT EXISTS
FOR (n:File) REQUIRE n.repository_id IS NOT NULL;

CREATE CONSTRAINT file_path_not_null IF NOT EXISTS
FOR (n:File) REQUIRE n.path IS NOT NULL;

CREATE CONSTRAINT commit_repository_not_null IF NOT EXISTS
FOR (n:Commit) REQUIRE n.repository_id IS NOT NULL;

CREATE CONSTRAINT commit_subject_not_null IF NOT EXISTS
FOR (n:Commit) REQUIRE n.subject IS NOT NULL;

CREATE CONSTRAINT pr_number_not_null IF NOT EXISTS
FOR (n:PR) REQUIRE n.number IS NOT NULL;

CREATE CONSTRAINT pr_repository_not_null IF NOT EXISTS
FOR (n:PR) REQUIRE n.repository_id IS NOT NULL;

CREATE CONSTRAINT deployment_tag_not_null IF NOT EXISTS
FOR (n:Deployment) REQUIRE n.tag IS NOT NULL;

CREATE CONSTRAINT incident_number_not_null IF NOT EXISTS
FOR (n:Incident) REQUIRE n.number IS NOT NULL;

CREATE CONSTRAINT incident_title_not_null IF NOT EXISTS
FOR (n:Incident) REQUIRE n.title IS NOT NULL;

CREATE CONSTRAINT evidence_kind_not_null IF NOT EXISTS
FOR (n:Evidence) REQUIRE n.kind IS NOT NULL;

CREATE CONSTRAINT answer_question_not_null IF NOT EXISTS
FOR (n:Answer) REQUIRE n.question IS NOT NULL;

// ---------------------------------------------------------------------
// Provenance: an unprovenanced node is unauditable, so it must not exist.
// ---------------------------------------------------------------------
CREATE CONSTRAINT repository_provenance_source IF NOT EXISTS
FOR (n:Repository) REQUIRE n.prov_source IS NOT NULL;

CREATE CONSTRAINT file_provenance_source IF NOT EXISTS
FOR (n:File) REQUIRE n.prov_source IS NOT NULL;

CREATE CONSTRAINT commit_provenance_source IF NOT EXISTS
FOR (n:Commit) REQUIRE n.prov_source IS NOT NULL;

CREATE CONSTRAINT pr_provenance_source IF NOT EXISTS
FOR (n:PR) REQUIRE n.prov_source IS NOT NULL;

CREATE CONSTRAINT deployment_provenance_source IF NOT EXISTS
FOR (n:Deployment) REQUIRE n.prov_source IS NOT NULL;

CREATE CONSTRAINT incident_provenance_source IF NOT EXISTS
FOR (n:Incident) REQUIRE n.prov_source IS NOT NULL;

CREATE CONSTRAINT evidence_provenance_source IF NOT EXISTS
FOR (n:Evidence) REQUIRE n.prov_source IS NOT NULL;

CREATE CONSTRAINT answer_provenance_source IF NOT EXISTS
FOR (n:Answer) REQUIRE n.prov_source IS NOT NULL;
