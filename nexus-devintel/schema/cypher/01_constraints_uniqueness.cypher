// =====================================================================
// NEXUS-DevIntel — 01_constraints_uniqueness.cypher
// Uniqueness constraints, one per node label.
//
// Requires: Neo4j Community Edition (all editions support uniqueness).
// Idempotent: every statement uses IF NOT EXISTS.
//
// Apply with:
//   docker exec -i nexus-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
//     < schema/cypher/01_constraints_uniqueness.cypher
//
// These are not optional. Without a uniqueness constraint on `id`:
//   * MERGE can create duplicate nodes under concurrency, silently splitting
//     the graph (two (:File) nodes for the same path);
//   * the loader loses the unique index that makes the batch UNWIND/MERGE fast.
//
// Id scheme (see schema/README.md):
//   Repository : owner/name
//   File       : owner/name::path
//   Commit     : <full sha>
//   PR         : owner/name#number
//   Deployment : owner/name@tag
//   Incident   : owner/name#issue-number
//   Person     : normalized email (fallback: login)
//   Evidence   : ev-<slug>-<n>
//   Answer     : ans-<timestamp>-<n>
//   CodeChunk  : owner/name::path#L<start>-L<end>   (reserved, pgvector layer)
// =====================================================================

CREATE CONSTRAINT repository_id_unique IF NOT EXISTS
FOR (n:Repository) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT file_id_unique IF NOT EXISTS
FOR (n:File) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT commit_id_unique IF NOT EXISTS
FOR (n:Commit) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT pr_id_unique IF NOT EXISTS
FOR (n:PR) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT deployment_id_unique IF NOT EXISTS
FOR (n:Deployment) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT incident_id_unique IF NOT EXISTS
FOR (n:Incident) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT person_id_unique IF NOT EXISTS
FOR (n:Person) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT evidence_id_unique IF NOT EXISTS
FOR (n:Evidence) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT answer_id_unique IF NOT EXISTS
FOR (n:Answer) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT code_chunk_id_unique IF NOT EXISTS
FOR (n:CodeChunk) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT schema_meta_id_unique IF NOT EXISTS
FOR (n:SchemaMeta) REQUIRE n.id IS UNIQUE;
