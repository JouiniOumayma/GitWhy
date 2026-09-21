// =====================================================================
// NEXUS-DevIntel — 02_indexes.cypher
// Lookup / range / fulltext indexes sized for the agent's traversals.
// Idempotent: every statement uses IF NOT EXISTS.
// =====================================================================

// ---------------------------------------------------------------------
// 1. Range indexes — the entry points of the agent's questions.
// ---------------------------------------------------------------------

// "every file of this repository" / "files of the httpie package"
CREATE INDEX file_repository_idx IF NOT EXISTS
FOR (n:File) ON (n.repository_id);

CREATE INDEX file_top_package_idx IF NOT EXISTS
FOR (n:File) ON (n.top_package);

// Change Impact Analysis starts from a path typed by the user.
CREATE INDEX file_path_idx IF NOT EXISTS
FOR (n:File) ON (n.path);

// Last-modified lookups (blame-style questions).
CREATE INDEX file_last_modified_idx IF NOT EXISTS
FOR (n:File) ON (n.last_modified_at);

// Pre-computed blast radius: "which files ripple the most?"
CREATE INDEX file_impact_idx IF NOT EXISTS
FOR (n:File) ON (n.transitive_dependents);

// ---------------------------------------------------------------------
// 2. Commit / PR index.
// ---------------------------------------------------------------------

CREATE INDEX commit_committed_at_idx IF NOT EXISTS
FOR (n:Commit) ON (n.committed_at);

CREATE INDEX commit_repository_idx IF NOT EXISTS
FOR (n:Commit) ON (n.repository_id);

CREATE INDEX pr_repository_state_idx IF NOT EXISTS
FOR (n:PR) ON (n.repository_id, n.state);

CREATE INDEX pr_merged_at_idx IF NOT EXISTS
FOR (n:PR) ON (n.merged_at);

// ---------------------------------------------------------------------
// 3. Incident index — severity/status filtering for triage questions.
// ---------------------------------------------------------------------

CREATE INDEX incident_status_severity_idx IF NOT EXISTS
FOR (n:Incident) ON (n.status, n.severity);

CREATE INDEX incident_opened_at_idx IF NOT EXISTS
FOR (n:Incident) ON (n.opened_at);

CREATE INDEX incident_repository_idx IF NOT EXISTS
FOR (n:Incident) ON (n.repository_id);

// ---------------------------------------------------------------------
// 4. Deployment index — "what shipped in 3.2.3?" and temporal windows.
// ---------------------------------------------------------------------

CREATE INDEX deployment_repository_version_idx IF NOT EXISTS
FOR (n:Deployment) ON (n.repository_id, n.version);

CREATE INDEX deployment_created_at_idx IF NOT EXISTS
FOR (n:Deployment) ON (n.created_at);

// ---------------------------------------------------------------------
// 5. Answer / Evidence index — replaying past audits.
// ---------------------------------------------------------------------

CREATE INDEX answer_intent_created_idx IF NOT EXISTS
FOR (n:Answer) ON (n.intent, n.created_at);

CREATE INDEX answer_confidence_idx IF NOT EXISTS
FOR (n:Answer) ON (n.confidence_score);

CREATE INDEX evidence_answer_idx IF NOT EXISTS
FOR (n:Evidence) ON (n.answer_id);

CREATE INDEX evidence_path_idx IF NOT EXISTS
FOR (n:Evidence) ON (n.path_id);

CREATE INDEX evidence_score_idx IF NOT EXISTS
FOR (n:Evidence) ON (n.score);

// ---------------------------------------------------------------------
// 6. Fulltext indexes — lexical retrieval + keyword boosting on top of
//    the pgvector similarity search (hybrid retrieval).
// ---------------------------------------------------------------------

CREATE FULLTEXT INDEX incident_fulltext_idx IF NOT EXISTS
FOR (n:Incident) ON EACH [n.title, n.body];

CREATE FULLTEXT INDEX file_fulltext_idx IF NOT EXISTS
FOR (n:File) ON EACH [n.path, n.module_path];

CREATE FULLTEXT INDEX evidence_fulltext_idx IF NOT EXISTS
FOR (n:Evidence) ON EACH [n.text, n.rationale];

CREATE FULLTEXT INDEX commit_fulltext_idx IF NOT EXISTS
FOR (n:Commit) ON EACH [n.subject, n.body];

CREATE FULLTEXT INDEX pr_fulltext_idx IF NOT EXISTS
FOR (n:PR) ON EACH [n.title, n.body];

// ---------------------------------------------------------------------
// 7. Relationship indexes — filtering the traversal itself.
// ---------------------------------------------------------------------

// "imports on line < N" is a common debugger-driven question.
CREATE INDEX imports_line_idx IF NOT EXISTS
FOR ()-[r:IMPORTS]-() ON (r.line);

// Churn-based ranking of a change set.
CREATE INDEX modifies_churn_idx IF NOT EXISTS
FOR ()-[r:MODIFIES]-() ON (r.churn);

// Trust-weighting of the Incident <-> PR/Commit links (see LinkMethod).
CREATE INDEX closes_confidence_idx IF NOT EXISTS
FOR ()-[r:CLOSES]-() ON (r.confidence);

// ---------------------------------------------------------------------
// 8. Optional — native Neo4j vector index.
//    NEXUS-DevIntel keeps embeddings in PostgreSQL/pgvector (see
//    schema/postgres/001_init.sql). Uncomment if you also want the
//    sub-graph retrieval to be vector-native; requires Neo4j >= 5.11.
// ---------------------------------------------------------------------
// CREATE VECTOR INDEX evidence_embedding_idx IF NOT EXISTS
// FOR (n:Evidence) ON (n.embedding)
// OPTIONS { indexConfig: {
//   `vector.dimensions`: 1024,
//   `vector.similarity_function`: 'cosine'
// }};
