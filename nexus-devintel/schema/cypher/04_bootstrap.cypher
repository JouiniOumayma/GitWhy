// =====================================================================
// NEXUS-DevIntel — 03_bootstrap.cypher
// Run *after* 01_constraints.cypher and 02_indexes.cypher.
//   1. stamps the schema version into the graph
//   2. seeds the two repositories used by the fixtures
//   3. sanity-checks that constraints and indexes are in place
// Idempotent: safe to re-run.
// Apply with: docker exec -i nexus-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD < schema/cypher/03_bootstrap.cypher
// =====================================================================

// ---------------------------------------------------------------------
// 1. Schema metadata — the agent refuses to answer if the graph was built
//    with a schema version it does not know.
// ---------------------------------------------------------------------
MERGE (m:SchemaMeta {id: 'nexus-devintel'})
SET m.schema_version = '0.1.0',
    m.contracts      = ['Repository', 'File', 'Commit', 'PR', 'Deployment',
                        'Incident', 'Person', 'Evidence', 'Answer'],
    m.relations      = ['CONTAINS', 'HAS_COMMIT', 'HAS_DEPLOYMENT', 'IMPORTS',
                        'PARENT_OF', 'MODIFIES', 'AUTHORED', 'COMMITTED',
                        'OPENED', 'MERGED_INTO', 'TOUCHES', 'CLOSES',
                        'REFERENCES', 'REPORTED', 'AFFECTS', 'OBSERVED_IN',
                        'CAUSED_BY', 'DEPLOYED_AT', 'DEPLOYED_AS', 'PRECEDES',
                        'SUPPORTED_BY', 'DERIVED_FROM', 'HAS_CHUNK'],
    m.updated_at     = datetime();

// ---------------------------------------------------------------------
// 2. Repository seeds (the fixtures reference them; ingestion MERGEs the
//    real metrics on top, it never duplicates the node).
// ---------------------------------------------------------------------
MERGE (r:Repository {id: 'httpie/cli'})
ON CREATE SET r.host = 'github.com', r.owner = 'httpie', r.name = 'cli',
              r.default_branch = 'master', r.primary_language = 'Python',
              r.url = 'https://github.com/httpie/cli',
              r.prov_source = 'manual', r.prov_confidence = 1.0,
              r.prov_extractor = 'schema-bootstrap',
              r.prov_extractor_version = '0.1.0',
              r.prov_ingested_at = datetime()
SET r.is_demo_primary = true;

MERGE (r:Repository {id: 'psf/requests'})
ON CREATE SET r.host = 'github.com', r.owner = 'psf', r.name = 'requests',
              r.default_branch = 'main', r.primary_language = 'Python',
              r.url = 'https://github.com/psf/requests',
              r.prov_source = 'manual', r.prov_confidence = 1.0,
              r.prov_extractor = 'schema-bootstrap',
              r.prov_extractor_version = '0.1.0',
              r.prov_ingested_at = datetime()
SET r.is_demo_secondary = true;

// ---------------------------------------------------------------------
// 3. Sanity checks.
// ---------------------------------------------------------------------

// 3a. Constraints expected on a healthy graph.
SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties
WHERE labelsOrTypes IS NOT NULL
RETURN name, type, labelsOrTypes, properties
ORDER BY name;

// 3b. Indexes (must include the fulltext ones used by hybrid retrieval).
SHOW INDEXES YIELD name, type, labelsOrTypes, state
WHERE type = 'FULLTEXT' OR type = 'RANGE'
RETURN name, type, labelsOrTypes, state
ORDER BY type, name;

// 3c. Schema version currently installed.
MATCH (m:SchemaMeta {id: 'nexus-devintel'})
RETURN m.schema_version AS schema_version, m.updated_at AS updated_at;
