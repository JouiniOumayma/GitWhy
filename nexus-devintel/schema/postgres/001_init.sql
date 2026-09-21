-- =====================================================================
-- NEXUS-DevIntel — PostgreSQL / pgvector bootstrap
-- Mounted into /docker-entrypoint-initdb.d: executed once, on first start
-- of an empty data volume.
--
-- Two responsibilities:
--   1. store the embeddings the graph cannot hold (code chunks, evidence)
--   2. expose the similarity-search primitives the retriever calls
--
-- Dimensions: 1024 matches BAAI/bge-m3 and intfloat/e5-large-v2.
-- For OpenAI text-embedding-3-small (1536) change VECTOR_DIM below AND
-- the dimension in the table definitions, then re-create the volume.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------
-- 1. code_chunks — the embeddable unit produced by chunking each File.
--    `file_id` mirrors (:File).id so a chunk can always be re-attached to
--    its graph node:  "httpie/cli::httpie/utils.py"
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS code_chunks (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id   text        NOT NULL,
    file_id         text        NOT NULL,
    path            text        NOT NULL,
    module_path     text,
    language        text        DEFAULT 'python',
    symbol          text,        -- function/class the chunk belongs to, when known
    start_line      integer     NOT NULL CHECK (start_line >= 1),
    end_line        integer     NOT NULL CHECK (end_line >= start_line),
    content         text        NOT NULL,
    content_hash    text        NOT NULL,   -- sha256, for idempotent re-ingestion
    token_count     integer,
    commit_id       text,                   -- blob revision the chunk was taken from
    -- vector payload
    embedding       vector(1024),
    embedding_model text        DEFAULT 'BAAI/bge-m3',
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (file_id, start_line, end_line)
);

CREATE INDEX IF NOT EXISTS code_chunks_file_idx
    ON code_chunks (file_id);

CREATE INDEX IF NOT EXISTS code_chunks_repository_idx
    ON code_chunks (repository_id);

CREATE INDEX IF NOT EXISTS code_chunks_path_trgm_idx
    ON code_chunks USING gin (path gin_trgm_ops);

-- Lexical half of the hybrid retrieval.
CREATE INDEX IF NOT EXISTS code_chunks_fts_idx
    ON code_chunks USING gin (to_tsvector('english', content));

-- Approximate nearest neighbour (cosine). HNSW is preferred over IVFFlat
-- here because the corpus grows continuously during ingestion and HNSW
-- needs no periodic re-training.
CREATE INDEX IF NOT EXISTS code_chunks_embedding_hnsw_idx
    ON code_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------
-- 2. evidence_embeddings — one row per (:Evidence) node.
--    `evidence_id` is NOT a foreign key on purpose: evidence can outlive a
--    re-ingestion of the graph, and the ids are the join key.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence_embeddings (
    evidence_id     text        PRIMARY KEY,
    repository_id   text,
    answer_id       text,
    path_id         text,
    kind            text,        -- mirrors EvidenceKind
    node_kind       text,        -- mirrors NodeKind of the referenced node
    node_id         text,
    text            text        NOT NULL,
    metadata        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector(1024),
    embedding_model text        DEFAULT 'BAAI/bge-m3',
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS evidence_embeddings_answer_idx
    ON evidence_embeddings (answer_id);

CREATE INDEX IF NOT EXISTS evidence_embeddings_node_idx
    ON evidence_embeddings (node_id);

CREATE INDEX IF NOT EXISTS evidence_embeddings_fts_idx
    ON evidence_embeddings USING gin (to_tsvector('english', text));

CREATE INDEX IF NOT EXISTS evidence_embeddings_hnsw_idx
    ON evidence_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------
-- 3. ingestion_runs — provenance ledger for the ingestion pipeline.
--    Every node carries prov_* properties in Neo4j; this table records the
--    runs themselves so an answer can be traced back to a graph snapshot.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id     text        NOT NULL,
    revision          text,                  -- commit sha the snapshot was taken at
    extractor         text        NOT NULL DEFAULT 'nexus-devintel.ingestion',
    extractor_version text        NOT NULL DEFAULT '0.1.0',
    status            text        NOT NULL DEFAULT 'running'
                      CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz,
    stats             jsonb       NOT NULL DEFAULT '{}'::jsonb,
    error             text
);

CREATE INDEX IF NOT EXISTS ingestion_runs_repository_idx
    ON ingestion_runs (repository_id, started_at DESC);

-- ---------------------------------------------------------------------
-- 4. Retrieval primitives.
--    The retriever (LangGraph node) calls these; keeping the SQL here means
--    the embedding dimension and the distance operator live in one place.
-- ---------------------------------------------------------------------

-- 4a. Pure vector search over code chunks.
--
-- The DROP before each CREATE is what makes this file *re-runnable*: it is mounted
-- into /docker-entrypoint-initdb.d (first boot only), but during Week 2 you will
-- iterate on these functions and want to apply them with a plain
--   docker exec -i nexus-postgres psql -U nexus -d nexus < schema/postgres/001_init.sql
-- without recreating the volume. `CREATE OR REPLACE` alone is not enough: adding
-- an output column changes the return type and is rejected.
DROP FUNCTION IF EXISTS match_code_chunks(vector, integer, text, text);

CREATE OR REPLACE FUNCTION match_code_chunks(
    query_embedding vector(1024),
    match_count     integer DEFAULT 10,
    filter_repository text DEFAULT NULL,
    filter_path_regex text DEFAULT NULL
)
RETURNS TABLE (
    chunk_id     uuid,
    file_id      text,
    path         text,
    symbol       text,
    start_line   integer,
    end_line     integer,
    content      text,
    similarity   double precision
)
LANGUAGE sql STABLE
AS $$
    SELECT c.id,
           c.file_id,
           c.path,
           c.symbol,
           c.start_line,
           c.end_line,
           c.content,
           1 - (c.embedding <=> query_embedding) AS similarity
    FROM code_chunks c
    WHERE c.embedding IS NOT NULL
      AND (filter_repository IS NULL OR c.repository_id = filter_repository)
      AND (filter_path_regex IS NULL OR c.path ~ filter_path_regex)
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- 4b. Pure vector search over evidence (used to re-use earlier investigations).
DROP FUNCTION IF EXISTS match_evidence(vector, integer, text);

CREATE OR REPLACE FUNCTION match_evidence(
    query_embedding vector(1024),
    match_count     integer DEFAULT 10,
    filter_answer   text DEFAULT NULL
)
RETURNS TABLE (
    evidence_id  text,
    node_kind    text,
    node_id      text,
    text_content text,
    similarity   double precision
)
LANGUAGE sql STABLE
AS $$
    SELECT e.evidence_id,
           e.node_kind,
           e.node_id,
           e.text,
           1 - (e.embedding <=> query_embedding) AS similarity
    FROM evidence_embeddings e
    WHERE e.embedding IS NOT NULL
      AND (filter_answer IS NULL OR e.answer_id = filter_answer)
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- 4c. Hybrid retrieval: reciprocal rank fusion of the lexical and the
--     vector ranking. k = 60 as in the original RRF paper.
DROP FUNCTION IF EXISTS hybrid_search_code_chunks(text, vector, integer, integer, text);

CREATE OR REPLACE FUNCTION hybrid_search_code_chunks(
    query_text      text,
    query_embedding vector(1024),
    match_count     integer DEFAULT 10,
    rrf_k           integer DEFAULT 60,
    filter_repository text DEFAULT NULL
)
RETURNS TABLE (
    chunk_id   uuid,
    file_id    text,
    path       text,
    symbol     text,
    start_line integer,
    end_line   integer,
    content    text,
    rrf_score  double precision,
    vector_rank integer,
    lexical_rank integer
)
LANGUAGE sql STABLE
AS $$
WITH lexical AS (
    SELECT c.id,
           row_number() OVER (
               ORDER BY ts_rank_cd(to_tsvector('english', c.content),
                                   plainto_tsquery('english', query_text)) DESC
           ) AS rank
    FROM code_chunks c
    WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
      AND (filter_repository IS NULL OR c.repository_id = filter_repository)
    LIMIT 100
),
vector AS (
    SELECT c.id,
           row_number() OVER (ORDER BY c.embedding <=> query_embedding) AS rank
    FROM code_chunks c
    WHERE c.embedding IS NOT NULL
      AND (filter_repository IS NULL OR c.repository_id = filter_repository)
    LIMIT 100
)
SELECT c.id,
       c.file_id,
       c.path,
       c.symbol,
       c.start_line,
       c.end_line,
       c.content,
       COALESCE(1.0 / (rrf_k + l.rank), 0) + COALESCE(1.0 / (rrf_k + v.rank), 0) AS rrf_score,
       v.rank  AS vector_rank,
       l.rank  AS lexical_rank
FROM code_chunks c
LEFT JOIN lexical l ON l.id = c.id
LEFT JOIN vector  v ON v.id = c.id
WHERE l.id IS NOT NULL OR v.id IS NOT NULL
ORDER BY rrf_score DESC
LIMIT match_count;
$$;

-- ---------------------------------------------------------------------
-- 5. Reporting helpers.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_ingestion_health AS
SELECT repository_id,
       count(*)                                            AS runs,
       max(started_at)                                      AS last_run_at,
       count(*) FILTER (WHERE status = 'failed')            AS failed_runs,
       (SELECT count(*) FROM code_chunks c WHERE c.repository_id = r.repository_id)
                                                            AS chunks,
       (SELECT count(*) FROM evidence_embeddings e WHERE e.repository_id = r.repository_id)
                                                            AS evidence_rows
FROM ingestion_runs r
GROUP BY repository_id;
