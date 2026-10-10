-- =====================================================================
-- NEXUS-DevIntel — migration 1024 (bge-m3) -> 384 (MiniLM-L6-v2)
--
-- Fresh volumes already get vector(384) from 001_init.sql; this file migrates
-- an EXISTING database created with the old 1024-dim layout, without losing
-- the ingestion_runs ledger:
--
--   docker exec -i nexus-postgres psql -U nexus -d nexus \
--     < schema/postgres/002_mini.sql
--
-- It drops the 1024-dim vector columns (+ their HNSW indexes, which embed
-- the dimension) and recreates them at 384, then re-applies the retrieval
-- functions whose signatures carry vector(1024). code_chunks rows survive
-- (content, hashes, metadata) but their embeddings are cleared: re-run
--   python scripts/index_embeddings.py
-- to re-embed with MiniLM (~1 min for 774 chunks on CPU).
-- =====================================================================

DROP INDEX IF EXISTS code_chunks_embedding_hnsw_idx;
DROP INDEX IF EXISTS evidence_embeddings_hnsw_idx;

ALTER TABLE code_chunks DROP COLUMN IF EXISTS embedding;
ALTER TABLE code_chunks
    ADD COLUMN embedding vector(384),
    ALTER COLUMN embedding_model SET DEFAULT 'sentence-transformers/all-MiniLM-L6-v2';

ALTER TABLE evidence_embeddings DROP COLUMN IF EXISTS embedding;
ALTER TABLE evidence_embeddings
    ADD COLUMN embedding vector(384),
    ALTER COLUMN embedding_model SET DEFAULT 'sentence-transformers/all-MiniLM-L6-v2';

-- Stored rows (if any) were 1024-dim: force a clean re-embed.
UPDATE code_chunks SET embedding = NULL, embedding_model = 'sentence-transformers/all-MiniLM-L6-v2';
UPDATE evidence_embeddings SET embedding = NULL, embedding_model = 'sentence-transformers/all-MiniLM-L6-v2';

CREATE INDEX IF NOT EXISTS code_chunks_embedding_hnsw_idx
    ON code_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS evidence_embeddings_hnsw_idx
    ON evidence_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Re-apply the retrieval functions at the new dimension (same bodies as
-- 001_init.sql; DROP first because the return/signature type changes).
DROP FUNCTION IF EXISTS match_code_chunks(vector, integer, text, text);

CREATE OR REPLACE FUNCTION match_code_chunks(
    query_embedding vector(384),
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

DROP FUNCTION IF EXISTS match_evidence(vector, integer, text);

CREATE OR REPLACE FUNCTION match_evidence(
    query_embedding vector(384),
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

DROP FUNCTION IF EXISTS hybrid_search_code_chunks(text, vector, integer, integer, text);

CREATE OR REPLACE FUNCTION hybrid_search_code_chunks(
    query_text      text,
    query_embedding vector(384),
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
