-- Run once on your PostgreSQL database (Neon, Supabase, local, etc.)
-- Inferred from the notebook's INSERT/SELECT statements.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS webpage_chunks (
    id         SERIAL PRIMARY KEY,
    source_url TEXT    NOT NULL,
    page_title TEXT,
    chunk_id   INTEGER NOT NULL,
    content    TEXT    NOT NULL,
    embedding  VECTOR(384) NOT NULL,   -- all-MiniLM-L6-v2 produces 384 dimensions
    UNIQUE (source_url, chunk_id)      -- required by ON CONFLICT (source_url, chunk_id)
);

-- Optional: speeds up similarity search on large tables
CREATE INDEX IF NOT EXISTS webpage_chunks_embedding_idx
    ON webpage_chunks USING hnsw (embedding vector_cosine_ops);
