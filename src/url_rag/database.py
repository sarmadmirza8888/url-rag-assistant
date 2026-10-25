"""PostgreSQL + pgvector storage and similarity search.

Expects the table created by db/schema.sql. Connection string comes from the
DATABASE_URL environment variable (or Kaggle Secrets when running on Kaggle).
"""
import os

_conn = None

INSERT_SQL = """
    INSERT INTO webpage_chunks
        (source_url, page_title, chunk_id, content, embedding)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (source_url, chunk_id) DO NOTHING
"""

SEARCH_SQL = """
    SELECT
        chunk_id,
        content,
        1 - (embedding <=> %s) AS similarity
    FROM webpage_chunks
    WHERE source_url = %s
    ORDER BY embedding <=> %s
    LIMIT %s
"""


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    try:  # Kaggle notebooks keep secrets here
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("DATABASE_URL")
    except Exception as exc:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
        ) from exc


def get_connection():
    """Return a live connection, reconnecting if it was closed."""
    global _conn
    if _conn is None or _conn.closed:
        import psycopg
        from pgvector.psycopg import register_vector

        _conn = psycopg.connect(_database_url())
        register_vector(_conn)
    return _conn


def store_chunks(records) -> None:
    """records: iterable of (source_url, page_title, chunk_id, content, embedding)."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.executemany(INSERT_SQL, list(records))
    conn.commit()


def search_chunks(query_embedding, source_url: str, top_k: int = 3):
    """Return the top_k most similar chunks for one source URL."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(SEARCH_SQL, (query_embedding, source_url, query_embedding, top_k))
        rows = cur.fetchall()
    return [
        {"chunk_id": r[0], "text": r[1], "similarity": float(r[2])} for r in rows
    ]
