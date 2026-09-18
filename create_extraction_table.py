from sqlalchemy import text

from backend.db.database import engine


sql = text("""
CREATE TABLE IF NOT EXISTS evidence_extraction (
    chunk_id BIGINT PRIMARY KEY
        REFERENCES evidence_chunk(id)
        ON DELETE CASCADE,

    source_id TEXT NOT NULL
        REFERENCES source(id)
        ON DELETE CASCADE,

    status TEXT NOT NULL DEFAULT 'pending',

    claims_found INTEGER NOT NULL DEFAULT 0,

    error TEXT,

    attempts INTEGER NOT NULL DEFAULT 0,

    processed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
""")


with engine.begin() as connection:
    connection.execute(sql)


print("evidence_extraction table ready.")