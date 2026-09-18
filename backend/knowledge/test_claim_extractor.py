from sqlalchemy import text

from backend.db.database import engine
from backend.knowledge.claim_extractor import extract_claims


def get_nodes():
    sql = text("""
        SELECT
            id,
            canonical_name
        FROM node
        ORDER BY kind, id
    """)

    with engine.connect() as connection:
        rows = connection.execute(sql).fetchall()

    return [
        {
            "id": row.id,
            "canonical_name": row.canonical_name,
        }
        for row in rows
    ]


def get_test_chunk():
    sql = text("""
        SELECT
            ec.id,
            ec.source_id,
            ec.text
        FROM evidence_chunk ec
        WHERE ec.source_id = 'S15'
        ORDER BY ec.id
        LIMIT 1
    """)

    with engine.connect() as connection:
        return connection.execute(sql).fetchone()


nodes = get_nodes()
chunk = get_test_chunk()

print("=" * 70)
print("TEST EVIDENCE")
print("=" * 70)

print("Chunk ID:", chunk.id)
print("Source:", chunk.source_id)
print()
print(chunk.text)

print()
print("=" * 70)
print("EXTRACTED CLAIMS")
print("=" * 70)

claims = extract_claims(
    evidence_text=chunk.text,
    source_id=chunk.source_id,
    available_nodes=nodes,
)

for i, claim in enumerate(claims, start=1):
    print()
    print(f"CLAIM {i}")
    print("-" * 40)
    print(claim.model_dump_json(indent=2))