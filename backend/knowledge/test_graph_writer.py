from sqlalchemy import text

from backend.db.database import engine
from backend.knowledge.claim_extractor import extract_claims
from backend.knowledge.graph_writer import write_claim


def get_nodes():
    sql = text("""
        SELECT
            id,
            canonical_name
        FROM node
        ORDER BY kind, id
    """)

    with engine.connect() as connection:
        rows = connection.execute(
            sql
        ).fetchall()

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
            id,
            source_id,
            text
        FROM evidence_chunk
        WHERE source_id = 'S15'
        ORDER BY id
        LIMIT 1
    """)

    with engine.connect() as connection:
        return connection.execute(sql).fetchone()


nodes = get_nodes()
chunk = get_test_chunk()

print("=" * 70)
print("EXTRACTING CLAIMS")
print("=" * 70)

claims = extract_claims(
    evidence_text=chunk.text,
    source_id=chunk.source_id,
    available_nodes=nodes,
)

print()
print(f"Claims extracted: {len(claims)}")

print()
print("=" * 70)
print("WRITING CLAIMS TO GRAPH")
print("=" * 70)

for claim in claims:
    print()
    print(claim.model_dump())

    write_claim(
        source_id=chunk.source_id,
        chunk_id=chunk.id,
        claim=claim,
    )

print()
print("=" * 70)
print("DONE")
print("=" * 70)