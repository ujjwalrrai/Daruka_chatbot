from sqlalchemy import text

from backend.db.database import engine


sql = text("""
    SELECT
        id,
        src_node,
        dst_node,
        relation,
        effect_type,
        effect_size,
        source_id,
        chunk_id,
        verification_status,
        extraction_method
    FROM edge
    WHERE source_id = 'S15'
    ORDER BY id DESC
    LIMIT 10
""")


with engine.connect() as connection:
    rows = connection.execute(sql).fetchall()


print("=" * 80)
print("S15 EDGES")
print("=" * 80)

for row in rows:
    print(row)