from sqlalchemy import text
from backend.db.database import engine

sql = text("""
    SELECT
        t.typname AS enum_name,
        e.enumlabel AS enum_value
    FROM pg_type t
    JOIN pg_enum e
        ON t.oid = e.enumtypid
    WHERE t.typtype = 'e'
    ORDER BY t.typname, e.enumsortorder
""")

with engine.connect() as connection:
    rows = connection.execute(sql).fetchall()

for row in rows:
    print(row)