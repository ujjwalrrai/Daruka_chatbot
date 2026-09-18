from sqlalchemy import text

from backend.db.database import engine


with engine.connect() as connection:

    print("=" * 60)
    print("NODE COLUMNS")
    print("=" * 60)

    node_rows = connection.execute(
        text(
            """
            SELECT
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_name = 'node'
            ORDER BY ordinal_position
            """
        )
    ).fetchall()

    for row in node_rows:
        print(row)

    print()
    print("=" * 60)
    print("EDGE COLUMNS")
    print("=" * 60)

    edge_rows = connection.execute(
        text(
            """
            SELECT
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_name = 'edge'
            ORDER BY ordinal_position
            """
        )
    ).fetchall()

    for row in edge_rows:
        print(row)