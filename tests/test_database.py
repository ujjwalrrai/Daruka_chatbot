from sqlalchemy import text

from backend.db.database import engine


with engine.connect() as connection:

    result = connection.execute(
        text("SELECT version();")
    )

    print("\nPostgreSQL connection successful!\n")
    print(result.fetchone())