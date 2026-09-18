from sqlalchemy import create_engine


DATABASE_URL = (
    "postgresql+psycopg://"
    "cee:cee@localhost:5432/cee"
)


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)