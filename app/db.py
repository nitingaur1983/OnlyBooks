import os

from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# A full DATABASE_URL env var wins if set (e.g. for a hosted DB later).
# Otherwise build one from discrete PG_* env vars, defaulting to this
# project's local dev Postgres instance. URL.create() percent-encodes the
# password for us, which matters here since it contains "@".
_DATABASE_URL = os.getenv("DATABASE_URL")

if _DATABASE_URL:
    _url = _DATABASE_URL
else:
    _url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("PGUSER", "app_user"),
        password=os.getenv("PGPASSWORD", "jaiganesh219@"),
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        database=os.getenv("PGDATABASE", "only_books_db"),
    )

# All tables live in their own Postgres schema rather than "public" — see
# README > Database setup. The app_user role's search_path is also set to
# this schema, so plain (unqualified) SQL still works in psql.
SCHEMA_NAME = os.getenv("PGSCHEMA", "only_books_schema")

engine = create_engine(_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA_NAME)
