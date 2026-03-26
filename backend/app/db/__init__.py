"""Database package -- SQLite schema, seeding, and connection helpers."""

from backend.app.db.database import get_db, init_db, movie_count

__all__ = ["get_db", "init_db", "movie_count"]
