"""SQLite database helpers.

Uses plain ``sqlite3`` for synchronous access.  The schema is intentionally
minimal -- a single ``movies`` table that caches the processed metadata CSV
plus any TMDb image URLs we have fetched.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

_CREATE_MOVIES_TABLE = """\
CREATE TABLE IF NOT EXISTS movies (
    movie_id    INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL,
    year        INTEGER,
    genres      TEXT    NOT NULL DEFAULT '[]',   -- JSON array
    tmdb_id     INTEGER,
    imdb_id     TEXT,
    avg_rating  REAL,
    num_ratings INTEGER DEFAULT 0,
    poster_url  TEXT,
    backdrop_url TEXT,
    tags        TEXT    NOT NULL DEFAULT ''
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_movies_title ON movies(title COLLATE NOCASE);",
    "CREATE INDEX IF NOT EXISTS idx_movies_avg_rating ON movies(avg_rating DESC);",
    "CREATE INDEX IF NOT EXISTS idx_movies_num_ratings ON movies(num_ratings DESC);",
    "CREATE INDEX IF NOT EXISTS idx_movies_year ON movies(year);",
    "CREATE INDEX IF NOT EXISTS idx_movies_tmdb_id ON movies(tmdb_id);",
    # Text index on genres to speed up LIKE-based genre filtering
    "CREATE INDEX IF NOT EXISTS idx_movies_genres ON movies(genres);",
]

_CREATE_FTS_TABLE = """\
CREATE VIRTUAL TABLE IF NOT EXISTS movies_fts USING fts5(
    title, genres, tags,
    content=movies,
    content_rowid=movie_id
);
"""


def get_db_path() -> Path:
    """Return the resolved SQLite database path."""
    return get_settings().DB_PATH


def get_db() -> sqlite3.Connection:
    """Open (or reuse) a connection to the SQLite database.

    Returns a connection with ``row_factory`` set to ``sqlite3.Row`` so that
    query results behave like dicts.
    """
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """Create the schema if it does not already exist."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    try:
        conn.execute(_CREATE_MOVIES_TABLE)
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)
        conn.execute(_CREATE_FTS_TABLE)
        conn.commit()
        logger.info("Database initialised at %s", db_path)
    finally:
        conn.close()


def rebuild_fts_index() -> int:
    """Rebuild the FTS5 index from the movies table.

    Creates the virtual table if it does not exist, then issues a full
    rebuild so every row in ``movies`` is indexed.

    Returns
    -------
    int
        Number of rows now present in the FTS index.
    """
    conn = get_db()
    try:
        conn.execute(_CREATE_FTS_TABLE)
        conn.execute("INSERT INTO movies_fts(movies_fts) VALUES('rebuild');")
        conn.commit()
        row = conn.execute("SELECT COUNT(*) AS cnt FROM movies_fts;").fetchone()
        count = row["cnt"] if row else 0
        logger.info("FTS5 index rebuilt -- %d rows indexed", count)
        return count
    finally:
        conn.close()


def movie_count() -> int:
    """Return the total number of rows in the movies table."""
    conn = get_db()
    try:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM movies;").fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()
