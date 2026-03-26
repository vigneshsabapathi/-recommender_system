"""Seed the SQLite movies table from the processed metadata CSV.

Can be run standalone::

    python -m backend.app.db.seed

Or invoked programmatically via :func:`seed_database`.
"""

from __future__ import annotations

import ast
import json
import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd

from backend.app.config import get_settings
from backend.app.db.database import get_db, init_db, movie_count, rebuild_fts_index

logger = logging.getLogger(__name__)


def _parse_genres(raw: str | list | float) -> list[str]:
    """Normalise the genres column into a Python list of strings."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or pd.isna(raw):
        return []
    raw = raw.strip()
    # Try JSON first (e.g. '["Action","Drama"]')
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                pass
    # Pipe-separated fallback (e.g. "Action|Drama")
    return [g.strip() for g in raw.split("|") if g.strip()]


def _safe_int(val, default=None):
    """Convert to int, returning *default* on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default=None):
    """Convert to float, returning *default* on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def seed_database(force: bool = False) -> int:
    """Read ``movie_metadata.csv`` and insert rows into the movies table.

    Parameters
    ----------
    force : bool
        If ``True``, drop and recreate the table before seeding.

    Returns
    -------
    int
        Number of rows inserted.
    """
    settings = get_settings()
    csv_path = settings.DATA_DIR / "processed" / "movie_metadata.csv"

    if not csv_path.exists():
        logger.warning("Metadata CSV not found at %s -- skipping seed", csv_path)
        return 0

    init_db()

    # Skip if already seeded (unless forced)
    if not force and movie_count() > 0:
        count = movie_count()
        logger.info("Database already seeded (%d movies) -- skipping", count)
        return count

    logger.info("Reading metadata from %s ...", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from CSV", len(df))

    conn = get_db()
    try:
        if force:
            conn.execute("DELETE FROM movies;")
            conn.commit()

        inserted = 0
        batch: list[tuple] = []
        batch_size = 500

        for _, row in df.iterrows():
            movie_id = _safe_int(row.get("movieId"))
            if movie_id is None:
                continue

            title = str(row.get("title", "")).strip()
            if not title:
                continue

            # Prefer genres_list column if available, else genres
            genres_raw = row.get("genres_list", row.get("genres", ""))
            genres = _parse_genres(genres_raw)

            year = _safe_int(row.get("year"))
            tmdb_id = _safe_int(row.get("tmdbId"))
            imdb_id = row.get("imdbId")
            if pd.notna(imdb_id):
                imdb_id = str(int(imdb_id)) if isinstance(imdb_id, float) else str(imdb_id)
                # Ensure 'tt' prefix
                if not str(imdb_id).startswith("tt"):
                    imdb_id = f"tt{int(imdb_id):07d}"
            else:
                imdb_id = None

            avg_rating = _safe_float(row.get("avg_rating"))
            num_ratings = _safe_int(row.get("num_ratings"), default=0)
            tags = str(row.get("tags_combined", "")).strip() if pd.notna(row.get("tags_combined")) else ""

            batch.append((
                movie_id,
                title,
                year,
                json.dumps(genres),
                tmdb_id,
                imdb_id,
                avg_rating,
                num_ratings,
                None,   # poster_url  (filled later by TMDb)
                None,   # backdrop_url
                tags,
            ))

            if len(batch) >= batch_size:
                _insert_batch(conn, batch)
                inserted += len(batch)
                batch.clear()

        # Final partial batch
        if batch:
            _insert_batch(conn, batch)
            inserted += len(batch)

        conn.commit()
        logger.info("Seeded %d movies into the database", inserted)

        # Build the FTS5 full-text search index
        fts_count = rebuild_fts_index()
        logger.info("FTS5 index populated with %d rows", fts_count)

        return inserted

    finally:
        conn.close()


def _insert_batch(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Insert a batch of rows using INSERT OR REPLACE."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO movies
            (movie_id, title, year, genres, tmdb_id, imdb_id,
             avg_rating, num_ratings, poster_url, backdrop_url, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _fetch_tmdb_posters() -> None:
    """Optionally backfill poster/backdrop URLs from TMDb.

    Only runs if ``TMDB_API_KEY`` is configured.  Respects TMDb rate limits
    (~40 requests/10 s) and persists URLs into the SQLite movies table.
    """
    settings = get_settings()
    if not settings.TMDB_API_KEY:
        logger.info("TMDB_API_KEY not set -- skipping poster fetch")
        return

    import httpx

    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT movie_id, tmdb_id FROM movies WHERE tmdb_id IS NOT NULL AND poster_url IS NULL"
        ).fetchall()

        if not rows:
            logger.info("No movies need TMDb poster fetch")
            return

        logger.info("Fetching TMDb posters for %d movies ...", len(rows))
        base = "https://api.themoviedb.org/3/movie"
        headers = {
            "Authorization": f"Bearer {settings.TMDB_API_KEY}",
            "accept": "application/json",
        }

        with httpx.Client(timeout=10.0) as client:
            for i, row in enumerate(rows):
                tmdb_id = row["tmdb_id"]
                try:
                    resp = client.get(f"{base}/{tmdb_id}", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        poster = data.get("poster_path")
                        backdrop = data.get("backdrop_path")
                        poster_url = f"https://image.tmdb.org/t/p/w342{poster}" if poster else None
                        backdrop_url = f"https://image.tmdb.org/t/p/w1280{backdrop}" if backdrop else None
                        conn.execute(
                            "UPDATE movies SET poster_url = ?, backdrop_url = ? WHERE movie_id = ?",
                            (poster_url, backdrop_url, row["movie_id"]),
                        )
                    elif resp.status_code == 429:
                        logger.warning("TMDb rate limit hit -- sleeping 10s")
                        time.sleep(10)
                except httpx.HTTPError as exc:
                    logger.warning("TMDb request failed for tmdb_id=%s: %s", tmdb_id, exc)

                # Commit every 50 and rate-limit
                if (i + 1) % 50 == 0:
                    conn.commit()
                    logger.info("  Fetched %d / %d posters", i + 1, len(rows))
                    time.sleep(0.3)

        conn.commit()
        logger.info("TMDb poster fetch complete")
    finally:
        conn.close()


# ---- CLI entry point ----
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    force = "--force" in sys.argv
    count = seed_database(force=force)
    print(f"Seeded {count} movies")

    if "--posters" in sys.argv:
        _fetch_tmdb_posters()
