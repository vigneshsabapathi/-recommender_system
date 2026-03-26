"""Movie catalogue service backed by SQLite."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from typing import Any

from backend.app.db.database import get_db
from backend.app.schemas.common import PaginatedResponse, PaginationMeta
from backend.app.schemas.movie import MovieCard, MovieDetail

logger = logging.getLogger(__name__)

# Valid sort columns exposed to the API
_SORT_OPTIONS = {
    "avg_rating": "avg_rating DESC",
    "num_ratings": "num_ratings DESC",
    "year": "year DESC",
    "title": "title ASC",
}


def _row_to_card(row: sqlite3.Row) -> MovieCard:
    """Convert a SQLite Row into a MovieCard schema."""
    genres = _parse_genres(row["genres"])
    return MovieCard(
        id=row["movie_id"],
        title=row["title"],
        year=row["year"],
        genres=genres,
        poster_url=row["poster_url"],
        avg_rating=row["avg_rating"],
    )


def _row_to_detail(row: sqlite3.Row) -> MovieDetail:
    """Convert a SQLite Row into a MovieDetail schema."""
    genres = _parse_genres(row["genres"])
    tags_raw = row["tags"] or ""
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
    return MovieDetail(
        id=row["movie_id"],
        title=row["title"],
        year=row["year"],
        genres=genres,
        poster_url=row["poster_url"],
        avg_rating=row["avg_rating"],
        backdrop_url=row["backdrop_url"],
        overview=None,  # not stored locally
        num_ratings=row["num_ratings"],
        tags=tags[:20],  # cap tag list
        imdb_id=row["imdb_id"],
        tmdb_id=row["tmdb_id"],
    )


def _parse_genres(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [g.strip() for g in raw.split("|") if g.strip()]


class MovieService:
    """Read-only service for movie metadata queries."""

    # ------------------------------------------------------------------
    # Single-movie lookups
    # ------------------------------------------------------------------
    def get_movie(self, movie_id: int) -> MovieDetail | None:
        """Fetch full details for a single movie by ID."""
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT * FROM movies WHERE movie_id = ?", (movie_id,)
            ).fetchone()
            return _row_to_detail(row) if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search_movies(self, query: str, limit: int = 20) -> list[MovieCard]:
        """Full-text title search (case-insensitive LIKE)."""
        conn = get_db()
        try:
            rows = conn.execute(
                """
                SELECT * FROM movies
                WHERE title LIKE ?
                ORDER BY num_ratings DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()
            return [_row_to_card(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Browse / paginated listing
    # ------------------------------------------------------------------
    def get_movies(
        self,
        genre: str | None = None,
        page: int = 1,
        per_page: int = 20,
        sort_by: str = "num_ratings",
    ) -> PaginatedResponse:
        """Return a paginated, optionally genre-filtered movie listing."""
        order_clause = _SORT_OPTIONS.get(sort_by, "num_ratings DESC")
        conditions: list[str] = []
        params: list[Any] = []

        if genre:
            # JSON array stored as text; use LIKE for lightweight genre filter
            conditions.append("genres LIKE ?")
            params.append(f'%"{genre}"%')

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        conn = get_db()
        try:
            # Total count
            count_row = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM movies {where}", params
            ).fetchone()
            total_items = count_row["cnt"] if count_row else 0

            total_pages = max(1, math.ceil(total_items / per_page))
            offset = (page - 1) * per_page

            rows = conn.execute(
                f"SELECT * FROM movies {where} ORDER BY {order_clause} LIMIT ? OFFSET ?",
                params + [per_page, offset],
            ).fetchall()

            cards = [_row_to_card(r) for r in rows]

            return PaginatedResponse(
                items=cards,
                meta=PaginationMeta(
                    page=page,
                    per_page=per_page,
                    total_items=total_items,
                    total_pages=total_pages,
                ),
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Popular movies
    # ------------------------------------------------------------------
    def get_popular_movies(self, n: int = 20) -> list[MovieCard]:
        """Top-*n* movies by number of ratings."""
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT * FROM movies ORDER BY num_ratings DESC LIMIT ?", (n,)
            ).fetchall()
            return [_row_to_card(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Batch lookups (used by recommender service)
    # ------------------------------------------------------------------
    def get_movie_cards(self, movie_ids: list[int]) -> dict[int, MovieCard]:
        """Fetch MovieCards for a list of IDs in one query."""
        if not movie_ids:
            return {}
        conn = get_db()
        try:
            placeholders = ",".join("?" for _ in movie_ids)
            rows = conn.execute(
                f"SELECT * FROM movies WHERE movie_id IN ({placeholders})",
                movie_ids,
            ).fetchall()
            return {r["movie_id"]: _row_to_card(r) for r in rows}
        finally:
            conn.close()
