"""TMDb image-URL resolver with in-memory + SQLite caching."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import httpx

from backend.app.config import get_settings
from backend.app.db.database import get_db

logger = logging.getLogger(__name__)

_POSTER_BASE = "https://image.tmdb.org/t/p"
_API_BASE = "https://api.themoviedb.org/3/movie"


class TMDbService:
    """Fetches and caches movie poster / backdrop URLs from TMDb.

    Provides a two-layer cache:
      1. In-memory dict for the current process lifetime.
      2. SQLite ``movies`` table for cross-restart persistence.
    """

    def __init__(self) -> None:
        self._cache: dict[int, dict[str, str | None]] = {}
        settings = get_settings()
        self._api_key: str = settings.TMDB_API_KEY
        self._enabled: bool = bool(self._api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_movie_images(self, tmdb_id: int | None) -> dict[str, str | None]:
        """Return ``{"poster_url": ..., "backdrop_url": ...}`` for a TMDb ID.

        Falls back gracefully:
          - Returns cached values if available.
          - Returns ``{None, None}`` if the API key is missing or the
            request fails.
        """
        if tmdb_id is None:
            return {"poster_url": None, "backdrop_url": None}

        # Layer 1: in-memory cache
        if tmdb_id in self._cache:
            return self._cache[tmdb_id]

        # Layer 2: SQLite cache
        db_hit = self._lookup_db(tmdb_id)
        if db_hit is not None:
            self._cache[tmdb_id] = db_hit
            return db_hit

        # Layer 3: live API call
        if not self._enabled:
            result: dict[str, str | None] = {"poster_url": None, "backdrop_url": None}
            self._cache[tmdb_id] = result
            return result

        result = self._fetch_from_api(tmdb_id)
        self._cache[tmdb_id] = result

        # Persist to SQLite
        self._persist_db(tmdb_id, result)

        return result

    def build_poster_url(self, path: str | None, size: str = "w342") -> str | None:
        """Construct a full poster URL from a TMDb ``poster_path``."""
        if not path:
            return None
        return f"{_POSTER_BASE}/{size}/{path.lstrip('/')}"

    def build_backdrop_url(self, path: str | None, size: str = "w1280") -> str | None:
        """Construct a full backdrop URL from a TMDb ``backdrop_path``."""
        if not path:
            return None
        return f"{_POSTER_BASE}/{size}/{path.lstrip('/')}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _lookup_db(self, tmdb_id: int) -> dict[str, str | None] | None:
        """Check the SQLite movies table for cached image URLs."""
        try:
            conn = get_db()
            try:
                row = conn.execute(
                    "SELECT poster_url, backdrop_url FROM movies WHERE tmdb_id = ?",
                    (tmdb_id,),
                ).fetchone()
                if row and (row["poster_url"] or row["backdrop_url"]):
                    return {
                        "poster_url": row["poster_url"],
                        "backdrop_url": row["backdrop_url"],
                    }
            finally:
                conn.close()
        except Exception:
            pass
        return None

    def _persist_db(self, tmdb_id: int, images: dict[str, str | None]) -> None:
        """Write fetched URLs back to the movies table."""
        try:
            conn = get_db()
            try:
                conn.execute(
                    "UPDATE movies SET poster_url = ?, backdrop_url = ? WHERE tmdb_id = ?",
                    (images.get("poster_url"), images.get("backdrop_url"), tmdb_id),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def _fetch_from_api(self, tmdb_id: int) -> dict[str, str | None]:
        """Call the TMDb ``/movie/{id}`` endpoint."""
        result: dict[str, str | None] = {"poster_url": None, "backdrop_url": None}
        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "accept": "application/json",
            }
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{_API_BASE}/{tmdb_id}", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                poster_path = data.get("poster_path")
                backdrop_path = data.get("backdrop_path")
                result["poster_url"] = self.build_poster_url(poster_path)
                result["backdrop_url"] = self.build_backdrop_url(backdrop_path)
            else:
                logger.debug("TMDb returned %d for tmdb_id=%s", resp.status_code, tmdb_id)
        except httpx.HTTPError as exc:
            logger.warning("TMDb API error for tmdb_id=%s: %s", tmdb_id, exc)
        return result
