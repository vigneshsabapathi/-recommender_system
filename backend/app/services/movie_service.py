"""Movie catalogue service backed by SQLite."""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
import sqlite3
from typing import Any

from backend.app.db.database import get_db
from backend.app.schemas.common import PaginatedResponse, PaginationMeta
from backend.app.schemas.movie import MovieCard, MovieDetail

logger = logging.getLogger(__name__)

# Known genre names for genre-keyword detection
_KNOWN_GENRES = {
    "action", "adventure", "animation", "children", "comedy", "crime",
    "documentary", "drama", "fantasy", "film-noir", "horror", "imax",
    "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western",
}

# Valid sort columns exposed to the API
_SORT_OPTIONS = {
    "avg_rating": "avg_rating DESC",
    "num_ratings": "num_ratings DESC",
    "year": "year DESC",
    "title": "title ASC",
}

# Genre conflict map: when filtering by a genre, exclude movies that also
# belong to these conflicting genres so that rows feel "pure".
GENRE_CONFLICTS: dict[str, list[str]] = {
    "Comedy": ["Horror", "Thriller"],
    "Horror": ["Comedy", "Children"],
    "Children": ["Horror", "Thriller"],
    "Romance": ["Horror", "Thriller"],
    "Animation": ["Horror", "Thriller"],
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
        """Search movies using FTS5 full-text search with fuzzy fallback.

        Strategy (applied in order, results are merged):
          1. FTS5 exact / phrase match
          2. FTS5 prefix match  (``query*``)
          3. Genre keyword match (e.g. "action movies")
          4. Enhanced LIKE fallback (title + genres, year extraction)
          5. Python-side fuzzy matching via ``difflib`` for typo tolerance

        All results are de-duplicated and sorted by popularity
        (``num_ratings DESC``) so the most well-known titles surface first.
        """
        query = query.strip()
        if not query:
            return []

        conn = get_db()
        try:
            seen_ids: set[int] = set()
            results: list[MovieCard] = []
            # Track which search tier found each movie (lower = better match)
            tier_map: dict[int, int] = {}

            def _collect(rows: list[sqlite3.Row], tier: int) -> None:
                for r in rows:
                    mid = r["movie_id"]
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        tier_map[mid] = tier
                        results.append(_row_to_card(r))

            # ---- 1. FTS5 exact match (tier 0 -- best) ----
            fts_rows = self._fts_search(conn, query, limit)
            _collect(fts_rows, tier=0)

            # ---- 2. FTS5 prefix match (tier 1) ----
            if len(results) < limit:
                prefix_query = " ".join(
                    f"{self._fts_escape(tok)}*"
                    for tok in query.split() if tok
                )
                if prefix_query:
                    prefix_rows = self._fts_search(conn, prefix_query, limit)
                    _collect(prefix_rows, tier=1)

            # ---- 3. Genre keyword match (tier 2) ----
            genre_results = self._genre_keyword_search(conn, query, limit)
            _collect(genre_results, tier=2)

            # ---- 4. Enhanced LIKE fallback (tier 3) ----
            like_rows = self._like_search(conn, query, limit)
            _collect(like_rows, tier=3)

            # ---- 5. Fuzzy matching / typo tolerance (tier 4) ----
            if len(results) < limit:
                fuzzy_rows = self._fuzzy_search(conn, query, limit)
                _collect(fuzzy_rows, tier=4)

            # Sort: primary key is tier (exact FTS first), secondary is
            # popularity (num_ratings DESC) within each tier.
            if results:
                id_list = [c.id for c in results]
                placeholders = ",".join("?" for _ in id_list)
                rating_rows = conn.execute(
                    f"SELECT movie_id, num_ratings FROM movies WHERE movie_id IN ({placeholders})",
                    id_list,
                ).fetchall()
                rating_map = {r["movie_id"]: r["num_ratings"] or 0 for r in rating_rows}
                results.sort(
                    key=lambda c: (tier_map.get(c.id, 99), -(rating_map.get(c.id, 0))),
                )

            return results[:limit]
        finally:
            conn.close()

    # ----- FTS5 helpers -----

    @staticmethod
    def _fts_escape(token: str) -> str:
        """Escape special FTS5 characters in a single token."""
        # Remove characters that are FTS5 operators
        return re.sub(r'["\'\*\(\)\-\+\^]', "", token).strip()

    @staticmethod
    def _fts_available(conn: sqlite3.Connection) -> bool:
        """Return True if the movies_fts virtual table exists."""
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='movies_fts';"
        ).fetchone()
        return row is not None

    def _fts_search(
        self, conn: sqlite3.Connection, fts_query: str, limit: int
    ) -> list[sqlite3.Row]:
        """Run an FTS5 query and return matching movie rows."""
        if not self._fts_available(conn):
            return []
        # Sanitise: build a safe query string
        safe_tokens = [self._fts_escape(t) for t in fts_query.split() if t]
        safe_tokens = [t for t in safe_tokens if t]
        if not safe_tokens:
            return []
        # If the caller already prepared a prefix query (contains *), pass through
        if "*" in fts_query:
            safe_query = fts_query
        else:
            safe_query = " ".join(f'"{t}"' for t in safe_tokens)
        try:
            return conn.execute(
                """
                SELECT m.*
                FROM movies_fts fts
                JOIN movies m ON m.movie_id = fts.rowid
                WHERE movies_fts MATCH ?
                ORDER BY fts.rank, m.num_ratings DESC
                LIMIT ?
                """,
                (safe_query, limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.debug("FTS5 query failed (%s): %s", safe_query, exc)
            return []

    # ----- Genre keyword search -----

    @staticmethod
    def _genre_keyword_search(
        conn: sqlite3.Connection, query: str, limit: int
    ) -> list[sqlite3.Row]:
        """If the query contains genre names, return top movies of that genre."""
        tokens = query.lower().split()
        matched_genres = [t for t in tokens if t in _KNOWN_GENRES]
        if not matched_genres:
            return []
        # Build conditions: movie must match ALL detected genres
        conditions = []
        params: list[Any] = []
        for genre in matched_genres:
            conditions.append("LOWER(genres) LIKE ?")
            params.append(f"%{genre}%")
        where = " AND ".join(conditions)
        params.append(limit)
        return conn.execute(
            f"""
            SELECT * FROM movies
            WHERE {where}
            ORDER BY num_ratings DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    # ----- Enhanced LIKE fallback -----

    @staticmethod
    def _like_search(
        conn: sqlite3.Connection, query: str, limit: int
    ) -> list[sqlite3.Row]:
        """Enhanced LIKE search: extracts year, searches title and genres."""
        # Extract a trailing 4-digit year (e.g. "matrix 1999")
        year_match = re.search(r"\b((?:19|20)\d{2})\b", query)
        year = int(year_match.group(1)) if year_match else None
        text = re.sub(r"\b(?:19|20)\d{2}\b", "", query).strip() if year else query

        conditions = []
        params: list[Any] = []

        if text:
            conditions.append("(title LIKE ? OR genres LIKE ?)")
            params.extend([f"%{text}%", f"%{text}%"])
        if year:
            conditions.append("year = ?")
            params.append(year)

        if not conditions:
            return []

        where = " AND ".join(conditions)
        params.append(limit)
        return conn.execute(
            f"""
            SELECT * FROM movies
            WHERE {where}
            ORDER BY num_ratings DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    # ----- Python-side fuzzy matching -----

    @staticmethod
    def _fuzzy_search(
        conn: sqlite3.Connection, query: str, limit: int
    ) -> list[sqlite3.Row]:
        """Use ``difflib`` for typo-tolerant search.

        Two-pass approach:
          1. Match the full query against full titles (catches "Toy Stiry" -> "Toy Story (1995)").
          2. Match each query token against individual words extracted from
             titles (catches "matrx" -> "Matrix" in "The Matrix (1999)").

        Results are scored by similarity so the best fuzzy matches surface
        first, with popularity (``num_ratings``) as a tiebreaker.

        Candidate sets are loaded from the DB using LIKE filters based on the
        query's first characters to keep memory usage reasonable.
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        # movie_id -> best similarity score seen so far (higher = better)
        scored: dict[int, float] = {}

        # --- Helper: fetch candidates whose title contains a LIKE pattern ---
        def _fetch_candidates(like_pattern: str) -> list[sqlite3.Row]:
            return conn.execute(
                "SELECT movie_id, title FROM movies WHERE LOWER(title) LIKE ? LIMIT 5000",
                (like_pattern,),
            ).fetchall()

        # --- Pass 1: full-title fuzzy match ---
        tokens = query_lower.split()
        all_candidates: dict[str, int] = {}
        for tok in tokens:
            if len(tok) < 2:
                continue
            for row in _fetch_candidates(f"%{tok[:3]}%"):
                all_candidates[row["title"]] = row["movie_id"]

        if all_candidates:
            # Score every candidate title against the query (case-insensitive)
            for title, mid in all_candidates.items():
                ratio = difflib.SequenceMatcher(
                    None, query_lower, title.lower()
                ).ratio()
                if ratio >= 0.55:
                    scored[mid] = max(scored.get(mid, 0.0), ratio)

        # --- Pass 2: per-word fuzzy match (handles "matrx" -> "Matrix") ---
        # Build a word -> movie_id(s) mapping from the candidate pool
        word_to_ids: dict[str, list[int]] = {}
        for title, mid in all_candidates.items():
            clean = re.sub(r"\(\d{4}\)", "", title).strip()
            for word in clean.split():
                word_clean = re.sub(r"[^a-zA-Z0-9]", "", word)
                if len(word_clean) >= 3:
                    key = word_clean.lower()
                    word_to_ids.setdefault(key, []).append(mid)

        unique_words = list(word_to_ids.keys())
        for tok in tokens:
            if len(tok) < 2:
                continue
            # Use a high cutoff (0.7) to avoid false positives like
            # "amateur" matching "matrx" (which had 0.667 at cutoff=0.5).
            close_words = difflib.get_close_matches(
                tok.lower(), unique_words, n=limit * 2, cutoff=0.7
            )
            for cw in close_words:
                word_ratio = difflib.SequenceMatcher(
                    None, tok.lower(), cw
                ).ratio()
                for mid in word_to_ids[cw]:
                    scored[mid] = max(scored.get(mid, 0.0), word_ratio)

        if not scored:
            return []

        # Take the top candidates sorted by similarity score descending
        top_ids = sorted(scored, key=lambda mid: scored[mid], reverse=True)[
            :limit
        ]

        placeholders = ",".join("?" for _ in top_ids)
        rows = conn.execute(
            f"SELECT * FROM movies WHERE movie_id IN ({placeholders})",
            top_ids,
        ).fetchall()

        # Re-sort the SQL rows to match our similarity-based ordering
        id_rank = {mid: i for i, mid in enumerate(top_ids)}
        return sorted(rows, key=lambda r: id_rank.get(r["movie_id"], 999))

    # ------------------------------------------------------------------
    # Browse / paginated listing
    # ------------------------------------------------------------------
    def get_movies(
        self,
        genre: str | None = None,
        page: int = 1,
        per_page: int = 20,
        sort_by: str = "num_ratings",
        exclude_conflicting: bool = True,
    ) -> PaginatedResponse:
        """Return a paginated, optionally genre-filtered movie listing.

        When *genre* is provided and *exclude_conflicting* is ``True``,
        movies that also belong to a conflicting genre (see
        ``GENRE_CONFLICTS``) are removed so that genre rows feel "pure".

        Results are additionally sorted so that movies whose **primary**
        (first-listed) genre matches the requested genre rank higher.
        """
        order_clause = _SORT_OPTIONS.get(sort_by, "num_ratings DESC")
        conditions: list[str] = []
        params: list[Any] = []

        if genre:
            # JSON array stored as text; use LIKE for lightweight genre filter
            conditions.append("genres LIKE ?")
            params.append(f'%"{genre}"%')

            # Exclude movies that also contain a conflicting genre
            if exclude_conflicting:
                conflicts = GENRE_CONFLICTS.get(genre, [])
                for conflict in conflicts:
                    conditions.append("genres NOT LIKE ?")
                    params.append(f'%"{conflict}"%')

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # When filtering by genre, prioritise movies where the target genre
        # is the *primary* (first-listed) genre.  The JSON array starts with
        # '["Genre"' so we can cheaply check with LIKE.
        if genre:
            primary_expr = f"""CASE WHEN genres LIKE ? THEN 0 ELSE 1 END"""
            full_order = f"{primary_expr}, {order_clause}"
            order_params: list[Any] = [f'["{genre}"%']
        else:
            full_order = order_clause
            order_params = []

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
                f"SELECT * FROM movies {where} ORDER BY {full_order} LIMIT ? OFFSET ?",
                params + order_params + [per_page, offset],
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
