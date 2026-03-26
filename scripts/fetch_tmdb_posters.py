"""Fetch TMDb poster and backdrop URLs for all movies in the SQLite database.

Usage (from project root):
    source venv/Scripts/activate && python scripts/fetch_tmdb_posters.py

Features:
  - Async concurrent fetching via aiohttp with semaphore-based rate limiting
  - Batch commits every 100 movies for resumability
  - Skips movies that already have poster_url (safe to re-run)
  - Graceful error handling for invalid/missing tmdbIds
  - tqdm progress bar
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import aiohttp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "backend" / "movies.db"
ENV_PATH = PROJECT_ROOT / ".env"

TMDB_API_BASE = "https://api.themoviedb.org/3/movie"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"
BACKDROP_BASE = "https://image.tmdb.org/t/p/w1280"

# Rate-limiting: TMDb allows ~40 req/s; we stay at ~35
MAX_CONCURRENT = 35
REQUEST_DELAY = 1.0 / MAX_CONCURRENT  # minimum gap between request starts
BATCH_COMMIT_SIZE = 100

# Retry / timeout
REQUEST_TIMEOUT = 10
MAX_RETRIES = 2
RETRY_DELAY = 2.0
RATE_LIMIT_SLEEP = 10  # seconds to wait on HTTP 429

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_tmdb_posters")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    """Load TMDB_API_KEY from .env file or environment variable."""
    # Check environment first
    key = os.environ.get("TMDB_API_KEY")
    if key:
        return key

    # Parse .env file
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TMDB_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    return key

    logger.error("TMDB_API_KEY not found in environment or %s", ENV_PATH)
    sys.exit(1)


def get_movies_needing_posters(conn: sqlite3.Connection) -> list[tuple[int, int]]:
    """Return (movie_id, tmdb_id) for movies that still need poster URLs."""
    rows = conn.execute(
        """
        SELECT movie_id, tmdb_id
        FROM movies
        WHERE tmdb_id IS NOT NULL
          AND poster_url IS NULL
        ORDER BY movie_id
        """
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


# ---------------------------------------------------------------------------
# Async fetching
# ---------------------------------------------------------------------------
async def fetch_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    movie_id: int,
    tmdb_id: int,
    api_key: str,
    results: dict,
    errors: list,
    pbar: tqdm,
) -> None:
    """Fetch poster/backdrop for a single movie from TMDb."""
    url = f"{TMDB_API_BASE}/{tmdb_id}"
    params = {"api_key": api_key}

    for attempt in range(MAX_RETRIES + 1):
        async with semaphore:
            try:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        poster_path = data.get("poster_path")
                        backdrop_path = data.get("backdrop_path")
                        poster_url = f"{POSTER_BASE}{poster_path}" if poster_path else None
                        backdrop_url = f"{BACKDROP_BASE}{backdrop_path}" if backdrop_path else None
                        results[movie_id] = (poster_url, backdrop_url)
                        pbar.update(1)
                        return

                    elif resp.status == 429:
                        # Rate limit hit -- back off
                        retry_after = int(resp.headers.get("Retry-After", RATE_LIMIT_SLEEP))
                        logger.warning(
                            "Rate limit (429) for tmdb_id=%s, sleeping %ds (attempt %d)",
                            tmdb_id, retry_after, attempt + 1,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    elif resp.status == 404:
                        # Movie not found on TMDb -- skip
                        results[movie_id] = (None, None)
                        pbar.update(1)
                        return

                    else:
                        logger.debug(
                            "HTTP %d for tmdb_id=%s (attempt %d)", resp.status, tmdb_id, attempt + 1,
                        )
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY)
                            continue
                        # Final attempt failed
                        errors.append((movie_id, tmdb_id, f"HTTP {resp.status}"))
                        pbar.update(1)
                        return

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < MAX_RETRIES:
                    logger.debug(
                        "Error for tmdb_id=%s: %s (attempt %d, retrying)", tmdb_id, exc, attempt + 1,
                    )
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                errors.append((movie_id, tmdb_id, str(exc)))
                pbar.update(1)
                return


async def fetch_all(
    movies: list[tuple[int, int]],
    api_key: str,
    conn: sqlite3.Connection,
) -> tuple[int, int]:
    """Fetch poster URLs for all movies, commit in batches. Returns (updated, failed)."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results: dict[int, tuple[str | None, str | None]] = {}
    errors: list[tuple[int, int, str]] = []

    updated_count = 0
    failed_count = 0

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, limit_per_host=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in chunks to allow periodic DB commits
        chunk_size = BATCH_COMMIT_SIZE
        total_chunks = (len(movies) + chunk_size - 1) // chunk_size

        pbar = tqdm(total=len(movies), desc="Fetching posters", unit="movie", ncols=100)

        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(movies))
            chunk = movies[start:end]

            # Create tasks for this chunk
            tasks = [
                fetch_one(session, semaphore, mid, tid, api_key, results, errors, pbar)
                for mid, tid in chunk
            ]
            await asyncio.gather(*tasks)

            # Commit this batch to DB
            batch_updates = []
            for mid, _ in chunk:
                if mid in results:
                    poster_url, backdrop_url = results[mid]
                    if poster_url is not None:
                        batch_updates.append((poster_url, backdrop_url, mid))
                        updated_count += 1
                    else:
                        failed_count += 1
                    del results[mid]

            if batch_updates:
                conn.executemany(
                    "UPDATE movies SET poster_url = ?, backdrop_url = ? WHERE movie_id = ?",
                    batch_updates,
                )
                conn.commit()

        pbar.close()

    # Log errors summary
    if errors:
        logger.warning("%d movies had fetch errors:", len(errors))
        for mid, tid, err in errors[:20]:
            logger.warning("  movie_id=%d, tmdb_id=%d: %s", mid, tid, err)
        if len(errors) > 20:
            logger.warning("  ... and %d more errors", len(errors) - 20)
        failed_count += len(errors)

    return updated_count, failed_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    api_key = load_api_key()
    logger.info("Loaded TMDb API key: %s...%s", api_key[:4], api_key[-4:])

    if not DB_PATH.exists():
        logger.error("Database not found at %s", DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        movies = get_movies_needing_posters(conn)
        logger.info(
            "Found %d movies needing poster URLs (out of %d total with tmdb_id)",
            len(movies),
            conn.execute("SELECT count(*) FROM movies WHERE tmdb_id IS NOT NULL").fetchone()[0],
        )

        if not movies:
            logger.info("Nothing to do -- all movies already have poster URLs!")
            return

        start_time = time.time()
        updated, failed = asyncio.run(fetch_all(movies, api_key, conn))
        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("Done in %.1f seconds (%.1f movies/sec)", elapsed, len(movies) / elapsed)
        logger.info("  Updated: %d", updated)
        logger.info("  Failed/no poster: %d", failed)
        logger.info("=" * 60)

        # Final verification
        with_poster = conn.execute(
            "SELECT count(*) FROM movies WHERE poster_url IS NOT NULL"
        ).fetchone()[0]
        total = conn.execute("SELECT count(*) FROM movies").fetchone()[0]
        logger.info("Database now has %d / %d movies with poster URLs", with_poster, total)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
