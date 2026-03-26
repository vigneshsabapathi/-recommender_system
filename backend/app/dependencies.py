"""FastAPI dependency-injection providers.

Singletons are created once during the lifespan startup event and stored
on ``app.state``.  The dependency functions retrieve them from the running
application instance so that routers stay decoupled from global state.
"""

from __future__ import annotations

from functools import lru_cache

from backend.app.services.movie_service import MovieService
from backend.app.services.recommender_service import RecommenderService
from backend.app.services.tmdb_service import TMDbService

# Module-level singletons initialised by the lifespan handler in main.py.
_recommender_service: RecommenderService | None = None
_movie_service: MovieService | None = None
_tmdb_service: TMDbService | None = None


def init_services() -> dict[str, object]:
    """Create singleton service instances.

    Called once during application startup.  Returns a mapping for
    convenient logging / inspection.
    """
    global _recommender_service, _movie_service, _tmdb_service

    _recommender_service = RecommenderService()
    _movie_service = MovieService()
    _tmdb_service = TMDbService()

    return {
        "recommender_service": _recommender_service,
        "movie_service": _movie_service,
        "tmdb_service": _tmdb_service,
    }


# ------------------------------------------------------------------
# FastAPI Depends() callables
# ------------------------------------------------------------------

def get_recommender_service() -> RecommenderService:
    """Provide the RecommenderService singleton."""
    assert _recommender_service is not None, "RecommenderService not initialised"
    return _recommender_service


def get_movie_service() -> MovieService:
    """Provide the MovieService singleton."""
    assert _movie_service is not None, "MovieService not initialised"
    return _movie_service


def get_tmdb_service() -> TMDbService:
    """Provide the TMDbService singleton."""
    assert _tmdb_service is not None, "TMDbService not initialised"
    return _tmdb_service
