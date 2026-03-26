"""Shared pytest fixtures for backend tests.

Provides a FastAPI TestClient with mocked service dependencies so tests
run without ML model artefacts or a real database.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.dependencies import (
    get_movie_service,
    get_recommender_service,
    get_tmdb_service,
)
from backend.app.routers import health, movies, recommendations, similar
from backend.app.schemas.common import PaginatedResponse, PaginationMeta
from backend.app.schemas.movie import MovieCard, MovieDetail


# ---------------------------------------------------------------------------
# Fake / stub services
# ---------------------------------------------------------------------------

class FakeMovieService:
    """In-memory stub for MovieService."""

    _MOVIES = {
        1: MovieDetail(
            id=1, title="Toy Story (1995)", year=1995,
            genres=["Animation", "Comedy", "Family"],
            poster_url=None, avg_rating=3.9, num_ratings=215,
            tags=["pixar", "fun"], imdb_id="tt0114709", tmdb_id=862,
        ),
        2: MovieDetail(
            id=2, title="Jumanji (1995)", year=1995,
            genres=["Adventure", "Children", "Fantasy"],
            poster_url=None, avg_rating=3.2, num_ratings=110,
            tags=["board game"], imdb_id="tt0113497", tmdb_id=8844,
        ),
        3: MovieDetail(
            id=3, title="Grumpier Old Men (1995)", year=1995,
            genres=["Comedy", "Romance"],
            poster_url=None, avg_rating=3.2, num_ratings=52,
            tags=[], imdb_id="tt0113228", tmdb_id=15602,
        ),
    }

    def get_movie(self, movie_id: int) -> MovieDetail | None:
        return self._MOVIES.get(movie_id)

    def search_movies(self, query: str, limit: int = 20) -> list[MovieCard]:
        results = [
            MovieCard(id=m.id, title=m.title, year=m.year, genres=m.genres,
                      poster_url=m.poster_url, avg_rating=m.avg_rating)
            for m in self._MOVIES.values()
            if query.lower() in m.title.lower()
        ]
        return results[:limit]

    def get_movies(self, genre=None, page=1, per_page=20, sort_by="num_ratings"):
        items = list(self._MOVIES.values())
        if genre:
            items = [m for m in items if genre in m.genres]
        cards = [
            MovieCard(id=m.id, title=m.title, year=m.year, genres=m.genres,
                      poster_url=m.poster_url, avg_rating=m.avg_rating)
            for m in items
        ]
        return PaginatedResponse(
            items=cards,
            meta=PaginationMeta(page=page, per_page=per_page,
                                total_items=len(cards), total_pages=1),
        )

    def get_popular_movies(self, n: int = 20) -> list[MovieCard]:
        return [
            MovieCard(id=m.id, title=m.title, year=m.year, genres=m.genres,
                      poster_url=m.poster_url, avg_rating=m.avg_rating)
            for m in list(self._MOVIES.values())[:n]
        ]

    def get_movie_cards(self, movie_ids: list[int]) -> dict[int, MovieCard]:
        result = {}
        for mid in movie_ids:
            m = self._MOVIES.get(mid)
            if m:
                result[mid] = MovieCard(
                    id=m.id, title=m.title, year=m.year, genres=m.genres,
                    poster_url=m.poster_url, avg_rating=m.avg_rating,
                )
        return result


class FakeRecommenderService:
    """Stub that returns deterministic recommendations."""

    @property
    def loaded_algorithms(self) -> list[str]:
        return ["collaborative", "content_based", "hybrid"]

    def load_models(self) -> list[str]:
        return self.loaded_algorithms

    def get_recommendations(self, user_id, algorithm="hybrid", n=20, explain=False):
        return [
            {
                "movie_id": 1,
                "score": 0.95,
                "predicted_rating": 4.2,
                "explanation": [
                    {"algorithm": algorithm, "score": 0.95, "reason": "test explanation"}
                ] if explain else [],
            },
            {
                "movie_id": 2,
                "score": 0.82,
                "predicted_rating": 3.8,
                "explanation": [],
            },
        ][:n]

    def get_similar(self, movie_id, algorithm="collaborative", n=20):
        return [
            {"movie_id": 2, "score": 0.91, "predicted_rating": None, "explanation": []},
            {"movie_id": 3, "score": 0.78, "predicted_rating": None, "explanation": []},
        ][:n]


class FakeTMDbService:
    def get_movie_images(self, tmdb_id):
        return {"poster_url": None, "backdrop_url": None}


# ---------------------------------------------------------------------------
# Test app builder (no lifespan, no real DB or models)
# ---------------------------------------------------------------------------

def _build_test_app() -> FastAPI:
    """Build a minimal FastAPI app wired to fake services.

    Deliberately avoids ``create_app()`` so the lifespan handler (which
    tries to load real models and seed the DB) never runs.
    """
    settings = get_settings()

    app = FastAPI(title="Test Movie Recommender API")

    prefix = settings.API_V1_PREFIX
    app.include_router(health.router, prefix=prefix)
    app.include_router(recommendations.router, prefix=prefix)
    app.include_router(similar.router, prefix=prefix)
    app.include_router(movies.router, prefix=prefix)

    # Wire fake services into the DI system
    fake_rec = FakeRecommenderService()
    fake_movie = FakeMovieService()
    fake_tmdb = FakeTMDbService()

    app.dependency_overrides[get_recommender_service] = lambda: fake_rec
    app.dependency_overrides[get_movie_service] = lambda: fake_movie
    app.dependency_overrides[get_tmdb_service] = lambda: fake_tmdb

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> TestClient:
    """TestClient with dependency overrides for all three services."""
    app = _build_test_app()
    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc
