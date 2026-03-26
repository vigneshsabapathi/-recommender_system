"""Similar-movies endpoint."""

from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, Depends, Query

from backend.app.dependencies import get_movie_service, get_recommender_service
from backend.app.schemas.movie import MovieCard
from backend.app.schemas.recommendation import (
    MovieRecommendation,
    RecommendationMeta,
    SimilarMoviesResponse,
)
from backend.app.services.movie_service import MovieService
from backend.app.services.recommender_service import RecommenderService

router = APIRouter(tags=["similar"])

AlgorithmParam = Literal["collaborative", "content_based", "als", "hybrid", "blended_similar"]


@router.get(
    "/similar/{movie_id}",
    response_model=SimilarMoviesResponse,
    summary="Find similar movies",
    description="Return the top-N most similar movies to the given movie.",
)
def get_similar_movies(
    movie_id: int,
    algorithm: AlgorithmParam = Query("blended_similar", description="Similarity algorithm"),
    n: int = Query(20, ge=1, le=100, description="Number of similar movies"),
    rec_svc: RecommenderService = Depends(get_recommender_service),
    movie_svc: MovieService = Depends(get_movie_service),
) -> SimilarMoviesResponse:
    t0 = time.perf_counter()

    raw_sims = rec_svc.get_similar(
        movie_id=movie_id,
        algorithm=algorithm,
        n=n,
    )

    # Batch-fetch movie cards
    sim_ids = [r["movie_id"] for r in raw_sims]
    cards = movie_svc.get_movie_cards(sim_ids)

    similar: list[MovieRecommendation] = []
    for rec in raw_sims:
        mid = rec["movie_id"]
        card = cards.get(mid)
        if card is None:
            card = MovieCard(id=mid, title=f"Movie {mid}")

        similar.append(
            MovieRecommendation(
                movie=card,
                score=rec["score"],
                predicted_rating=rec.get("predicted_rating"),
                explanation=[],
            )
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return SimilarMoviesResponse(
        movie_id=movie_id,
        algorithm=algorithm,
        similar=similar,
        metadata=RecommendationMeta(
            total=len(similar),
            processing_time_ms=round(elapsed_ms, 2),
        ),
    )
