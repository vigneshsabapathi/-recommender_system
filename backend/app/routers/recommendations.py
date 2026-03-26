"""Recommendation endpoint -- personalised movie suggestions for a user."""

from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.dependencies import get_movie_service, get_recommender_service
from backend.app.schemas.movie import MovieCard
from backend.app.schemas.recommendation import (
    ExplanationItem,
    MovieRecommendation,
    RecommendationMeta,
    RecommendationResponse,
)
from backend.app.services.movie_service import MovieService
from backend.app.services.recommender_service import RecommenderService, VALID_ALGORITHMS

router = APIRouter(tags=["recommendations"])

AlgorithmParam = Literal["collaborative", "content_based", "als", "hybrid"]


@router.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
    summary="Get personalised recommendations",
    description="Return top-N movie recommendations for a user, using the specified algorithm.",
)
def get_recommendations(
    user_id: int,
    algorithm: AlgorithmParam = Query("hybrid", description="Recommendation algorithm"),
    n: int = Query(20, ge=1, le=100, description="Number of recommendations"),
    explain: bool = Query(False, description="Include per-item explanations"),
    rec_svc: RecommenderService = Depends(get_recommender_service),
    movie_svc: MovieService = Depends(get_movie_service),
) -> RecommendationResponse:
    t0 = time.perf_counter()

    raw_recs = rec_svc.get_recommendations(
        user_id=user_id,
        algorithm=algorithm,
        n=n,
        explain=explain,
    )

    # Batch-fetch movie metadata
    movie_ids = [r["movie_id"] for r in raw_recs]
    cards = movie_svc.get_movie_cards(movie_ids)

    recommendations: list[MovieRecommendation] = []
    for rec in raw_recs:
        mid = rec["movie_id"]
        card = cards.get(mid)
        if card is None:
            # Movie not in DB -- build a minimal placeholder
            card = MovieCard(id=mid, title=f"Movie {mid}")

        explanation_items = [
            ExplanationItem(
                algorithm=e.get("algorithm", algorithm),
                score=e.get("score", 0.0),
                reason=e.get("reason", ""),
            )
            for e in rec.get("explanation", [])
        ]

        recommendations.append(
            MovieRecommendation(
                movie=card,
                score=rec["score"],
                predicted_rating=rec.get("predicted_rating"),
                explanation=explanation_items,
            )
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return RecommendationResponse(
        user_id=user_id,
        algorithm=algorithm,
        recommendations=recommendations,
        metadata=RecommendationMeta(
            total=len(recommendations),
            processing_time_ms=round(elapsed_ms, 2),
        ),
    )
