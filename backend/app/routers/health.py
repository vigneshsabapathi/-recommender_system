"""Health-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.config import get_settings
from backend.app.db.database import movie_count
from backend.app.dependencies import get_recommender_service
from backend.app.services.recommender_service import RecommenderService

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Service health check",
    description="Returns service status, loaded model list, movie count, and version.",
)
def health_check(
    rec_svc: RecommenderService = Depends(get_recommender_service),
) -> dict:
    settings = get_settings()
    loaded = rec_svc.loaded_algorithms

    return {
        "status": "healthy" if loaded else "degraded",
        "models_loaded": loaded,
        "movie_count": movie_count(),
        "version": settings.APP_VERSION,
    }
