"""Public schema re-exports."""

from backend.app.schemas.common import ErrorResponse, PaginatedResponse, PaginationMeta
from backend.app.schemas.movie import MovieCard, MovieDetail
from backend.app.schemas.recommendation import (
    ExplanationItem,
    MovieRecommendation,
    RecommendationMeta,
    RecommendationResponse,
    SimilarMoviesResponse,
)

__all__ = [
    "ErrorResponse",
    "PaginatedResponse",
    "PaginationMeta",
    "MovieCard",
    "MovieDetail",
    "ExplanationItem",
    "MovieRecommendation",
    "RecommendationMeta",
    "RecommendationResponse",
    "SimilarMoviesResponse",
]
