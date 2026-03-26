"""Recommendation and similarity response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.app.schemas.movie import MovieCard


class ExplanationItem(BaseModel):
    """One component of a recommendation explanation."""

    algorithm: str
    score: float
    reason: str = ""


class MovieRecommendation(BaseModel):
    """A single recommended movie with scoring metadata."""

    movie: MovieCard
    score: float = Field(description="Normalised relevance score")
    predicted_rating: float | None = Field(
        default=None, ge=0.5, le=5.0,
        description="Predicted star rating on the 0.5-5.0 scale",
    )
    explanation: list[ExplanationItem] = Field(default_factory=list)


class RecommendationMeta(BaseModel):
    """Metadata block for a recommendation response."""

    total: int = Field(ge=0)
    processing_time_ms: float = Field(ge=0.0)


class RecommendationResponse(BaseModel):
    """Top-level response from the recommendations endpoint."""

    user_id: int
    algorithm: str
    recommendations: list[MovieRecommendation] = Field(default_factory=list)
    metadata: RecommendationMeta


class SimilarMoviesResponse(BaseModel):
    """Top-level response from the similar-movies endpoint."""

    movie_id: int
    algorithm: str
    similar: list[MovieRecommendation] = Field(default_factory=list)
    metadata: RecommendationMeta
