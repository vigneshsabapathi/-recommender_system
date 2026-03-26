"""Movie-related Pydantic schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MovieCard(BaseModel):
    """Compact movie representation for list views and cards."""

    id: int = Field(description="MovieLens movie ID")
    title: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    poster_url: str | None = None
    avg_rating: float | None = Field(default=None, ge=0.0, le=5.0)


class MovieDetail(MovieCard):
    """Extended movie representation for the detail page."""

    backdrop_url: str | None = None
    overview: str | None = None
    num_ratings: int | None = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list)
    imdb_id: str | None = None
    tmdb_id: int | None = None
