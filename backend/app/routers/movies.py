"""Movie browsing, search, and detail endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.dependencies import get_movie_service
from backend.app.schemas.common import PaginatedResponse
from backend.app.schemas.movie import MovieCard, MovieDetail
from backend.app.services.movie_service import MovieService

router = APIRouter(tags=["movies"])


@router.get(
    "/movies/search",
    response_model=list[MovieCard],
    summary="Search movies by title",
)
def search_movies(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    movie_svc: MovieService = Depends(get_movie_service),
) -> list[MovieCard]:
    return movie_svc.search_movies(query=q, limit=limit)


@router.get(
    "/movies/{movie_id}",
    response_model=MovieDetail,
    summary="Get movie details",
)
def get_movie_detail(
    movie_id: int,
    movie_svc: MovieService = Depends(get_movie_service),
) -> MovieDetail:
    movie = movie_svc.get_movie(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    return movie


@router.get(
    "/movies",
    response_model=PaginatedResponse,
    summary="Browse movies",
    description="Paginated movie listing with optional genre filter and sort.",
)
def list_movies(
    genre: str | None = Query(None, description="Filter by genre name"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort_by: Literal["avg_rating", "num_ratings", "year", "title"] = Query("num_ratings"),
    movie_svc: MovieService = Depends(get_movie_service),
) -> PaginatedResponse:
    return movie_svc.get_movies(
        genre=genre,
        page=page,
        per_page=per_page,
        sort_by=sort_by,
    )
