"""Shared response schemas used across multiple endpoints."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorResponse(BaseModel):
    """Standard error envelope returned by all error handlers."""

    detail: str
    status_code: int = Field(ge=400, le=599)


class PaginationMeta(BaseModel):
    """Pagination metadata included in paginated responses."""

    page: int = Field(ge=1)
    per_page: int = Field(ge=1, le=100)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated wrapper.

    The ``items`` list holds the current page of results.  ``meta`` provides
    pagination bookkeeping so the client can render page controls.
    """

    items: list[Any] = Field(default_factory=list)
    meta: PaginationMeta
