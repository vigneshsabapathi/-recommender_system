"""FastAPI application entry point.

Startup sequence:
  1. Initialise SQLite schema and seed movie metadata.
  2. Create service singletons (movie, tmdb, recommender).
  3. Load ML models from disk.
  4. Mount routers and middleware.

Run with::

    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import get_settings
from backend.app.db.database import init_db
from backend.app.db.seed import seed_database
from backend.app.dependencies import init_services
from backend.app.routers import health, movies, recommendations, similar

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler -- runs on startup and shutdown."""
    settings = get_settings()
    t0 = time.perf_counter()

    # 1. Database
    logger.info("Initialising database ...")
    init_db()
    seed_database()

    # 2. Services
    logger.info("Creating services ...")
    services = init_services()

    # 3. Load ML models
    logger.info("Loading ML models from %s ...", settings.MODEL_DIR)
    rec_svc = services["recommender_service"]
    loaded = rec_svc.load_models()
    logger.info("Loaded models: %s", loaded)

    elapsed = time.perf_counter() - t0
    logger.info("Startup complete in %.2f s", elapsed)

    yield

    logger.info("Shutting down ...")


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------
def create_app() -> FastAPI:
    """Build and return the configured FastAPI instance."""
    settings = get_settings()

    application = FastAPI(
        title="Movie Recommender API",
        description="REST API serving collaborative, content-based, ALS, and hybrid movie recommendations.",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # -- CORS --
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Routers --
    prefix = settings.API_V1_PREFIX
    application.include_router(health.router, prefix=prefix)
    application.include_router(recommendations.router, prefix=prefix)
    application.include_router(similar.router, prefix=prefix)
    application.include_router(movies.router, prefix=prefix)

    # -- Exception handlers --
    @application.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "status_code": 400},
        )

    @application.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "status_code": 500},
        )

    return application


# Module-level app instance for uvicorn
app = create_app()
