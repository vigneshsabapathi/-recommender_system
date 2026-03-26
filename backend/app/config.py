"""Application configuration via Pydantic Settings.

All settings can be overridden by environment variables (case-insensitive)
or a ``.env`` file in the project root.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve project root as three levels up from this file:
# backend/app/config.py -> backend/app -> backend -> project_root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Backend configuration."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Paths ---
    PROJECT_ROOT: Path = _PROJECT_ROOT
    MODEL_DIR: Path = _PROJECT_ROOT / "models"
    DATA_DIR: Path = _PROJECT_ROOT / "data"
    DB_PATH: Path = _PROJECT_ROOT / "backend" / "movies.db"

    # --- API keys ---
    TMDB_API_KEY: str = ""

    # --- CORS ---
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # --- Server ---
    API_V1_PREFIX: str = "/api/v1"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton for the application settings."""
    return Settings()
