"""Central configuration using Pydantic Settings."""

from pathlib import Path
from pydantic_settings import BaseSettings
import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_params() -> dict:
    """Load hyperparameters from params.yaml."""
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # Paths
    root_dir: Path = ROOT_DIR
    data_raw_dir: Path = ROOT_DIR / "data" / "raw"
    data_processed_dir: Path = ROOT_DIR / "data" / "processed"
    models_dir: Path = ROOT_DIR / "models"

    # DagsHub / MLflow
    dagshub_token: str = ""
    mlflow_tracking_uri: str = (
        "https://dagshub.com/vigneshsabapathi/recommender_system.mlflow"
    )

    # TMDb
    tmdb_api_key: str = ""

    # Dataset
    movielens_url: str = (
        "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    )

    model_config = {"env_file": str(ROOT_DIR / ".env"), "extra": "ignore"}


settings = Settings()
