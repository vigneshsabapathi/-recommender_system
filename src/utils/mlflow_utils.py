"""DagsHub/MLflow experiment tracking utilities."""

import dagshub
import mlflow


def init_tracking(experiment_name: str = "recommender_system") -> None:
    """Initialize DagsHub + MLflow tracking."""
    dagshub.init(
        repo_name="recommender_system",
        repo_owner="vigneshsabapathi",
        mlflow=True,
    )
    mlflow.set_experiment(experiment_name)


def log_model_metrics(model_name: str, metrics: dict, params: dict) -> None:
    """Log metrics and params for a model run."""
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
