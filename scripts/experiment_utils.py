"""Shared experiment utilities - MLflow logging, evaluation, encoding fixes."""

import os
import sys
import io
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Fix Windows encoding for MLflow emoji output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Set DagsHub credentials from .env
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")
os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "vigneshsabapathi")
os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", os.environ.get("DAGSHUB_USER_TOKEN", ""))


def init_mlflow(experiment_name: str):
    """Initialize MLflow with DagsHub tracking."""
    import dagshub
    import mlflow

    dagshub.init(
        repo_owner="vigneshsabapathi",
        repo_name="recommender_system",
        mlflow=True,
    )
    mlflow.set_experiment(experiment_name)
    return mlflow


def load_test_data():
    """Load test ratings and return DataFrame."""
    processed = ROOT_DIR / "data" / "processed"
    test_df = pd.read_csv(processed / "test_ratings.csv")
    return test_df


def load_train_data():
    """Load train ratings and return DataFrame."""
    processed = ROOT_DIR / "data" / "processed"
    train_df = pd.read_csv(processed / "train_ratings.csv")
    return train_df


def load_models():
    """Load all trained models. Returns dict of model_name -> model."""
    from src.models.collaborative import ItemItemCF
    from src.models.content_based import ContentBasedRecommender
    from src.models.als_model import SparkALSRecommender
    from src.models.hybrid import WeightedHybridRecommender

    models_dir = ROOT_DIR / "models"
    loaded = {}

    try:
        cf = ItemItemCF.load(models_dir / "collaborative")
        loaded["collaborative"] = cf
    except Exception as e:
        print(f"  Warning: Could not load CF model: {e}")

    try:
        cb = ContentBasedRecommender.load(models_dir / "content_based")
        loaded["content_based"] = cb
    except Exception as e:
        print(f"  Warning: Could not load Content model: {e}")

    try:
        als = SparkALSRecommender.load(models_dir / "als")
        loaded["als"] = als
    except Exception as e:
        print(f"  Warning: Could not load ALS model: {e}")

    if len(loaded) >= 2:
        try:
            hybrid = WeightedHybridRecommender(
                cf_model=loaded.get("collaborative"),
                content_model=loaded.get("content_based"),
                als_model=loaded.get("als"),
            )
            loaded["hybrid"] = hybrid
        except Exception as e:
            print(f"  Warning: Could not create Hybrid model: {e}")

    return loaded


def evaluate_model_on_users(
    model, test_df, train_df, k=10, max_users=500, relevance_threshold=4.0, seed=42
):
    """Evaluate a single model on sampled test users.

    Returns dict of metrics.
    """
    from src.evaluation.metrics import (
        rmse, precision_at_k, recall_at_k, ndcg_at_k,
        catalog_coverage, mean_average_precision,
    )

    rng = np.random.RandomState(seed)
    test_users = test_df["userId"].unique()

    # Only evaluate users the model knows about
    if hasattr(model, "user_id_to_idx"):
        known_users = set(model.user_id_to_idx.keys())
        test_users = np.array([u for u in test_users if u in known_users])

    if len(test_users) > max_users:
        test_users = rng.choice(test_users, size=max_users, replace=False)

    # Build per-user test sets
    test_grouped = test_df.groupby("userId")
    train_grouped = train_df.groupby("userId")

    all_rmse_preds = []
    all_rmse_true = []
    all_precisions = []
    all_recalls = []
    all_ndcgs = []
    all_rec_lists = []
    all_relevant_lists = []
    users_evaluated = 0
    users_skipped = 0

    for uid in test_users:
        try:
            if uid not in test_grouped.groups:
                continue

            user_test = test_grouped.get_group(uid)
            relevant = user_test[user_test["rating"] >= relevance_threshold]["movieId"].tolist()

            # Get recommendations
            recs = model.recommend(int(uid), n=k, exclude_seen=True)
            if not recs:
                users_skipped += 1
                continue

            rec_ids = [r[0] for r in recs]
            all_rec_lists.append(rec_ids)

            # Ranking metrics
            if relevant:
                all_precisions.append(precision_at_k(rec_ids, relevant, k))
                all_recalls.append(recall_at_k(rec_ids, relevant, k))
                all_ndcgs.append(ndcg_at_k(rec_ids, relevant, k))
                all_relevant_lists.append(relevant)

            # Rating prediction (RMSE)
            for _, row in user_test.iterrows():
                try:
                    pred = model.predict_rating(int(uid), int(row["movieId"]))
                    if pred is not None and not np.isnan(pred):
                        all_rmse_true.append(row["rating"])
                        all_rmse_preds.append(pred)
                except Exception:
                    pass

            users_evaluated += 1
        except Exception:
            users_skipped += 1

    # Aggregate metrics
    metrics = {
        "users_evaluated": users_evaluated,
        "users_skipped": users_skipped,
    }

    if all_rmse_true:
        metrics["rmse"] = float(rmse(np.array(all_rmse_true), np.array(all_rmse_preds)))
        metrics["n_rating_predictions"] = len(all_rmse_true)

    if all_precisions:
        metrics[f"precision_at_{k}"] = float(np.mean(all_precisions))
    if all_recalls:
        metrics[f"recall_at_{k}"] = float(np.mean(all_recalls))
    if all_ndcgs:
        metrics[f"ndcg_at_{k}"] = float(np.mean(all_ndcgs))
    if all_relevant_lists:
        metrics[f"map_at_{k}"] = float(mean_average_precision(all_rec_lists, all_relevant_lists, k))

    # Catalog coverage
    if all_rec_lists:
        n_movies = len(set(test_df["movieId"].unique()))
        metrics["catalog_coverage"] = float(catalog_coverage(all_rec_lists, n_movies))

    return metrics


class PopularityBaseline:
    """Simple baseline: recommend the most popular movies to everyone."""

    def __init__(self, train_df):
        popular = train_df.groupby("movieId")["rating"].agg(["count", "mean"])
        popular = popular.sort_values("count", ascending=False)
        self.popular_ids = popular.index.tolist()
        self.global_mean = float(train_df["rating"].mean())
        self.movie_means = train_df.groupby("movieId")["rating"].mean().to_dict()
        # NOTE: Do NOT set user_id_to_idx so that evaluate_model_on_users
        # does not filter out test users (popularity works for any user).

    def recommend(self, user_id, n=20, exclude_seen=True):
        return [(mid, 1.0 - i / len(self.popular_ids)) for i, mid in enumerate(self.popular_ids[:n])]

    def predict_rating(self, user_id, movie_id):
        return self.movie_means.get(movie_id, self.global_mean)

    def similar_items(self, movie_id, n=20):
        return []

    def explain(self, user_id, movie_id):
        return {"algorithm": "popularity", "reason": "Most popular movies"}
