"""Evaluation pipeline for all recommender models.

Loads trained model artefacts, runs them against the held-out test set,
computes a comprehensive suite of metrics, logs everything to MLflow via
DagsHub, and writes an ``evaluation_summary.json`` to the ``models/``
directory.

Usage as a DVC stage::

    python -m src.pipeline.evaluate
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from src.evaluation.metrics import (
    catalog_coverage,
    intra_list_diversity,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)
from src.evaluation.verification import cosine_euclidean_correlation
from src.models.base import BaseRecommender
from src.utils.config import load_params, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum test users to evaluate per model (for speed)
_MAX_TEST_USERS = 500
# Default cut-off for ranking metrics
_K = 10
# Relevance threshold: a test rating >= this value is "relevant"
_RELEVANCE_THRESHOLD = 4.0


# ------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------

def _load_json_id_map(path: Path) -> dict[int, int]:
    """Load a JSON id-to-index mapping, converting string keys to int."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def _load_test_data(processed_dir: Path) -> pd.DataFrame:
    """Load the held-out test ratings."""
    test_path = processed_dir / "test_ratings.csv"
    logger.info("Loading test ratings from %s", test_path)
    df = pd.read_csv(test_path)
    logger.info("  Test rows: %d", len(df))
    return df


def _build_user_relevance(
    test_df: pd.DataFrame,
    threshold: float = _RELEVANCE_THRESHOLD,
) -> dict[int, list[int]]:
    """Build a mapping of user_id -> list of relevant movie IDs.

    A movie is relevant if the user rated it at or above *threshold* in
    the test set.
    """
    relevant = test_df[test_df["rating"] >= threshold]
    user_relevant: dict[int, list[int]] = {}
    for uid, grp in relevant.groupby("userId"):
        user_relevant[int(uid)] = grp["movieId"].astype(int).tolist()
    return user_relevant


def _build_user_test_pairs(
    test_df: pd.DataFrame,
) -> dict[int, list[tuple[int, float]]]:
    """Build user_id -> [(movie_id, actual_rating), ...] from test set."""
    pairs: dict[int, list[tuple[int, float]]] = {}
    for _, row in test_df.iterrows():
        uid = int(row["userId"])
        mid = int(row["movieId"])
        rating = float(row["rating"])
        pairs.setdefault(uid, []).append((mid, rating))
    return pairs


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "collaborative": (
        "src.models.collaborative",
        "ItemItemCF",
    ),
    "content_based": (
        "src.models.content_based",
        "ContentBasedRecommender",
    ),
    "als": (
        "src.models.als_model",
        "SparkALSRecommender",
    ),
    "hybrid": (
        "src.models.hybrid",
        "WeightedHybridRecommender",
    ),
}


def _load_model(model_type: str, models_dir: Path) -> BaseRecommender | None:
    """Attempt to load a trained model from disk.

    Returns ``None`` (instead of raising) if the model directory is
    missing or the artefacts are incomplete, so the evaluation pipeline
    can skip unavailable models gracefully.
    """
    model_dir = models_dir / model_type

    if not model_dir.exists():
        logger.warning("Model directory does not exist: %s -- skipping.", model_dir)
        return None

    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        logger.warning("No meta.json in %s -- skipping.", model_dir)
        return None

    if model_type not in _MODEL_REGISTRY:
        logger.warning("Unknown model type '%s' -- skipping.", model_type)
        return None

    module_path, class_name = _MODEL_REGISTRY[model_type]

    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        model = cls.load(model_dir)
        logger.info("Loaded %s from %s", model_type, model_dir)
        return model
    except Exception as exc:
        logger.warning("Failed to load %s: %s -- skipping.", model_type, exc)
        return None


# ------------------------------------------------------------------
# Per-model evaluation
# ------------------------------------------------------------------

def _evaluate_single_model(
    model_name: str,
    model: BaseRecommender,
    test_df: pd.DataFrame,
    user_relevance: dict[int, list[int]],
    user_test_pairs: dict[int, list[tuple[int, float]]],
    n_total_movies: int,
    similarity_matrix: Any | None,
    id_to_idx: dict[int, int] | None,
    k: int = _K,
    max_users: int = _MAX_TEST_USERS,
) -> dict[str, Any]:
    """Run the full evaluation suite for one model.

    Returns a dict of computed metric values.
    """
    logger.info("=" * 60)
    logger.info("Evaluating model: %s", model_name)
    logger.info("=" * 60)

    t0 = time.time()

    # Determine the set of test users to evaluate.  Intersect with users
    # the model knows about (via user_id_to_idx if available) to avoid
    # wasting time on guaranteed cold-start fallbacks.
    all_test_users = list(user_test_pairs.keys())
    model_known_users: set[int] | None = None
    if hasattr(model, "user_id_to_idx"):
        model_known_users = set(model.user_id_to_idx.keys())

    if model_known_users is not None:
        eligible = [u for u in all_test_users if u in model_known_users]
    else:
        eligible = all_test_users

    # Sample if needed
    rng = np.random.RandomState(42)
    if len(eligible) > max_users:
        sample_idx = rng.choice(len(eligible), size=max_users, replace=False)
        eval_users = [eligible[i] for i in sorted(sample_idx)]
    else:
        eval_users = eligible

    logger.info(
        "  Test users total: %d  |  eligible (known to model): %d  |  sampled: %d",
        len(all_test_users),
        len(eligible),
        len(eval_users),
    )

    # Accumulators
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    all_rec_lists: list[list[int]] = []
    all_rel_lists: list[list[int]] = []
    diversity_values: list[float] = []
    users_evaluated = 0
    users_skipped = 0

    for user_id in tqdm(eval_users, desc=f"  {model_name}", unit="user"):
        try:
            # --- Rating prediction (RMSE) ---
            pairs = user_test_pairs.get(user_id, [])
            for movie_id, actual_rating in pairs:
                try:
                    predicted = model.predict_rating(user_id, movie_id)
                    y_true_all.append(actual_rating)
                    y_pred_all.append(predicted)
                except Exception:
                    # Model cannot predict for this pair -- skip
                    pass

            # --- Ranking metrics ---
            try:
                recs = model.recommend(user_id, n=k, exclude_seen=True)
                rec_ids = [mid for mid, _ in recs]
            except Exception:
                rec_ids = []

            relevant = user_relevance.get(user_id, [])
            all_rec_lists.append(rec_ids)
            all_rel_lists.append(relevant)

            # --- Intra-list diversity ---
            if (
                similarity_matrix is not None
                and id_to_idx is not None
                and len(rec_ids) >= 2
            ):
                try:
                    div = intra_list_diversity(rec_ids, similarity_matrix, id_to_idx)
                    diversity_values.append(div)
                except Exception:
                    pass

            users_evaluated += 1

        except Exception as exc:
            logger.debug(
                "  Skipping user %d for %s: %s", user_id, model_name, exc,
            )
            users_skipped += 1

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    results: dict[str, Any] = {"model": model_name}

    # RMSE
    if y_true_all:
        results["rmse"] = rmse(np.array(y_true_all), np.array(y_pred_all))
        results["n_rating_predictions"] = len(y_true_all)
    else:
        results["rmse"] = None
        results["n_rating_predictions"] = 0

    # Precision, Recall, NDCG (averaged over users)
    if all_rec_lists:
        p_values = [
            precision_at_k(rec, rel, k)
            for rec, rel in zip(all_rec_lists, all_rel_lists)
        ]
        r_values = [
            recall_at_k(rec, rel, k)
            for rec, rel in zip(all_rec_lists, all_rel_lists)
        ]
        ndcg_values = [
            ndcg_at_k(rec, rel, k)
            for rec, rel in zip(all_rec_lists, all_rel_lists)
        ]

        results[f"precision_at_{k}"] = float(np.mean(p_values))
        results[f"recall_at_{k}"] = float(np.mean(r_values))
        results[f"ndcg_at_{k}"] = float(np.mean(ndcg_values))
        results[f"map_at_{k}"] = mean_average_precision(
            all_rec_lists, all_rel_lists, k,
        )
    else:
        results[f"precision_at_{k}"] = 0.0
        results[f"recall_at_{k}"] = 0.0
        results[f"ndcg_at_{k}"] = 0.0
        results[f"map_at_{k}"] = 0.0

    # Catalog coverage
    results["catalog_coverage"] = catalog_coverage(all_rec_lists, n_total_movies)

    # Average intra-list diversity
    if diversity_values:
        results["avg_intra_list_diversity"] = float(np.mean(diversity_values))
    else:
        results["avg_intra_list_diversity"] = None

    results["users_evaluated"] = users_evaluated
    results["users_skipped"] = users_skipped
    results["eval_time_s"] = round(elapsed, 2)

    logger.info("  Results for %s:", model_name)
    for key, val in results.items():
        if key == "model":
            continue
        if isinstance(val, float):
            logger.info("    %-28s %.4f", key, val)
        else:
            logger.info("    %-28s %s", key, val)

    return results


# ------------------------------------------------------------------
# Similarity matrix loader (for diversity metric)
# ------------------------------------------------------------------

def _get_similarity_matrix_and_map(
    models_dir: Path,
) -> tuple[Any | None, dict[int, int] | None]:
    """Try to load an item-item similarity matrix for diversity computation.

    Prefers the collaborative model's similarity matrix because it is
    based on rating co-occurrence.  Falls back to content-based.
    """
    for model_type, filename in [
        ("collaborative", "similarity.npz"),
        ("content_based", "content_similarity.npz"),
    ]:
        sim_path = models_dir / model_type / filename
        meta_path = models_dir / model_type / "meta.json"

        if sim_path.exists() and meta_path.exists():
            try:
                sim = sp.load_npz(sim_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                id_map = {
                    int(k): int(v) for k, v in meta["movie_id_to_idx"].items()
                }
                logger.info(
                    "Loaded similarity matrix from %s (%d x %d, nnz=%d)",
                    sim_path, sim.shape[0], sim.shape[1], sim.nnz,
                )
                return sim, id_map
            except Exception as exc:
                logger.warning("Could not load similarity from %s: %s", sim_path, exc)

    logger.info("No similarity matrix available -- diversity will be skipped.")
    return None, None


# ------------------------------------------------------------------
# Content features loader (for verification)
# ------------------------------------------------------------------

def _load_content_features(processed_dir: Path, models_dir: Path):
    """Load content features for cosine-Euclidean verification.

    Tries the processed data directory first, then the content-based
    model artefacts.
    """
    for path in [
        processed_dir / "content_features.npz",
        models_dir / "content_based" / "content_features.npz",
    ]:
        if path.exists():
            try:
                mat = sp.load_npz(path)
                logger.info(
                    "Loaded content features from %s (%d x %d)",
                    path, mat.shape[0], mat.shape[1],
                )
                return mat
            except Exception as exc:
                logger.warning("Could not load %s: %s", path, exc)

    return None


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def evaluate_all_models() -> dict:
    """Run the full evaluation suite across all available models.

    Returns
    -------
    dict
        Top-level keys: ``"models"`` (per-model metric dicts),
        ``"verification"`` (cosine-Euclidean result), ``"metadata"``.
    """
    params = load_params()
    processed_dir = settings.data_processed_dir
    models_dir = settings.models_dir

    logger.info("=" * 70)
    logger.info("EVALUATION PIPELINE START")
    logger.info("=" * 70)
    logger.info("  Processed data dir : %s", processed_dir)
    logger.info("  Models dir         : %s", models_dir)

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    test_df = _load_test_data(processed_dir)
    user_relevance = _build_user_relevance(test_df)
    user_test_pairs = _build_user_test_pairs(test_df)

    # Determine total catalog size from ID map
    movie_id_map_path = processed_dir / "movie_id_to_idx.json"
    if movie_id_map_path.exists():
        movie_id_map = _load_json_id_map(movie_id_map_path)
        n_total_movies = len(movie_id_map)
    else:
        n_total_movies = test_df["movieId"].nunique()

    logger.info(
        "  Test users: %d  |  Users with relevant items: %d  |  Catalog size: %d",
        len(user_test_pairs),
        len(user_relevance),
        n_total_movies,
    )

    # ------------------------------------------------------------------
    # Load similarity matrix for diversity computation
    # ------------------------------------------------------------------
    similarity_matrix, sim_id_to_idx = _get_similarity_matrix_and_map(models_dir)

    # ------------------------------------------------------------------
    # Evaluate each model
    # ------------------------------------------------------------------
    model_types = ["collaborative", "content_based", "als", "hybrid"]
    all_results: dict[str, dict[str, Any]] = {}

    for model_type in model_types:
        model = _load_model(model_type, models_dir)
        if model is None:
            logger.info("Skipping %s (not available).", model_type)
            continue

        results = _evaluate_single_model(
            model_name=model_type,
            model=model,
            test_df=test_df,
            user_relevance=user_relevance,
            user_test_pairs=user_test_pairs,
            n_total_movies=n_total_movies,
            similarity_matrix=similarity_matrix,
            id_to_idx=sim_id_to_idx,
            k=_K,
            max_users=_MAX_TEST_USERS,
        )
        all_results[model_type] = results

    # ------------------------------------------------------------------
    # Cosine-Euclidean verification
    # ------------------------------------------------------------------
    verification_result: dict[str, Any] = {}
    content_features = _load_content_features(processed_dir, models_dir)
    if content_features is not None:
        logger.info("Running cosine-Euclidean verification ...")
        try:
            verification_result = cosine_euclidean_correlation(
                content_features, sample_size=1000,
            )
            # Remove large lists from the summary to keep it concise
            verification_summary = {
                k: v
                for k, v in verification_result.items()
                if k not in ("cosine_values", "euclidean_values")
            }
            logger.info("Verification summary: %s", verification_summary)
        except Exception as exc:
            logger.warning("Verification failed: %s", exc)
            verification_result = {"error": str(exc)}
    else:
        logger.info("No content features found -- skipping verification.")

    # ------------------------------------------------------------------
    # Log to MLflow / DagsHub
    # ------------------------------------------------------------------
    _log_to_mlflow(all_results, verification_result, params)

    # ------------------------------------------------------------------
    # Save evaluation summary
    # ------------------------------------------------------------------
    summary = {
        "models": all_results,
        "verification": {
            k: v
            for k, v in verification_result.items()
            if k not in ("cosine_values", "euclidean_values")
        },
        "metadata": {
            "k": _K,
            "relevance_threshold": _RELEVANCE_THRESHOLD,
            "max_test_users": _MAX_TEST_USERS,
            "n_total_movies": n_total_movies,
            "test_rows": len(test_df),
        },
    }

    summary_path = models_dir / "evaluation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    logger.info("Evaluation summary saved to %s", summary_path)
    logger.info("=" * 70)
    logger.info("EVALUATION PIPELINE COMPLETE")
    logger.info("=" * 70)

    return summary


# ------------------------------------------------------------------
# MLflow logging
# ------------------------------------------------------------------

def _log_to_mlflow(
    model_results: dict[str, dict],
    verification: dict,
    params: dict,
) -> None:
    """Log evaluation metrics to MLflow via DagsHub."""
    try:
        from src.utils.mlflow_utils import init_tracking, log_model_metrics

        init_tracking()

        for model_name, results in model_results.items():
            # Collect only numeric metrics
            metrics = {
                k: v
                for k, v in results.items()
                if isinstance(v, (int, float)) and v is not None
            }
            # Prefix metrics with eval_ to distinguish from training metrics
            prefixed = {f"eval_{k}": v for k, v in metrics.items()}

            model_params = params.get(
                model_name.replace("_based", ""),
                params.get(model_name, {}),
            )
            if isinstance(model_params, dict):
                # Flatten nested params for MLflow
                flat_params = {
                    str(k): str(v) for k, v in model_params.items()
                }
            else:
                flat_params = {}

            flat_params["eval_k"] = str(_K)
            flat_params["eval_relevance_threshold"] = str(_RELEVANCE_THRESHOLD)

            log_model_metrics(
                model_name=f"eval_{model_name}",
                metrics=prefixed,
                params=flat_params,
            )

        # Log verification as a separate run
        if verification and "error" not in verification:
            verification_metrics = {
                k: v
                for k, v in verification.items()
                if isinstance(v, (int, float)) and k not in (
                    "cosine_values", "euclidean_values",
                )
            }
            if verification_metrics:
                log_model_metrics(
                    model_name="eval_cosine_euclidean_verification",
                    metrics=verification_metrics,
                    params={"sample_size": "1000"},
                )

        logger.info("Evaluation metrics logged to MLflow.")

    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): %s", exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ------------------------------------------------------------------
# CLI entry point (for DVC pipeline)
# ------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_all_models()
