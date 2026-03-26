"""Training pipeline entry point for DVC stages.

Loads parameters and data artefacts, trains the specified model, saves
artefacts to ``models/{model_type}/``, and logs metrics to MLflow via
DagsHub.

Supported model types: ``collaborative``, ``content_based``, ``als``.

Usage as a DVC stage::

    python -m src.pipeline.train collaborative
    python -m src.pipeline.train content_based
    python -m src.pipeline.train als
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.config import load_params, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_MODEL_TYPES = {"collaborative", "content_based", "als"}


# ------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------
def _load_json_id_map(path: Path) -> dict[int, int]:
    """Load a JSON id-to-index mapping, converting string keys to int."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def _load_common_data(processed_dir: Path) -> tuple[
    sp.csr_matrix,
    dict[int, int],
    dict[int, int],
]:
    """Load user-item matrix and ID maps shared by multiple models."""
    logger.info("Loading user-item matrix ...")
    ui_matrix = sp.load_npz(processed_dir / "user_item_matrix.npz")
    logger.info("  Shape: %s, nnz: %d", ui_matrix.shape, ui_matrix.nnz)

    movie_id_map = _load_json_id_map(processed_dir / "movie_id_to_idx.json")
    user_id_map = _load_json_id_map(processed_dir / "user_id_to_idx.json")
    logger.info("  Movies: %d, Users: %d", len(movie_id_map), len(user_id_map))

    return ui_matrix, movie_id_map, user_id_map


# ------------------------------------------------------------------
# Per-model trainers
# ------------------------------------------------------------------
def _train_collaborative(params: dict, processed_dir: Path, output_dir: Path) -> dict:
    """Train the item-item collaborative filtering model."""
    from src.models.collaborative import ItemItemCF

    cf_params = params.get("collaborative", {})
    top_k = cf_params.get("top_k_similar", 50)

    ui_matrix, movie_id_map, user_id_map = _load_common_data(processed_dir)

    model = ItemItemCF(top_k=top_k)

    t0 = time.time()
    model.fit(ui_matrix, movie_id_map=movie_id_map, user_id_map=user_id_map)
    elapsed = time.time() - t0

    model.save(output_dir)

    metrics = {
        "training_time_s": round(elapsed, 2),
        "n_users": model.n_users,
        "n_items": model.n_items,
        "similarity_nnz": model.similarity.nnz,
        "top_k": top_k,
    }
    return metrics


def _train_content_based(params: dict, processed_dir: Path, output_dir: Path) -> dict:
    """Train the content-based recommender."""
    from src.models.content_based import ContentBasedRecommender

    cb_params = params.get("content_based", {})
    top_k = cb_params.get("top_k_similar", 50)

    ui_matrix, movie_id_map, user_id_map = _load_common_data(processed_dir)

    logger.info("Loading content features ...")
    content_features = sp.load_npz(processed_dir / "content_features.npz")
    logger.info("  Shape: %s, nnz: %d", content_features.shape, content_features.nnz)

    # Align content features to movie_id_map ordering
    # content_features rows are ordered by content_movie_id_to_idx
    content_mid_map_path = processed_dir / "content_movie_id_to_idx.json"
    if content_mid_map_path.exists():
        content_mid_map = _load_json_id_map(content_mid_map_path)
    else:
        # Assume same ordering as movie_id_map
        content_mid_map = movie_id_map

    # Reindex content features to match movie_id_map order
    content_aligned = _align_content_features(
        content_features, content_mid_map, movie_id_map,
    )

    # Load genome tags for explainability (optional)
    genome_tags = None
    genome_tags_path = processed_dir / "genome_tags.json"
    if genome_tags_path.exists():
        with open(genome_tags_path, "r", encoding="utf-8") as f:
            genome_tags = {int(k): v for k, v in json.load(f).items()}

    model = ContentBasedRecommender(top_k=top_k)

    t0 = time.time()
    model.fit(
        content_aligned,
        user_item_matrix=ui_matrix,
        movie_id_map=movie_id_map,
        user_id_map=user_id_map,
        genome_tags=genome_tags,
    )
    elapsed = time.time() - t0

    model.save(output_dir)

    metrics = {
        "training_time_s": round(elapsed, 2),
        "n_users": model.n_users,
        "n_items": model.n_items,
        "content_features_shape": list(content_aligned.shape),
        "similarity_nnz": model.content_similarity.nnz,
        "top_k": top_k,
    }
    return metrics


def _train_als(params: dict, processed_dir: Path, output_dir: Path) -> dict:
    """Train the ALS matrix-factorisation model."""
    from src.models.als_model import SparkALSRecommender

    als_params = params.get("als", {})

    ui_matrix, movie_id_map, user_id_map = _load_common_data(processed_dir)

    logger.info("Loading training ratings for Spark ALS ...")
    train_df = pd.read_csv(
        processed_dir / "train_ratings.csv",
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
    )
    logger.info("  Rows: %d", len(train_df))

    model = SparkALSRecommender()

    t0 = time.time()
    model.fit(
        train_df,
        params=als_params,
        user_item_matrix=ui_matrix,
        movie_id_map=movie_id_map,
        user_id_map=user_id_map,
    )
    elapsed = time.time() - t0

    model.save(output_dir)

    metrics = {
        "training_time_s": round(elapsed, 2),
        "n_users": model.n_users,
        "n_items": model.n_items,
        "rank": model.rank,
    }
    return metrics


# ------------------------------------------------------------------
# Content feature alignment
# ------------------------------------------------------------------
def _align_content_features(
    content_features: sp.csr_matrix,
    content_mid_map: dict[int, int],
    target_mid_map: dict[int, int],
) -> sp.csr_matrix:
    """Reindex content feature rows to match *target_mid_map* ordering.

    If the two maps already share the same ordering, the original matrix
    is returned as-is.  Otherwise, rows are permuted (or zero-padded for
    movies missing from the content matrix).
    """
    # Quick check: are the two maps identical?
    if content_mid_map == target_mid_map:
        logger.info("  Content features already aligned with movie_id_map.")
        return content_features

    n_target = len(target_mid_map)
    n_features = content_features.shape[1]

    # Build reverse map: content_idx -> content row
    idx_to_content_mid = {v: k for k, v in content_mid_map.items()}

    # Build permutation
    rows: list[np.ndarray] = []
    for target_mid in sorted(target_mid_map, key=target_mid_map.get):
        target_idx = target_mid_map[target_mid]
        if target_mid in content_mid_map:
            content_idx = content_mid_map[target_mid]
            rows.append(content_features[content_idx].toarray().flatten())
        else:
            rows.append(np.zeros(n_features, dtype=np.float32))

    aligned = sp.csr_matrix(np.vstack(rows), dtype=np.float32)
    logger.info(
        "  Aligned content features: %s -> %s",
        content_features.shape, aligned.shape,
    )
    return aligned


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def train_model(model_type: str) -> None:
    """Train a single model and save artefacts.

    Parameters
    ----------
    model_type : str
        One of ``"collaborative"``, ``"content_based"``, ``"als"``.
    """
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be one of: {sorted(_VALID_MODEL_TYPES)}"
        )

    params = load_params()
    processed_dir = settings.data_processed_dir
    output_dir = settings.models_dir / model_type

    logger.info("=" * 70)
    logger.info("TRAINING MODEL: %s", model_type)
    logger.info("=" * 70)
    logger.info("  Processed data dir : %s", processed_dir)
    logger.info("  Model output dir   : %s", output_dir)

    # Dispatch to the appropriate trainer
    trainers = {
        "collaborative": _train_collaborative,
        "content_based": _train_content_based,
        "als": _train_als,
    }
    metrics = trainers[model_type](params, processed_dir, output_dir)

    logger.info("Training metrics for %s: %s", model_type, metrics)

    # Log to MLflow / DagsHub
    try:
        from src.utils.mlflow_utils import init_tracking, log_model_metrics

        init_tracking()
        log_model_metrics(
            model_name=model_type,
            metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            params=params.get(model_type.replace("_based", ""), params.get(model_type, {})),
        )
        logger.info("Metrics logged to MLflow.")
    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): %s", exc)

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE: %s", model_type)
    logger.info("=" * 70)


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python -m src.pipeline.train <model_type>")
        print(f"  model_type: {sorted(_VALID_MODEL_TYPES)}")
        sys.exit(1)

    train_model(sys.argv[1])
