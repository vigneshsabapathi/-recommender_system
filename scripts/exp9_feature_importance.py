"""Experiment 9: Feature Importance -- isolate each feature source and measure its
standalone contribution to the content-based model.

Configurations:
  F1 (full)        : Genre TF-IDF (weight 0.3) + Genome Tags (weight 0.7)
  F2 (genre_only)  : Genre TF-IDF only (weight 1.0)
  F3 (genome_only) : Genome Tags only (weight 1.0)
"""
import time
import json
import sys
import os

# Force unbuffered output so progress is visible in background runs
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.experiment_utils import *

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from pathlib import Path

from src.data.feature_engineering import (
    build_genre_tfidf_features,
    build_genome_feature_matrix,
    build_content_features,
)
from src.models.content_based import ContentBasedRecommender


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

K = 10
MAX_USERS = 300
SEED = 42


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Feature configurations
# ---------------------------------------------------------------------------
FEATURE_CONFIGS = [
    {
        "name": "full",
        "description": "Genre TF-IDF (0.3) + Genome Tags (0.7)",
        "use_genre": True,
        "use_genome": True,
        "genre_weight": 0.3,
        "genome_weight": 0.7,
    },
    {
        "name": "genre_only",
        "description": "Genre TF-IDF only (1.0)",
        "use_genre": True,
        "use_genome": False,
        "genre_weight": 1.0,
        "genome_weight": 0.0,
    },
    {
        "name": "genome_only",
        "description": "Genome Tags only (1.0)",
        "use_genre": False,
        "use_genome": True,
        "genre_weight": 0.0,
        "genome_weight": 1.0,
    },
]


def load_ui_matrix():
    """Load user-item matrix and ID mappings from data/processed/."""
    ui_matrix = sp.load_npz(PROCESSED / "user_item_matrix.npz")

    with open(PROCESSED / "movie_id_to_idx.json") as f:
        movie_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}

    with open(PROCESSED / "user_id_to_idx.json") as f:
        user_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}

    return ui_matrix, movie_id_to_idx, user_id_to_idx


def build_feature_matrix(cfg, movie_metadata, movie_ids_ordered, genome_matrix_full, tfidf_full, tfidf_movie_ids):
    """Build the content feature matrix for a given configuration.

    Parameters
    ----------
    cfg : dict
        Feature configuration.
    movie_metadata : pd.DataFrame
        Movie metadata with genres and tag_str.
    movie_ids_ordered : list[int]
        Movie IDs in user-item matrix column order.
    genome_matrix_full : np.ndarray
        Full genome matrix aligned to tfidf_movie_ids order.
    tfidf_full : sparse matrix
        Full TF-IDF matrix aligned to tfidf_movie_ids order.
    tfidf_movie_ids : list[int]
        Movie IDs corresponding to rows of tfidf_full / genome_matrix_full.

    Returns
    -------
    sp.csr_matrix
        Content feature matrix with rows aligned to movie_ids_ordered.
    """
    # Build mapping from tfidf_movie_ids to row index
    tfidf_mid_to_row = {mid: idx for idx, mid in enumerate(tfidf_movie_ids)}

    # We need to reindex features to match user-item matrix's movie_id_to_idx order
    n_movies = len(movie_ids_ordered)

    if cfg["use_genre"] and cfg["use_genome"]:
        # Full: use build_content_features, then reindex
        # First reindex both matrices to movie_ids_ordered
        tfidf_reindexed = _reindex_sparse(tfidf_full, tfidf_mid_to_row, movie_ids_ordered)
        genome_reindexed = _reindex_dense(genome_matrix_full, tfidf_mid_to_row, movie_ids_ordered)

        combined = build_content_features(
            tfidf_reindexed, genome_reindexed,
            genre_weight=cfg["genre_weight"],
            genome_weight=cfg["genome_weight"],
        )
        return combined

    elif cfg["use_genre"] and not cfg["use_genome"]:
        # Genre only: reindex TF-IDF, L2-normalize
        tfidf_reindexed = _reindex_sparse(tfidf_full, tfidf_mid_to_row, movie_ids_ordered)
        normed = normalize(tfidf_reindexed, norm="l2", axis=1)
        return normed

    elif not cfg["use_genre"] and cfg["use_genome"]:
        # Genome only: reindex genome, L2-normalize, convert to sparse
        genome_reindexed = _reindex_dense(genome_matrix_full, tfidf_mid_to_row, movie_ids_ordered)
        genome_sparse = sp.csr_matrix(genome_reindexed)
        normed = normalize(genome_sparse, norm="l2", axis=1)
        return normed

    else:
        raise ValueError("At least one feature source must be enabled")


def _reindex_sparse(matrix, src_mid_to_row, target_movie_ids):
    """Reindex a sparse matrix from source order to target movie ID order.

    Movies not present in the source get all-zero rows.
    """
    n_target = len(target_movie_ids)
    n_features = matrix.shape[1]

    rows = []
    for mid in target_movie_ids:
        if mid in src_mid_to_row:
            rows.append(matrix[src_mid_to_row[mid]])
        else:
            rows.append(sp.csr_matrix((1, n_features), dtype=np.float32))

    return sp.vstack(rows, format="csr")


def _reindex_dense(matrix, src_mid_to_row, target_movie_ids):
    """Reindex a dense matrix from source order to target movie ID order.

    Movies not present in the source get all-zero rows.
    """
    n_target = len(target_movie_ids)
    n_features = matrix.shape[1]
    result = np.zeros((n_target, n_features), dtype=np.float32)

    for i, mid in enumerate(target_movie_ids):
        if mid in src_mid_to_row:
            result[i] = matrix[src_mid_to_row[mid]]

    return result


def run_experiment():
    mlflow = init_mlflow("feature-importance")

    log("=" * 70)
    log("EXPERIMENT 9: Feature Importance Analysis")
    log("=" * 70)

    # ------------------------------------------------------------------
    # Load shared data
    # ------------------------------------------------------------------
    log("\nLoading data...")

    test_df = load_test_data()
    train_df = load_train_data()
    log(f"  Test ratings:  {len(test_df):,}")
    log(f"  Train ratings: {len(train_df):,}")

    ui_matrix, movie_id_to_idx, user_id_to_idx = load_ui_matrix()
    log(f"  User-item matrix: {ui_matrix.shape[0]:,} users x {ui_matrix.shape[1]:,} movies")

    # Ordered movie IDs matching user-item matrix columns
    idx_to_movie_id = {v: k for k, v in movie_id_to_idx.items()}
    movie_ids_ordered = [idx_to_movie_id[i] for i in range(len(idx_to_movie_id))]

    # Load movie metadata
    movie_metadata = pd.read_csv(PROCESSED / "movie_metadata.csv")
    log(f"  Movie metadata: {len(movie_metadata):,} movies")

    # ------------------------------------------------------------------
    # Build raw feature matrices once (shared across configs)
    # ------------------------------------------------------------------
    log("\nBuilding TF-IDF genre features...")
    tfidf_full, tfidf_movie_ids = build_genre_tfidf_features(movie_metadata)
    log(f"  TF-IDF shape: {tfidf_full.shape}")

    log("\nBuilding genome feature matrix (this may take a few minutes)...")
    genome_scores_path = RAW / "genome-scores.csv"
    genome_matrix_full = build_genome_feature_matrix(genome_scores_path, tfidf_movie_ids)
    log(f"  Genome shape: {genome_matrix_full.shape}")

    # ------------------------------------------------------------------
    # Run each feature configuration
    # ------------------------------------------------------------------
    all_results = []
    baseline_ndcg = None

    for cfg in FEATURE_CONFIGS:
        run_name = f"F_{cfg['name']}"
        log(f"\n{'='*60}")
        log(f"Config: {cfg['name']}  --  {cfg['description']}")
        log(f"{'='*60}")

        # Build content features for this config
        log("  Building feature matrix...")
        t0 = time.time()
        content_features = build_feature_matrix(
            cfg, movie_metadata, movie_ids_ordered,
            genome_matrix_full, tfidf_full, tfidf_movie_ids,
        )
        build_time = time.time() - t0
        log(f"  Feature matrix: {content_features.shape}  ({build_time:.1f}s)")

        # Train content-based model
        log("  Training ContentBasedRecommender...")
        t0 = time.time()
        model = ContentBasedRecommender(top_k=50)
        model.fit(
            train_data=content_features,
            user_item_matrix=ui_matrix,
            movie_id_map=movie_id_to_idx,
            user_id_map=user_id_to_idx,
        )
        train_time = time.time() - t0
        log(f"  Training done ({train_time:.1f}s)")

        # Evaluate
        log(f"  Evaluating on {MAX_USERS} test users (K={K})...")
        t0 = time.time()
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "experiment_type": "feature_importance",
                "config_name": cfg["name"],
            })
            mlflow.log_params({
                "config_name": cfg["name"],
                "description": cfg["description"],
                "use_genre": cfg["use_genre"],
                "use_genome": cfg["use_genome"],
                "genre_weight": cfg["genre_weight"],
                "genome_weight": cfg["genome_weight"],
                "content_features_shape": str(content_features.shape),
                "eval_k": K,
                "max_test_users": MAX_USERS,
                "relevance_threshold": 4.0,
                "seed": SEED,
                "top_k_similarity": 50,
            })

            metrics = evaluate_model_on_users(
                model, test_df, train_df,
                k=K, max_users=MAX_USERS, relevance_threshold=4.0, seed=SEED,
            )
            eval_time = time.time() - t0
            metrics["eval_time_s"] = eval_time
            metrics["build_time_s"] = build_time
            metrics["train_time_s"] = train_time
            metrics["total_time_s"] = build_time + train_time + eval_time
            metrics["n_content_features"] = content_features.shape[1]

            ndcg_key = f"ndcg_at_{K}"
            current_ndcg = metrics.get(ndcg_key, 0.0)

            # Record baseline NDCG from full config
            if cfg["name"] == "full":
                baseline_ndcg = current_ndcg

            # Compute delta vs full
            if baseline_ndcg is not None:
                delta = current_ndcg - baseline_ndcg
            else:
                delta = 0.0
            metrics["delta_ndcg_vs_full"] = delta

            mlflow.log_metrics(metrics)

            prec_key = f"precision_at_{K}"
            recall_key = f"recall_at_{K}"
            map_key = f"map_at_{K}"
            log(f"  RMSE:             {metrics.get('rmse', 'N/A')}")
            log(f"  Precision@{K}:     {metrics.get(prec_key, 'N/A')}")
            log(f"  Recall@{K}:        {metrics.get(recall_key, 'N/A')}")
            log(f"  NDCG@{K}:          {metrics.get(ndcg_key, 'N/A')}")
            log(f"  MAP@{K}:           {metrics.get(map_key, 'N/A')}")
            log(f"  Coverage:         {metrics.get('catalog_coverage', 'N/A')}")
            log(f"  Delta NDCG:       {delta:+.4f}")
            log(f"  Eval time:        {eval_time:.1f}s")

            all_results.append({
                "name": cfg["name"],
                "description": cfg["description"],
                "n_features": content_features.shape[1],
                "rmse": metrics.get("rmse"),
                "ndcg": metrics.get(ndcg_key),
                "precision": metrics.get(prec_key),
                "recall": metrics.get(recall_key),
                "map": metrics.get(map_key),
                "coverage": metrics.get("catalog_coverage"),
                "delta_ndcg": delta,
                "users_eval": metrics.get("users_evaluated"),
                "total_time_s": metrics["total_time_s"],
            })

    # ------------------------------------------------------------------
    # Summary Table
    # ------------------------------------------------------------------
    log("\n")
    log("=" * 120)
    log("FEATURE IMPORTANCE RESULTS")
    log("=" * 120)

    header = (
        f"{'Config':<14} {'Features':>8}  {'RMSE':>8}  {'NDCG@10':>8}  "
        f"{'P@10':>8}  {'R@10':>8}  {'MAP@10':>8}  {'Coverage':>8}  "
        f"{'dNDCG':>8}  {'Users':>5}  {'Time(s)':>7}"
    )
    log(header)
    log("-" * 120)

    for r in all_results:
        def fmt(v, digits=4):
            return f"{v:.{digits}f}" if v is not None else "N/A"

        delta_str = f"{r['delta_ndcg']:+.4f}" if r["delta_ndcg"] is not None else "N/A"
        line = (
            f"{r['name']:<14} {r['n_features']:>8}  {fmt(r['rmse']):>8}  "
            f"{fmt(r['ndcg']):>8}  {fmt(r['precision']):>8}  "
            f"{fmt(r['recall']):>8}  {fmt(r['map']):>8}  "
            f"{fmt(r['coverage']):>8}  {delta_str:>8}  "
            f"{str(r.get('users_eval', 'N/A')):>5}  {r['total_time_s']:>7.1f}"
        )
        log(line)

    log("=" * 120)

    # Feature source contribution analysis
    log("\nFeature Source Contribution (standalone NDCG@10 vs full model):")
    log("-" * 55)
    for r in all_results:
        if r["name"] == "full":
            continue
        delta_str = f"{r['delta_ndcg']:+.4f}" if r["delta_ndcg"] is not None else "N/A"
        ndcg_str = f"{r['ndcg']:.4f}" if r["ndcg"] is not None else "N/A"
        log(f"  {r['name']:<14} NDCG@10 = {ndcg_str}  (vs full: {delta_str})")

    if len(all_results) >= 3:
        full_ndcg = all_results[0].get("ndcg", 0) or 0
        genre_ndcg = all_results[1].get("ndcg", 0) or 0
        genome_ndcg = all_results[2].get("ndcg", 0) or 0

        if genre_ndcg > genome_ndcg:
            log(f"\n  Genre TF-IDF alone ({genre_ndcg:.4f}) outperforms Genome Tags alone ({genome_ndcg:.4f})")
        elif genome_ndcg > genre_ndcg:
            log(f"\n  Genome Tags alone ({genome_ndcg:.4f}) outperforms Genre TF-IDF alone ({genre_ndcg:.4f})")
        else:
            log(f"\n  Both sources perform equally ({genre_ndcg:.4f})")

        if full_ndcg > max(genre_ndcg, genome_ndcg):
            log(f"  Combining both sources (full: {full_ndcg:.4f}) is better than either alone.")
        else:
            best_single = "genome_only" if genome_ndcg >= genre_ndcg else "genre_only"
            log(f"  Best single source ({best_single}) matches or exceeds the combined model.")

    log("\n\nDone! View results at:")
    log("https://dagshub.com/vigneshsabapathi/recommender_system.mlflow")


if __name__ == "__main__":
    run_experiment()
