"""Alternating Least Squares (ALS) recommender using PySpark.

Trains a matrix-factorisation model via :class:`pyspark.ml.recommendation.ALS`,
then extracts the learned user and item factor matrices as NumPy arrays so
that inference can happen in pure NumPy without a live Spark session.

Typical usage::

    from src.models.als_model import SparkALSRecommender
    model = SparkALSRecommender()
    model.fit(ratings_df, params={"rank": 64, "max_iter": 15, "reg_param": 0.1})
    recs = model.recommend(user_id=42, n=10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.models.base import BaseRecommender
from src.utils.logger import get_logger

logger = get_logger(__name__)

_GLOBAL_MEAN_FALLBACK = 3.5


class SparkALSRecommender(BaseRecommender):
    """ALS matrix-factorisation recommender backed by PySpark.

    The Spark session is created during :meth:`fit` and stopped as soon
    as factor matrices have been extracted, so the model is self-contained
    in NumPy for all downstream operations.
    """

    def __init__(self):
        # Factor matrices  (populated by fit / load)
        self.user_factors: np.ndarray | None = None  # (n_users, rank)
        self.item_factors: np.ndarray | None = None  # (n_items, rank)
        self.item_factors_normed: np.ndarray | None = None  # L2-normed, for similarity

        # ID mappings
        self.movie_id_to_idx: dict[int, int] = {}
        self.user_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}
        self.idx_to_user_id: dict[int, int] = {}

        # Metadata
        self.rank: int = 0
        self.n_users: int = 0
        self.n_items: int = 0
        self.global_mean: float = _GLOBAL_MEAN_FALLBACK
        self._popular_items: list[int] = []

        # Sparse user-item matrix for seen-item tracking
        self.user_item_matrix: sp.csr_matrix | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(  # type: ignore[override]
        self,
        train_data: pd.DataFrame,
        params: dict | None = None,
        user_item_matrix: sp.csr_matrix | None = None,
        movie_id_map: dict[int, int] | None = None,
        user_id_map: dict[int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Train an ALS model with PySpark and extract factor matrices.

        Parameters
        ----------
        train_data : pd.DataFrame
            Ratings dataframe with columns ``userId``, ``movieId``, ``rating``.
        params : dict, optional
            ALS hyperparameters: ``rank``, ``max_iter``, ``reg_param``,
            ``cold_start_strategy``.
        user_item_matrix : scipy.sparse.csr_matrix, optional
            Pre-built user-item matrix for seen-item tracking.
        movie_id_map : dict[int, int], optional
            Maps original ``movieId`` to matrix column index.
        user_id_map : dict[int, int], optional
            Maps original ``userId`` to matrix row index.
        """
        params = params or {}
        rank = params.get("rank", 64)
        max_iter = params.get("max_iter", 15)
        reg_param = params.get("reg_param", 0.1)
        cold_start_strategy = params.get("cold_start_strategy", "drop")

        self.rank = rank

        # Store ID mappings
        if movie_id_map is not None:
            self.movie_id_to_idx = {int(k): int(v) for k, v in movie_id_map.items()}
        if user_id_map is not None:
            self.user_id_to_idx = {int(k): int(v) for k, v in user_id_map.items()}

        if user_item_matrix is not None:
            self.user_item_matrix = user_item_matrix

        # Global mean from training data
        if len(train_data) > 0:
            self.global_mean = float(train_data["rating"].mean())

        # ------------------------------------------------------------------
        # PySpark ALS training
        # ------------------------------------------------------------------
        logger.info(
            "Starting PySpark ALS training (rank=%d, maxIter=%d, regParam=%.4f) ...",
            rank, max_iter, reg_param,
        )

        from pyspark.sql import SparkSession
        from pyspark.ml.recommendation import ALS as SparkALS

        spark = SparkSession.builder \
            .master("local[*]") \
            .appName("recommender_als") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.ui.showConsoleProgress", "false") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        try:
            # Prepare Spark DataFrame
            ratings_pd = train_data[["userId", "movieId", "rating"]].copy()
            ratings_pd["userId"] = ratings_pd["userId"].astype(int)
            ratings_pd["movieId"] = ratings_pd["movieId"].astype(int)
            ratings_pd["rating"] = ratings_pd["rating"].astype(float)

            spark_df = spark.createDataFrame(ratings_pd)

            # Train ALS
            als = SparkALS(
                rank=rank,
                maxIter=max_iter,
                regParam=reg_param,
                userCol="userId",
                itemCol="movieId",
                ratingCol="rating",
                coldStartStrategy=cold_start_strategy,
                nonneg=False,
                seed=42,
            )
            model = als.fit(spark_df)

            # Extract factor matrices as pandas, then numpy
            user_factors_df = model.userFactors.toPandas()
            item_factors_df = model.itemFactors.toPandas()

            logger.info(
                "ALS training complete. User factors: %d, Item factors: %d",
                len(user_factors_df), len(item_factors_df),
            )

        finally:
            spark.stop()
            logger.info("SparkSession stopped.")

        # ------------------------------------------------------------------
        # Build numpy factor matrices with consistent indexing
        # ------------------------------------------------------------------
        self._build_factor_matrices(user_factors_df, item_factors_df)

        # Precompute L2-normalised item factors for similarity queries
        self.item_factors_normed = normalize(self.item_factors, norm="l2", axis=1)

        # Popularity fallback
        self._popular_items = self._compute_popular_items(train_data)

        logger.info(
            "Factor matrices built: users=%s, items=%s, rank=%d",
            self.user_factors.shape, self.item_factors.shape, self.rank,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        n: int = 20,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Recommend via dot product of user and item factor vectors."""
        if user_id not in self.user_id_to_idx:
            return self._popular_fallback(n)

        user_idx = self.user_id_to_idx[user_id]
        user_vec = self.user_factors[user_idx]  # (rank,)

        # Dot product with all item factors
        scores = self.item_factors @ user_vec  # (n_items,)

        # Exclude seen items
        if exclude_seen and self.user_item_matrix is not None:
            seen_indices = self.user_item_matrix[user_idx].indices
            scores[seen_indices] = -np.inf

        # Top-n
        if n < len(scores):
            top_indices = np.argpartition(scores, -n)[-n:]
        else:
            top_indices = np.arange(len(scores))

        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        top_indices = top_indices[:n]

        results: list[tuple[int, float]] = []
        for idx in top_indices:
            if np.isinf(scores[idx]):
                continue
            movie_id = self.idx_to_movie_id.get(int(idx))
            if movie_id is not None:
                results.append((movie_id, float(scores[idx])))

        return results

    def similar_items(self, movie_id: int, n: int = 20) -> list[tuple[int, float]]:
        """Find similar items via cosine similarity in the latent factor space."""
        if movie_id not in self.movie_id_to_idx:
            return []

        item_idx = self.movie_id_to_idx[movie_id]
        item_vec = self.item_factors_normed[item_idx]  # (rank,)

        # Cosine similarity with all items (already L2-normed)
        similarities = self.item_factors_normed @ item_vec  # (n_items,)
        similarities[item_idx] = -np.inf  # exclude self

        if n < len(similarities):
            top_indices = np.argpartition(similarities, -n)[-n:]
        else:
            top_indices = np.arange(len(similarities))

        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        top_indices = top_indices[:n]

        results: list[tuple[int, float]] = []
        for idx in top_indices:
            if np.isinf(similarities[idx]):
                continue
            mid = self.idx_to_movie_id.get(int(idx))
            if mid is not None:
                results.append((mid, float(similarities[idx])))

        return results

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict a single rating via dot product of latent factors."""
        if user_id not in self.user_id_to_idx or movie_id not in self.movie_id_to_idx:
            return self.global_mean

        user_idx = self.user_id_to_idx[user_id]
        movie_idx = self.movie_id_to_idx[movie_id]

        score = float(np.dot(self.user_factors[user_idx], self.item_factors[movie_idx]))
        return float(np.clip(score, 0.5, 5.0))

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def explain(self, user_id: int, movie_id: int) -> dict:
        """Explain the recommendation via latent factor decomposition."""
        result: dict[str, Any] = {
            "user_id": user_id,
            "movie_id": movie_id,
            "score": self.predict_rating(user_id, movie_id),
            "method": "als_matrix_factorisation",
            "rank": self.rank,
        }

        if user_id not in self.user_id_to_idx or movie_id not in self.movie_id_to_idx:
            result["note"] = "unknown user or movie -- returned global mean"
            return result

        user_idx = self.user_id_to_idx[user_id]
        movie_idx = self.movie_id_to_idx[movie_id]

        user_vec = self.user_factors[user_idx]
        item_vec = self.item_factors[movie_idx]

        # Per-dimension contributions
        contributions = user_vec * item_vec
        top_dims = np.argsort(np.abs(contributions))[::-1][:10]

        result["latent_factor_contributions"] = [
            {
                "dimension": int(d),
                "user_value": round(float(user_vec[d]), 4),
                "item_value": round(float(item_vec[d]), 4),
                "contribution": round(float(contributions[d]), 4),
            }
            for d in top_dims
        ]

        result["dot_product"] = round(float(np.dot(user_vec, item_vec)), 4)

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Save factor matrices and metadata to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "user_factors.npy", self.user_factors)
        np.save(path / "item_factors.npy", self.item_factors)

        if self.user_item_matrix is not None:
            sp.save_npz(path / "user_item_matrix.npz", self.user_item_matrix)

        meta = {
            "movie_id_to_idx": {str(k): v for k, v in self.movie_id_to_idx.items()},
            "user_id_to_idx": {str(k): v for k, v in self.user_id_to_idx.items()},
            "rank": self.rank,
            "global_mean": self.global_mean,
            "n_items": self.n_items,
            "n_users": self.n_users,
            "popular_items": self._popular_items[:200],
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        logger.info("SparkALSRecommender saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "SparkALSRecommender":
        """Load a previously saved SparkALSRecommender."""
        path = Path(path)

        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = cls()
        model.user_factors = np.load(path / "user_factors.npy")
        model.item_factors = np.load(path / "item_factors.npy")
        model.item_factors_normed = normalize(model.item_factors, norm="l2", axis=1)

        ui_path = path / "user_item_matrix.npz"
        if ui_path.exists():
            model.user_item_matrix = sp.load_npz(ui_path)

        model.movie_id_to_idx = {int(k): int(v) for k, v in meta["movie_id_to_idx"].items()}
        model.user_id_to_idx = {int(k): int(v) for k, v in meta["user_id_to_idx"].items()}
        model.idx_to_movie_id = {v: k for k, v in model.movie_id_to_idx.items()}
        model.idx_to_user_id = {v: k for k, v in model.user_id_to_idx.items()}
        model.rank = meta.get("rank", model.user_factors.shape[1])
        model.global_mean = meta.get("global_mean", _GLOBAL_MEAN_FALLBACK)
        model.n_items = meta.get("n_items", model.item_factors.shape[0])
        model.n_users = meta.get("n_users", model.user_factors.shape[0])
        model._popular_items = meta.get("popular_items", [])

        logger.info("SparkALSRecommender loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_factor_matrices(
        self,
        user_factors_df: pd.DataFrame,
        item_factors_df: pd.DataFrame,
    ) -> None:
        """Align Spark ALS output with the project's ID-mapping convention.

        Spark ALS assigns its own integer IDs, which match the original
        ``userId`` / ``movieId`` columns.  This method builds dense NumPy
        arrays whose rows correspond to the project's index maps.
        """
        # Build id-to-idx maps from training data if not already set
        spark_user_ids = user_factors_df["id"].values
        spark_item_ids = item_factors_df["id"].values

        # If no external mapping was provided, create one from Spark output
        if not self.user_id_to_idx:
            all_user_ids = sorted(spark_user_ids)
            self.user_id_to_idx = {int(uid): idx for idx, uid in enumerate(all_user_ids)}
        if not self.movie_id_to_idx:
            all_item_ids = sorted(spark_item_ids)
            self.movie_id_to_idx = {int(mid): idx for idx, mid in enumerate(all_item_ids)}

        self.idx_to_movie_id = {v: k for k, v in self.movie_id_to_idx.items()}
        self.idx_to_user_id = {v: k for k, v in self.user_id_to_idx.items()}
        self.n_users = len(self.user_id_to_idx)
        self.n_items = len(self.movie_id_to_idx)

        # Allocate factor matrices
        self.user_factors = np.zeros((self.n_users, self.rank), dtype=np.float32)
        self.item_factors = np.zeros((self.n_items, self.rank), dtype=np.float32)

        # Fill from Spark output
        for _, row in user_factors_df.iterrows():
            uid = int(row["id"])
            if uid in self.user_id_to_idx:
                idx = self.user_id_to_idx[uid]
                self.user_factors[idx] = np.array(row["features"], dtype=np.float32)

        for _, row in item_factors_df.iterrows():
            mid = int(row["id"])
            if mid in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[mid]
                self.item_factors[idx] = np.array(row["features"], dtype=np.float32)

        # Log coverage
        user_coverage = (np.linalg.norm(self.user_factors, axis=1) > 0).sum()
        item_coverage = (np.linalg.norm(self.item_factors, axis=1) > 0).sum()
        logger.info(
            "Factor coverage: %d/%d users, %d/%d items",
            user_coverage, self.n_users, item_coverage, self.n_items,
        )

    def _compute_popular_items(
        self,
        ratings_df: pd.DataFrame,
        top_n: int = 200,
    ) -> list[int]:
        """Compute popular items from training ratings for cold-start fallback."""
        if ratings_df is None or len(ratings_df) == 0:
            return []

        counts = ratings_df["movieId"].value_counts()
        return counts.head(top_n).index.tolist()

    def _popular_fallback(self, n: int) -> list[tuple[int, float]]:
        """Return top-n popular items as fallback for unknown users."""
        return [(int(mid), 0.0) for mid in self._popular_items[:n]]
