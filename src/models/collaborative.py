"""Item-item collaborative filtering recommender.

Uses cosine similarity between item rating vectors to recommend movies.
The similarity matrix is kept sparse (only top-K neighbours per item) so
that memory stays manageable for datasets with tens of thousands of items.

Typical usage::

    from src.models.collaborative import ItemItemCF
    model = ItemItemCF()
    model.fit(user_item_matrix, movie_id_map, user_id_map)
    recs = model.recommend(user_id=42, n=10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseRecommender
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default number of neighbours retained per item
_DEFAULT_TOP_K = 50
# Batch size for computing similarity rows
_DEFAULT_BATCH_SIZE = 1000
# Global mean rating used as fallback prediction
_GLOBAL_MEAN_FALLBACK = 3.5


class ItemItemCF(BaseRecommender):
    """Item-item collaborative filtering with sparse similarity storage.

    Parameters
    ----------
    top_k : int
        Number of most-similar neighbours to retain per item.
    batch_size : int
        Number of items to process at once when computing similarities.
        Controls peak memory consumption.
    """

    def __init__(self, top_k: int = _DEFAULT_TOP_K, batch_size: int = _DEFAULT_BATCH_SIZE):
        self.top_k = top_k
        self.batch_size = batch_size

        # Populated by fit()
        self.similarity: sp.csr_matrix | None = None
        self.user_item_matrix: sp.csr_matrix | None = None
        self.movie_id_to_idx: dict[int, int] = {}
        self.user_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}
        self.idx_to_user_id: dict[int, int] = {}
        self.n_items: int = 0
        self.n_users: int = 0
        self.global_mean: float = _GLOBAL_MEAN_FALLBACK
        self._popular_items: list[int] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(  # type: ignore[override]
        self,
        train_data: sp.csr_matrix,
        movie_id_map: dict[int, int] | None = None,
        user_id_map: dict[int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Compute the top-K item-item cosine similarity matrix.

        Parameters
        ----------
        train_data : scipy.sparse.csr_matrix
            User-item rating matrix of shape ``(n_users, n_items)``.
        movie_id_map : dict[int, int]
            Maps original ``movieId`` to column index in *train_data*.
        user_id_map : dict[int, int]
            Maps original ``userId`` to row index in *train_data*.
        """
        self.user_item_matrix = train_data.tocsr()
        self.movie_id_to_idx = {int(k): int(v) for k, v in (movie_id_map or {}).items()}
        self.user_id_to_idx = {int(k): int(v) for k, v in (user_id_map or {}).items()}
        self.idx_to_movie_id = {v: k for k, v in self.movie_id_to_idx.items()}
        self.idx_to_user_id = {v: k for k, v in self.user_id_to_idx.items()}
        self.n_users, self.n_items = self.user_item_matrix.shape

        # Global mean rating (for cold-start fallback)
        nnz = self.user_item_matrix.nnz
        if nnz > 0:
            self.global_mean = float(self.user_item_matrix.data.sum() / nnz)

        # Precompute popularity ranking for cold-start users
        self._popular_items = self._compute_popular_items()

        logger.info(
            "Computing item-item similarity (top_k=%d, batch=%d) for %d items ...",
            self.top_k, self.batch_size, self.n_items,
        )

        # Item vectors: transpose so each row is an item
        item_vectors = self.user_item_matrix.T.tocsr()  # (n_items, n_users)

        # Build sparse similarity in COO then convert to CSR
        sim_rows: list[np.ndarray] = []
        sim_cols: list[np.ndarray] = []
        sim_vals: list[np.ndarray] = []

        n_batches = (self.n_items + self.batch_size - 1) // self.batch_size
        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, self.n_items)

            # Cosine similarity of this batch against ALL items
            batch_sim = cosine_similarity(item_vectors[start:end], item_vectors)

            # Zero out self-similarity
            for local_i in range(end - start):
                batch_sim[local_i, start + local_i] = 0.0

            # Keep only top-K per row using argpartition (O(n) per row)
            k = min(self.top_k, self.n_items - 1)
            if k <= 0:
                continue

            for local_i in range(end - start):
                row = batch_sim[local_i]
                if k < len(row):
                    top_k_idx = np.argpartition(row, -k)[-k:]
                else:
                    top_k_idx = np.arange(len(row))
                top_k_vals = row[top_k_idx]

                # Filter out zero/negative similarities
                mask = top_k_vals > 0
                top_k_idx = top_k_idx[mask]
                top_k_vals = top_k_vals[mask]

                global_i = start + local_i
                sim_rows.append(np.full(len(top_k_idx), global_i, dtype=np.int32))
                sim_cols.append(top_k_idx.astype(np.int32))
                sim_vals.append(top_k_vals.astype(np.float32))

            if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
                logger.info(
                    "  Similarity batch %d/%d (items %d-%d)",
                    batch_idx + 1, n_batches, start, end - 1,
                )

        # Assemble sparse matrix
        if sim_rows:
            all_rows = np.concatenate(sim_rows)
            all_cols = np.concatenate(sim_cols)
            all_vals = np.concatenate(sim_vals)
        else:
            all_rows = np.array([], dtype=np.int32)
            all_cols = np.array([], dtype=np.int32)
            all_vals = np.array([], dtype=np.float32)

        self.similarity = sp.csr_matrix(
            (all_vals, (all_rows, all_cols)),
            shape=(self.n_items, self.n_items),
            dtype=np.float32,
        )

        logger.info(
            "Item-item similarity matrix: %d x %d, nnz=%d",
            self.similarity.shape[0], self.similarity.shape[1], self.similarity.nnz,
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
        """Generate recommendations via similarity-weighted neighbour aggregation."""
        if user_id not in self.user_id_to_idx:
            return self._popular_fallback(n)

        user_idx = self.user_id_to_idx[user_id]
        user_ratings = self.user_item_matrix[user_idx]  # 1 x n_items sparse

        # Items the user has rated
        rated_indices = user_ratings.indices
        rated_values = user_ratings.data

        if len(rated_indices) == 0:
            return self._popular_fallback(n)

        # For each candidate item, accumulate weighted scores from rated items
        # scores[j] = sum_{i in rated} rating[i] * sim[i, j]
        # weights[j] = sum_{i in rated} |sim[i, j]|
        scores = np.zeros(self.n_items, dtype=np.float64)
        weights = np.zeros(self.n_items, dtype=np.float64)

        for rated_idx, rating_val in zip(rated_indices, rated_values):
            sim_row = self.similarity[rated_idx]  # sparse 1 x n_items
            neighbour_indices = sim_row.indices
            neighbour_sims = sim_row.data

            scores[neighbour_indices] += rating_val * neighbour_sims
            weights[neighbour_indices] += np.abs(neighbour_sims)

        # Normalise
        nonzero_mask = weights > 0
        scores[nonzero_mask] /= weights[nonzero_mask]

        # Exclude already-seen items
        if exclude_seen:
            scores[rated_indices] = -np.inf

        # Top-n
        if n < len(scores):
            top_indices = np.argpartition(scores, -n)[-n:]
        else:
            top_indices = np.arange(len(scores))

        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        top_indices = top_indices[:n]

        results: list[tuple[int, float]] = []
        for idx in top_indices:
            if scores[idx] <= 0 or scores[idx] == -np.inf:
                continue
            movie_id = self.idx_to_movie_id.get(int(idx))
            if movie_id is not None:
                results.append((movie_id, float(scores[idx])))

        return results

    def similar_items(self, movie_id: int, n: int = 20) -> list[tuple[int, float]]:
        """Direct lookup in the precomputed sparse similarity matrix."""
        if movie_id not in self.movie_id_to_idx:
            return []

        item_idx = self.movie_id_to_idx[movie_id]
        sim_row = self.similarity[item_idx]

        indices = sim_row.indices
        values = sim_row.data

        if len(indices) == 0:
            return []

        # Sort by descending similarity
        if n < len(indices):
            top_k = np.argpartition(values, -n)[-n:]
        else:
            top_k = np.arange(len(indices))

        top_k = top_k[np.argsort(values[top_k])[::-1]]
        top_k = top_k[:n]

        results: list[tuple[int, float]] = []
        for k in top_k:
            mid = self.idx_to_movie_id.get(int(indices[k]))
            if mid is not None:
                results.append((mid, float(values[k])))

        return results

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict a single rating for evaluation (RMSE)."""
        if user_id not in self.user_id_to_idx or movie_id not in self.movie_id_to_idx:
            return self.global_mean

        user_idx = self.user_id_to_idx[user_id]
        movie_idx = self.movie_id_to_idx[movie_id]

        user_ratings = self.user_item_matrix[user_idx]
        rated_indices = user_ratings.indices
        rated_values = user_ratings.data

        if len(rated_indices) == 0:
            return self.global_mean

        sim_row = self.similarity[movie_idx]

        # Find intersection: items the user has rated that are also neighbours
        weighted_sum = 0.0
        weight_total = 0.0

        for rated_idx, rating_val in zip(rated_indices, rated_values):
            sim_val = sim_row[0, rated_idx] if sp.issparse(sim_row) else sim_row[rated_idx]
            if isinstance(sim_val, np.matrix):
                sim_val = float(sim_val.item())
            else:
                sim_val = float(sim_val)
            if sim_val > 0:
                weighted_sum += rating_val * sim_val
                weight_total += abs(sim_val)

        if weight_total > 0:
            prediction = weighted_sum / weight_total
            # Clamp to valid rating range
            return float(np.clip(prediction, 0.5, 5.0))

        return self.global_mean

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def explain(self, user_id: int, movie_id: int) -> dict:
        """Explain which rated movies contributed most to this recommendation."""
        result: dict[str, Any] = {
            "user_id": user_id,
            "movie_id": movie_id,
            "score": self.predict_rating(user_id, movie_id),
            "method": "item_item_collaborative_filtering",
            "contributing_items": [],
        }

        if user_id not in self.user_id_to_idx or movie_id not in self.movie_id_to_idx:
            result["note"] = "unknown user or movie -- returned global mean"
            return result

        user_idx = self.user_id_to_idx[user_id]
        movie_idx = self.movie_id_to_idx[movie_id]

        user_ratings = self.user_item_matrix[user_idx]
        rated_indices = user_ratings.indices
        rated_values = user_ratings.data

        sim_row = self.similarity[movie_idx]
        contributions: list[dict] = []

        for rated_idx, rating_val in zip(rated_indices, rated_values):
            sim_val = sim_row[0, rated_idx] if sp.issparse(sim_row) else sim_row[rated_idx]
            if isinstance(sim_val, np.matrix):
                sim_val = float(sim_val.item())
            else:
                sim_val = float(sim_val)
            if sim_val > 0:
                mid = self.idx_to_movie_id.get(int(rated_idx), rated_idx)
                contributions.append({
                    "movie_id": mid,
                    "user_rating": float(rating_val),
                    "similarity": round(sim_val, 4),
                    "weighted_contribution": round(float(rating_val * sim_val), 4),
                })

        # Sort by weighted contribution descending
        contributions.sort(key=lambda c: c["weighted_contribution"], reverse=True)
        result["contributing_items"] = contributions[:10]

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Save model artefacts to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Sparse similarity matrix
        sp.save_npz(path / "similarity.npz", self.similarity)

        # User-item matrix (needed for inference)
        sp.save_npz(path / "user_item_matrix.npz", self.user_item_matrix)

        # ID mappings and metadata
        meta = {
            "movie_id_to_idx": {str(k): v for k, v in self.movie_id_to_idx.items()},
            "user_id_to_idx": {str(k): v for k, v in self.user_id_to_idx.items()},
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "global_mean": self.global_mean,
            "n_items": self.n_items,
            "n_users": self.n_users,
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # Popular items list
        joblib.dump(self._popular_items, path / "popular_items.joblib")

        logger.info("ItemItemCF saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ItemItemCF":
        """Load a previously saved ItemItemCF model."""
        path = Path(path)

        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = cls(
            top_k=meta.get("top_k", _DEFAULT_TOP_K),
            batch_size=meta.get("batch_size", _DEFAULT_BATCH_SIZE),
        )
        model.similarity = sp.load_npz(path / "similarity.npz")
        model.user_item_matrix = sp.load_npz(path / "user_item_matrix.npz")

        model.movie_id_to_idx = {int(k): int(v) for k, v in meta["movie_id_to_idx"].items()}
        model.user_id_to_idx = {int(k): int(v) for k, v in meta["user_id_to_idx"].items()}
        model.idx_to_movie_id = {v: k for k, v in model.movie_id_to_idx.items()}
        model.idx_to_user_id = {v: k for k, v in model.user_id_to_idx.items()}
        model.global_mean = meta.get("global_mean", _GLOBAL_MEAN_FALLBACK)
        model.n_items = meta.get("n_items", model.similarity.shape[0])
        model.n_users = meta.get("n_users", model.user_item_matrix.shape[0])
        model._popular_items = joblib.load(path / "popular_items.joblib")

        logger.info("ItemItemCF loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_popular_items(self, top_n: int = 200) -> list[int]:
        """Compute a ranked list of popular items by number of ratings.

        Used as a fallback for cold-start or unknown users.
        """
        if self.user_item_matrix is None:
            return []

        # Number of non-zero entries per column (item)
        counts = np.diff(self.user_item_matrix.tocsc().indptr)
        if top_n < len(counts):
            top_indices = np.argpartition(counts, -top_n)[-top_n:]
        else:
            top_indices = np.arange(len(counts))

        top_indices = top_indices[np.argsort(counts[top_indices])[::-1]]

        return [
            self.idx_to_movie_id[int(idx)]
            for idx in top_indices
            if int(idx) in self.idx_to_movie_id
        ]

    def _popular_fallback(self, n: int) -> list[tuple[int, float]]:
        """Return the top-n popular items with a synthetic score."""
        return [(mid, 0.0) for mid in self._popular_items[:n]]
