"""Content-based recommender using TF-IDF genre and genome tag features.

Builds user profiles as rating-weighted averages of content feature vectors,
then ranks candidates by cosine similarity to the user profile.

Typical usage::

    from src.models.content_based import ContentBasedRecommender
    model = ContentBasedRecommender()
    model.fit(content_features, user_item_matrix, movie_id_map, user_id_map)
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
from sklearn.preprocessing import normalize

from src.models.base import BaseRecommender
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TOP_K = 50
_DEFAULT_BATCH_SIZE = 1000
_GLOBAL_MEAN_FALLBACK = 3.5


class ContentBasedRecommender(BaseRecommender):
    """Content-based filtering using combined TF-IDF + genome features.

    Parameters
    ----------
    top_k : int
        Number of nearest content neighbours to retain per item.
    batch_size : int
        Batch size for computing the content similarity matrix.
    """

    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        self.top_k = top_k
        self.batch_size = batch_size

        # Populated by fit()
        self.content_features: sp.csr_matrix | None = None
        self.content_similarity: sp.csr_matrix | None = None
        self.user_profiles: np.ndarray | None = None
        self.user_item_matrix: sp.csr_matrix | None = None
        self.movie_id_to_idx: dict[int, int] = {}
        self.user_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}
        self.idx_to_user_id: dict[int, int] = {}
        self.genome_tags: dict[int, str] | None = None
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
        user_item_matrix: sp.csr_matrix | None = None,
        movie_id_map: dict[int, int] | None = None,
        user_id_map: dict[int, int] | None = None,
        genome_tags: dict[int, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit the content-based model.

        Parameters
        ----------
        train_data : scipy.sparse.csr_matrix
            Content feature matrix of shape ``(n_items, n_features)``.
            Rows correspond to items in *movie_id_map* order.
        user_item_matrix : scipy.sparse.csr_matrix
            User-item rating matrix of shape ``(n_users, n_items)``.
        movie_id_map : dict[int, int]
            Maps original ``movieId`` to item index.
        user_id_map : dict[int, int]
            Maps original ``userId`` to user index.
        genome_tags : dict[int, str], optional
            Maps genome ``tagId`` to tag name, for explainability.
        """
        self.content_features = sp.csr_matrix(train_data)
        self.user_item_matrix = sp.csr_matrix(user_item_matrix) if user_item_matrix is not None else None
        self.movie_id_to_idx = {int(k): int(v) for k, v in (movie_id_map or {}).items()}
        self.user_id_to_idx = {int(k): int(v) for k, v in (user_id_map or {}).items()}
        self.idx_to_movie_id = {v: k for k, v in self.movie_id_to_idx.items()}
        self.idx_to_user_id = {v: k for k, v in self.user_id_to_idx.items()}
        self.genome_tags = genome_tags
        self.n_items = self.content_features.shape[0]

        if self.user_item_matrix is not None:
            self.n_users = self.user_item_matrix.shape[0]
            nnz = self.user_item_matrix.nnz
            if nnz > 0:
                self.global_mean = float(self.user_item_matrix.data.sum() / nnz)

        # Popularity fallback
        self._popular_items = self._compute_popular_items()

        # ---- Build content similarity (sparse, top-K) ----
        logger.info(
            "Computing content similarity (top_k=%d, batch=%d) for %d items ...",
            self.top_k, self.batch_size, self.n_items,
        )
        self.content_similarity = self._build_sparse_similarity(
            self.content_features, self.top_k, self.batch_size,
        )
        logger.info(
            "Content similarity matrix: %d x %d, nnz=%d",
            self.content_similarity.shape[0],
            self.content_similarity.shape[1],
            self.content_similarity.nnz,
        )

        # ---- Build user profiles ----
        if self.user_item_matrix is not None:
            logger.info("Building user profiles (%d users) ...", self.n_users)
            self.user_profiles = self._build_user_profiles()
            logger.info("User profiles shape: %s", self.user_profiles.shape)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        n: int = 20,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Recommend items by cosine similarity between user profile and items."""
        if user_id not in self.user_id_to_idx or self.user_profiles is None:
            return self._popular_fallback(n)

        user_idx = self.user_id_to_idx[user_id]
        user_profile = self.user_profiles[user_idx]

        # If user profile is all zeros (no ratings), fall back
        if np.allclose(user_profile, 0):
            return self._popular_fallback(n)

        # Cosine similarity between user profile and all item features
        # user_profile: (1, n_features), content_features: (n_items, n_features)
        profile_sparse = sp.csr_matrix(user_profile.reshape(1, -1))
        scores = cosine_similarity(profile_sparse, self.content_features).flatten()

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
            if scores[idx] <= 0 or np.isinf(scores[idx]):
                continue
            movie_id = self.idx_to_movie_id.get(int(idx))
            if movie_id is not None:
                results.append((movie_id, float(scores[idx])))

        return results

    def similar_items(self, movie_id: int, n: int = 20) -> list[tuple[int, float]]:
        """Content-based similarity lookup from the precomputed sparse matrix."""
        if movie_id not in self.movie_id_to_idx:
            return []

        item_idx = self.movie_id_to_idx[movie_id]
        sim_row = self.content_similarity[item_idx]

        indices = sim_row.indices
        values = sim_row.data

        if len(indices) == 0:
            return []

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
        """Predict a single rating using content similarity to user profile.

        The prediction is scaled to the rating range [0.5, 5.0] using the
        cosine similarity score and the user's mean rating.
        """
        if user_id not in self.user_id_to_idx or movie_id not in self.movie_id_to_idx:
            return self.global_mean

        if self.user_profiles is None:
            return self.global_mean

        user_idx = self.user_id_to_idx[user_id]
        movie_idx = self.movie_id_to_idx[movie_id]

        user_profile = self.user_profiles[user_idx]
        if np.allclose(user_profile, 0):
            return self.global_mean

        # Cosine similarity between user profile and item feature vector
        item_vec = self.content_features[movie_idx].toarray().flatten()
        profile_norm = np.linalg.norm(user_profile)
        item_norm = np.linalg.norm(item_vec)

        if profile_norm == 0 or item_norm == 0:
            return self.global_mean

        cos_sim = float(np.dot(user_profile, item_vec) / (profile_norm * item_norm))

        # Scale: map cosine sim [-1, 1] to rating [0.5, 5.0]
        # Use user's mean rating as the centre point
        user_ratings = self.user_item_matrix[user_idx]
        if user_ratings.nnz > 0:
            user_mean = float(user_ratings.data.mean())
        else:
            user_mean = self.global_mean

        # cos_sim=0 -> user_mean, cos_sim=1 -> 5.0, cos_sim=-1 -> 0.5
        prediction = user_mean + cos_sim * (5.0 - user_mean) if cos_sim >= 0 else user_mean + cos_sim * (user_mean - 0.5)
        return float(np.clip(prediction, 0.5, 5.0))

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def explain(self, user_id: int, movie_id: int) -> dict:
        """Explain which content features drive this recommendation."""
        result: dict[str, Any] = {
            "user_id": user_id,
            "movie_id": movie_id,
            "score": self.predict_rating(user_id, movie_id),
            "method": "content_based_filtering",
            "top_features": [],
            "similar_items_by_content": [],
        }

        if movie_id not in self.movie_id_to_idx:
            result["note"] = "unknown movie -- returned global mean"
            return result

        # Top similar items by content
        result["similar_items_by_content"] = [
            {"movie_id": mid, "similarity": round(sim, 4)}
            for mid, sim in self.similar_items(movie_id, n=5)
        ]

        if user_id not in self.user_id_to_idx or self.user_profiles is None:
            result["note"] = "unknown user -- no profile available"
            return result

        user_idx = self.user_id_to_idx[user_id]
        movie_idx = self.movie_id_to_idx[movie_id]

        user_profile = self.user_profiles[user_idx]
        item_vec = self.content_features[movie_idx].toarray().flatten()

        # Element-wise contribution: which feature dimensions align most
        contribution = user_profile * item_vec
        top_feat_indices = np.argsort(np.abs(contribution))[::-1][:20]

        top_features: list[dict] = []
        for fi in top_feat_indices:
            if abs(contribution[fi]) < 1e-6:
                break
            feat_info: dict[str, Any] = {
                "feature_index": int(fi),
                "contribution": round(float(contribution[fi]), 6),
                "user_weight": round(float(user_profile[fi]), 6),
                "item_weight": round(float(item_vec[fi]), 6),
            }
            top_features.append(feat_info)

        result["top_features"] = top_features
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Save model artefacts to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        sp.save_npz(path / "content_features.npz", self.content_features)
        sp.save_npz(path / "content_similarity.npz", self.content_similarity)

        if self.user_item_matrix is not None:
            sp.save_npz(path / "user_item_matrix.npz", self.user_item_matrix)

        if self.user_profiles is not None:
            np.save(path / "user_profiles.npy", self.user_profiles)

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

        if self.genome_tags is not None:
            with open(path / "genome_tags.json", "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in self.genome_tags.items()}, f)

        joblib.dump(self._popular_items, path / "popular_items.joblib")
        logger.info("ContentBasedRecommender saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ContentBasedRecommender":
        """Load a previously saved ContentBasedRecommender."""
        path = Path(path)

        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = cls(
            top_k=meta.get("top_k", _DEFAULT_TOP_K),
            batch_size=meta.get("batch_size", _DEFAULT_BATCH_SIZE),
        )
        model.content_features = sp.load_npz(path / "content_features.npz")
        model.content_similarity = sp.load_npz(path / "content_similarity.npz")

        ui_path = path / "user_item_matrix.npz"
        if ui_path.exists():
            model.user_item_matrix = sp.load_npz(ui_path)

        profiles_path = path / "user_profiles.npy"
        if profiles_path.exists():
            model.user_profiles = np.load(profiles_path)

        model.movie_id_to_idx = {int(k): int(v) for k, v in meta["movie_id_to_idx"].items()}
        model.user_id_to_idx = {int(k): int(v) for k, v in meta["user_id_to_idx"].items()}
        model.idx_to_movie_id = {v: k for k, v in model.movie_id_to_idx.items()}
        model.idx_to_user_id = {v: k for k, v in model.user_id_to_idx.items()}
        model.global_mean = meta.get("global_mean", _GLOBAL_MEAN_FALLBACK)
        model.n_items = meta.get("n_items", model.content_features.shape[0])
        model.n_users = meta.get("n_users", 0)

        genome_path = path / "genome_tags.json"
        if genome_path.exists():
            with open(genome_path, "r", encoding="utf-8") as f:
                model.genome_tags = {int(k): v for k, v in json.load(f).items()}

        model._popular_items = joblib.load(path / "popular_items.joblib")
        logger.info("ContentBasedRecommender loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_sparse_similarity(
        self,
        feature_matrix: sp.csr_matrix,
        top_k: int,
        batch_size: int,
    ) -> sp.csr_matrix:
        """Compute a top-K sparse cosine similarity matrix in batches."""
        n = feature_matrix.shape[0]

        sim_rows: list[np.ndarray] = []
        sim_cols: list[np.ndarray] = []
        sim_vals: list[np.ndarray] = []

        n_batches = (n + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)

            batch_sim = cosine_similarity(feature_matrix[start:end], feature_matrix)

            # Zero out self-similarity
            for local_i in range(end - start):
                batch_sim[local_i, start + local_i] = 0.0

            k = min(top_k, n - 1)
            if k <= 0:
                continue

            for local_i in range(end - start):
                row = batch_sim[local_i]
                if k < len(row):
                    top_k_idx = np.argpartition(row, -k)[-k:]
                else:
                    top_k_idx = np.arange(len(row))
                top_k_vals = row[top_k_idx]

                mask = top_k_vals > 0
                top_k_idx = top_k_idx[mask]
                top_k_vals = top_k_vals[mask]

                global_i = start + local_i
                sim_rows.append(np.full(len(top_k_idx), global_i, dtype=np.int32))
                sim_cols.append(top_k_idx.astype(np.int32))
                sim_vals.append(top_k_vals.astype(np.float32))

            if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
                logger.info(
                    "  Content similarity batch %d/%d (items %d-%d)",
                    batch_idx + 1, n_batches, start, end - 1,
                )

        if sim_rows:
            all_rows = np.concatenate(sim_rows)
            all_cols = np.concatenate(sim_cols)
            all_vals = np.concatenate(sim_vals)
        else:
            all_rows = np.array([], dtype=np.int32)
            all_cols = np.array([], dtype=np.int32)
            all_vals = np.array([], dtype=np.float32)

        return sp.csr_matrix(
            (all_vals, (all_rows, all_cols)),
            shape=(n, n),
            dtype=np.float32,
        )

    def _build_user_profiles(self) -> np.ndarray:
        """Build user profiles as rating-weighted averages of content vectors.

        For each user, the profile is the mean of their rated item vectors,
        weighted by rating value.  The result is L2-normalised per user.
        """
        n_features = self.content_features.shape[1]
        profiles = np.zeros((self.n_users, n_features), dtype=np.float32)

        # Convert content features to dense once (fits in memory for ~27K items)
        # If content matrix is very large, process in user batches
        content_dense = self.content_features.toarray()  # (n_items, n_features)

        for user_idx in range(self.n_users):
            user_ratings = self.user_item_matrix[user_idx]
            rated_indices = user_ratings.indices
            rated_values = user_ratings.data

            if len(rated_indices) == 0:
                continue

            # Weighted average
            rated_features = content_dense[rated_indices]  # (n_rated, n_features)
            weights = rated_values.reshape(-1, 1)  # (n_rated, 1)
            weighted_sum = (rated_features * weights).sum(axis=0)
            weight_total = weights.sum()

            if weight_total > 0:
                profiles[user_idx] = weighted_sum / weight_total

        # L2 normalise non-zero profiles
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        profiles /= norms

        return profiles

    def _compute_popular_items(self, top_n: int = 200) -> list[int]:
        """Compute popular items by rating count for cold-start fallback."""
        if self.user_item_matrix is None:
            return []

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
        """Return top-n popular items as fallback for unknown users."""
        return [(mid, 0.0) for mid in self._popular_items[:n]]
