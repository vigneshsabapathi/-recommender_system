"""Cosine similarity vs Euclidean distance verification.

For L2-normalised feature vectors the following identity holds exactly::

    ||a - b||^2  =  2 * (1 - cos(a, b))

This module verifies that the content feature representations in the
recommender system respect this relationship by sampling random item
pairs, computing both metrics, and reporting the Pearson correlation
between observed Euclidean distances and the theoretical values derived
from cosine similarities.

A Pearson correlation close to **-1** between raw cosine similarity and
raw Euclidean distance (or close to **+1** between ``2*(1-cos)`` and
``d^2``) confirms that the feature space behaves as expected.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize

from src.utils.logger import get_logger

logger = get_logger(__name__)


def cosine_euclidean_correlation(
    feature_matrix,
    sample_size: int = 1000,
    seed: int = 42,
) -> dict:
    """Sample random pairs and compare cosine similarity with Euclidean distance.

    Parameters
    ----------
    feature_matrix
        Item feature matrix of shape ``(n_items, n_features)``.  Dense or
        sparse.  Rows that are all-zero are excluded from sampling.
    sample_size : int
        Number of random pairs to draw.  Capped at
        ``n_items * (n_items - 1) / 2`` for small matrices.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``pearson_correlation``
            Pearson r between cosine similarity and Euclidean distance
            (expected to be close to -1.0 for L2-normalised vectors).
        ``pairs_sampled``
            Number of pairs actually evaluated.
        ``cosine_values``
            List of cosine similarity values.
        ``euclidean_values``
            List of Euclidean distance values.
        ``theoretical_match``
            ``True`` when the observed ``d^2`` values match the
            theoretical ``2*(1 - cos_sim)`` within a loose tolerance.
        ``mean_abs_error``
            Average absolute difference between observed ``d^2`` and
            theoretical ``2*(1 - cos_sim)`` on the L2-normalised vectors.
    """
    rng = np.random.RandomState(seed)

    # Convert to dense numpy if sparse
    if sp.issparse(feature_matrix):
        feature_matrix = feature_matrix.toarray()
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)

    n_items, n_features = feature_matrix.shape
    logger.info(
        "Verification: feature matrix shape (%d, %d)", n_items, n_features,
    )

    # Exclude all-zero rows (items with no features)
    row_norms = np.linalg.norm(feature_matrix, axis=1)
    valid_indices = np.where(row_norms > 0)[0]
    n_valid = len(valid_indices)

    if n_valid < 2:
        logger.warning("Fewer than 2 non-zero rows -- cannot compute pairs.")
        return {
            "pearson_correlation": 0.0,
            "pairs_sampled": 0,
            "cosine_values": [],
            "euclidean_values": [],
            "theoretical_match": False,
            "mean_abs_error": float("inf"),
        }

    # Cap sample size at the total number of unique pairs
    max_pairs = n_valid * (n_valid - 1) // 2
    sample_size = min(sample_size, max_pairs)

    # Sample random pairs (without replacement by rejection)
    pairs_set: set[tuple[int, int]] = set()
    while len(pairs_set) < sample_size:
        batch = rng.randint(0, n_valid, size=(sample_size * 2, 2))
        for a, b in batch:
            if a == b:
                continue
            pair = (min(a, b), max(a, b))
            pairs_set.add(pair)
            if len(pairs_set) >= sample_size:
                break

    pairs = list(pairs_set)[:sample_size]
    idx_a = valid_indices[[p[0] for p in pairs]]
    idx_b = valid_indices[[p[1] for p in pairs]]

    logger.info("Sampled %d pairs for cosine-Euclidean verification.", len(pairs))

    # ------------------------------------------------------------------
    # Raw feature space (not necessarily normalised)
    # ------------------------------------------------------------------
    vecs_a = feature_matrix[idx_a]  # (sample_size, n_features)
    vecs_b = feature_matrix[idx_b]

    # Cosine similarity per pair (row-wise)
    cos_values = np.array([
        float(cosine_similarity(vecs_a[i:i + 1], vecs_b[i:i + 1])[0, 0])
        for i in range(len(pairs))
    ], dtype=np.float64)

    # Euclidean distance per pair
    euc_values = np.array([
        float(euclidean_distances(vecs_a[i:i + 1], vecs_b[i:i + 1])[0, 0])
        for i in range(len(pairs))
    ], dtype=np.float64)

    # Pearson correlation between cosine similarity and Euclidean distance
    # For L2-normalised vectors this should be strongly negative.
    pearson_r = _safe_pearson(cos_values, euc_values)

    # ------------------------------------------------------------------
    # Theoretical check on L2-normalised versions
    # ------------------------------------------------------------------
    normed_matrix = normalize(feature_matrix[valid_indices], norm="l2", axis=1)

    normed_a = normed_matrix[[p[0] for p in pairs]]
    normed_b = normed_matrix[[p[1] for p in pairs]]

    cos_normed = np.array([
        float(cosine_similarity(normed_a[i:i + 1], normed_b[i:i + 1])[0, 0])
        for i in range(len(pairs))
    ], dtype=np.float64)

    # Squared Euclidean on normalised vectors
    diff = normed_a - normed_b
    euc_sq_normed = np.sum(diff ** 2, axis=1)

    # Theoretical: d^2 = 2 * (1 - cos_sim) for unit vectors
    theoretical_euc_sq = 2.0 * (1.0 - cos_normed)

    abs_errors = np.abs(euc_sq_normed - theoretical_euc_sq)
    mean_abs_error = float(np.mean(abs_errors))

    # Tolerance: allow for floating-point accumulation
    theoretical_match = bool(mean_abs_error < 1e-6)

    logger.info(
        "Pearson(cos, euc) = %.4f  |  Theoretical match = %s  "
        "(mean |d^2 - 2(1-cos)| = %.2e)",
        pearson_r, theoretical_match, mean_abs_error,
    )

    return {
        "pearson_correlation": round(float(pearson_r), 6),
        "pairs_sampled": len(pairs),
        "cosine_values": [round(float(v), 6) for v in cos_values],
        "euclidean_values": [round(float(v), 6) for v in euc_values],
        "theoretical_match": theoretical_match,
        "mean_abs_error": round(mean_abs_error, 10),
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with guard against constant arrays."""
    if len(x) < 2:
        return 0.0
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
