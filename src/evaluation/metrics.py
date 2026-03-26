"""Evaluation metrics for the recommender system.

Provides both rating-prediction accuracy metrics (RMSE) and ranking-quality
metrics (Precision@K, Recall@K, NDCG@K, MAP@K) as well as beyond-accuracy
metrics (catalog coverage, intra-list diversity).

All functions operate on plain Python / NumPy types and are stateless so they
can be composed freely in evaluation pipelines.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Rating-prediction accuracy
# ------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error for rating prediction.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth ratings.
    y_pred : np.ndarray
        Predicted ratings (same length as *y_true*).

    Returns
    -------
    float
        RMSE value.  Returns ``0.0`` if the inputs are empty.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        return 0.0

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ------------------------------------------------------------------
# Ranking quality
# ------------------------------------------------------------------

def precision_at_k(
    recommended: list[int],
    relevant: list[int],
    k: int,
) -> float:
    """Fraction of top-*k* recommendations that are relevant.

    An item is considered *relevant* if it appears in the test set with a
    rating >= 4.0 (the caller is responsible for this filtering).

    Parameters
    ----------
    recommended : list[int]
        Ordered list of recommended movie IDs (most relevant first).
    relevant : list[int]
        Set of movie IDs the user found relevant in the test period.
    k : int
        Cut-off position.

    Returns
    -------
    float
        Precision@k in [0, 1].  Returns ``0.0`` when *k* is zero or
        *recommended* is empty.
    """
    if k <= 0 or not recommended:
        return 0.0

    top_k = recommended[:k]
    relevant_set = set(relevant)
    n_hits = sum(1 for item in top_k if item in relevant_set)
    return n_hits / k


def recall_at_k(
    recommended: list[int],
    relevant: list[int],
    k: int,
) -> float:
    """Fraction of relevant items that appear in top-*k* recommendations.

    Parameters
    ----------
    recommended : list[int]
        Ordered list of recommended movie IDs.
    relevant : list[int]
        Movie IDs the user found relevant.
    k : int
        Cut-off position.

    Returns
    -------
    float
        Recall@k in [0, 1].  Returns ``0.0`` when there are no relevant
        items or *k* is zero.
    """
    if k <= 0 or not recommended or not relevant:
        return 0.0

    top_k = recommended[:k]
    relevant_set = set(relevant)
    n_hits = sum(1 for item in top_k if item in relevant_set)
    return n_hits / len(relevant_set)


def ndcg_at_k(
    recommended: list[int],
    relevant: list[int],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at *k*.

    Uses binary relevance: an item is relevant (gain = 1) if it is in the
    *relevant* set, otherwise gain = 0.

    Parameters
    ----------
    recommended : list[int]
        Ordered list of recommended movie IDs.
    relevant : list[int]
        Movie IDs the user found relevant.
    k : int
        Cut-off position.

    Returns
    -------
    float
        NDCG@k in [0, 1].  Returns ``0.0`` when there are no relevant
        items or *k* is zero.
    """
    if k <= 0 or not recommended or not relevant:
        return 0.0

    relevant_set = set(relevant)
    top_k = recommended[:k]

    # DCG: sum of 1 / log2(rank + 1) for each hit
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG: the best possible ordering puts all relevant items first
    n_relevant_in_k = min(len(relevant_set), k)
    idcg = 0.0
    for i in range(n_relevant_in_k):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def _average_precision(
    recommended: list[int],
    relevant: list[int],
    k: int,
) -> float:
    """Average Precision at *k* for a single user.

    AP@k is the mean of precision values computed at each position where a
    relevant item appears in the top-*k* recommendations.

    Parameters
    ----------
    recommended : list[int]
        Ordered list of recommended movie IDs.
    relevant : list[int]
        Movie IDs the user found relevant.
    k : int
        Cut-off position.

    Returns
    -------
    float
        AP@k in [0, 1].
    """
    if k <= 0 or not recommended or not relevant:
        return 0.0

    relevant_set = set(relevant)
    top_k = recommended[:k]

    cumulative_hits = 0
    precision_sum = 0.0

    for i, item in enumerate(top_k):
        if item in relevant_set:
            cumulative_hits += 1
            precision_sum += cumulative_hits / (i + 1)

    # Normalise by the total number of relevant items (not just those in
    # the top-k) so that users with many relevant items are not penalised
    # when k is small.
    n_relevant = min(len(relevant_set), k)
    if n_relevant == 0:
        return 0.0

    return float(precision_sum / n_relevant)


def mean_average_precision(
    all_recommended: list[list[int]],
    all_relevant: list[list[int]],
    k: int,
) -> float:
    """Mean Average Precision across all users.

    Parameters
    ----------
    all_recommended : list[list[int]]
        Per-user ordered recommendation lists.
    all_relevant : list[list[int]]
        Per-user relevant-item lists (same ordering as *all_recommended*).
    k : int
        Cut-off position.

    Returns
    -------
    float
        MAP@k in [0, 1].  Returns ``0.0`` if no users are provided.
    """
    if not all_recommended:
        return 0.0

    ap_values = [
        _average_precision(rec, rel, k)
        for rec, rel in zip(all_recommended, all_relevant)
    ]
    return float(np.mean(ap_values))


# ------------------------------------------------------------------
# Beyond-accuracy (diversity & coverage)
# ------------------------------------------------------------------

def catalog_coverage(
    all_recommendations: list[list[int]],
    n_total_movies: int,
) -> float:
    """Fraction of the catalog that appears in any recommendation list.

    Parameters
    ----------
    all_recommendations : list[list[int]]
        Per-user recommendation lists.
    n_total_movies : int
        Total number of movies in the catalog.

    Returns
    -------
    float
        Coverage in [0, 1].  Returns ``0.0`` if the catalog is empty.
    """
    if n_total_movies <= 0:
        return 0.0

    unique_items: set[int] = set()
    for recs in all_recommendations:
        unique_items.update(recs)

    return len(unique_items) / n_total_movies


def intra_list_diversity(
    recommendations: list[int],
    similarity_matrix,
    id_to_idx: dict[int, int],
) -> float:
    """Average dissimilarity between items in a single recommendation list.

    Diversity = 1 - (average pairwise cosine similarity).

    Parameters
    ----------
    recommendations : list[int]
        Movie IDs in the recommendation list.
    similarity_matrix
        Precomputed item-item similarity matrix (dense or sparse).  Rows
        and columns are indexed by the internal matrix indices.
    id_to_idx : dict[int, int]
        Maps original movie IDs to matrix indices.

    Returns
    -------
    float
        Diversity score in [0, 1].  Returns ``0.0`` if the list has fewer
        than two valid items.
    """
    # Map movie IDs to matrix indices, skipping unknowns
    indices = [id_to_idx[mid] for mid in recommendations if mid in id_to_idx]

    if len(indices) < 2:
        return 0.0

    # Compute average pairwise similarity
    total_sim = 0.0
    n_pairs = 0

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_i, idx_j = indices[i], indices[j]
            sim_val = similarity_matrix[idx_i, idx_j]
            # Handle sparse matrix or numpy matrix returns
            if hasattr(sim_val, "item"):
                sim_val = float(sim_val.item())
            elif hasattr(sim_val, "A"):
                sim_val = float(sim_val.A.flat[0])
            else:
                sim_val = float(sim_val)
            total_sim += sim_val
            n_pairs += 1

    if n_pairs == 0:
        return 0.0

    avg_similarity = total_sim / n_pairs
    return float(1.0 - avg_similarity)
