"""Build feature matrices for content-based and collaborative filtering.

This module produces:

* **User-item interaction matrix** (sparse CSR) for collaborative filtering.
* **TF-IDF genre features** (sparse) from movie metadata.
* **Genome tag features** (dense float32) from ``genome-scores.csv``.
* **Combined content features** (sparse, L2-normalised).

All artefacts are saved to *data/processed/* as ``.npz`` / ``.json`` files.

Typical usage as a DVC pipeline stage::

    python -m src.data.feature_engineering
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.utils.config import load_params, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------
# 1. User-item interaction matrix
# -----------------------------------------------------------------------
def build_user_item_matrix(
    ratings_df: pd.DataFrame,
) -> tuple[sp.csr_matrix, dict[int, int], dict[int, int]]:
    """Build a sparse user-item rating matrix from a ratings dataframe.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Must contain ``userId``, ``movieId``, ``rating``.

    Returns
    -------
    matrix : scipy.sparse.csr_matrix
        Shape ``(n_users, n_movies)`` with ratings as values.
    user_id_to_idx : dict[int, int]
        Maps original ``userId`` to row index.
    movie_id_to_idx : dict[int, int]
        Maps original ``movieId`` to column index.
    """
    user_ids = sorted(ratings_df["userId"].unique())
    movie_ids = sorted(ratings_df["movieId"].unique())

    user_id_to_idx: dict[int, int] = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_id_to_idx: dict[int, int] = {mid: idx for idx, mid in enumerate(movie_ids)}

    row_indices = ratings_df["userId"].map(user_id_to_idx).values
    col_indices = ratings_df["movieId"].map(movie_id_to_idx).values
    data = ratings_df["rating"].values.astype(np.float32)

    matrix = sp.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32,
    )

    logger.info(
        "User-item matrix: %s users x %s movies, %s non-zero entries (density=%.4f%%)",
        f"{matrix.shape[0]:,}",
        f"{matrix.shape[1]:,}",
        f"{matrix.nnz:,}",
        100.0 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
    )

    return matrix, user_id_to_idx, movie_id_to_idx


# -----------------------------------------------------------------------
# 2. TF-IDF genre features
# -----------------------------------------------------------------------
def build_genre_tfidf_features(
    movie_metadata: pd.DataFrame,
    max_features: int = 5000,
) -> tuple[sp.csr_matrix, list[int]]:
    """Build TF-IDF features from movie genre and tag text.

    The text document for each movie is a concatenation of its genres
    (pipe-separated, turned into space-separated) and its aggregated tags.

    Parameters
    ----------
    movie_metadata : pd.DataFrame
        Must contain ``movieId``, ``genres``, ``tag_str`` columns.
    max_features : int
        Maximum vocabulary size for the TF-IDF vectoriser.

    Returns
    -------
    tfidf_matrix : scipy.sparse.csr_matrix
        Shape ``(n_movies, n_features)``.
    movie_ids : list[int]
        Ordered list of ``movieId`` values corresponding to rows.
    """
    df = movie_metadata.copy()
    df = df.sort_values("movieId").reset_index(drop=True)

    # Build text corpus: genres (space-separated) + tag string
    df["genres_text"] = df["genres"].fillna("").str.replace("|", " ", regex=False)
    df["tag_str"] = df["tag_str"].fillna("")
    corpus = (df["genres_text"] + " " + df["tag_str"]).tolist()

    vectoriser = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b",
    )
    tfidf_matrix = vectoriser.fit_transform(corpus)

    movie_ids = df["movieId"].tolist()

    logger.info(
        "TF-IDF genre features: %s movies x %s features",
        f"{tfidf_matrix.shape[0]:,}",
        f"{tfidf_matrix.shape[1]:,}",
    )

    return tfidf_matrix, movie_ids


# -----------------------------------------------------------------------
# 3. Genome tag features
# -----------------------------------------------------------------------
def build_genome_feature_matrix(
    genome_scores_path: Path,
    movie_ids: list[int],
    chunk_size: int = 500_000,
) -> np.ndarray:
    """Build a dense genome-tag relevance matrix from ``genome-scores.csv``.

    The file is ~260 MB so it is read in chunks to limit peak memory.

    Parameters
    ----------
    genome_scores_path : Path
        Path to ``genome-scores.csv``.
    movie_ids : list[int]
        Ordered list of movie IDs that define row order. Movies without
        genome scores receive an all-zeros row.
    chunk_size : int
        Number of CSV rows per chunk.

    Returns
    -------
    np.ndarray
        Shape ``(len(movie_ids), n_tags)`` with dtype ``float32``.
        ``n_tags`` is typically 1128 for ML-20M.
    """
    logger.info("Building genome feature matrix from %s ...", genome_scores_path)

    movie_id_set = set(movie_ids)
    movie_id_to_row: dict[int, int] = {mid: idx for idx, mid in enumerate(movie_ids)}

    # --- First pass: discover the tag IDs ---
    tag_ids: set[int] = set()
    for chunk in pd.read_csv(
        genome_scores_path,
        dtype={"movieId": np.int32, "tagId": np.int32, "relevance": np.float32},
        chunksize=chunk_size,
    ):
        tag_ids.update(chunk["tagId"].unique())

    sorted_tag_ids = sorted(tag_ids)
    tag_id_to_col: dict[int, int] = {tid: idx for idx, tid in enumerate(sorted_tag_ids)}
    n_tags = len(sorted_tag_ids)

    logger.info("  Genome tags discovered: %d", n_tags)

    # --- Second pass: populate the matrix ---
    matrix = np.zeros((len(movie_ids), n_tags), dtype=np.float32)
    rows_filled = 0

    total_chunks = None
    try:
        # Estimate total rows for the progress bar
        import csv as _csv

        with open(genome_scores_path, "r", encoding="utf-8") as _fh:
            # Count lines without loading everything into memory
            total_lines = sum(1 for _ in _fh) - 1  # exclude header
        total_chunks = (total_lines + chunk_size - 1) // chunk_size
    except Exception:
        pass

    for chunk in tqdm(
        pd.read_csv(
            genome_scores_path,
            dtype={"movieId": np.int32, "tagId": np.int32, "relevance": np.float32},
            chunksize=chunk_size,
        ),
        desc="Genome scores",
        total=total_chunks,
    ):
        # Keep only movies in our filtered set
        mask = chunk["movieId"].isin(movie_id_set)
        filtered = chunk.loc[mask]
        if filtered.empty:
            continue

        rows = filtered["movieId"].map(movie_id_to_row).values
        cols = filtered["tagId"].map(tag_id_to_col).values
        vals = filtered["relevance"].values

        # Vectorised assignment
        matrix[rows, cols] = vals
        rows_filled += len(filtered)

    movies_with_scores = int((matrix.sum(axis=1) > 0).sum())
    logger.info(
        "  Genome matrix: %s movies x %d tags  (%d movies have scores)",
        f"{matrix.shape[0]:,}",
        n_tags,
        movies_with_scores,
    )

    return matrix


# -----------------------------------------------------------------------
# 4. Combined content features
# -----------------------------------------------------------------------
def build_content_features(
    tfidf_matrix: sp.spmatrix,
    genome_matrix: np.ndarray,
    genre_weight: float = 0.3,
    genome_weight: float = 0.7,
) -> sp.csr_matrix:
    """Combine TF-IDF and genome features into a single L2-normalised matrix.

    Parameters
    ----------
    tfidf_matrix : sparse matrix
        Shape ``(n_movies, n_tfidf_features)``.
    genome_matrix : np.ndarray
        Shape ``(n_movies, n_genome_tags)``.
    genre_weight : float
        Weight applied to the TF-IDF block.
    genome_weight : float
        Weight applied to the genome block.

    Returns
    -------
    scipy.sparse.csr_matrix
        L2-row-normalised combined feature matrix.
    """
    assert tfidf_matrix.shape[0] == genome_matrix.shape[0], (
        f"Row count mismatch: TF-IDF has {tfidf_matrix.shape[0]}, "
        f"genome has {genome_matrix.shape[0]}"
    )

    # Normalise each block individually before weighting
    tfidf_normed = normalize(tfidf_matrix, norm="l2", axis=1)
    genome_sparse = sp.csr_matrix(genome_matrix)
    genome_normed = normalize(genome_sparse, norm="l2", axis=1)

    # Weighted horizontal stack
    combined = sp.hstack(
        [tfidf_normed * genre_weight, genome_normed * genome_weight],
        format="csr",
    )

    # Final L2 normalisation
    combined = normalize(combined, norm="l2", axis=1)

    logger.info(
        "Combined content features: %s movies x %s features",
        f"{combined.shape[0]:,}",
        f"{combined.shape[1]:,}",
    )

    return combined


# -----------------------------------------------------------------------
# 5. Orchestrator
# -----------------------------------------------------------------------
def run_feature_engineering(params: dict) -> None:
    """Execute the full feature engineering pipeline.

    Reads cleaned artefacts from *data/processed/* (produced by the
    preprocessing stage), builds feature matrices, and saves them.

    Parameters
    ----------
    params : dict
        Contents of *params.yaml*.
    """
    processed_dir: Path = settings.data_processed_dir
    raw_dir: Path = settings.data_raw_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    content_params = params.get("content_based", {})
    max_features = content_params.get("tfidf_max_features", 5000)
    genre_weight = content_params.get("genre_weight", 0.3)
    genome_weight = content_params.get("genome_weight", 0.7)

    # ------------------------------------------------------------------
    # Load preprocessed data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading preprocessed artefacts")
    logger.info("=" * 60)

    train_df = pd.read_csv(
        processed_dir / "train_ratings.csv",
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
    )
    movie_metadata = pd.read_csv(processed_dir / "movie_metadata.csv")

    logger.info(
        "  Train ratings: %s rows | Movie metadata: %s rows",
        f"{len(train_df):,}",
        f"{len(movie_metadata):,}",
    )

    # ------------------------------------------------------------------
    # 1. User-item matrix (from training data only)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1/4: Build user-item interaction matrix")
    logger.info("=" * 60)

    ui_matrix, user_id_to_idx, movie_id_to_idx = build_user_item_matrix(train_df)

    sp.save_npz(processed_dir / "user_item_matrix.npz", ui_matrix)
    _save_json(processed_dir / "user_id_to_idx.json", user_id_to_idx)
    _save_json(processed_dir / "movie_id_to_idx.json", movie_id_to_idx)
    logger.info("  Saved user_item_matrix.npz, user/movie id maps.")

    # ------------------------------------------------------------------
    # 2. TF-IDF genre features
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2/4: Build TF-IDF genre features")
    logger.info("=" * 60)

    tfidf_matrix, tfidf_movie_ids = build_genre_tfidf_features(
        movie_metadata, max_features=max_features
    )

    sp.save_npz(processed_dir / "tfidf_matrix.npz", tfidf_matrix)
    _save_json(processed_dir / "tfidf_movie_ids.json", tfidf_movie_ids)
    logger.info("  Saved tfidf_matrix.npz")

    # ------------------------------------------------------------------
    # 3. Genome features
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3/4: Build genome feature matrix")
    logger.info("=" * 60)

    genome_scores_path = raw_dir / "genome-scores.csv"
    if genome_scores_path.exists():
        genome_matrix = build_genome_feature_matrix(
            genome_scores_path, tfidf_movie_ids
        )
        np.save(processed_dir / "genome_matrix.npy", genome_matrix)
        logger.info("  Saved genome_matrix.npy")
    else:
        logger.warning(
            "  genome-scores.csv not found at %s -- skipping genome features.",
            genome_scores_path,
        )
        genome_matrix = np.zeros(
            (len(tfidf_movie_ids), 1128), dtype=np.float32
        )
        np.save(processed_dir / "genome_matrix.npy", genome_matrix)
        logger.info("  Saved zero-filled genome_matrix.npy (fallback)")

    # ------------------------------------------------------------------
    # 4. Combined content features
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4/4: Build combined content features")
    logger.info("=" * 60)

    content_matrix = build_content_features(
        tfidf_matrix, genome_matrix,
        genre_weight=genre_weight,
        genome_weight=genome_weight,
    )

    sp.save_npz(processed_dir / "content_features.npz", content_matrix)

    # Also save the movie-id-to-content-row mapping (same order as tfidf)
    content_movie_id_to_idx = {mid: idx for idx, mid in enumerate(tfidf_movie_ids)}
    _save_json(processed_dir / "content_movie_id_to_idx.json", content_movie_id_to_idx)
    logger.info("  Saved content_features.npz, content_movie_id_to_idx.json")

    logger.info("Feature engineering complete.")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _save_json(path: Path, obj: dict | list) -> None:
    """Serialise *obj* to a JSON file, converting int keys to strings."""
    # JSON keys must be strings
    if isinstance(obj, dict):
        obj = {str(k): v for k, v in obj.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


# -----------------------------------------------------------------------
# DVC / CLI entry-point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    params = load_params()
    run_feature_engineering(params)
