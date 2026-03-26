"""Temporal train/test splitting for time-aware evaluation.

This module provides a standalone, reusable temporal split function that
can be called from evaluation code or imported by other pipeline stages.

Typical usage as a DVC pipeline stage::

    python -m src.evaluation.temporal_split
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import load_params, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def temporal_split(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a ratings dataframe chronologically.

    Rows are sorted by ``timestamp`` and the last *test_ratio* fraction
    of rows is assigned to the test set.  This mirrors real-world usage
    where a recommender system must predict future interactions based on
    past data.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Must contain at least a ``timestamp`` column. Other expected
        columns are ``userId``, ``movieId``, ``rating``.
    test_ratio : float
        Fraction of rows (by chronological order) to hold out as the
        test set.  Must be in ``(0, 1)``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` -- both reset-indexed.

    Raises
    ------
    ValueError
        If *test_ratio* is not in the open interval ``(0, 1)``.
    KeyError
        If ``timestamp`` column is missing.
    """
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(
            f"test_ratio must be in (0, 1), got {test_ratio}"
        )

    if "timestamp" not in ratings_df.columns:
        raise KeyError(
            "ratings_df must contain a 'timestamp' column for temporal splitting"
        )

    sorted_df = ratings_df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1.0 - test_ratio))

    train_df = sorted_df.iloc[:split_idx].reset_index(drop=True)
    test_df = sorted_df.iloc[split_idx:].reset_index(drop=True)

    # Log summary statistics
    if "timestamp" in train_df.columns and len(train_df) > 0:
        train_min_ts = train_df["timestamp"].min()
        train_max_ts = train_df["timestamp"].max()
        test_min_ts = test_df["timestamp"].min()
        test_max_ts = test_df["timestamp"].max()

        logger.info(
            "Temporal split (ratio=%.2f): "
            "train=%s rows [ts %s..%s], test=%s rows [ts %s..%s]",
            test_ratio,
            f"{len(train_df):,}",
            train_min_ts,
            train_max_ts,
            f"{len(test_df):,}",
            test_min_ts,
            test_max_ts,
        )
    else:
        logger.info(
            "Temporal split (ratio=%.2f): train=%s rows, test=%s rows",
            test_ratio,
            f"{len(train_df):,}",
            f"{len(test_df):,}",
        )

    # Report user/movie overlap
    if {"userId", "movieId"}.issubset(train_df.columns):
        train_users = set(train_df["userId"].unique())
        test_users = set(test_df["userId"].unique())
        cold_start_users = test_users - train_users

        train_movies = set(train_df["movieId"].unique())
        test_movies = set(test_df["movieId"].unique())
        cold_start_movies = test_movies - train_movies

        logger.info(
            "  Cold-start users in test: %d / %d (%.1f%%)",
            len(cold_start_users),
            len(test_users),
            100.0 * len(cold_start_users) / max(len(test_users), 1),
        )
        logger.info(
            "  Cold-start movies in test: %d / %d (%.1f%%)",
            len(cold_start_movies),
            len(test_movies),
            100.0 * len(cold_start_movies) / max(len(test_movies), 1),
        )

    return train_df, test_df


def run_temporal_split(params: dict) -> None:
    """Execute temporal splitting as a standalone pipeline stage.

    Reads ``ratings_clean.csv`` from *data/processed/*, splits it, and
    writes ``train_ratings.csv`` and ``test_ratings.csv``.

    Parameters
    ----------
    params : dict
        Contents of *params.yaml*.
    """
    processed_dir: Path = settings.data_processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    data_params = params.get("data", {})
    test_ratio = data_params.get("test_ratio", 0.2)

    ratings_path = processed_dir / "ratings_clean.csv"
    logger.info("Loading cleaned ratings from %s", ratings_path)
    ratings_df = pd.read_csv(ratings_path)

    train_df, test_df = temporal_split(ratings_df, test_ratio=test_ratio)

    train_path = processed_dir / "train_ratings.csv"
    test_path = processed_dir / "test_ratings.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved %s and %s", train_path.name, test_path.name)


# -----------------------------------------------------------------------
# DVC / CLI entry-point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    params = load_params()
    run_temporal_split(params)
