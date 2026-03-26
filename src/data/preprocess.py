"""Clean and prepare raw MovieLens 20M data for modelling.

Pipeline stage responsibilities:
1. Load raw CSVs and filter sparse users/movies.
2. Parse genre strings into clean multi-hot lists.
3. Merge movie metadata (movies + links + tags, extract year).
4. Create a temporal train/test split.
5. Persist cleaned artefacts to *data/processed/*.

Typical usage as a DVC pipeline stage::

    python -m src.data.preprocess
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_params, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Year pattern: matches "(1995)" or "(2003)" at the end of a movie title
_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


# -----------------------------------------------------------------------
# 1. Ratings cleaning
# -----------------------------------------------------------------------
def load_and_clean_ratings(
    raw_dir: Path,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 50,
) -> pd.DataFrame:
    """Load *ratings.csv* and filter out sparse users and movies.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the raw CSV files.
    min_user_ratings : int
        Minimum number of ratings a user must have to be kept.
    min_movie_ratings : int
        Minimum number of ratings a movie must have to be kept.

    Returns
    -------
    pd.DataFrame
        Filtered ratings with columns ``[userId, movieId, rating, timestamp]``.
    """
    ratings_path = raw_dir / "ratings.csv"
    logger.info("Loading ratings from %s", ratings_path)

    ratings_df = pd.read_csv(
        ratings_path,
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
        parse_dates=False,
    )
    initial_rows = len(ratings_df)
    logger.info("  Raw ratings: %s rows", f"{initial_rows:,}")

    # --- Iterative filtering (repeat until stable) ---
    prev_rows = -1
    iteration = 0
    while len(ratings_df) != prev_rows:
        prev_rows = len(ratings_df)
        iteration += 1

        # Users with enough ratings
        user_counts = ratings_df["userId"].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        ratings_df = ratings_df[ratings_df["userId"].isin(valid_users)]

        # Movies with enough ratings
        movie_counts = ratings_df["movieId"].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movies)]

        logger.info(
            "  Filter pass %d: %s rows (removed %s)",
            iteration,
            f"{len(ratings_df):,}",
            f"{prev_rows - len(ratings_df):,}",
        )

    removed = initial_rows - len(ratings_df)
    logger.info(
        "  Filtering complete: kept %s / %s rows (removed %s)",
        f"{len(ratings_df):,}",
        f"{initial_rows:,}",
        f"{removed:,}",
    )

    # Sort chronologically
    ratings_df = ratings_df.sort_values("timestamp").reset_index(drop=True)
    return ratings_df


# -----------------------------------------------------------------------
# 2. Genre parsing
# -----------------------------------------------------------------------
def parse_genres(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Split pipe-separated genres and handle ``(no genres listed)``.

    Adds a ``genres_list`` column (list[str]) and replaces the raw
    ``genres`` column with a cleaned pipe-joined string.

    Parameters
    ----------
    movies_df : pd.DataFrame
        Must contain a ``genres`` column.

    Returns
    -------
    pd.DataFrame
        Copy of the input with ``genres`` cleaned and ``genres_list`` added.
    """
    df = movies_df.copy()

    def _split(g: str) -> list[str]:
        if not isinstance(g, str) or g.strip() == "(no genres listed)":
            return []
        return [genre.strip() for genre in g.split("|") if genre.strip()]

    df["genres_list"] = df["genres"].apply(_split)
    df["genres"] = df["genres_list"].apply(lambda gs: "|".join(gs) if gs else "")
    return df


# -----------------------------------------------------------------------
# 3. Metadata merging
# -----------------------------------------------------------------------
def merge_movie_metadata(
    movies_df: pd.DataFrame,
    links_df: pd.DataFrame,
    tags_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join movies with links and aggregated tags; extract year from title.

    Parameters
    ----------
    movies_df : pd.DataFrame
        Must contain ``movieId``, ``title``, ``genres``.
    links_df : pd.DataFrame
        Must contain ``movieId``, ``imdbId``, ``tmdbId``.
    tags_df : pd.DataFrame
        Must contain ``movieId``, ``tag``.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with extra columns ``year``, ``clean_title``,
        ``imdbId``, ``tmdbId``, ``tag_str``.
    """
    df = movies_df.copy()

    # --- Extract year from title ---
    def _extract_year(title: str) -> int | None:
        m = _YEAR_RE.search(str(title))
        if m:
            return int(m.group(1))
        return None

    df["year"] = df["title"].apply(_extract_year)
    df["clean_title"] = df["title"].apply(
        lambda t: _YEAR_RE.sub("", str(t)).strip()
    )

    # --- Merge links ---
    links_clean = links_df[["movieId", "imdbId", "tmdbId"]].copy()
    # tmdbId can have NaN -- keep as nullable int
    links_clean["imdbId"] = links_clean["imdbId"].astype("Int64")
    links_clean["tmdbId"] = links_clean["tmdbId"].astype("Int64")
    df = df.merge(links_clean, on="movieId", how="left")

    # --- Aggregate tags per movie ---
    if tags_df is not None and len(tags_df) > 0:
        tag_agg = (
            tags_df.dropna(subset=["tag"])
            .groupby("movieId")["tag"]
            .apply(lambda ts: " | ".join(ts.astype(str).unique()))
            .reset_index()
            .rename(columns={"tag": "tag_str"})
        )
        df = df.merge(tag_agg, on="movieId", how="left")
    else:
        df["tag_str"] = ""

    df["tag_str"] = df["tag_str"].fillna("")
    return df


# -----------------------------------------------------------------------
# 4. Temporal train/test split
# -----------------------------------------------------------------------
def temporal_train_test_split(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings chronologically: last *test_ratio* fraction as test.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Must contain a ``timestamp`` column. Assumed already sorted.
    test_ratio : float
        Fraction of rows (by time order) to hold out for testing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)``
    """
    sorted_df = ratings_df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1.0 - test_ratio))

    train_df = sorted_df.iloc[:split_idx].reset_index(drop=True)
    test_df = sorted_df.iloc[split_idx:].reset_index(drop=True)

    logger.info(
        "Temporal split: train=%s, test=%s (ratio=%.2f)",
        f"{len(train_df):,}",
        f"{len(test_df):,}",
        test_ratio,
    )
    return train_df, test_df


# -----------------------------------------------------------------------
# 5. Orchestrator
# -----------------------------------------------------------------------
def run_preprocessing(params: dict) -> None:
    """Execute the full preprocessing pipeline and persist artefacts.

    Parameters
    ----------
    params : dict
        Contents of *params.yaml*.
    """
    raw_dir: Path = settings.data_raw_dir
    processed_dir: Path = settings.data_processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    data_params = params.get("data", {})
    test_ratio = data_params.get("test_ratio", 0.2)
    min_user_ratings = data_params.get("min_user_ratings", 20)
    min_movie_ratings = data_params.get("min_movie_ratings", 50)

    # --- 1. Ratings ---
    logger.info("=" * 60)
    logger.info("STEP 1/4: Load and clean ratings")
    logger.info("=" * 60)
    ratings_df = load_and_clean_ratings(
        raw_dir,
        min_user_ratings=min_user_ratings,
        min_movie_ratings=min_movie_ratings,
    )

    # --- 2. Movies / genres ---
    logger.info("=" * 60)
    logger.info("STEP 2/4: Parse genres and merge metadata")
    logger.info("=" * 60)
    movies_df = pd.read_csv(raw_dir / "movies.csv")
    links_df = pd.read_csv(raw_dir / "links.csv")
    tags_df = pd.read_csv(raw_dir / "tags.csv")

    movies_df = parse_genres(movies_df)
    movie_metadata = merge_movie_metadata(movies_df, links_df, tags_df)

    # Keep only movies that survived the ratings filter
    surviving_movies = set(ratings_df["movieId"].unique())
    movie_metadata = movie_metadata[
        movie_metadata["movieId"].isin(surviving_movies)
    ].reset_index(drop=True)
    logger.info(
        "  Movie metadata after filtering: %s movies", f"{len(movie_metadata):,}"
    )

    # --- 3. Temporal split ---
    logger.info("=" * 60)
    logger.info("STEP 3/4: Temporal train/test split")
    logger.info("=" * 60)
    train_df, test_df = temporal_train_test_split(ratings_df, test_ratio=test_ratio)

    # --- 4. Save ---
    logger.info("=" * 60)
    logger.info("STEP 4/4: Saving processed artefacts")
    logger.info("=" * 60)

    train_path = processed_dir / "train_ratings.csv"
    test_path = processed_dir / "test_ratings.csv"
    ratings_path = processed_dir / "ratings_clean.csv"
    metadata_path = processed_dir / "movie_metadata.csv"

    train_df.to_csv(train_path, index=False)
    logger.info("  Saved %s (%s rows)", train_path.name, f"{len(train_df):,}")

    test_df.to_csv(test_path, index=False)
    logger.info("  Saved %s (%s rows)", test_path.name, f"{len(test_df):,}")

    ratings_df.to_csv(ratings_path, index=False)
    logger.info("  Saved %s (%s rows)", ratings_path.name, f"{len(ratings_df):,}")

    movie_metadata.to_csv(metadata_path, index=False)
    logger.info("  Saved %s (%s rows)", metadata_path.name, f"{len(movie_metadata):,}")

    logger.info("Preprocessing complete.")


# -----------------------------------------------------------------------
# DVC / CLI entry-point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    params = load_params()
    run_preprocessing(params)
