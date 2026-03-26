"""Download and validate the MovieLens 20M dataset.

This module handles fetching the ML-20M zip archive from GroupLens,
extracting it to *data/raw/*, and validating that critical CSV files
contain the expected number of rows.

Typical usage as a DVC pipeline stage::

    python -m src.data.ingest
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from src.utils.config import load_params, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Expected files and approximate row counts (excluding header).
# These are *lower bounds* -- the real counts can be slightly higher.
# ---------------------------------------------------------------------------
EXPECTED_ROWS: dict[str, int] = {
    "ratings.csv": 19_000_000,
    "movies.csv": 25_000,
    "genome-scores.csv": 11_000_000,
    "genome-tags.csv": 1_000,
    "links.csv": 25_000,
    "tags.csv": 400_000,
}


class _DownloadProgressBar(tqdm):
    """tqdm wrapper that works as a urllib reporthook."""

    def update_to(
        self, blocks: int = 1, block_size: int = 1, total_size: int = -1
    ) -> None:
        if total_size > 0:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a tqdm progress bar."""
    logger.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with _DownloadProgressBar(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=dest.name
    ) as pbar:
        urllib.request.urlretrieve(url, dest, reporthook=pbar.update_to)

    logger.info("Download complete: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


def _count_lines(path: Path) -> int:
    """Return number of data lines (total lines minus header)."""
    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        for _ in fh:
            count += 1
    # Subtract 1 for the header row.
    return max(count - 1, 0)


def _validate_dataset(raw_dir: Path) -> dict[str, Path]:
    """Validate that extracted CSVs have the expected row counts.

    Returns a dict mapping each filename to its absolute ``Path``.

    Raises
    ------
    FileNotFoundError
        If a required CSV is missing.
    ValueError
        If a CSV has fewer rows than the expected minimum.
    """
    file_paths: dict[str, Path] = {}

    for filename, min_rows in EXPECTED_ROWS.items():
        csv_path = raw_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected file not found after extraction: {csv_path}"
            )

        row_count = _count_lines(csv_path)
        if row_count < min_rows:
            raise ValueError(
                f"{filename}: expected >= {min_rows:,} rows, found {row_count:,}"
            )

        logger.info("  %-25s %12s rows  [OK]", filename, f"{row_count:,}")
        file_paths[filename] = csv_path

    return file_paths


def download_movielens(data_dir: Path | None = None) -> dict[str, Path]:
    """Download the MovieLens 20M dataset and validate it.

    Parameters
    ----------
    data_dir : Path, optional
        Root data directory.  Defaults to ``settings.data_raw_dir``.

    Returns
    -------
    dict[str, Path]
        Mapping of CSV filename (e.g. ``"ratings.csv"``) to its path on disk.
    """
    raw_dir: Path = data_dir if data_dir is not None else settings.data_raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "ml-20m.zip"
    extract_marker = raw_dir / "ratings.csv"

    # ------------------------------------------------------------------
    # 1.  Download the zip if the extracted data is not already present.
    # ------------------------------------------------------------------
    if extract_marker.exists():
        logger.info("Dataset already extracted at %s -- skipping download.", raw_dir)
    else:
        if not zip_path.exists():
            _download_with_progress(settings.movielens_url, zip_path)
        else:
            logger.info("Zip already present at %s -- skipping download.", zip_path)

        # --------------------------------------------------------------
        # 2.  Extract.  The archive contains a top-level ``ml-20m/`` dir;
        #     we move the contents up into *raw_dir* directly.
        # --------------------------------------------------------------
        logger.info("Extracting %s ...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

        nested_dir = raw_dir / "ml-20m"
        if nested_dir.is_dir():
            for item in nested_dir.iterdir():
                dest = raw_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            nested_dir.rmdir()

        logger.info("Extraction complete.")

    # ------------------------------------------------------------------
    # 3.  Validate row counts.
    # ------------------------------------------------------------------
    logger.info("Validating dataset at %s ...", raw_dir)
    file_paths = _validate_dataset(raw_dir)

    # Optionally remove the zip to save disk space.
    if zip_path.exists():
        logger.info("Removing zip archive to save disk space.")
        zip_path.unlink()

    logger.info("MovieLens 20M dataset ready at %s", raw_dir)
    return file_paths


# -----------------------------------------------------------------------
# DVC / CLI entry-point
# -----------------------------------------------------------------------
def run_ingest(params: dict | None = None) -> dict[str, Path]:
    """Entry-point for the DVC *ingest* stage."""
    _ = params  # params.yaml not needed here, but kept for pipeline symmetry
    return download_movielens()


if __name__ == "__main__":
    params = load_params()
    run_ingest(params)
