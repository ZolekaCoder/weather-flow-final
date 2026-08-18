"""Chronological train/val/test split over the processed ERA5 files.

Split boundaries are defined once in config.SPLIT_YEARS and reused here,
so training code can simply call list_split_files("train") and get a
ready-to-use list of monthly NetCDF files.
"""

from src.data_prep.config import PROCESSED_DIR, SPLIT_YEARS


def get_split(year: int) -> str:
    """Return which split ("train", "val", or "test") a given year belongs to."""
    for split_name, years in SPLIT_YEARS.items():
        if year in years:
            return split_name
    raise ValueError(f"Year {year} is not covered by any split")


def list_split_files(split_name: str, processed_dir=PROCESSED_DIR):
    """Return sorted processed NetCDF file paths for a split ("train"/"val"/"test")."""
    years = SPLIT_YEARS[split_name]
    files = []
    for year in years:
        year_dir = processed_dir / str(year)
        if year_dir.exists():
            files.extend(sorted(year_dir.glob("*.nc")))
    return files
