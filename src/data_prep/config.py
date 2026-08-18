"""Shared constants for the ERA5 data pipeline.

Kept in one place so the download, preprocessing, and split scripts
all agree on the region, variables, years, and folder layout.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "era5"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "era5"

# CDS API area filter: [North, West, South, East], covers South Africa.
AREA = [-22, 16, -35, 33]

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

START_YEAR = 2011
END_YEAR = 2025

# Chronological train/val/test split, reused by both preprocessing and training.
SPLIT_YEARS = {
    "train": range(2011, 2020),  # 2011-2019
    "val": range(2020, 2023),    # 2020-2022
    "test": range(2023, 2026),   # 2023-2025
}
