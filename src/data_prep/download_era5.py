"""Download ERA5 hourly single-levels data for South Africa from the CDS API.

Downloads one NetCDF file per month, organised by year, into
data/raw/era5/<year>/era5_sa_<year>_<month>.nc

Requires a CDS API account and a ~/.cdsapirc file with your API key.
See: https://cds.climate.copernicus.eu/how-to-api
"""

import argparse
import calendar
import tempfile
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr

from src.data_prep.config import AREA, END_YEAR, RAW_DIR, START_YEAR, VARIABLES


def month_target_path(year: int, month: int):
    """Return the raw NetCDF path for a given year/month, creating its folder."""
    year_dir = RAW_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    return year_dir / f"era5_sa_{year}_{month:02d}.nc"


def merge_if_zip(path: Path):
    """CDS sometimes delivers a "netcdf" request as a ZIP of several .nc files
    (instantaneous vs. accumulated variables split apart). Detect that case
    and merge them into the single NetCDF file the rest of the pipeline expects.
    """
    if not zipfile.is_zipfile(path):
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp_dir)
            members = [Path(tmp_dir) / name for name in zf.namelist()]

        merged = xr.merge([xr.open_dataset(member) for member in members])
        path.unlink()
        merged.to_netcdf(path)
        merged.close()


def download_month(client: cdsapi.Client, year: int, month: int):
    """Download one month of ERA5 data, skipping it if already present."""
    target = month_target_path(year, month)
    if target.exists():
        print(f"Skipping {target.name} (already exists)")
        return

    days_in_month = calendar.monthrange(year, month)[1]

    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": VARIABLES,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, days_in_month + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
    }

    print(f"Downloading {target.name} ...")
    client.retrieve("reanalysis-era5-single-levels", request, str(target))
    merge_if_zip(target)


def main(start_year: int, end_year: int):
    client = cdsapi.Client()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            download_month(client, year, month)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    args = parser.parse_args()
    main(args.start_year, args.end_year)
