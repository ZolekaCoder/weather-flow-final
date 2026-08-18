"""Convert raw monthly ERA5 files into cleaned, derived-variable NetCDF files.

For each file in data/raw/era5/<year>/, this:
  - converts temperature and dewpoint from Kelvin to Celsius
  - converts surface pressure from Pa to hPa
  - converts total precipitation from m to mm
  - derives wind speed and wind direction from the u/v wind components
  - derives relative humidity from temperature and dewpoint

Output mirrors the raw folder layout under data/processed/era5/<year>/.
"""

import numpy as np
import xarray as xr

from src.data_prep.config import PROCESSED_DIR, RAW_DIR


def relative_humidity(temperature_c, dewpoint_c):
    """Relative humidity (%) from temperature and dewpoint, via Magnus-Tetens."""
    vapour_pressure = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
    saturation_vapour_pressure = 6.112 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))
    return 100 * (vapour_pressure / saturation_vapour_pressure)


def wind_speed(u, v):
    """Wind speed (m/s) from the u and v wind components."""
    return np.sqrt(u**2 + v**2)


def wind_direction(u, v):
    """Meteorological wind direction (degrees, direction wind blows FROM)."""
    return (180 + np.degrees(np.arctan2(u, v))) % 360


def convert_units(raw: xr.Dataset) -> xr.Dataset:
    """Build a processed dataset with converted units and derived variables."""
    temperature_c = raw["t2m"] - 273.15
    dewpoint_c = raw["d2m"] - 273.15

    processed = xr.Dataset(coords=raw.coords)
    processed["temperature_c"] = temperature_c.assign_attrs(units="degC")
    processed["dewpoint_c"] = dewpoint_c.assign_attrs(units="degC")
    processed["pressure_hpa"] = (raw["sp"] / 100).assign_attrs(units="hPa")
    processed["precipitation_mm"] = (raw["tp"] * 1000).assign_attrs(units="mm")
    processed["wind_speed"] = wind_speed(raw["u10"], raw["v10"]).assign_attrs(units="m/s")
    processed["wind_direction"] = wind_direction(raw["u10"], raw["v10"]).assign_attrs(units="degrees")
    processed["relative_humidity"] = relative_humidity(temperature_c, dewpoint_c).assign_attrs(units="%")

    return processed


def process_file(raw_path, processed_path):
    """Load one raw monthly file, convert it, and save it to processed_path."""
    raw = xr.open_dataset(raw_path)
    processed = convert_units(raw)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_netcdf(processed_path)
    raw.close()


def main():
    raw_files = sorted(RAW_DIR.glob("*/*.nc"))
    for raw_path in raw_files:
        processed_path = PROCESSED_DIR / raw_path.parent.name / raw_path.name
        if processed_path.exists():
            print(f"Skipping {processed_path.name} (already exists)")
            continue
        print(f"Processing {raw_path.name} ...")
        process_file(raw_path, processed_path)


if __name__ == "__main__":
    main()
