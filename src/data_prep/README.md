# src/data_prep

Loading, imputation, train/val/test splits, and adjacency matrix construction.

Turns `data/raw/` into `data/processed/`. No modeling or training logic here.

## ERA5 pipeline

- `config.py` — shared constants: region, variables, years, split boundaries, folder paths.
- `download_era5.py` — downloads one raw monthly NetCDF file per month from the CDS API into `data/raw/era5/<year>/`.
- `preprocess.py` — converts units (Kelvin to Celsius, Pa to hPa, m to mm) and derives wind speed, wind direction, and relative humidity, saving to `data/processed/era5/<year>/`.
- `split.py` — chronological train (2011-2019) / val (2020-2022) / test (2023-2025) split over the processed files.

Run from the project root, e.g.:

```bash
python -m src.data_prep.download_era5
python -m src.data_prep.preprocess
```
