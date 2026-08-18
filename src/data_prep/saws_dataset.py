"""Windowed tensor dataset for LiteSTGNN over the finalized SAWS station
dataset (data/saws_stgnn_dataset/), one weather variable at a time.

Mirrors the shape of src/data_prep/dataset.py's ERA5 pipeline (reuses its
make_windows) but reads the SAWS station .npz files instead of ERA5
NetCDF grids, normalizes per-station rather than with a single global
scalar (see PerStationScaler), and additionally carries the real
observation mask through to the target windows, since SAWS deliberately
preserves missingness rather than treating IDW-imputed cells as ground
truth.

Data policy (see data/saws_stgnn_dataset/data_leakage_prevention.md):
  - Model INPUTS always come from the IDW-imputed values array.
  - Normalization is PER-STATION (PerStationScaler below), fit on the
    TRAIN split's ORIGINALLY OBSERVED values only (mask=True) for each
    station independently - never on imputed fill values, and never on
    val/test.
  - The target mask is carried alongside every window so loss/metrics can
    exclude originally-missing target cells later - this module does not
    apply that mask itself, it only exposes it.
"""

import numpy as np
import pandas as pd
import torch

from src.data_prep.config import PROJECT_ROOT
from src.data_prep.dataset import make_windows

SAWS_DIR = PROJECT_ROOT / "data" / "saws_stgnn_dataset"

BASE_TIMESTAMP = pd.Timestamp("2016-01-01 01:00:00")
N_TOTAL_HOURS = 87672  # 2016-01-01 01:00 -> 2026-01-01 00:00, per the dataset README

SPLIT_TIMESTAMP_FILES = {
    "train": "train_timestamps.csv",
    "val": "validation_timestamps.csv",
    "test": "test_timestamps.csv",
}


class PerStationScaler:
    """Per-station mean/std standardization: x_norm[..., i] = (x[..., i] -
    mean[i]) / std[i]. `mean`/`std` are [n_nodes] arrays, broadcasting
    against any tensor whose last dimension is the station axis
    ([T,N], [n_windows,L,N], ...).

    Fit on train-split, originally-observed values only (see `fit`) -
    never on IDW-imputed fill values, never on val/test.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray, observed: np.ndarray) -> "PerStationScaler":
        """values, observed: [T, n_nodes]. observed=True marks cells to
        include in each station's mean/std - originally-imputed cells
        (observed=False) are excluded from the fit even though `values`
        itself is the IDW-imputed array (imputed values are for model
        input, never for fitting statistics).
        """
        n_nodes = values.shape[1]
        mean = np.zeros(n_nodes, dtype=np.float64)
        std = np.zeros(n_nodes, dtype=np.float64)
        for i in range(n_nodes):
            station_observed = values[:, i][observed[:, i]]
            if station_observed.size == 0:
                raise ValueError(f"station index {i} has zero originally-observed values in the train split")
            mean[i] = station_observed.mean()
            std[i] = station_observed.std() + 1e-6
        return cls(mean, std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def as_dict(self, station_order):
        return {
            climate_number: {"mean": float(self.mean[i]), "std": float(self.std[i])}
            for i, climate_number in enumerate(station_order)
        }


def list_saws_variables():
    """Sorted list of variable names present in the SAWS imputed array,
    derived from the file's own keys (each "{climate_number}|{variable}")
    rather than hardcoded, so it can't drift out of sync with the dataset.
    """
    imputed = np.load(SAWS_DIR / "saws_30station_hourly_values_imputed.npz")
    return sorted({key.split("|", 1)[1] for key in imputed.keys()})


def load_station_order():
    """Deterministic node order: the row order of stations.csv."""
    stations = pd.read_csv(SAWS_DIR / "stations.csv")
    return stations["climate_number"].tolist()


def load_variable_arrays(variable: str, station_order):
    """[time, n_nodes] IDW-imputed values and real-observation mask for one
    variable, in `station_order`. Values come from the *_imputed.npz file;
    the mask comes from the separate mask file (never derived from NaNs in
    the imputed file, since that file has none by construction).
    """
    imputed = np.load(SAWS_DIR / "saws_30station_hourly_values_imputed.npz")
    mask = np.load(SAWS_DIR / "saws_30station_hourly_mask.npz")

    value_cols, mask_cols = [], []
    for climate_number in station_order:
        key = f"{climate_number}|{variable}"
        value_cols.append(imputed[key])
        mask_cols.append(mask[key])

    values = np.stack(value_cols, axis=1).astype(np.float32)  # [time, n_nodes]
    observed = np.stack(mask_cols, axis=1).astype(bool)  # [time, n_nodes]
    return values, observed


def split_row_selector(split_name: str):
    """Boolean [N_TOTAL_HOURS] selector for one split, from the dataset's
    own timestamp CSVs (exact recipe documented in the dataset README).
    """
    date_index = pd.date_range(BASE_TIMESTAMP, periods=N_TOTAL_HOURS, freq="h")
    ts = pd.read_csv(SAWS_DIR / SPLIT_TIMESTAMP_FILES[split_name], parse_dates=["timestamp"])
    return date_index.isin(ts["timestamp"])


class SAWSWindowDataset(torch.utils.data.Dataset):
    """(input, target, target_mask) windows for one variable, one split.

    Unlike src/data_prep/dataset.py's WindowTensorDataset, this also
    carries the target-side observation mask, since SAWS evaluation must
    exclude originally-missing target cells (see module docstring).
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, target_mask: np.ndarray, scaler: PerStationScaler):
        self.inputs = inputs
        self.targets = targets
        self.target_mask = target_mask
        self.scaler = scaler

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.inputs[idx]),
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.target_mask[idx]),
        )


def build_saws_splits(config):
    """(train_ds, val_ds, test_ds, n_nodes, scaler, station_order) for one
    SAWS variable, per `config`.

    Required config keys: "variable", "seq_len", "pred_len".
    Optional: "max_timesteps_per_split" (smoke-test-only truncation of each
    split's raw timeline before windowing - omit/None for the real run).
    """
    variable = config["variable"]
    seq_len = config["seq_len"]
    pred_len = config["pred_len"]
    max_timesteps = config.get("max_timesteps_per_split")

    station_order = load_station_order()
    values, observed = load_variable_arrays(variable, station_order)

    raw_slices = {}
    for split_name in ("train", "val", "test"):
        selector = split_row_selector(split_name)
        split_values = values[selector]
        split_observed = observed[selector]
        if max_timesteps is not None:
            split_values = split_values[:max_timesteps]
            split_observed = split_observed[:max_timesteps]
        raw_slices[split_name] = (split_values, split_observed)

    train_values, train_observed = raw_slices["train"]
    scaler = PerStationScaler.fit(train_values, train_observed)

    datasets = {}
    for split_name, (split_values, split_observed) in raw_slices.items():
        normalized = scaler.transform(split_values)
        inputs, targets = make_windows(normalized, seq_len, pred_len)
        # Reuse make_windows on the mask too, so window indices match
        # exactly; only the target-side windows are needed downstream.
        _, target_mask = make_windows(split_observed.astype(np.float32), seq_len, pred_len)
        target_mask = target_mask > 0.5
        datasets[split_name] = SAWSWindowDataset(inputs, targets, target_mask, scaler)

    n_nodes = len(station_order)
    return datasets["train"], datasets["val"], datasets["test"], n_nodes, scaler, station_order
